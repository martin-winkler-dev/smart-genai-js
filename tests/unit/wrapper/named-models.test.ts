// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'bun:test';
import { ThinkingLevel } from '@google/genai';
import { type Bucket, type Cost, createRPM } from 'smart-limit';
import { batchTiming } from '../../../src/core/batch/calls.js';
import { createGenaiClient } from '../../../src/core/client.js';
import { assistantMessage, toolMessage } from '../../../src/core/messages.js';
import type { GenaiClientOptions, GenaiMessage, GenaiModel, GenaiModelInput } from '../../../src/types.js';
import { createGenaiModels } from '../../../src/wrapper/named-models.js';
import {
    apiError,
    batchResponse,
    decisionError,
    decisionsReply,
    type GeminiReply,
    type GeminiRequest,
    generateBody,
    generateResponse,
    sentBatch,
    sentConfig,
    startFakeGemini,
} from '../fake-gemini.js';
import { fakeAuthClient, startFakeVertex } from '../fake-vertex.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const server = startFakeGemini();
const vertexServer = startFakeVertex();

afterAll(() => {
    server.stop();
    vertexServer.stop();
});
beforeEach(() => {
    server.received.length = 0;
    server.reply(() => generateResponse('ok'));
});

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates client options configured for fake server. */
function clientOptions(extra: Partial<GenaiClientOptions> = {}): GenaiClientOptions {
    return { apiKey: 'test-key', baseUrl: server.baseUrl, env: {}, ...extra };
}

describe('client defaults', () => {
    it('fill the request and config; the call wins per field, thinkingConfig per member', async () => {
        const client = createGenaiClient(
            clientOptions({
                defaults: {
                    model: 'gemini-3-x',
                    temperature: 0.5,
                    reasoning_effort: 'low',
                    web_search: true,
                    config: { topK: 3, thinkingConfig: { includeThoughts: true } },
                },
            }),
        );
        await client.generate({ messages: [{ role: 'user', content: 'hi' }], temperature: 0 });
        const first = server.received[0];
        expect(first?.path).toBe('/v1beta/models/gemini-3-x:generateContent');
        expect(first?.body?.generationConfig).toEqual({
            temperature: 0,
            topK: 3,
            thinkingConfig: { thinkingLevel: 'LOW', includeThoughts: true },
        });
        expect(first?.body?.tools).toEqual([{ googleSearch: {} }]);
        await client.generate(
            { model: 'gemini-2.5-y', messages: [{ role: 'user', content: 'hi' }], web_search: false },
            { config: { thinkingConfig: { includeThoughts: false } } },
        );
        const second = server.received[1];
        expect(second?.path).toBe('/v1beta/models/gemini-2.5-y:generateContent');
        expect(sentConfig(second).thinkingConfig).toEqual({ thinkingBudget: 1024, includeThoughts: false });
        expect(second?.body?.tools).toBeUndefined();
    });

    it('report a copy of the config in the settings, so changing it changes no later request', async () => {
        const config = { topK: 3, thinkingConfig: { includeThoughts: true } };
        const client = createGenaiClient(clientOptions({ defaults: { model: 'gemini-3-x', config } }));
        const first = await client.generate({ messages: [{ role: 'user', content: 'hi' }] });
        expect(first.ok && first.data.settings?.config).toEqual(config);
        const settings = first.ok ? first.data.settings?.config : undefined;
        if (settings?.thinkingConfig !== undefined) {
            settings.topK = 99;
            settings.thinkingConfig.includeThoughts = false;
        }
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { flash: { model: 'm', defaults: { config } } },
        });
        const named = await gemini.flash('hi');
        const own = named.ok ? named.data.settings?.config : undefined;
        if (own !== undefined) {
            own.topK = 77;
        }
        await client.generate({ messages: [{ role: 'user', content: 'hi' }] });
        await gemini.flash('hi');
        expect(config).toEqual({ topK: 3, thinkingConfig: { includeThoughts: true } });
        expect(server.received[2]?.body?.generationConfig).toEqual({
            topK: 3,
            thinkingConfig: { includeThoughts: true },
        });
        expect(server.received[3]?.body?.generationConfig).toEqual({
            topK: 3,
            thinkingConfig: { includeThoughts: true },
        });
    });

    it('keep copies of defaults and report copies in settings, all levels deep', async () => {
        const stop = ['END'];
        const safety = { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' };
        const defaults = { model: 'gemini-3-x', stop, config: { safetySettings: [safety] } };
        const client = createGenaiClient(clientOptions({ defaults } as never));
        const modelDefaults = { stop: ['STOP'], config: { safetySettings: [{ ...safety }] } };
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { flash: { model: 'm', defaults: modelDefaults as never } },
        });
        const first = await client.generate({ messages: [{ role: 'user', content: 'hi' }] });
        const named = await gemini.flash('hi');
        // change everything a caller could reach: the defaults objects and the settings of both results
        stop.push('LEAK');
        safety.threshold = 'BLOCK_LOW_AND_ABOVE';
        modelDefaults.stop.push('LEAK');
        modelDefaults.config.safetySettings.push({ ...safety });
        for (const result of [first, named]) {
            if (!result.ok) {
                throw new Error('expected an answer');
            }
            const settings = result.data.settings as {
                stop: string[];
                config: { safetySettings: { threshold: string }[] };
            };
            settings.stop.push('LEAK');
            (settings.config.safetySettings[0] as { threshold: string }).threshold = 'BLOCK_LOW_AND_ABOVE';
        }
        const again = await client.generate({ messages: [{ role: 'user', content: 'hi' }] });
        const namedAgain = await gemini.flash('hi');
        expect(again.ok && again.data.settings).toMatchObject({ stop: ['END'] });
        expect(namedAgain.ok && namedAgain.data.settings).toMatchObject({ stop: ['STOP'] });
        for (const sent of server.received.slice(2)) {
            expect(sent.body?.safetySettings).toEqual([
                { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
            ]);
        }
        expect(sentConfig(server.received[2]).stopSequences).toEqual(['END']);
        expect(sentConfig(server.received[3]).stopSequences).toEqual(['STOP']);
    });

    it('treat max_tokens and max_completion_tokens as one field', async () => {
        const client = createGenaiClient(clientOptions({ defaults: { model: 'gemini-3-x', max_tokens: 2000 } }));
        const result = await client.generate({
            messages: [{ role: 'user', content: 'hi' }],
            max_completion_tokens: 500,
        });
        expect(result).toMatchObject({ ok: true, data: { settings: { max_tokens: 500 } } });
        expect(server.received[0]?.body?.generationConfig).toEqual({ maxOutputTokens: 500 });
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { max_tokens: 2000 },
            models: { flash: { model: 'm', defaults: { max_completion_tokens: 300 } } },
        });
        expect(await gemini.flash('hi')).toMatchObject({ ok: true, data: { settings: { max_tokens: 300 } } });
        expect(await gemini.flash('hi', { max_tokens: 100 })).toMatchObject({
            ok: true,
            data: { settings: { max_tokens: 100 } },
        });
        expect(server.received.map((received) => received.body?.generationConfig)).toEqual([
            { maxOutputTokens: 500 },
            { maxOutputTokens: 300 },
            { maxOutputTokens: 100 },
        ]);
    });

    it('treat reasoning_effort, thinkingBudget and thinkingLevel as one setting; the call wins', async () => {
        const client = createGenaiClient(
            clientOptions({
                defaults: {
                    model: 'gemini-3-x',
                    reasoning_effort: 'low',
                    config: { thinkingConfig: { thinkingBudget: 100 } },
                },
            }),
        );
        const messages: GenaiMessage[] = [{ role: 'user', content: 'hi' }];
        const effort = await client.generate({ messages, reasoning_effort: 'high' });
        const budget = await client.generate({ messages }, { config: { thinkingConfig: { thinkingBudget: 10 } } });
        const level = await client.generate(
            { messages },
            { config: { thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM } } },
        );
        await client.generate({ messages });
        expect(effort.ok && effort.data.settings).toMatchObject({ reasoning_effort: 'high' });
        expect(budget.ok && budget.data.settings).not.toHaveProperty('reasoning_effort');
        expect(level.ok && level.data.settings?.config).toEqual({
            thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM },
        });
        expect(server.received.map((received) => sentConfig(received).thinkingConfig)).toEqual([
            { thinkingLevel: 'HIGH', includeThoughts: true },
            { thinkingBudget: 10 },
            { thinkingLevel: 'MEDIUM' },
            // both from the defaults: the config wins as within one call
            { thinkingBudget: 100, includeThoughts: true },
        ]);
    });

    it('resolve the thinking setting by source for named models too', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { config: { thinkingConfig: { thinkingBudget: 100, includeThoughts: false } } },
            models: {
                flash: { model: 'gemini-3-flash', defaults: { reasoning_effort: 'high' } },
                lite: {
                    model: 'gemini-3-flash',
                    defaults: { config: { thinkingConfig: { thinkingLevel: ThinkingLevel.LOW } } },
                },
            },
        });
        await gemini.flash('a');
        await gemini.flash('b', { config: { thinkingConfig: { thinkingBudget: 10 } } });
        await gemini.lite('c', { reasoning_effort: 'medium' });
        expect(server.received.map((received) => sentConfig(received).thinkingConfig)).toEqual([
            { thinkingLevel: 'HIGH', includeThoughts: false },
            { thinkingBudget: 10, includeThoughts: false },
            { thinkingLevel: 'MEDIUM', includeThoughts: false },
        ]);
    });

    it('are validated when the client is created', () => {
        const invalid: [unknown, string][] = [
            [{ messages: [] }, 'Option "defaults.messages" is not supported'],
            [{ reasoning_effort: 'ultra' }, 'Invalid value for option "reasoning_effort"'],
            [{ web_search: {} }, 'Invalid value for option "web_search": true or false'],
            [{ web_search_options: {} }, 'Option "defaults.web_search_options" is not supported'],
            [{ config: { abortSignal: null } }, 'Option "config.abortSignal" is not supported'],
        ];
        for (const [defaults, message] of invalid) {
            expect(() => createGenaiClient(clientOptions({ defaults: defaults as never }))).toThrow(
                new TypeError(message),
            );
        }
        expect(() => createGenaiClient(clientOptions())).not.toThrow();
    });

    it('return grounding sources with the answer', async () => {
        server.reply(() =>
            Response.json({
                ...generateBody('answer'),
                candidates: [
                    {
                        content: { role: 'model', parts: [{ text: 'answer' }] },
                        finishReason: 'STOP',
                        groundingMetadata: {
                            groundingChunks: [{ web: { uri: 'https://example.org/a', title: 'A' } }, { web: {} }],
                        },
                    },
                ],
            }),
        );
        const result = await createGenaiClient(clientOptions()).generate({
            model: 'm',
            messages: [{ role: 'user', content: 'q' }],
            web_search: true,
        });
        expect(result.ok && result.data.sources).toEqual([{ url: 'https://example.org/a', title: 'A' }]);
    });
});

describe('listModels', () => {
    it('reads every page and strips the resource prefix', async () => {
        server.reply(
            () =>
                Response.json({
                    models: [
                        {
                            name: 'models/gemini-3-x',
                            displayName: 'Gemini X',
                            inputTokenLimit: 1000,
                            outputTokenLimit: 100,
                            supportedGenerationMethods: ['generateContent', 'batchGenerateContent'],
                            thinking: true,
                        },
                    ],
                    nextPageToken: 'p2',
                }),
            () => Response.json({ models: [{ name: 'models/gemma-y' }, { displayName: 'no name' }, null] }),
        );
        const listed = await createGenaiClient(clientOptions()).listModels();
        expect(listed).toEqual({
            ok: true,
            data: [
                {
                    id: 'gemini-3-x',
                    displayName: 'Gemini X',
                    inputTokenLimit: 1000,
                    outputTokenLimit: 100,
                    actions: ['generateContent', 'batchGenerateContent'],
                    thinking: true,
                },
                { id: 'gemma-y', actions: [] },
            ],
        });
        expect(server.received.map((entry) => entry.method)).toEqual(['GET', 'GET']);
        server.reply(() => apiError(403));
        expect(await createGenaiClient(clientOptions()).listModels()).toMatchObject({
            ok: false,
            error: { code: 'http', status: 403 },
        });
    });
});

describe('createGenaiModels', () => {
    it('calls each name with its model id and the merged defaults; a string is the user message', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 0.5, max_tokens: 50, seed: 1 },
            models: {
                flash: { model: 'gemini-3-flash', defaults: { reasoning_effort: 'high', max_tokens: 80 } },
                lite: 'gemini-3-lite',
            },
        });
        expect(gemini.flash.id).toBe('gemini-3-flash');
        expect(await gemini.flash('hello')).toMatchObject({ ok: true, data: { text: 'ok' } });
        await gemini.lite('x', { temperature: 0 });
        const [flash, lite] = server.received;
        expect(flash?.path).toBe('/v1beta/models/gemini-3-flash:generateContent');
        expect(flash?.body?.contents).toEqual([{ role: 'user', parts: [{ text: 'hello' }] }]);
        expect(flash?.body?.generationConfig).toEqual({
            temperature: 0.5,
            maxOutputTokens: 80,
            seed: 1,
            thinkingConfig: { thinkingLevel: 'HIGH', includeThoughts: true },
        });
        expect(lite?.path).toBe('/v1beta/models/gemini-3-lite:generateContent');
        expect(lite?.body?.generationConfig).toEqual({ temperature: 0, maxOutputTokens: 50, seed: 1 });
        expect(typeof gemini.listModels).toBe('function');
        expect(gemini.client.mode).toBe('developer');
    });

    it('takes content parts or messages as input; options stay fixed when the model is handed on', async () => {
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'gemini-3-flash' } });
        const image = { type: 'image_url' as const, image_url: { url: 'data:image/png;base64,AA==' } };
        const conversation: GenaiMessage[] = [
            { role: 'system', content: 'be short' },
            { role: 'user', content: 'hi' },
            { role: 'assistant', content: 'hello' },
            { role: 'user', content: 'and?' },
        ];
        function fixed(input: GenaiModelInput) {
            return gemini.flash(input, { temperature: 0, config: { candidateCount: 1 } });
        }
        function run(call: GenaiModel) {
            return call('ho', { seed: 2 });
        }
        expect(await gemini.flash([{ type: 'text', text: 'what?' }, image])).toMatchObject({ ok: true });
        expect(await fixed(conversation)).toMatchObject({ ok: true });
        expect(await run(gemini.flash)).toMatchObject({ ok: true });
        const [parts, chat, handed] = server.received;
        expect(parts?.body?.contents).toEqual([
            { role: 'user', parts: [{ text: 'what?' }, { inlineData: { mimeType: 'image/png', data: 'AA==' } }] },
        ]);
        expect(chat?.body?.systemInstruction).toMatchObject({ parts: [{ text: 'be short' }] });
        expect(chat?.body?.contents).toHaveLength(3);
        expect(chat?.body?.generationConfig).toEqual({ temperature: 0, candidateCount: 1 });
        expect(handed?.body?.generationConfig).toEqual({ seed: 2 });
    });

    it('an option or config field set to undefined keeps the default; {} changes nothing', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 0.3, max_tokens: 100, config: { topK: 5 } },
            models: { flash: 'gemini-3-flash' },
        });
        await gemini.flash('a', { temperature: undefined, max_tokens: undefined, seed: 1 });
        await gemini.flash('b', {});
        await gemini.flash('c', { config: { topK: undefined, candidateCount: 1 } as never });
        const expected = { temperature: 0.3, maxOutputTokens: 100, topK: 5 };
        expect(server.received.map((received) => received.body?.generationConfig)).toEqual([
            { ...expected, seed: 1 },
            expected,
            { ...expected, candidateCount: 1 },
        ]);
    });

    it('takes a JSON schema in the options and answers its json', async () => {
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'gemini-3-flash' } });
        const schema = { type: 'object', properties: { kind: { type: 'string' } }, required: ['kind'] };
        const format = { type: 'json_schema' as const, json_schema: { name: 'word', schema } };
        function classify(input: GenaiModelInput) {
            return gemini.flash(input, { response_format: format });
        }
        server.reply(() => generateResponse('{"kind":"noun"}'));
        expect(await classify('house')).toMatchObject({ ok: true, data: { json: { kind: 'noun' } } });
        expect(server.received[0]?.body?.generationConfig).toEqual({
            responseMimeType: 'application/json',
            responseJsonSchema: schema,
        });
    });

    it('runs a tool loop: tools in the options, the conversation with assistantMessage and toolMessage', async () => {
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'gemini-3-flash' } });
        const tools = [{ type: 'function' as const, function: { name: 'get_time' } }];
        server.reply(
            () =>
                Response.json({
                    candidates: [
                        {
                            content: {
                                role: 'model',
                                parts: [
                                    {
                                        functionCall: { id: 'g-1', name: 'get_time', args: {} },
                                        thoughtSignature: 'c2ln',
                                    },
                                ],
                            },
                            finishReason: 'STOP',
                        },
                    ],
                }),
            () => generateResponse('noon'),
        );
        const conversation: GenaiMessage[] = [{ role: 'user', content: 'Time?' }];
        const first = await gemini.flash(conversation, { tools });
        if (!first.ok) {
            throw new Error('first call failed');
        }
        conversation.push(
            assistantMessage(first.data),
            ...(first.data.toolCalls ?? []).map((call) => toolMessage(call, { time: '12:00' })),
        );
        expect(await gemini.flash(conversation, { tools })).toMatchObject({ ok: true, data: { text: 'noon' } });
        expect(server.received[1]?.body?.contents).toEqual([
            { role: 'user', parts: [{ text: 'Time?' }] },
            {
                role: 'model',
                parts: [{ functionCall: { id: 'g-1', name: 'get_time', args: {} }, thoughtSignature: 'c2ln' }],
            },
            {
                role: 'user',
                parts: [{ functionResponse: { id: 'g-1', name: 'get_time', response: { time: '12:00' } } }],
            },
        ]);
    });

    it('retries a 429 after its retry delay and fails a 400 at once', async () => {
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'gemini-3-flash' } });
        server.reply(
            () => apiError(429, '0.05s'),
            () => generateResponse('second'),
        );
        expect(await gemini.flash('a')).toMatchObject({ ok: true, data: { text: 'second' } });
        expect(server.received).toHaveLength(2);
        server.received.length = 0;
        server.reply(
            () => generateResponse('', { finishReason: 'MALFORMED_FUNCTION_CALL' }),
            () => generateResponse('fixed'),
        );
        expect(await gemini.flash('tool-retry')).toMatchObject({ ok: true, data: { text: 'fixed', attempts: 2 } });
        server.received.length = 0;
        server.reply(() => apiError(400));
        expect(await gemini.flash('b')).toMatchObject({
            ok: false,
            error: { code: 'http', status: 400, retryable: false },
        });
        expect(server.received).toHaveLength(1);
    });

    it('reports the settings under the model name with the merged defaults, and the attempts of the call', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 0.5, config: { topK: 3 } },
            models: { flash: { model: 'gemini-3-flash', defaults: { max_tokens: 50, config: { topP: 0.9 } } } },
        });
        const tool = { type: 'function' as const, function: { name: 'lookup' } };
        server.reply(
            () => apiError(429, '0.01s'),
            () => generateResponse('second'),
        );
        expect(await gemini.flash('a', { reasoning_effort: 'low', tools: [tool], config: { seed: 1 } })).toMatchObject({
            ok: true,
            data: {
                text: 'second',
                settings: {
                    name: 'flash',
                    model: 'gemini-3-flash',
                    temperature: 0.5,
                    max_tokens: 50,
                    reasoning_effort: 'low',
                    tools: ['lookup'],
                    config: { topK: 3, topP: 0.9, seed: 1 },
                },
                attempts: 2,
            },
        });
        server.reply(() => apiError(400));
        expect(await gemini.flash('b')).toMatchObject({
            ok: false,
            error: {
                code: 'http',
                settings: { name: 'flash', model: 'gemini-3-flash', temperature: 0.5 },
                attempts: 1,
            },
        });
        const refused = createGenaiModels({
            client: clientOptions(),
            models: { tiny: { model: 'gemini-3-flash', modelLimits: { tpm: 10 } } },
        });
        expect(await refused.tiny('too long for ten tokens per minute')).toMatchObject({
            ok: false,
            error: { code: 'limiter', settings: { name: 'tiny', model: 'gemini-3-flash' }, attempts: 0 },
        });
        const invalid = await gemini.flash('c', { temperature: 'hot' as never });
        expect(invalid).toMatchObject({ ok: false, error: { code: 'invalid_request' } });
        expect(!invalid.ok && invalid.error.settings).toBeUndefined();
    });

    it('turns retries off with maxRetries 0 and reports refused calls as limiter errors', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxRetries: 0 },
            models: { flash: { model: 'gemini-3-flash', modelLimits: { tpm: 10 } } },
        });
        expect(await gemini.flash('too long for ten tokens per minute')).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'COST_EXCEEDS_LIMIT', retryable: false },
        });
        expect(server.received).toHaveLength(0);
        const noRetry = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxRetries: 0 },
            models: { flash: 'gemini-3-flash' },
        });
        server.reply(() => apiError(408));
        expect(await noRetry.flash('c')).toMatchObject({ ok: false, error: { code: 'http', status: 408 } });
        expect(server.received).toHaveLength(1);
    });

    it('books tokens with charsPerToken and a fixed amount per image', async () => {
        const options = { max_tokens: 1 };
        const coarse = createGenaiModels({
            client: clientOptions(),
            charsPerToken: 0.5,
            models: {
                coarse: { model: 'gemini-3-flash', modelLimits: { tpm: 10 } },
                fine: { model: 'gemini-3-flash', modelLimits: { tpm: 10 }, charsPerToken: 4 },
            },
        });
        expect(await coarse.coarse('abcdefgh', options)).toMatchObject({
            ok: false,
            error: { limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
        expect(await coarse.fine('abcdefgh', options)).toMatchObject({ ok: true });
        server.received.length = 0;
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: {
                fine: { model: 'gemini-3-flash', modelLimits: { tpm: 10 } },
                image: { model: 'gemini-3-flash', modelLimits: { tpm: 10 } },
            },
        });
        expect(await gemini.fine('abcdefgh', options)).toMatchObject({ ok: true });
        const image = { type: 'image_url' as const, image_url: { url: 'data:image/png;base64,AA==' } };
        expect(await gemini.image([image], { max_tokens: 1 })).toMatchObject({
            ok: false,
            error: { limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
        expect(server.received).toHaveLength(1);
    });

    it('books tool definitions, tool calls and tool results as prompt characters', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: {
                plain: { model: 'gemini-3-flash', modelLimits: { tpm: 12 } },
                tools: { model: 'gemini-3-flash', modelLimits: { tpm: 12 } },
                calls: { model: 'gemini-3-flash', modelLimits: { tpm: 12 } },
            },
        });
        const user = { role: 'user' as const, content: 'hi' };
        expect(await gemini.plain([user], { max_tokens: 1 })).toMatchObject({ ok: true });
        const tool = { type: 'function' as const, function: { name: 'get_time' } }; // 46 characters as JSON
        expect(await gemini.tools([user], { tools: [tool], max_tokens: 1 })).toMatchObject({
            ok: false,
            error: { limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
        const call = {
            id: 'c',
            type: 'function' as const,
            function: { name: 'get_time', arguments: '{"zone":"Europe/Rome"}' },
        };
        const history = [
            user,
            { role: 'assistant' as const, content: null, tool_calls: [call] },
            { role: 'tool' as const, tool_call_id: 'c', content: '12:00 in the afternoon, Rome' },
        ];
        expect(await gemini.calls(history, { max_tokens: 1 })).toMatchObject({
            ok: false,
            error: { limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
        expect(server.received).toHaveLength(1);
    });

    it('applies sharedLimits to all models together and modelLimits to one model', async () => {
        const shared = createGenaiModels({
            client: clientOptions(),
            sharedLimits: { rpm: 1 },
            limiterOptions: { maxWaitMs: 100 },
            models: { flash: 'gemini-3-flash', lite: 'gemini-3-lite' },
        });
        expect(await shared.flash('a')).toMatchObject({ ok: true });
        expect(await shared.lite('b')).toMatchObject({ ok: false, error: { limiterCode: 'MAX_WAIT_EXCEEDED' } });
        const own = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: {
                flash: { model: 'gemini-3-flash', modelLimits: { rpm: 1 } },
                lite: { model: 'gemini-3-lite', modelLimits: { rpm: 1 } },
            },
        });
        expect(await own.flash('a')).toMatchObject({ ok: true });
        expect(await own.lite('b')).toMatchObject({ ok: true });
        expect(await own.flash('c')).toMatchObject({ ok: false, error: { limiterCode: 'MAX_WAIT_EXCEEDED' } });
        expect(server.received).toHaveLength(3);
    });

    it('spaces calls with intervalMs: one call every 3 seconds', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { flash: { model: 'gemini-3-flash', modelLimits: { intervalMs: 3000 } } },
        });
        expect(await gemini.flash('a')).toMatchObject({ ok: true });
        expect(await gemini.flash('b')).toMatchObject({ ok: false, error: { limiterCode: 'MAX_WAIT_EXCEEDED' } });
    });

    it('accepts maxWaitMs: Infinity as no wait limit, rejects -Infinity and NaN', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: Number.POSITIVE_INFINITY },
            models: { flash: 'gemini-3-flash' },
        });
        expect(await gemini.flash('a')).toMatchObject({ ok: true });
        for (const maxWaitMs of [Number.NEGATIVE_INFINITY, Number.NaN]) {
            expect(() =>
                createGenaiModels({ client: clientOptions(), limiterOptions: { maxWaitMs }, models: { a: 'm' } }),
            ).toThrow(/^Invalid value for option "limiterOptions\.maxWaitMs"$/);
        }
        expect(() =>
            createGenaiModels({
                client: clientOptions(),
                limiterOptions: { maxRetries: Number.POSITIVE_INFINITY },
                models: { a: 'm' },
            }),
        ).toThrow(/^Invalid value for option "limiterOptions\.maxRetries"$/);
    });

    it('ends a call waiting for its limits or a lock as aborted when its signal fires', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { flash: { model: 'gemini-3-flash', modelLimits: { intervalMs: 3000 } }, lite: 'gemini-3-lite' },
        });
        expect(await gemini.flash('a')).toMatchObject({ ok: true });
        const startedAt = Date.now();
        expect(await gemini.flash('b', { signal: AbortSignal.timeout(50) })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false, settings: { name: 'flash' }, attempts: 0 },
        });
        server.reply(() => apiError(503));
        expect(await gemini.lite('c', { signal: AbortSignal.timeout(50) })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false, attempts: 1 },
        });
        expect(Date.now() - startedAt).toBeLessThan(1000);
    });

    it('counts the request in flight when the signal fires during it', async () => {
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'gemini-3-flash' } });
        server.reply(async () => {
            await Bun.sleep(300);
            return generateResponse('late');
        });
        expect(await gemini.flash('a', { signal: AbortSignal.timeout(50) })).toMatchObject({
            ok: false,
            error: { code: 'aborted', attempts: 1 },
        });
    });

    it('returns the last provider error when an outage outlasts maxWaitMs', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 1000 }, // the first lock (retryDelayMs, 2 s) is longer
            models: { flash: 'gemini-3-flash' },
        });
        server.reply(() => apiError(503));
        expect(await gemini.flash('a')).toMatchObject({
            ok: false,
            error: { code: 'http', status: 503, retryable: true, settings: { name: 'flash' }, attempts: 1 },
        });
    });

    /** Configures fake server delaying calls by callMs and recording call order and concurrency. */
    function queueServer(callMs = 0) {
        const texts: string[] = [];
        const runningAtStart: number[] = [];
        const errors: Response[] = [];
        let running = 0;
        server.reply(async (request) => {
            const contents = request.body?.contents as { parts: { text?: string }[] }[];
            texts.push(String(contents.at(-1)?.parts.at(-1)?.text));
            runningAtStart.push(running);
            running++;
            await Bun.sleep(callMs);
            running--;
            return errors.shift() ?? generateResponse('ok');
        });
        return { texts, runningAtStart, reply: (error: Response) => errors.push(error) };
    }

    it('starts the calls of one model in call order', async () => {
        const fake = queueServer();
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { flash: { model: 'm', modelLimits: { concurrency: 1 } } },
        });
        const texts = ['a', 'b', 'c', 'd', 'e'];
        await Promise.all(texts.map((text) => gemini.flash(text)));
        expect(fake.texts).toEqual(texts);
    });

    it('runs every call of a model at once without a concurrency limit', async () => {
        const fake = queueServer(20);
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'm' } });
        await Promise.all(Array.from({ length: 8 }, (_, index) => gemini.flash(String(index))));
        expect(fake.runningAtStart).toEqual([0, 1, 2, 3, 4, 5, 6, 7]);
    });

    it('sends one probe after a 429 until it passes, then the waiting calls together', async () => {
        const fake = queueServer(20);
        fake.reply(apiError(429, '0.1s'));
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'm' } });
        const first = gemini.flash('a');
        await Bun.sleep(50);
        const results = await Promise.all([first, gemini.flash('b'), gemini.flash('c'), gemini.flash('d')]);
        expect(results.every((result) => result.ok)).toBe(true);
        // the 429, the probe alone, then the other three at once
        expect(fake.runningAtStart).toEqual([0, 0, 0, 1, 2]);
    });

    it('ends calls without a request while a lock outlasts maxWaitMs, and sends again once it is near', async () => {
        const fake = queueServer();
        fake.reply(apiError(429, '0.3s'));
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100, maxRetries: 0 },
            models: { flash: 'm' },
        });
        expect(await gemini.flash('a')).toMatchObject({
            ok: false,
            error: { code: 'rate_limited', retryAfterMs: 300, attempts: 1 },
        });
        expect(await gemini.flash('b')).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED', attempts: 0 },
        });
        expect(fake.texts).toHaveLength(1);
        await Bun.sleep(250);
        expect(await gemini.flash('c')).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(fake.texts).toHaveLength(2);
    });

    it('gives back the token booking of a 429, so its retry fits the same tpm', async () => {
        const fake = queueServer();
        fake.reply(apiError(429, '0.05s'));
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 1000 },
            models: { flash: { model: 'm', modelLimits: { tpm: 1500 } } },
        });
        // books about 1000 tokens per attempt: a second booking fits the minute only if the first went back
        expect(await gemini.flash('a')).toMatchObject({ ok: true, data: { attempts: 2 } });
    });

    it('ends an aborted call that waits in the queue at once, without a request', async () => {
        const fake = queueServer(50);
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { flash: { model: 'm', modelLimits: { concurrency: 1 } } },
        });
        const first = gemini.flash('first');
        const controller = new AbortController();
        const second = gemini.flash('second', { signal: controller.signal });
        controller.abort();
        expect(await second).toMatchObject({ ok: false, error: { code: 'aborted', attempts: 0 } });
        expect(await first).toMatchObject({ ok: true });
        expect(fake.texts).toEqual(['first']);
    });

    it('resolves the thinking setting of batch items by source, for clients and named models', async () => {
        const defaults = { reasoning_effort: 'low', config: { thinkingConfig: { thinkingBudget: 100 } } } as const;
        const client = createGenaiClient(clientOptions({ defaults: { model: 'gemini-3-flash', ...defaults } }));
        const gemini = createGenaiModels({ client: clientOptions(), defaults, models: { flash: 'gemini-3-flash' } });
        const own = {
            key: 'own',
            request: { messages: [{ role: 'user' as const, content: 'x' }], reasoning_effort: 'high' as const },
        };
        const plain = { key: 'plain', request: { messages: [{ role: 'user' as const, content: 'y' }] } };
        for (const batch of [client.batch, gemini.flash.batch]) {
            expect(await batch?.submit([plain, own])).toMatchObject({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: expect.stringContaining('Option "reasoning_effort" of item "own" is not supported'),
                },
            });
        }
        expect(server.received).toHaveLength(0);
        server.reply(() => batchResponse('batches/b1', 'BATCH_STATE_PENDING'));
        const level = { thinkingConfig: { thinkingLevel: ThinkingLevel.MEDIUM } };
        for (const batch of [client.batch, gemini.flash.batch]) {
            expect(await batch?.submit([plain, own], { config: level })).toMatchObject({ ok: true });
            expect(await batch?.submit([plain])).toMatchObject({ ok: true });
        }
        const sent = server.received.map((received) =>
            sentBatch(received).inputConfig.requests.requests.map((entry) => entry.request.generationConfig),
        );
        const fromConfig = [
            { thinkingConfig: { thinkingLevel: 'MEDIUM' } },
            { thinkingConfig: { thinkingLevel: 'MEDIUM', includeThoughts: true } },
        ];
        // both from the defaults: the config wins as within one call
        const fromDefaults = [{ thinkingConfig: { thinkingBudget: 100, includeThoughts: true } }];
        expect(sent).toEqual([fromConfig, fromDefaults, fromConfig, fromDefaults]);
    });

    it('submits batches with the model id and defaults', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 0 },
            models: { flash: 'gemini-3-flash' },
        });
        server.reply(() => batchResponse('batches/b1', 'BATCH_STATE_PENDING'));
        const submitted = await gemini.flash.batch?.submit([
            { key: '1', request: { messages: [{ role: 'user', content: 'x' }] } },
        ]);
        expect(submitted).toMatchObject({ ok: true, data: { name: 'batches/b1' } });
        const sent = server.received[0];
        expect(sent?.path).toBe('/v1beta/models/gemini-3-flash:batchGenerateContent');
        expect(sentBatch(sent).inputConfig.requests.requests[0].request.generationConfig).toEqual({ temperature: 0 });
    });

    it('reports tool_choice in the settings only with tools, as a copy', async () => {
        const tools = [{ type: 'function' as const, function: { name: 'get_time' } }];
        const named = { type: 'function' as const, function: { name: 'get_time' } };
        const client = createGenaiClient(clientOptions());
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'gemini-3-flash' } });
        const off = await gemini.flash('hi', { tools, tool_choice: 'none' });
        expect(off.ok && off.data.settings).toMatchObject({ tools: ['get_time'], tool_choice: 'none' });
        const forced = await client.generate({
            model: 'gemini-3-flash',
            messages: [{ role: 'user', content: 'hi' }],
            tools,
            tool_choice: named,
        });
        expect(forced.ok && forced.data.settings?.tool_choice).toEqual(named);
        expect(forced.ok && forced.data.settings?.tool_choice).not.toBe(named);
        const unused = await gemini.flash('hi', { tool_choice: 'required' });
        expect(unused.ok && unused.data.settings).not.toHaveProperty('tool_choice');
        expect(server.received[2]?.body?.toolConfig).toBeUndefined();
    });

    it('rejects reserved model names at creation and keeps odd names as own members', async () => {
        const fake = queueServer();
        const reserved = ['client', 'listModels', 'then', 'toJSON', 'toString', 'valueOf', 'constructor'];
        for (const name of [...reserved, 'hasOwnProperty', '__proto__']) {
            const models = JSON.parse(JSON.stringify({ [name]: 'x' }));
            expect(() => createGenaiModels({ client: clientOptions(), models })).toThrow(
                `Invalid value for option "models.${name}": ${name} is reserved`,
            );
        }
        const models = JSON.parse('{"flash-lite": "lite-model", "38flash": "digit-model", "": "empty-model"}');
        const gemini = createGenaiModels({ client: clientOptions(), models }) as unknown as Record<string, GenaiModel>;
        expect(Object.getPrototypeOf(gemini)).toBe(Object.prototype);
        expect(await gemini['flash-lite']?.('a')).toMatchObject({
            ok: true,
            data: { settings: { name: 'flash-lite' } },
        });
        expect(await gemini['38flash']?.('b')).toMatchObject({
            ok: true,
            data: { settings: { model: 'digit-model' } },
        });
        expect(await gemini['']?.('c')).toMatchObject({ ok: true, data: { settings: { model: 'empty-model' } } });
        JSON.stringify({ gemini });
        expect(fake.texts).toEqual(['a', 'b', 'c']);
    });

    it('throws for invalid names, specs and limits; answers invalid_request for invalid requests', async () => {
        const client = clientOptions();
        const invalid: [unknown, RegExp][] = [
            [{ client, models: {} }, /option "models"/],
            [
                { client: createGenaiClient(client), models: { a: 'm' } },
                /^Invalid value for option "client": client options, not a client$/,
            ],
            [{ client, models: { toJSON: 'm' } }, /models.toJSON": toJSON is reserved/],
            [{ client, models: { client: 'm' } }, /models\.client/],
            [{ client, models: { a: { model: 'm', limit: {} } } }, /Option "models.a.limit" is not supported/],
            [{ client, models: { a: { model: 'm', defaults: { model: 'x' } } } }, /models\.a\.defaults\.model/],
            [
                { client, models: { a: { model: 'm', defaults: { model: '' } } } },
                /^Option "models\.a\.defaults\.model"/,
            ],
            [{ client, models: { a: 'm' }, defaults: { model: '' } }, /^Option "defaults\.model" is not supported$/],
            [{ client, models: { a: 'm' }, sharedLimits: { rpm: 0 } }, /sharedLimits\.rpm/],
            [
                { client, models: { a: { model: 'm', modelLimits: { rph: 1 } } } },
                /Option "models.a.modelLimits.rph" is not supported/,
            ],
            [
                { client, models: { a: { model: 'm', modelLimits: { intervalMs: 0.5 } } } },
                /models\.a\.modelLimits\.intervalMs/,
            ],
            [{ client, models: { a: 'm' }, limits: {} }, /Option "limits" is not supported/],
            [{ client, models: { a: 'm' }, charsPerToken: 0 }, /option "charsPerToken"/],
            [{ client, models: { a: { model: 'm', charsPerToken: -1 } } }, /models\.a\.charsPerToken/],
            [
                { client, models: { a: { model: 'm', modelLimits: { windows: [] } } } },
                /Option "models.a.modelLimits.windows" is not supported/,
            ],
            [{ client, models: { a: 'm' }, limiterOptions: { maxRetries: -1 } }, /limiterOptions\.maxRetries/],
            [{ client, models: { a: 'm' }, limiter: {} }, /Option "limiter" is not supported/],
            [
                { client, models: { a: 'm' }, limiterOptions: { timeoutMs: 1 } },
                /Option "limiterOptions.timeoutMs" is not supported/,
            ],
            [{ client: clientOptions({ defaults: { seed: 1 } }), models: { a: 'm' } }, /client\.defaults/],
            [{ client, models: { a: 'm' }, extra: 1 }, /Option "extra" is not supported/],
            [{ client, models: { a: 'm' }, sharedLimits: [] }, /^Invalid value for option "sharedLimits"$/],
            [{ client, models: { a: 'm' }, limiterOptions: [] }, /^Invalid value for option "limiterOptions"$/],
            [{ client, models: { a: [] } }, /^Invalid value for option "models\.a"$/],
            [
                { client, models: { a: { model: 'm', modelLimits: [] } } },
                /^Invalid value for option "models\.a\.modelLimits"$/,
            ],
            [{ client, models: { a: 'm' }, sharedLimits: null }, /^Invalid value for option "sharedLimits"$/],
            [{ client, models: { a: '' } }, /^Invalid value for option "models\.a"$/],
            [{ client, models: { a: { model: '' } } }, /^Invalid value for option "models\.a\.model"$/],
            [
                { client, models: { a: 'm' }, sharedLimits: { buckets: 'x' } },
                /^Invalid value for option "sharedLimits\.buckets"$/,
            ],
            [
                { client, models: { a: 'm' }, sharedLimits: { buckets: [createRPM(0)] } },
                /^Invalid value for option "sharedLimits":/,
            ],
            [
                {
                    client,
                    defaults: { web_search: true },
                    models: {
                        a: {
                            model: 'gemini-2.5-flash',
                            defaults: { tools: [{ type: 'function', function: { name: 'f' } }] },
                        },
                    },
                },
                /^Invalid value for option "models.a.defaults": Option "tools" is not supported together with "web_search" on Gemini 2.x$/,
            ],
        ];
        for (const [options, message] of invalid) {
            expect(() => createGenaiModels(options as never)).toThrow(message);
        }
        const gemini = createGenaiModels({ client, models: { a: 'm' } });
        const batch = gemini.a.batch;
        const fixed = 'Option "model" is not supported: the model of a name is fixed';
        const invalidCalls: [Promise<unknown> | undefined, string][] = [
            [gemini.a('x', { model: 'x' } as never), 'Option "model" is not supported: the name and input set it'],
            [gemini.a('x', { messages: [] } as never), 'Option "messages" is not supported: the name and input set it'],
            [gemini.a('x', { n: 2 } as never), 'Option "n" is not supported'],
            [gemini.a(42 as never), 'Invalid value for the input: a string, content parts or messages'],
            [gemini.a({ messages: [] } as never), 'Invalid value for the input: a string, content parts or messages'],
            [gemini.a([]), 'Invalid value for the input: a string, content parts or messages'],
            [
                gemini.a([
                    { role: 'user', content: 'x' },
                    { type: 'text', text: 'y' },
                ] as never),
                'Invalid value for the input: a string, content parts or messages',
            ],
            [gemini.a('x', null as never), 'Invalid value for the options: an object'],
            [gemini.a('x', { retries: 1 } as never), 'Option "retries" is not supported'],
            [gemini.a('x', { config: 'x' } as never), 'Invalid value for option "config"'],
            [gemini.a('x', { signal: 'x' } as never), 'Invalid value for option "signal"'],
            [batch?.submit(null as never), 'Invalid value for option "items"'],
            [batch?.submit([], null as never), 'Invalid value for option "options"'],
            [batch?.submit([], [] as never), 'Invalid value for option "options"'],
            [batch?.submit([{ key: 'k', request: { model: 'x', messages: [] } as never }]), fixed],
            [batch?.check(''), 'Invalid value for option "name"'],
        ];
        for (const [pending, message] of invalidCalls) {
            expect(await pending).toEqual({
                ok: false,
                error: { code: 'invalid_request', message, retryable: false },
            });
        }
        expect(batch?.requestBytes(42 as never)).toEqual({
            ok: false,
            error: { code: 'invalid_request', message: 'Invalid value for option "request"', retryable: false },
        });
        const bytes = batch?.requestBytes({ messages: [{ role: 'user', content: 'x' }] });
        expect(bytes?.ok && bytes.data).toBeGreaterThan(10);
        const withConfig = batch?.requestBytes(
            { messages: [{ role: 'user', content: 'x' }] },
            { systemInstruction: 'x'.repeat(1000) },
        );
        expect(withConfig?.ok && bytes?.ok && withConfig.data - bytes.data).toBeGreaterThan(900);
        expect(server.received).toHaveLength(0);
    });

    it('reports the model name first in the settings, also for errors', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 1 },
            models: { a: 'm' },
            limiterOptions: { maxRetries: 0 },
        });
        const ok = await gemini.a('x');
        expect(ok.ok && JSON.stringify(ok.data.settings)).toBe('{"name":"a","model":"m","temperature":1}');
        server.reply(() => apiError(400));
        const failed = await gemini.a('x');
        expect(!failed.ok && JSON.stringify(failed.error.settings)).toBe('{"name":"a","model":"m","temperature":1}');
    });
});

describe('batch calls of named models', () => {
    const timing = { ...batchTiming };

    beforeAll(() => {
        batchTiming.pollMs = 5;
        batchTiming.msPerSecond = 20;
    });
    afterAll(() => {
        Object.assign(batchTiming, timing);
    });

    /** Configures fake batch server taking batchMs and tracking peak concurrency and job sizes. */
    function batchServer(batchMs: number, errors: Response[] = []) {
        const jobs = new Map<string, { at: number; keys: string[]; done: boolean }>();
        const state = { submits: 0, maxRunning: 0, sizes: [] as number[] };
        let running = 0;
        server.reply((request) => {
            if (request.path.endsWith(':batchGenerateContent')) {
                state.submits++;
                const error = errors.shift();
                if (error !== undefined) {
                    return error;
                }
                const name = `batches/b${jobs.size + 1}`;
                const requests = sentBatch(request).inputConfig.requests.requests;
                jobs.set(name, { at: Date.now(), keys: requests.map((entry) => entry.metadata.key), done: false });
                state.sizes.push(requests.length);
                running++;
                state.maxRunning = Math.max(state.maxRunning, running);
                return batchResponse(name, 'BATCH_STATE_PENDING');
            }
            if (request.path.startsWith('/v1beta/batches/')) {
                const name = request.path.slice('/v1beta/'.length);
                const job = jobs.get(name);
                if (job === undefined || Date.now() - job.at < batchMs) {
                    return batchResponse(name, 'BATCH_STATE_RUNNING');
                }
                if (!job.done) {
                    job.done = true;
                    running--;
                }
                const responses = job.keys.map((key) => ({ response: generateBody('ok'), metadata: { key } }));
                return batchResponse(name, 'BATCH_STATE_SUCCEEDED', responses);
            }
            return generateResponse('ok');
        });
        return state;
    }

    it('take batch from the call or the defaults, so two names can share one model id', async () => {
        batchServer(1);
        const gemini = createGenaiModels({
            client: clientOptions({ batchWaitSec: 1 }),
            models: { flash: 'gemini-3-flash', flashBatch: { model: 'gemini-3-flash', defaults: { batch: true } } },
        });
        await gemini.flash('a');
        await gemini.flashBatch('b');
        await gemini.flash('c', { batch: true });
        await gemini.flashBatch('d', { batch: false });
        const calls = server.received.filter((request) => request.path.startsWith('/v1beta/models/'));
        expect(calls.map((request) => request.path)).toEqual([
            '/v1beta/models/gemini-3-flash:generateContent',
            '/v1beta/models/gemini-3-flash:batchGenerateContent',
            '/v1beta/models/gemini-3-flash:batchGenerateContent',
            '/v1beta/models/gemini-3-flash:generateContent',
        ]);
    });

    it('refuse a request too large for a batch like a failed request check, without a call', async () => {
        const huge = 'x'.repeat(21_000_000);
        const client = createGenaiClient(clientOptions());
        const refused = await client.generate({ model: 'm', batch: true, messages: [{ role: 'user', content: huge }] });
        expect(refused).toMatchObject({ ok: false, error: { code: 'invalid_request' } });
        expect(refused.ok || [refused.error.settings, refused.error.attempts]).toEqual([undefined, undefined]);
        const gemini = createGenaiModels({ client: clientOptions(), models: { flash: 'm' } });
        const named = await gemini.flash(huge, { batch: true });
        expect(named).toEqual(refused);
        expect(server.received).toHaveLength(0);
    });

    it('collect batch calls in one job despite the model concurrency; limits stay for normal calls', async () => {
        const state = batchServer(40);
        const gemini = createGenaiModels({
            client: clientOptions({ batchWaitSec: 1 }),
            sharedLimits: { rpm: 1, tpm: 10 },
            limiterOptions: { maxWaitMs: 10 },
            models: { flash: { model: 'm', modelLimits: { concurrency: 1, rpd: 1, intervalMs: 60_000 } } },
        });
        const batch = { batch: true };
        const results = await Promise.all([
            gemini.flash('x', batch),
            gemini.flash('y', batch),
            gemini.flash('z', batch),
        ]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(state).toEqual({ submits: 1, maxRunning: 1, sizes: [3] });
        expect(await gemini.flash('normal')).toMatchObject({ ok: false, error: { limiterCode: 'COST_EXCEEDS_LIMIT' } });
    });

    it('retry a retryable error per limiterOptions and fail a fatal one at once', async () => {
        const state = batchServer(1, [apiError(429, '0.001s'), apiError(400)]);
        const gemini = createGenaiModels({
            client: clientOptions({ batchWaitSec: 1 }),
            limiterOptions: { retryDelayMs: 1 },
            models: { flash: 'm' },
        });
        expect(await gemini.flash('x', { batch: true })).toMatchObject({
            ok: false,
            error: {
                code: 'http',
                status: 400,
                retryable: false,
                settings: { name: 'flash', model: 'm', batch: true },
                attempts: 2,
            },
        });
        expect(state.submits).toBe(2);
    });

    it('run a later batch while an earlier one waits for its retry, also with model concurrency 1', async () => {
        const state = batchServer(1, [apiError(503)]);
        const gemini = createGenaiModels({
            client: clientOptions({ batchWaitSec: 1 }),
            limiterOptions: { retryDelayMs: 300 },
            models: { flash: { model: 'm', modelLimits: { concurrency: 1 } } },
        });
        const batch = { batch: true };
        const first = gemini.flash('a', batch);
        await Bun.sleep(50);
        const start = Date.now();
        expect(await gemini.flash('b', batch)).toMatchObject({ ok: true });
        expect(Date.now() - start).toBeLessThan(200);
        expect(await first).toMatchObject({ ok: true, data: { attempts: 2 } });
        expect(state.submits).toBe(3);
    });

    it('leave batch defaults out of explicit batch items', async () => {
        server.reply(() => batchResponse('batches/b1', 'BATCH_STATE_PENDING'));
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { flashBatch: { model: 'gemini-3-flash', defaults: { batch: true } } },
        });
        expect(
            await gemini.flashBatch.batch?.submit([
                { key: '1', request: { messages: [{ role: 'user', content: 'x' }] } },
            ]),
        ).toMatchObject({ ok: true, data: { name: 'batches/b1' } });
    });
});

describe('embedding models', () => {
    /** Answers embed calls with one vector `[n]` per input (n from 1), other calls with text. */
    function embedReply(request: GeminiRequest): Response {
        if (!request.path.endsWith(':batchEmbedContents')) {
            return generateResponse('ok');
        }
        const requests = (request.body?.requests ?? []) as unknown[];
        return Response.json({ embeddings: requests.map((_, index) => ({ values: [index + 1] })) });
    }

    beforeEach(() => {
        vertexServer.received.length = 0;
        server.reply(embedReply);
    });

    /** Request entries of the received embed call at index. */
    function sentEmbeds(index: number): unknown {
        return server.received[index]?.body?.requests;
    }

    /** Configures fake server delaying calls by callMs, recording texts and running calls; errors answer first. */
    function embedServer(callMs = 0, errors: Response[] = []) {
        const texts: string[] = [];
        const runningAtStart: number[] = [];
        let running = 0;
        server.reply(async (request) => {
            const requests = (request.body?.requests ?? []) as { content: { parts: { text: string }[] } }[];
            texts.push(requests.map((entry) => entry.content.parts[0]?.text).join(','));
            runningAtStart.push(running);
            running++;
            await Bun.sleep(callMs);
            running--;
            return errors.shift() ?? embedReply(request);
        });
        return { texts, runningAtStart };
    }

    /** Vertex AI client options for fake Vertex server. */
    function vertexOptions(): GenaiClientOptions {
        return { vertex: { project: 'p', authClient: fakeAuthClient() }, baseUrl: vertexServer.baseUrl, env: {} };
    }

    it('throw for invalid embedding specs at creation', () => {
        const client = clientOptions();
        const invalid: [unknown, RegExp][] = [
            [{ kind: 'other', model: 'm' }, /^Invalid value for option "models\.e\.kind"$/],
            [{ kind: 1, model: 'm' }, /^Invalid value for option "models\.e\.kind"$/],
            [{ kind: 'embed', model: 'm', limit: {} }, /^Option "models\.e\.limit" is not supported$/],
            [{ kind: 'embed', model: '' }, /^Invalid value for option "models\.e\.model"$/],
            [{ kind: 'embed' }, /^Invalid value for option "models\.e\.model"$/],
            [{ kind: 'embed', model: 'm', defaults: null }, /^Invalid value for option "models\.e\.defaults"$/],
            [{ kind: 'embed', model: 'm', defaults: [] }, /^Invalid value for option "models\.e\.defaults"$/],
            [{ kind: 'embed', model: 'm', defaults: 'x' }, /^Invalid value for option "models\.e\.defaults"$/],
            [
                { kind: 'embed', model: 'm', defaults: { temperature: 0 } },
                /^Option "models\.e\.defaults\.temperature" is not supported$/,
            ],
            [
                { kind: 'embed', model: 'm', defaults: { taskType: 'NOPE' } },
                /^Invalid value for option "models\.e\.defaults": Invalid value for option "taskType"$/,
            ],
            [
                { kind: 'embed', model: 'm', defaults: { dimensions: 0 } },
                /^Invalid value for option "models\.e\.defaults": Invalid value for option "dimensions"/,
            ],
            [{ kind: 'embed', model: 'm', charsPerToken: 0 }, /^Invalid value for option "models\.e\.charsPerToken"$/],
            [
                { kind: 'embed', model: 'm', modelLimits: { rpm: 0 } },
                /^Invalid value for option "models\.e\.modelLimits\.rpm"$/,
            ],
        ];
        for (const [spec, message] of invalid) {
            function create() {
                return createGenaiModels({ client, models: { e: spec } } as never);
            }
            expect(create).toThrow(message);
            expect(create).toThrow(TypeError);
        }
    });

    it('embed string and array input with the model id, own defaults and per-call options', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 0.5, stop: ['END'] },
            models: {
                embed: {
                    kind: 'embed',
                    model: 'gemini-embedding-001',
                    defaults: { taskType: 'RETRIEVAL_DOCUMENT', dimensions: 8 },
                },
            },
        });
        const settings = { name: 'embed', model: 'gemini-embedding-001' };
        expect(await gemini.embed('one')).toEqual({
            ok: true,
            data: { embeddings: [[1]], model: 'gemini-embedding-001', ms: expect.any(Number), settings, attempts: 1 },
        });
        expect(server.received[0]?.path).toBe('/v1beta/models/gemini-embedding-001:batchEmbedContents');
        // own defaults only: no generation defaults of the instance
        expect(sentEmbeds(0)).toEqual([
            {
                model: 'models/gemini-embedding-001',
                content: { role: 'user', parts: [{ text: 'one' }] },
                taskType: 'RETRIEVAL_DOCUMENT',
                outputDimensionality: 8,
            },
        ]);
        expect(await gemini.embed(['a', 'b'], { taskType: 'CLUSTERING', dimensions: undefined })).toMatchObject({
            ok: true,
            data: { embeddings: [[1], [2]], settings, attempts: 1 },
        });
        expect(sentEmbeds(1)).toMatchObject([
            { taskType: 'CLUSTERING', outputDimensionality: 8 },
            { taskType: 'CLUSTERING', outputDimensionality: 8 },
        ]);
        await gemini.embed([{ text: 'c', title: 'T' }], { taskType: undefined, dimensions: 4 });
        expect(sentEmbeds(2)).toMatchObject([{ taskType: 'RETRIEVAL_DOCUMENT', title: 'T', outputDimensionality: 4 }]);
    });

    it('have an id, no batch, and work without this', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { embed: { kind: 'embed', model: 'gemini-embedding-001' } },
        });
        expect(gemini.embed.id).toBe('gemini-embedding-001');
        expect(Object.keys(gemini.embed)).toEqual(['id']);
        expect('batch' in gemini.embed).toBe(false);
        const { embed } = gemini;
        expect(await embed('x')).toMatchObject({ ok: true });
        expect(await embed.call(undefined, 'y')).toMatchObject({ ok: true });
    });

    it('answer invalid_request before the queue: no settings, no attempts, no booking and no request', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { embed: { kind: 'embed', model: 'gemini-embedding-001', modelLimits: { rpm: 1 } } },
        });
        const { embed } = gemini;
        const titled = [
            { text: 'a', title: 'A' },
            { text: 'b', title: 'B' },
        ];
        const invalidCalls: [Promise<unknown>, string][] = [
            [embed(''), 'Invalid value for option "input.0": a non-empty text'],
            [embed([]), 'Invalid value for option "input": 1 to 100 texts'],
            [embed(42 as never), 'Invalid value for option "input": 1 to 100 texts'],
            [embed('x', null as never), 'Invalid value for option "options"'],
            [embed('x', [] as never), 'Invalid value for option "options"'],
            [embed('x', { model: 'm' } as never), 'Option "model" is not supported'],
            [embed('x', { signal: 'x' } as never), 'Invalid value for option "signal"'],
            [embed('x', { dimensions: -1 }), 'Invalid value for option "dimensions": a positive integer'],
            [embed('x', { taskType: 'NOPE' as never }), 'Invalid value for option "taskType"'],
            [
                embed([{ text: 'a', title: 'A' }]),
                'Option "input.title" is not supported: only with taskType RETRIEVAL_DOCUMENT',
            ],
            [
                embed(titled, { taskType: 'RETRIEVAL_DOCUMENT' }),
                'Invalid value for option "input": gemini-embedding-001 takes one title per call',
            ],
        ];
        for (const [pending, message] of invalidCalls) {
            expect(await pending).toEqual({ ok: false, error: { code: 'invalid_request', message, retryable: false } });
        }
        expect(server.received).toHaveLength(0);
        expect(await embed('valid')).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(server.received).toHaveLength(1);
    });

    it('answer an aborted signal and Vertex multi-input as unsupported before the queue, aborted first', async () => {
        const settings = { name: 'embed', model: 'gemini-embedding-2' };
        const vertex = createGenaiModels({
            client: vertexOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { embed: { kind: 'embed', model: 'gemini-embedding-2', modelLimits: { rpm: 1 } } },
        });
        expect(await vertex.embed(['a', 'b'])).toEqual({
            ok: false,
            error: {
                code: 'unsupported',
                message: 'Vertex AI embeds one text per call for gemini-embedding-2; got 2',
                retryable: false,
                settings,
                attempts: 0,
            },
        });
        const controller = new AbortController();
        controller.abort();
        expect(await vertex.embed(['a', 'b'], { signal: controller.signal })).toEqual({
            ok: false,
            error: { code: 'aborted', message: 'Request aborted', retryable: false, settings, attempts: 0 },
        });
        expect(vertexServer.received).toHaveLength(0);
        // nothing booked: the one request per minute is still free
        expect(await vertex.embed('a')).toMatchObject({ ok: true, data: { settings, attempts: 1 } });
        expect(vertexServer.received).toHaveLength(1);
        const developer = createGenaiModels({
            client: clientOptions(),
            models: { embed: { kind: 'embed', model: 'gemini-embedding-2' } },
        });
        expect(await developer.embed(['a', 'b'], { signal: controller.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', settings, attempts: 0 },
        });
        expect(server.received).toHaveLength(0);
    });

    it('apply rpm and intervalMs of modelLimits', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: {
                rpm: { kind: 'embed', model: 'm', modelLimits: { rpm: 1 } },
                interval: { kind: 'embed', model: 'm', modelLimits: { intervalMs: 3000 } },
            },
        });
        for (const embed of [gemini.rpm, gemini.interval]) {
            expect(await embed('a')).toMatchObject({ ok: true });
            expect(await embed('b')).toMatchObject({
                ok: false,
                error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED', settings: { model: 'm' }, attempts: 0 },
            });
        }
        expect(server.received).toHaveLength(2);
    });

    it('book text and title characters per charsPerToken against tpm', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: {
                exact: { kind: 'embed', model: 'm', modelLimits: { tpm: 2 } },
                texts: { kind: 'embed', model: 'm', modelLimits: { tpm: 2 } },
                titles: { kind: 'embed', model: 'm', modelLimits: { tpm: 2 } },
                coarse: { kind: 'embed', model: 'm', modelLimits: { tpm: 2 }, charsPerToken: 8 },
            },
        });
        const exceeded = { ok: false, error: { limiterCode: 'COST_EXCEEDS_LIMIT', settings: { name: 'texts' } } };
        expect(await gemini.exact('abcdefgh')).toMatchObject({ ok: true });
        expect(await gemini.texts(['abcd', 'abcde'])).toMatchObject(exceeded);
        expect(
            await gemini.titles([{ text: 'abcd', title: 'abcde' }], { taskType: 'RETRIEVAL_DOCUMENT' }),
        ).toMatchObject({
            ok: false,
            error: { limiterCode: 'COST_EXCEEDS_LIMIT' },
        });
        expect(await gemini.coarse(['abcdefgh', 'abcdefgh'])).toMatchObject({ ok: true });
        expect(server.received).toHaveLength(2);
    });

    it('apply concurrency of modelLimits and end a waiting aborted call without a request', async () => {
        const fake = embedServer(30);
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { embed: { kind: 'embed', model: 'm', modelLimits: { concurrency: 1 } } },
        });
        const controller = new AbortController();
        const calls = [gemini.embed('a'), gemini.embed('b'), gemini.embed('c', { signal: controller.signal })];
        controller.abort();
        const [first, second, aborted] = await Promise.all(calls);
        expect([first?.ok, second?.ok]).toEqual([true, true]);
        expect(aborted).toMatchObject({
            ok: false,
            error: { code: 'aborted', settings: { name: 'embed' }, attempts: 0 },
        });
        expect(fake.texts).toEqual(['a', 'b']);
        expect(fake.runningAtStart).toEqual([0, 0]);
    });

    it('retry 429, 5xx and retryable errors with the attempts counted; fail a 400 at once', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { retryDelayMs: 10 },
            models: { embed: { kind: 'embed', model: 'm' } },
        });
        for (const error of [apiError(429, '0.01s'), apiError(503), apiError(408)]) {
            server.received.length = 0;
            embedServer(0, [error]);
            expect(await gemini.embed('a')).toMatchObject({
                ok: true,
                data: { embeddings: [[1]], settings: { name: 'embed', model: 'm' }, attempts: 2 },
            });
            expect(server.received).toHaveLength(2);
        }
        server.received.length = 0;
        embedServer(0, [apiError(400)]);
        expect(await gemini.embed('a')).toMatchObject({
            ok: false,
            error: { code: 'http', status: 400, settings: { name: 'embed', model: 'm' }, attempts: 1 },
        });
        expect(server.received).toHaveLength(1);
    });

    it('retry a network error after the global backoff', async () => {
        // closes the first connection at once (fast, unlike a refused port on Windows), then answers
        let connections = 0;
        const body = JSON.stringify({ embeddings: [{ values: [1] }] });
        const flaky = Bun.listen({
            hostname: '127.0.0.1',
            port: 0,
            socket: {
                open(socket) {
                    connections++;
                    if (connections === 1) {
                        socket.end();
                    }
                },
                data(socket) {
                    const head = `HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: ${body.length}`;
                    socket.end(`${head}\r\nConnection: close\r\n\r\n${body}`);
                },
            },
        });
        const gemini = createGenaiModels({
            client: clientOptions({ baseUrl: `http://127.0.0.1:${flaky.port}` }),
            limiterOptions: { retryDelayMs: 10 },
            models: { embed: { kind: 'embed', model: 'm' } },
        });
        try {
            expect(await gemini.embed('a')).toMatchObject({
                ok: true,
                data: { embeddings: [[1]], settings: { name: 'embed', model: 'm' }, attempts: 2 },
            });
            expect(connections).toBe(2);
        } finally {
            flaky.stop(true);
        }
    });

    it('give back the token booking of a 429, so its retry fits the same tpm', async () => {
        embedServer(0, [apiError(429, '0.05s')]);
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 1000 },
            models: { embed: { kind: 'embed', model: 'm', modelLimits: { tpm: 3 } } },
        });
        // books 2 tokens per attempt: a second booking fits the minute only if the first went back
        expect(await gemini.embed('abcdefgh')).toMatchObject({ ok: true, data: { attempts: 2 } });
    });

    it('count each input text as one request against rpm', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: {
                full: { kind: 'embed', model: 'm', modelLimits: { rpm: 3 } },
                fits: { kind: 'embed', model: 'm', modelLimits: { rpm: 3 } },
            },
        });
        expect(await gemini.full(['a', 'b'])).toMatchObject({ ok: true });
        expect(await gemini.full(['c', 'd'])).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED', attempts: 0 },
        });
        expect(await gemini.fits(['a', 'b'])).toMatchObject({ ok: true });
        expect(await gemini.fits('c')).toMatchObject({ ok: true });
        expect(server.received).toHaveLength(3);
    });

    it('answer more input texts than rps, rpm or rpd allow as COST_EXCEEDS_LIMIT without a request', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: {
                rps: { kind: 'embed', model: 'm', modelLimits: { rps: 1 } },
                rpm: { kind: 'embed', model: 'm', modelLimits: { rpm: 2 } },
                rpd: { kind: 'embed', model: 'm', modelLimits: { rpd: 2 } },
            },
        });
        for (const [name, embed] of [
            ['rps', gemini.rps],
            ['rpm', gemini.rpm],
            ['rpd', gemini.rpd],
        ] as const) {
            expect(await embed(['a', 'b', 'c'])).toEqual({
                ok: false,
                error: {
                    code: 'limiter',
                    message: expect.any(String),
                    retryable: false,
                    limiterCode: 'COST_EXCEEDS_LIMIT',
                    settings: { name, model: 'm' },
                    attempts: 0,
                },
            });
        }
        expect(server.received).toHaveLength(0);
    });

    it('book requests per input text in custom buckets', async () => {
        const booked: Cost[] = [];
        const bucket: Bucket = {
            getWaitMs: () => 0,
            reserve: (cost) => {
                booked.push(cost);
            },
            settle: () => undefined,
        };
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { embed: { kind: 'embed', model: 'm', modelLimits: { buckets: [bucket] } } },
        });
        expect(await gemini.embed(['a', 'b', 'c'])).toMatchObject({ ok: true });
        expect(await gemini.embed('d')).toMatchObject({ ok: true });
        expect(booked.map((cost) => cost.requests)).toEqual([3, 1]);
    });

    it('keep intervalMs per call: a multi-input call runs, the next call waits', async () => {
        const spaced = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { embed: { kind: 'embed', model: 'm', modelLimits: { intervalMs: 3000 } } },
        });
        expect(await spaced.embed(['a', 'b', 'c'])).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(await spaced.embed(['d', 'e'])).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED', attempts: 0 },
        });
        const fake = embedServer();
        const short = createGenaiModels({
            client: clientOptions(),
            models: { embed: { kind: 'embed', model: 'm', modelLimits: { intervalMs: 60 } } },
        });
        const started = performance.now();
        const results = await Promise.all([short.embed(['a', 'b']), short.embed(['c', 'd'])]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(fake.texts).toEqual(['a,b', 'c,d']);
        expect(performance.now() - started).toBeGreaterThanOrEqual(50);
    });

    it('count embed input texts against an rpm shared with generation models', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            sharedLimits: { rpm: 3 },
            limiterOptions: { maxWaitMs: 100 },
            models: { flash: 'gemini-3-flash', embed: { kind: 'embed', model: 'm' } },
        });
        expect(await gemini.embed(['a', 'b'])).toMatchObject({ ok: true });
        expect(await gemini.flash('c')).toMatchObject({ ok: true });
        expect(await gemini.flash('d')).toMatchObject({ ok: false, error: { limiterCode: 'MAX_WAIT_EXCEEDED' } });
        expect(server.received).toHaveLength(2);
    });

    it('keep sharedLimits.concurrency per call: concurrency 1 runs a multi-input call', async () => {
        const fake = embedServer(30);
        const gemini = createGenaiModels({
            client: clientOptions(),
            sharedLimits: { concurrency: 1 },
            models: { flash: 'gemini-3-flash', embed: { kind: 'embed', model: 'm' } },
        });
        const results = await Promise.all([gemini.embed(['a', 'b', 'c']), gemini.embed(['d', 'e'])]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(fake.texts).toEqual(['a,b,c', 'd,e']);
        expect(fake.runningAtStart).toEqual([0, 0]);
    });

    it('share sharedLimits with generation models', async () => {
        const rpm = createGenaiModels({
            client: clientOptions(),
            sharedLimits: { rpm: 1 },
            limiterOptions: { maxWaitMs: 100 },
            models: { flash: 'gemini-3-flash', embed: { kind: 'embed', model: 'm' } },
        });
        expect(await rpm.flash('a')).toMatchObject({ ok: true });
        expect(await rpm.embed('b')).toMatchObject({ ok: false, error: { limiterCode: 'MAX_WAIT_EXCEEDED' } });
        server.received.length = 0;
        let running = 0;
        const runningAtStart: number[] = [];
        server.reply(async (request) => {
            runningAtStart.push(running);
            running++;
            await Bun.sleep(30);
            running--;
            return embedReply(request);
        });
        const concurrency = createGenaiModels({
            client: clientOptions(),
            sharedLimits: { concurrency: 1 },
            models: { flash: 'gemini-3-flash', embed: { kind: 'embed', model: 'm' } },
        });
        const results = await Promise.all([concurrency.flash('a'), concurrency.embed('b'), concurrency.flash('c')]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(runningAtStart).toEqual([0, 0, 0]);
    });
});

describe('decision models', () => {
    /** Yes/no question used where the question kind does not matter. */
    const NOUL = { type: 'noul', instructions: 'Is it damaged?' } as const;

    beforeEach(() => {
        vertexServer.received.length = 0;
        server.reply(decisionsReply());
    });

    /** Vertex AI client options for fake Vertex server. */
    function vertexOptions(): GenaiClientOptions {
        return { vertex: { project: 'p', authClient: fakeAuthClient() }, baseUrl: vertexServer.baseUrl, env: {} };
    }

    /** Configures fake server delaying calls by callMs, recording running calls; errors answer first. */
    function decideServer(callMs = 0, errors: Response[] = []) {
        const runningAtStart: number[] = [];
        let running = 0;
        const reply = decisionsReply();
        server.reply(async (request) => {
            runningAtStart.push(running);
            running++;
            await Bun.sleep(callMs);
            running--;
            return errors.shift() ?? reply(request);
        });
        return { runningAtStart };
    }

    /** Configures fake server answering decisions with fixed input token usage. */
    function usageServer(inputTokens: number): void {
        server.reply(() =>
            Response.json({
                answers: { a: { type: 'noul', noul: 0.8, confidence: 0.8 } },
                usage: { total_input_tokens: inputTokens, total_output_tokens: 0 },
            }),
        );
    }

    it('throw for invalid decision specs at creation', () => {
        const client = clientOptions();
        const invalid: [unknown, RegExp][] = [
            [{ kind: 'decide', defaults: {} }, /^Option "models\.d\.defaults" is not supported$/],
            [{ kind: 'decide', modelLimit: {} }, /^Option "models\.d\.modelLimit" is not supported$/],
            [{ kind: 'decide', model: '' }, /^Invalid value for option "models\.d\.model"$/],
            [{ kind: 'decide', model: '  ' }, /^Invalid value for option "models\.d\.model"$/],
            [{ kind: 'decide', model: 1 }, /^Invalid value for option "models\.d\.model"$/],
            [{ kind: 'decide', charsPerToken: 0 }, /^Invalid value for option "models\.d\.charsPerToken"$/],
            [{ kind: 'decide', modelLimits: { rpm: 0 } }, /^Invalid value for option "models\.d\.modelLimits\.rpm"$/],
        ];
        for (const [spec, message] of invalid) {
            function create() {
                return createGenaiModels({ client, models: { d: spec } } as never);
            }
            expect(create).toThrow(message);
            expect(create).toThrow(TypeError);
        }
    });

    it('decide with the default or own model id and return the answers; generation defaults not applied', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            defaults: { temperature: 0.5, stop: ['END'] },
            models: {
                decide: { kind: 'decide' },
                own: { kind: 'decide', model: 'models/gemini-decision-1' },
            },
        });
        expect(gemini.decide.id).toBe('gemini-decision-1');
        expect(gemini.own.id).toBe('models/gemini-decision-1');
        const questions = {
            damaged: NOUL,
            intent: { type: 'choice', instructions: 'Intent?', criteria: { refund: '', exchange: '' } },
        } as const;
        expect(await gemini.decide({ input: 'The mug arrived cracked.', questions })).toEqual({
            ok: true,
            data: {
                answers: {
                    damaged: { type: 'noul', noul: 0.8, confidence: 0.8 },
                    intent: {
                        type: 'choice',
                        choice: 'refund',
                        probabilities: { refund: 0.7, exchange: 0.3 },
                        confidence: 0.7,
                    },
                },
                usage: { inputTokens: 6, outputTokens: 0 },
                model: 'gemini-decision-1',
                ms: expect.any(Number),
                settings: { name: 'decide', model: 'gemini-decision-1' },
                attempts: 1,
            },
        });
        expect(server.received[0]?.path).toBe('/v1beta/decisions');
        // own fields only: no generation defaults of the instance
        expect(server.received[0]?.body).toEqual({
            model: 'gemini-decision-1',
            input: 'The mug arrived cracked.',
            questions,
        });
        expect(await gemini.own({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: true,
            data: { model: 'models/gemini-decision-1', settings: { name: 'own', model: 'models/gemini-decision-1' } },
        });
        expect(server.received[1]?.body).toMatchObject({ model: 'models/gemini-decision-1' });
    });

    it('have an id, no batch, and work without this', async () => {
        const gemini = createGenaiModels({ client: clientOptions(), models: { decide: { kind: 'decide' } } });
        expect(Object.keys(gemini.decide)).toEqual(['id']);
        expect('batch' in gemini.decide).toBe(false);
        const { decide } = gemini;
        expect(await decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({ ok: true });
        expect(await decide.call(undefined, { input: 'y', questions: { a: NOUL } })).toMatchObject({ ok: true });
    });

    it('answer invalid_request before the queue: no settings, no attempts, no booking and no request', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { decide: { kind: 'decide', modelLimits: { rpm: 1 } } },
        });
        const { decide } = gemini;
        const valid = { input: 'x', questions: { a: NOUL } };
        const invalidCalls: [Promise<unknown>, string][] = [
            [decide(null as never), 'Invalid value for option "request"'],
            [decide('x' as never), 'Invalid value for option "request"'],
            [decide([] as never), 'Invalid value for option "request"'],
            [
                decide({ ...valid, model: 'gemini-decision-1' } as never),
                'Option "model" is not supported: the model of a name is fixed',
            ],
            [decide({ ...valid, extra: 1 } as never), 'Option "request.extra" is not supported'],
            [
                decide({ input: 'x', questions: { a: { type: 'rank', instructions: 'i' } } } as never),
                'Invalid value for option "questions.a.type": noul, choice or score',
            ],
            [
                decide({ input: ' ', questions: { a: NOUL } }),
                'Invalid value for option "input": a non-empty text or 1 or more parts',
            ],
            [decide(valid, null as never), 'Invalid value for option "options"'],
            [decide(valid, [] as never), 'Invalid value for option "options"'],
            [decide(valid, { config: {} } as never), 'Option "config" is not supported'],
            [decide(valid, { signal: 'x' } as never), 'Invalid value for option "signal"'],
        ];
        for (const [pending, message] of invalidCalls) {
            expect(await pending).toEqual({ ok: false, error: { code: 'invalid_request', message, retryable: false } });
        }
        expect(server.received).toHaveLength(0);
        // nothing booked: the one request per minute is still free
        expect(await decide(valid)).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(server.received).toHaveLength(1);
    });

    it('answer Vertex as unsupported and an aborted signal as aborted before the queue, aborted first', async () => {
        const settings = { name: 'decide', model: 'gemini-decision-1' };
        const vertex = createGenaiModels({ client: vertexOptions(), models: { decide: { kind: 'decide' } } });
        const valid = { input: 'x', questions: { a: NOUL } };
        expect(await vertex.decide(valid)).toEqual({
            ok: false,
            error: {
                code: 'unsupported',
                message: 'Decisions API is not supported on Vertex AI: Gemini Developer API only',
                retryable: false,
                settings,
                attempts: 0,
            },
        });
        const controller = new AbortController();
        controller.abort();
        expect(await vertex.decide(valid, { signal: controller.signal })).toEqual({
            ok: false,
            error: { code: 'aborted', message: 'Request aborted', retryable: false, settings, attempts: 0 },
        });
        expect(await vertex.decide({ input: 'x', questions: {} })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request' },
        });
        expect(vertexServer.received).toHaveLength(0);
        const developer = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { decide: { kind: 'decide', modelLimits: { rpm: 1 } } },
        });
        expect(await developer.decide(valid, { signal: controller.signal })).toEqual({
            ok: false,
            error: { code: 'aborted', message: 'Request aborted', retryable: false, settings, attempts: 0 },
        });
        expect(await developer.decide(valid)).toMatchObject({ ok: true, data: { settings, attempts: 1 } });
        expect(server.received).toHaveLength(1);
    });

    it('apply rpm and concurrency of modelLimits', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { rpm: { kind: 'decide', modelLimits: { rpm: 1 } } },
        });
        const valid = { input: 'x', questions: { a: NOUL } };
        expect(await gemini.rpm(valid)).toMatchObject({ ok: true });
        expect(await gemini.rpm(valid)).toMatchObject({
            ok: false,
            error: {
                code: 'limiter',
                limiterCode: 'MAX_WAIT_EXCEEDED',
                settings: { name: 'rpm', model: 'gemini-decision-1' },
                attempts: 0,
            },
        });
        expect(server.received).toHaveLength(1);
        const fake = decideServer(30);
        const single = createGenaiModels({
            client: clientOptions(),
            models: { decide: { kind: 'decide', modelLimits: { concurrency: 1 } } },
        });
        const results = await Promise.all([single.decide(valid), single.decide(valid), single.decide(valid)]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(fake.runningAtStart).toEqual([0, 0, 0]);
    });

    it('book input, instructions and criteria characters per charsPerToken and images against tpm', async () => {
        const model = { kind: 'decide', modelLimits: { tpm: 3 } } as const;
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: {
                noul: model,
                noulOver: model,
                choice: model,
                choiceOver: model,
                score: model,
                scoreOver: model,
                coarse: { ...model, charsPerToken: 8 },
                image: { kind: 'decide', modelLimits: { tpm: 1121 } },
                imageOver: { kind: 'decide', modelLimits: { tpm: 1121 } },
            },
        });
        const exceeded = { ok: false, error: { code: 'limiter', limiterCode: 'COST_EXCEEDS_LIMIT' } };
        /** Noul request of 12 characters, or 13 with `over`. */
        function noul(over: boolean) {
            const criteria = { true_description: 'abc', false_description: over ? 'ab' : 'a' };
            return { input: 'abcd', questions: { q: { type: 'noul', instructions: 'abcd', criteria } } } as const;
        }
        // 12 characters: 3 tokens; 13: 4 tokens
        expect(await gemini.noul(noul(false))).toMatchObject({ ok: true });
        expect(await gemini.noulOver(noul(true))).toMatchObject(exceeded);
        expect(await gemini.coarse(noul(true))).toMatchObject({ ok: true });
        const choiceInstructions = [{ type: 'text', text: 'ab' }] as const;
        expect(
            await gemini.choice({
                input: 'abcd',
                questions: { q: { type: 'choice', instructions: choiceInstructions, criteria: { ab: '', cd: 'ab' } } },
            }),
        ).toMatchObject({ ok: true });
        expect(
            await gemini.choiceOver({
                input: 'abcd',
                questions: { q: { type: 'choice', instructions: choiceInstructions, criteria: { abc: '', cd: 'ab' } } },
            }),
        ).toMatchObject(exceeded);
        const scoreInstructions = { type: 'text', text: 'abcd' } as const;
        expect(
            await gemini.score({
                input: [{ type: 'text', text: 'abcd' }],
                questions: { q: { type: 'score', instructions: scoreInstructions, criteria: ['ab', 'cd'] } },
            }),
        ).toMatchObject({ ok: true });
        expect(
            await gemini.scoreOver({
                input: [{ type: 'text', text: 'abcd' }],
                questions: { q: { type: 'score', instructions: scoreInstructions, criteria: ['abc', 'cd'] } },
            }),
        ).toMatchObject(exceeded);
        // 1120 per image plus 4 or 5 characters
        const image = { type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } } as const;
        expect(
            await gemini.image({
                input: [{ type: 'text', text: 'ab' }, image],
                questions: { q: { ...NOUL, instructions: 'ab' } },
            }),
        ).toMatchObject({ ok: true });
        expect(
            await gemini.imageOver({
                input: [{ type: 'text', text: 'abc' }, image],
                questions: { q: { ...NOUL, instructions: 'ab' } },
            }),
        ).toMatchObject(exceeded);
        expect(server.received).toHaveLength(5);
    });

    it('settle the token booking against total_input_tokens of the response', async () => {
        // books 6 tokens (24 characters) per call against tpm 10
        const request = {
            input: 'abcdefghijkl',
            questions: { a: { type: 'noul', instructions: 'abcdefghijkl' } },
        } as const;
        const options = { client: clientOptions(), limiterOptions: { maxWaitMs: 100 } };
        usageServer(1);
        const low = createGenaiModels({ ...options, models: { decide: { kind: 'decide', modelLimits: { tpm: 10 } } } });
        expect(await low.decide(request)).toMatchObject({ ok: true, data: { usage: { inputTokens: 1 } } });
        // settled to 1: the next booking of 6 fits
        expect(await low.decide(request)).toMatchObject({ ok: true });
        usageServer(5);
        const high = createGenaiModels({
            ...options,
            models: { decide: { kind: 'decide', modelLimits: { tpm: 10 } } },
        });
        expect(await high.decide(request)).toMatchObject({ ok: true, data: { usage: { inputTokens: 5 } } });
        // settled to 5: the next booking of 6 exceeds the minute
        expect(await high.decide(request)).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED' },
        });
        expect(server.received).toHaveLength(3);
    });

    it('keep the estimated token booking when the response has no usage counts', async () => {
        // books 6 tokens (24 characters) per call against tpm 10
        const request = {
            input: 'abcdefghijkl',
            questions: { a: { type: 'noul', instructions: 'abcdefghijkl' } },
        } as const;
        server.reply(() => Response.json({ answers: { a: { type: 'noul', noul: 0.8, confidence: 0.8 } } }));
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 100 },
            models: { decide: { kind: 'decide', modelLimits: { tpm: 10 } } },
        });
        expect(await gemini.decide(request)).toMatchObject({
            ok: true,
            data: { usage: { inputTokens: null, outputTokens: null }, attempts: 1 },
        });
        // estimate of 6 kept: the next booking of 6 exceeds the minute
        expect(await gemini.decide(request)).toMatchObject({
            ok: false,
            error: { code: 'limiter', limiterCode: 'MAX_WAIT_EXCEEDED' },
        });
        expect(server.received).toHaveLength(1);
    });

    it('end a waiting aborted call without a request', async () => {
        const fake = decideServer(30);
        const gemini = createGenaiModels({
            client: clientOptions(),
            models: { decide: { kind: 'decide', modelLimits: { concurrency: 1 } } },
        });
        const valid = { input: 'x', questions: { a: NOUL } };
        const controller = new AbortController();
        const calls = [gemini.decide(valid), gemini.decide(valid), gemini.decide(valid, { signal: controller.signal })];
        controller.abort();
        const [first, second, aborted] = await Promise.all(calls);
        expect([first?.ok, second?.ok]).toEqual([true, true]);
        expect(aborted).toMatchObject({
            ok: false,
            error: { code: 'aborted', settings: { name: 'decide', model: 'gemini-decision-1' }, attempts: 0 },
        });
        expect(server.received).toHaveLength(2);
        expect(fake.runningAtStart).toEqual([0, 0]);
    });

    it('retry 429, 5xx and retryable errors with the attempts counted; fail a 400 at once', async () => {
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { retryDelayMs: 10 },
            models: { decide: { kind: 'decide' } },
        });
        const settings = { name: 'decide', model: 'gemini-decision-1' };
        const valid = { input: 'x', questions: { a: NOUL } };
        for (const error of [apiError(429, '0.01s'), apiError(503), apiError(408)]) {
            server.received.length = 0;
            decideServer(0, [error]);
            expect(await gemini.decide(valid)).toMatchObject({
                ok: true,
                data: { answers: { a: { type: 'noul' } }, settings, attempts: 2 },
            });
            expect(server.received).toHaveLength(2);
        }
        server.received.length = 0;
        decideServer(0, [decisionError(400, 'invalid_request', 'Bad.')]);
        expect(await gemini.decide(valid)).toMatchObject({
            ok: false,
            error: { code: 'http', status: 400, settings, attempts: 1 },
        });
        expect(server.received).toHaveLength(1);
    });

    it('give back the token booking of a 429, so its retry fits the same tpm', async () => {
        decideServer(0, [apiError(429, '0.05s')]);
        const gemini = createGenaiModels({
            client: clientOptions(),
            limiterOptions: { maxWaitMs: 1000 },
            models: { decide: { kind: 'decide', modelLimits: { tpm: 3 } } },
        });
        // books 2 tokens per attempt: a second booking fits the minute only if the first went back
        expect(
            await gemini.decide({ input: 'abcd', questions: { a: { type: 'noul', instructions: 'abcd' } } }),
        ).toMatchObject({
            ok: true,
            data: { attempts: 2 },
        });
    });
});

describe('named models of mixed kinds', () => {
    /** Answers decisions, embeddings (one vector per input) and generation calls by path. */
    function mixedReply(): GeminiReply {
        const decisions = decisionsReply();
        return (request) => {
            if (request.path.endsWith(':batchEmbedContents')) {
                const requests = (request.body?.requests ?? []) as unknown[];
                return Response.json({ embeddings: requests.map(() => ({ values: [1] })) });
            }
            return request.path === '/v1beta/decisions' ? decisions(request) : generateResponse('ok');
        };
    }

    const models = {
        flash: 'gemini-3-flash',
        embed: { kind: 'embed', model: 'm' },
        decide: { kind: 'decide' },
    } as const;
    const question = { input: 'x', questions: { a: { type: 'noul', instructions: 'i' } } } as const;

    it('share sharedLimits across generation, embedding and decision models', async () => {
        server.reply(mixedReply());
        const rpm = createGenaiModels({
            client: clientOptions(),
            sharedLimits: { rpm: 2 },
            limiterOptions: { maxWaitMs: 100 },
            models,
        });
        expect(await rpm.flash('a')).toMatchObject({ ok: true });
        expect(await rpm.embed('b')).toMatchObject({ ok: true });
        expect(await rpm.decide(question)).toMatchObject({ ok: false, error: { limiterCode: 'MAX_WAIT_EXCEEDED' } });
        server.received.length = 0;
        const reply = mixedReply();
        let running = 0;
        const runningAtStart: number[] = [];
        server.reply(async (request) => {
            runningAtStart.push(running);
            running++;
            await Bun.sleep(30);
            running--;
            return reply(request);
        });
        const concurrency = createGenaiModels({ client: clientOptions(), sharedLimits: { concurrency: 1 }, models });
        const results = await Promise.all([
            concurrency.flash('a'),
            concurrency.embed('b'),
            concurrency.decide(question),
        ]);
        expect(results.every((result) => result.ok)).toBe(true);
        expect(runningAtStart).toEqual([0, 0, 0]);
    });

    it('keep the 429 backoff of one model to its own queue', async () => {
        const reply = mixedReply();
        let embedCalls = 0;
        let firstEmbed: () => void = () => undefined;
        const embedSeen = new Promise<void>((resolve) => {
            firstEmbed = resolve;
        });
        server.reply((request) => {
            if (request.path.endsWith(':batchEmbedContents') && embedCalls++ === 0) {
                firstEmbed();
                return apiError(429, '0.3s');
            }
            return reply(request);
        });
        const gemini = createGenaiModels({ client: clientOptions(), sharedLimits: { rpm: 100 }, models });
        const embedding = gemini.embed('a');
        await embedSeen;
        const startedAt = performance.now();
        expect(await gemini.flash('b')).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(await gemini.decide(question)).toMatchObject({ ok: true, data: { attempts: 1 } });
        expect(performance.now() - startedAt).toBeLessThan(250);
        expect(await embedding).toMatchObject({ ok: true, data: { attempts: 2 } });
        expect(server.received.map((request) => request.path.split('/').pop())).toEqual([
            'm:batchEmbedContents',
            'gemini-3-flash:generateContent',
            'decisions',
            'm:batchEmbedContents',
        ]);
    });
});
