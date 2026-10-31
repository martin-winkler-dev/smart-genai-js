// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeAll, beforeEach, describe, expect, it } from 'bun:test';
import { createRPM } from 'smart-limit';
import { batchTiming } from '../../../src/core/batch/calls.js';
import { createGenaiClient } from '../../../src/core/client.js';
import { assistantMessage, toolMessage } from '../../../src/core/messages.js';
import type { GenaiClientOptions, GenaiMessage, GenaiModel, GenaiModelInput } from '../../../src/types.js';
import { createGenaiModels } from '../../../src/wrapper/named-models.js';
import { apiError, batchResponse, generateBody, generateResponse, startFakeGemini } from '../fake-gemini.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const server = startFakeGemini();

afterAll(() => {
    server.stop();
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
        expect(second?.body?.generationConfig.thinkingConfig).toEqual({ thinkingBudget: 1024, includeThoughts: false });
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
        expect(server.received[2]?.body?.generationConfig.stopSequences).toEqual(['END']);
        expect(server.received[3]?.body?.generationConfig.stopSequences).toEqual(['STOP']);
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
        expect(sent?.body?.batch.inputConfig.requests.requests[0].request.generationConfig).toEqual({ temperature: 0 });
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

    /** Configures fake batch server taking batchMs and tracking peak concurrency. */
    function batchServer(batchMs: number, errors: Response[] = []) {
        const jobs = new Map<string, { at: number; keys: string[]; done: boolean }>();
        const state = { submits: 0, maxRunning: 0 };
        let running = 0;
        server.reply((request) => {
            if (request.path.endsWith(':batchGenerateContent')) {
                state.submits++;
                const error = errors.shift();
                if (error !== undefined) {
                    return error;
                }
                const name = `batches/b${jobs.size + 1}`;
                const requests = request.body?.batch.inputConfig.requests.requests as { metadata: { key: string } }[];
                jobs.set(name, { at: Date.now(), keys: requests.map((entry) => entry.metadata.key), done: false });
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

    it('wait only for the model concurrency, also longer than maxWaitMs; other limits stay for normal calls', async () => {
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
        expect(state).toEqual({ submits: 3, maxRunning: 1 });
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

    it('run a later batch while an earlier one waits for its retry, so the retry wait frees the slot', async () => {
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
