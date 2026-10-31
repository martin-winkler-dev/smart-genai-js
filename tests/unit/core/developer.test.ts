// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeEach, describe, expect, it } from 'bun:test';
import { createGenaiClient } from '../../../src/core/client.js';
import type { GenaiClient, GenaiRequest, GenaiTool } from '../../../src/types.js';
import { apiError, batchResponse, generateBody, generateResponse, startFakeGemini } from '../fake-gemini.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const TEST_KEY = 'test-gemini-key';
const server = startFakeGemini();

afterAll(() => {
    server.stop();
});
beforeEach(() => {
    server.received.length = 0;
});

const schema = { type: 'object', properties: { kind: { type: 'string', enum: ['noun', 'verb'] } }, required: ['kind'] };

const request: GenaiRequest = {
    model: 'gemini-x',
    messages: [
        { role: 'system', content: 'Classify the word.' },
        { role: 'user', content: 'blorf' },
    ],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema } },
};

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates Developer API client configured for fake server. */
function client(baseUrl = server.baseUrl): GenaiClient {
    return createGenaiClient({ apiKey: TEST_KEY, baseUrl, env: {} });
}

describe('createGenaiClient', () => {
    it('takes the key from the option or env and rejects invalid options', () => {
        expect(createGenaiClient({ env: { GEMINI_API_KEY: 'k' } }).mode).toBe('developer');
        const invalid: [unknown, string][] = [
            [{ env: {} }, 'Invalid value for option "apiKey": set it or env GEMINI_API_KEY'],
            [{ apiKey: '' }, 'Invalid value for option "apiKey"'],
            [{ apiKey: 'k', model: 'x' }, 'Option "model" is not supported'],
            [{ apiKey: 'k', timeoutMs: 0 }, 'Invalid value for option "timeoutMs"'],
            [{ apiKey: 'k', baseUrl: 'no url' }, 'Invalid value for option "baseUrl"'],
            [{ apiKey: 'k', vertex: true }, 'Option "apiKey" is not supported with "vertex"'],
            [{ vertex: { region: 'x' } }, 'Option "vertex.region" is not supported'],
            [{ vertex: { project: '' } }, 'Invalid value for option "vertex.project"'],
            [
                { vertex: true, env: {} },
                'Invalid value for option "vertex.project": set it or env GOOGLE_CLOUD_PROJECT',
            ],
            [
                { vertex: { project: 'p', bucket: 'gs://b' }, env: {} },
                'Invalid value for option "vertex.bucket": a bucket name without gs://',
            ],
            [null, 'Invalid value for option "options"'],
        ];
        for (const [options, message] of invalid) {
            expect(() => createGenaiClient(options as never)).toThrow(new TypeError(message));
        }
        expect(() => createGenaiClient({ vertex: { project: 'p', credentialsFile: 'missing.json' }, env: {} })).toThrow(
            /vertex\.credentialsFile.*missing\.json is missing or empty/,
        );
    });
});

describe('generate on the Developer API', () => {
    it('sends the converted request with the key and returns text, json, model and usage', async () => {
        server.reply(() => generateResponse('{"kind":"noun"}', { thought: 'thinking...' }));
        const result = await client().generate({ ...request, temperature: 0.2 }, { config: { seed: 3 } });
        expect(result).toEqual({
            ok: true,
            data: {
                text: '{"kind":"noun"}',
                json: { kind: 'noun' },
                reasoning: 'thinking...',
                model: 'gemini-fake-001',
                usage: { inputTokens: 12, outputTokens: 4, reasoningTokens: 3 },
                finishReason: 'STOP',
                ms: expect.any(Number),
                settings: { model: 'gemini-x', temperature: 0.2, json: true, config: { seed: 3 } },
                attempts: 1,
            },
        });
        const sent = server.received[0];
        expect(sent?.path).toBe('/v1beta/models/gemini-x:generateContent');
        expect(sent?.headers.get('x-goog-api-key')).toBe(TEST_KEY);
        expect(sent?.body?.contents).toEqual([{ role: 'user', parts: [{ text: 'blorf' }] }]);
        expect(sent?.body?.systemInstruction).toEqual({ role: 'user', parts: [{ text: 'Classify the word.' }] });
        expect(sent?.body?.generationConfig).toEqual({
            responseMimeType: 'application/json',
            responseJsonSchema: schema,
            temperature: 0.2,
            seed: 3,
        });
    });

    it('returns plain text without json or reasoning for a text request and tolerates a code fence for JSON', async () => {
        server.reply(() => generateResponse('hello'));
        const plain = await client().generate({ model: 'm', messages: [{ role: 'user', content: 'hi' }] });
        expect(plain.ok && plain.data).toMatchObject({ text: 'hello' });
        expect(plain.ok && 'json' in plain.data).toBe(false);
        expect(plain.ok && 'reasoning' in plain.data).toBe(false);
        server.reply(() => generateResponse('```json\n{"kind":"verb"}\n```'));
        expect(await client().generate(request)).toMatchObject({ ok: true, data: { json: { kind: 'verb' } } });
        server.reply(() => generateResponse('```json{"kind":"noun"}```'));
        expect(await client().generate(request)).toMatchObject({ ok: true, data: { json: { kind: 'noun' } } });
        server.reply(() => generateResponse('```JSON\n{"kind":"adverb"}\n```'));
        expect(await client().generate(request)).toMatchObject({ ok: true, data: { json: { kind: 'adverb' } } });
    });

    it('maps truncated, blocked and unparsable answers with text and usage', async () => {
        server.reply(() => generateResponse('{"kind":', { finishReason: 'MAX_TOKENS' }));
        expect(await client().generate(request)).toMatchObject({
            ok: false,
            error: { code: 'truncated', retryable: false, text: '{"kind":', model: 'gemini-fake-001' },
        });
        server.reply(() =>
            Response.json({ ...generateBody(''), candidates: [], promptFeedback: { blockReason: 'SAFETY' } }),
        );
        expect(await client().generate(request)).toMatchObject({
            ok: false,
            error: { code: 'blocked', message: 'Prompt blocked: SAFETY' },
        });
        server.reply(() => generateResponse('', { finishReason: 'RECITATION' }));
        expect(await client().generate(request)).toMatchObject({ ok: false, error: { code: 'blocked' } });
        server.reply(() => generateResponse('{"kind":"no', { finishReason: 'SAFETY' }));
        expect(await client().generate(request)).toMatchObject({
            ok: false,
            error: { code: 'blocked', text: '{"kind":"no' },
        });
        server.reply(() => Response.json({ ...generateBody(''), candidates: [] }));
        expect(await client().generate(request)).toMatchObject({ ok: false, error: { code: 'unexpected' } });
        server.reply(() => generateResponse('not json'));
        const broken = await client().generate(request);
        expect(broken).toMatchObject({ ok: false, error: { code: 'invalid_json', text: 'not json' } });
        expect(!broken.ok && broken.error.usage).toEqual({ inputTokens: 12, outputTokens: 4, reasoningTokens: 3 });
    });

    it('puts the answer with its reasoning on answer errors as output, and the settings on every error', async () => {
        function answered(text: string, finishReason: string) {
            return {
                text,
                reasoning: 'thinking...',
                model: 'gemini-fake-001',
                usage: { inputTokens: 12, outputTokens: 4, reasoningTokens: 3 },
                finishReason,
                ms: expect.any(Number),
            };
        }
        const settings = { model: 'gemini-x', max_tokens: 5, json: true };
        const cases: [string, string, string][] = [
            ['{"kind":', 'MAX_TOKENS', 'truncated'],
            ['{"kind":"no', 'SAFETY', 'blocked'],
            ['not json', 'STOP', 'invalid_json'],
        ];
        for (const [text, finishReason, code] of cases) {
            server.reply(() => generateResponse(text, { finishReason, thought: 'thinking...' }));
            const result = await client().generate({ ...request, max_completion_tokens: 5 });
            expect(result).toMatchObject({ ok: false, error: { code, text, settings, attempts: 1 } });
            expect(!result.ok && result.error.output).toEqual(answered(text, finishReason));
        }
        server.reply(() => apiError(400));
        const http = await client().generate(request);
        expect(http).toMatchObject({
            ok: false,
            error: { code: 'http', settings: { model: 'gemini-x' }, attempts: 1 },
        });
        expect(!http.ok && http.error.output).toBeUndefined();
        const invalid = await client().generate({ ...request, temperature: 'hot' as never });
        expect(invalid).toMatchObject({ ok: false, error: { code: 'invalid_request' } });
        expect(!invalid.ok && invalid.error.settings).toBeUndefined();
    });

    it('maps HTTP errors with one request each: 429 with RetryInfo, 503 retryable, 400 fatal', async () => {
        server.reply(() => apiError(429, '17s'));
        expect(await client().generate(request)).toMatchObject({
            ok: false,
            error: { code: 'rate_limited', retryable: true, status: 429, retryAfterMs: 17_000 },
        });
        server.reply(() => apiError(503));
        expect(await client().generate(request)).toMatchObject({
            ok: false,
            error: { code: 'http', retryable: true, status: 503 },
        });
        server.reply(() => apiError(400));
        const fatal = await client().generate(request);
        expect(fatal).toMatchObject({ ok: false, error: { code: 'http', retryable: false, status: 400 } });
        expect(server.received).toHaveLength(3);
        expect(JSON.stringify(fatal)).not.toContain(TEST_KEY);
    });

    it('maps an aborted call and an unreachable server, and answers invalid_request for invalid input', async () => {
        const controller = new AbortController();
        controller.abort();
        expect(await client().generate(request, { signal: controller.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false, attempts: 0 },
        });
        expect(server.received).toHaveLength(0);
        expect(await client('http://127.0.0.1:9').generate(request)).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true },
        });
        server.received.length = 0;
        const invalid: [Promise<unknown>, string][] = [
            [client().generate({ ...request, n: 2 } as GenaiRequest), 'Option "n" is not supported'],
            [
                createGenaiClient({
                    apiKey: TEST_KEY,
                    baseUrl: server.baseUrl,
                    env: {},
                    defaults: { temperature: 0 },
                }).generate({ ...request, typo: undefined } as GenaiRequest),
                'Option "typo" is not supported',
            ],
            [client().generate(request, { retries: 1 } as never), 'Option "retries" is not supported'],
            [
                client().generate(request, { config: { abortSignal: controller.signal } }),
                'Option "config.abortSignal" is not supported',
            ],
            [client().generate(request, { config: 'x' } as never), 'Invalid value for option "config"'],
            [client().generate(null as never), 'Invalid value for option "request"'],
            [client().generate(request, null as never), 'Invalid value for option "options"'],
            [client().listModels({ page: 1 } as never), 'Option "page" is not supported'],
            [client().listModels({ signal: 'x' } as never), 'Invalid value for option "signal"'],
            [client().generate(request, { signal: 'x' } as never), 'Invalid value for option "signal"'],
        ];
        for (const [pending, message] of invalid) {
            expect(await pending).toEqual({
                ok: false,
                error: { code: 'invalid_request', message, retryable: false },
            });
        }
        expect(server.received).toHaveLength(0);
    });
});

describe('tool calls on the Developer API', () => {
    const weather: GenaiTool = {
        type: 'function',
        function: {
            name: 'get_weather',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
    };

    /** Returns response with two function calls, first containing thought signature without id. */
    function callsResponse(): Response {
        return Response.json({
            candidates: [
                {
                    content: {
                        role: 'model',
                        parts: [
                            { functionCall: { name: 'get_weather', args: { city: 'Rome' } }, thoughtSignature: 'c2ln' },
                            { functionCall: { id: 'g-2', name: 'get_weather', args: { city: 'Oslo' } } },
                        ],
                    },
                    finishReason: 'STOP',
                },
            ],
            usageMetadata: { promptTokenCount: 30, candidatesTokenCount: 10 },
            modelVersion: 'gemini-fake-001',
        });
    }

    it('runs a round trip: calls, tool results, final answer', async () => {
        server.reply(callsResponse, () => generateResponse('Rome 21 °C, Oslo 5 °C'));
        const messages: GenaiRequest['messages'] = [{ role: 'user', content: 'Weather in Rome and Oslo?' }];
        const asked = await client().generate({ model: 'gemini-3-flash', messages, tools: [weather] });
        expect(asked).toMatchObject({ ok: true, data: { text: '', finishReason: 'STOP' } });
        const toolCalls = asked.ok ? (asked.data.toolCalls ?? []) : [];
        expect(toolCalls).toEqual([
            {
                id: expect.stringMatching(/^call_[0-9a-f]{32}$/),
                type: 'function',
                function: { name: 'get_weather', arguments: '{"city":"Rome"}' },
                extra_content: { google: { thought_signature: 'c2ln' } },
            },
            { id: 'g-2', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Oslo"}' } },
        ]);
        expect(server.received[0]?.body?.tools).toEqual([
            { functionDeclarations: [{ name: 'get_weather', parametersJsonSchema: weather.function.parameters }] },
        ]);
        const first = toolCalls[0]?.id as string;
        messages.push(
            { role: 'assistant', content: asked.ok ? asked.data.text : '', tool_calls: toolCalls },
            { role: 'tool', tool_call_id: first, content: '{"celsius":21}' },
            { role: 'tool', tool_call_id: 'g-2', content: '5 °C' },
        );
        const answered = await client().generate({ model: 'gemini-3-flash', messages, tools: [weather] });
        expect(answered).toMatchObject({ ok: true, data: { text: 'Rome 21 °C, Oslo 5 °C' } });
        expect(answered.ok && 'toolCalls' in answered.data).toBe(false);
        expect(server.received[1]?.body?.contents).toEqual([
            { role: 'user', parts: [{ text: 'Weather in Rome and Oslo?' }] },
            {
                role: 'model',
                parts: [
                    {
                        functionCall: { id: first, name: 'get_weather', args: { city: 'Rome' } },
                        thoughtSignature: 'c2ln',
                    },
                    { functionCall: { id: 'g-2', name: 'get_weather', args: { city: 'Oslo' } } },
                ],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { id: first, name: 'get_weather', response: { celsius: 21 } } },
                    { functionResponse: { id: 'g-2', name: 'get_weather', response: { output: '5 °C' } } },
                ],
            },
        ]);
    });

    it('returns calls instead of json for a JSON request and maps failed tool calls', async () => {
        server.reply(callsResponse);
        const called = await client().generate({ ...request, tools: [weather], tool_choice: 'auto' });
        expect(called).toMatchObject({
            ok: true,
            data: { toolCalls: [{ function: { name: 'get_weather' } }, { id: 'g-2' }] },
        });
        expect(called.ok && 'json' in called.data).toBe(false);
        expect(server.received[0]?.body?.toolConfig).toEqual({ functionCallingConfig: { mode: 'AUTO' } });
        server.reply(() => generateResponse('', { finishReason: 'MALFORMED_FUNCTION_CALL' }));
        expect(await client().generate({ ...request, tools: [weather] })).toMatchObject({
            ok: false,
            error: {
                code: 'unexpected',
                retryable: true,
                message: 'Tool call failed, finish reason MALFORMED_FUNCTION_CALL',
            },
        });
        server.reply(() => generateResponse('', { finishReason: 'MISSING_THOUGHT_SIGNATURE' }));
        expect(await client().generate({ ...request, tools: [weather] })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', retryable: false },
        });
    });
});

describe('batch on the Developer API', () => {
    it('submits inline requests with key and format in metadata', async () => {
        server.reply(() => batchResponse('batches/b1', 'BATCH_STATE_PENDING'));
        const batch = client().batch;
        const plain: GenaiRequest = {
            model: 'gemini-x',
            messages: [{ role: 'user', content: 'snarf' }],
            temperature: 0,
        };
        const submitted = await batch?.submit([
            { key: '1', request },
            { key: '2', request: plain },
        ]);
        expect(submitted).toEqual({
            ok: true,
            data: { name: 'batches/b1', state: 'JOB_STATE_PENDING', phase: 'running' },
        });
        const sent = server.received[0];
        expect(sent?.path).toBe('/v1beta/models/gemini-x:batchGenerateContent');
        expect(sent?.body?.batch.displayName).toBe('smart-genai 2 requests');
        const requests = sent?.body?.batch.inputConfig.requests.requests;
        expect(requests.map((entry: { metadata: unknown }) => entry.metadata)).toEqual([
            { key: '1', format: 'json' },
            { key: '2' },
        ]);
        expect(requests[0].request.generationConfig.responseJsonSchema).toEqual(schema);
        expect(requests[1].request.generationConfig.temperature).toBe(0);
        const bytes = batch?.requestBytes(request);
        expect(bytes?.ok && bytes.data).toBeGreaterThan(100);
        expect(batch?.requestBytes({ ...request, n: 2 } as GenaiRequest)).toEqual({
            ok: false,
            error: { code: 'invalid_request', message: 'Option "n" is not supported', retryable: false },
        });
        expect(batch?.maxBytes).toBeLessThan(20_000_000);
        expect(
            await batch?.submit([
                { key: '1', request },
                { key: '2', request: { ...plain, model: 'other' } },
            ]),
        ).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Invalid value for option "model": a batch takes one model',
                retryable: false,
            },
        });
    });

    it('polls once and maps results by key; item errors by rpc code; keyless responses are skipped', async () => {
        const batch = client().batch;
        server.reply(() => batchResponse('batches/b1', 'BATCH_STATE_RUNNING'));
        expect(await batch?.check('batches/b1')).toEqual({
            ok: true,
            data: { name: 'batches/b1', state: 'JOB_STATE_RUNNING', phase: 'running' },
        });
        expect([server.received[0]?.method, server.received[0]?.path]).toEqual(['GET', '/v1beta/batches/b1']);
        server.reply(() =>
            batchResponse('batches/b1', 'BATCH_STATE_SUCCEEDED', [
                { response: generateBody('{"kind":"verb"}'), metadata: { key: '2', format: 'json' } },
                { response: generateBody('plain'), metadata: { key: '1' } },
                { response: generateBody('{}') },
                { error: { code: 13, message: 'internal' }, metadata: { key: '4' } },
                { error: { code: 3, message: 'bad schema' }, metadata: { key: '5' } },
                { error: { code: 8, message: 'quota' }, metadata: { key: '6' } },
            ]),
        );
        const done = await batch?.check('batches/b1');
        const results = done?.ok ? done.data.results : [];
        expect(results?.map((item) => item.key)).toEqual(['2', '1', '4', '5', '6']);
        expect(results?.[0]?.result).toMatchObject({
            ok: true,
            data: { json: { kind: 'verb' }, ms: 0, model: 'gemini-fake-001' },
        });
        expect(results?.[1]?.result).toMatchObject({ ok: true, data: { text: 'plain' } });
        expect(results?.[2]?.result).toMatchObject({
            ok: false,
            error: { code: 'http', status: 500, retryable: true },
        });
        expect(results?.[3]?.result).toMatchObject({
            ok: false,
            error: { code: 'http', status: 400, retryable: false },
        });
        expect(results?.[4]?.result).toMatchObject({ ok: false, error: { code: 'rate_limited', retryable: true } });
    });

    it('answers invalid_request for unknown batch options, items and names before any call', async () => {
        const batch = client().batch;
        const withDefaults = createGenaiClient({
            apiKey: TEST_KEY,
            baseUrl: server.baseUrl,
            env: {},
            defaults: { temperature: 0 },
        }).batch;
        const invalid: [Promise<unknown> | undefined, string][] = [
            [
                batch?.submit([{ key: '1', request }], { singal: undefined } as never),
                'Option "singal" is not supported',
            ],
            [batch?.submit([]), 'Invalid value for option "items"'],
            [batch?.submit(null as never), 'Invalid value for option "items"'],
            [batch?.submit([{ key: '1', request }], [] as never), 'Invalid value for option "options"'],
            [batch?.submit([{ key: '1', request }], { displayName: '' }), 'Invalid value for option "displayName"'],
            [
                batch?.submit([{ key: '1', request }], { displayName: 5 } as never),
                'Invalid value for option "displayName"',
            ],
            [withDefaults?.submit([{ key: '1', request }], null as never), 'Invalid value for option "options"'],
            [withDefaults?.submit([{ key: '1', request }], [] as never), 'Invalid value for option "options"'],
            [batch?.check(''), 'Invalid value for option "name"'],
            [batch?.check('batches/b1', { wait: true } as never), 'Option "wait" is not supported'],
        ];
        for (const [pending, message] of invalid) {
            expect(await pending).toEqual({
                ok: false,
                error: { code: 'invalid_request', message, retryable: false },
            });
        }
        expect(server.received).toHaveLength(0);
    });

    it('reports failed and expired jobs without results, and API errors', async () => {
        const batch = client().batch;
        server.reply(() => batchResponse('batches/b2', 'BATCH_STATE_EXPIRED'));
        expect(await batch?.check('batches/b2')).toEqual({
            ok: true,
            data: { name: 'batches/b2', state: 'JOB_STATE_EXPIRED', phase: 'failed' },
        });
        server.reply(() => apiError(404));
        expect(await batch?.check('batches/b2')).toMatchObject({ ok: false, error: { code: 'http', status: 404 } });
        expect(await batch?.submit([{ key: '1', request }])).toMatchObject({ ok: false, error: { code: 'http' } });
        expect(await batch?.cancel('batches/b2')).toMatchObject({ ok: false, error: { code: 'http', status: 404 } });
        server.reply(() => Response.json({}));
        expect(await batch?.submit([{ key: '1', request }])).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'Batch without name', retryable: false },
        });
        server.reply(() =>
            Response.json({
                name: 'batches/b3',
                metadata: {
                    '@type': 'type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch',
                    name: 'batches/b3',
                    state: 'BATCH_STATE_SUCCEEDED',
                    output: { responsesFile: 'files/batch-out' },
                },
            }),
        );
        expect(await batch?.check('batches/b3')).toEqual({
            ok: false,
            error: {
                code: 'unexpected',
                message: 'Batch results are in the file files/batch-out, not inline',
                retryable: false,
            },
        });
        server.reply(() => batchResponse('batches/b4', 'BATCH_STATE_SUCCEEDED', [{ metadata: { key: '1' } }]));
        const emptyItem = await batch?.check('batches/b4');
        expect(emptyItem?.ok && emptyItem.data.results).toEqual([
            {
                key: '1',
                result: {
                    ok: false,
                    error: { code: 'unexpected', message: 'No response in the batch', retryable: false },
                },
            },
        ]);
    });

    it('answers unsupported for removeFiles without any call', async () => {
        server.received.length = 0;
        expect(await client().batch?.removeFiles()).toEqual({
            ok: false,
            error: { code: 'unsupported', message: 'Developer API batches use no bucket', retryable: false },
        });
        expect(server.received.length).toBe(0);
    });
});
