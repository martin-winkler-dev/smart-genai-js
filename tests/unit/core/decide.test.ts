// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeEach, describe, expect, it } from 'bun:test';
import { createGenaiClient } from '../../../src/core/client.js';
import { MAX_DECIDE_QUESTIONS } from '../../../src/core/decide.js';
import type { GenaiClient, GenaiDecideRequest } from '../../../src/types.js';
import { apiError, decisionError, decisionsReply, startFakeGemini } from '../fake-gemini.js';
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
    vertexServer.received.length = 0;
    server.reply(decisionsReply());
});

/** Base64 payload of a 1x1 PNG. */
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

/** Yes/no question used where the question kind does not matter. */
const NOUL = { type: 'noul', instructions: 'Is it damaged?' } as const;

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates Developer API client configured for fake server. */
function client(baseUrl = server.baseUrl, apiKey = 'test-key'): GenaiClient {
    return createGenaiClient({ apiKey, baseUrl, env: {} });
}

/** Choice criteria with `count` options `o0`, `o1`, …; descriptions empty. */
function options(count: number): Record<string, string> {
    return Object.fromEntries(Array.from({ length: count }, (_, index) => [`o${index}`, '']));
}

describe('decide on the Developer API', () => {
    it('sends a noul question to POST /v1beta/decisions with the key header and returns its answer', async () => {
        const result = await client().decide({
            input: 'The blue mug arrived cracked.',
            questions: {
                is_damaged: {
                    type: 'noul',
                    instructions: 'Did the item arrive damaged?',
                    criteria: { true_description: 'Broken or faulty.', false_description: 'Other complaint.' },
                },
            },
        });
        expect(result).toMatchObject({
            ok: true,
            data: {
                answers: { is_damaged: { type: 'noul', noul: 0.8, confidence: 0.8 } },
                usage: { inputTokens: 8, outputTokens: 0 },
                model: 'gemini-decision-1',
            },
        });
        expect(result.ok && typeof result.data.ms).toBe('number');
        const [sent] = server.received;
        expect(server.received).toHaveLength(1);
        expect(sent?.method).toBe('POST');
        expect(sent?.path).toBe('/v1beta/decisions');
        expect(sent?.search).toBe('');
        expect(sent?.headers.get('x-goog-api-key')).toBe('test-key');
        expect(sent?.headers.get('content-type')).toBe('application/json');
        expect(sent?.body).toEqual({
            model: 'gemini-decision-1',
            input: 'The blue mug arrived cracked.',
            questions: {
                is_damaged: {
                    type: 'noul',
                    instructions: 'Did the item arrive damaged?',
                    criteria: { true_description: 'Broken or faulty.', false_description: 'Other complaint.' },
                },
            },
        });
    });

    it('answers choice and score questions with probabilities, legend and confidence', async () => {
        const choice = await client().decide({
            input: 'Refund me now!',
            questions: {
                intent: {
                    type: 'choice',
                    instructions: 'Intent?',
                    criteria: { refund: 'Wants money back', other: '' },
                },
            },
        });
        expect(choice).toMatchObject({
            ok: true,
            data: {
                answers: {
                    intent: { type: 'choice', choice: 'refund', probabilities: { refund: 0.7, other: 0.3 } },
                },
            },
        });
        const score = await client().decide({
            input: 'Refund me now!',
            questions: {
                anger: { type: 'score', instructions: 'How angry?', criteria: ['calm', 'annoyed', 'furious'] },
            },
        });
        expect(score).toMatchObject({
            ok: true,
            data: {
                answers: {
                    anger: {
                        type: 'score',
                        score: 1.625,
                        probabilities: { 0: 0.125, 1: 0.125, 2: 0.75 },
                        legend: { 0: 'calm', 1: 'annoyed', 2: 'furious' },
                        confidence: 0.75,
                    },
                },
            },
        });
    });

    it('sends mixed kinds in one request and keeps a models/ prefix', async () => {
        const result = await client().decide({
            model: 'models/gemini-decision-1',
            input: 'text',
            questions: {
                a: NOUL,
                b: { type: 'choice', instructions: [{ type: 'text', text: 'Pick' }], criteria: { x: '', y: '' } },
                c: { type: 'score', instructions: { type: 'text', text: 'Rate' }, criteria: ['low', 'high'] },
            },
        });
        expect(result).toMatchObject({
            ok: true,
            data: {
                answers: { a: { type: 'noul' }, b: { type: 'choice', choice: 'x' }, c: { type: 'score' } },
                model: 'models/gemini-decision-1',
            },
        });
        expect(server.received).toHaveLength(1);
        expect(server.received[0]?.body?.model).toBe('models/gemini-decision-1');
    });

    it('accepts partial or undefined noul criteria, 255 choice options and a base URL with trailing slash', async () => {
        const result = await client(`${server.baseUrl}/`).decide({
            input: 'x',
            questions: {
                a: { ...NOUL, criteria: { true_description: 'Broken.' } },
                b: { ...NOUL, criteria: { true_description: undefined, false_description: 'Fine.' } as never },
                c: { type: 'choice', instructions: 'i', criteria: options(255) },
            },
        });
        expect(result).toMatchObject({
            ok: true,
            data: { answers: { a: { type: 'noul' }, b: { type: 'noul' }, c: { type: 'choice', choice: 'o0' } } },
        });
        expect(server.received[0]?.path).toBe('/v1beta/decisions');
        expect(server.received[0]?.body?.questions).toMatchObject({
            b: { ...NOUL, criteria: { false_description: 'Fine.' } },
        });
    });

    it('converts chat image parts to image blocks and passes text and media blocks through', async () => {
        const result = await client().decide({
            input: [
                { type: 'text', text: 'Look:' },
                { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG}`, detail: 'high' } },
                { type: 'image', data: PNG, mime_type: 'image/png', resolution: 'ultra_high' },
                { type: 'document', uri: 'https://example.com/a.pdf', mime_type: 'application/pdf' },
                { type: 'video', data: 'AAAA', mime_type: 'video/mp4', resolution: 'low' },
                { type: 'audio', data: 'AAAA', mime_type: 'audio/wav' },
                { type: 'document', data: 'YSxi', mime_type: 'text/csv' },
            ],
            questions: { a: NOUL },
        });
        expect(result).toMatchObject({ ok: true, data: { usage: { inputTokens: 2 + 6 * 258, outputTokens: 0 } } });
        expect(server.received[0]?.body?.input).toEqual([
            { type: 'text', text: 'Look:' },
            { type: 'image', data: PNG, mime_type: 'image/png' },
            { type: 'image', data: PNG, mime_type: 'image/png', resolution: 'ultra_high' },
            { type: 'document', uri: 'https://example.com/a.pdf', mime_type: 'application/pdf' },
            { type: 'video', data: 'AAAA', mime_type: 'video/mp4', resolution: 'low' },
            { type: 'audio', data: 'AAAA', mime_type: 'audio/wav' },
            { type: 'document', data: 'YSxi', mime_type: 'text/csv' },
        ]);
    });

    it('maps 400 invalid_request, 404 not_found and an invalid key to http errors with the API message', async () => {
        server.reply(() => decisionError(400, 'invalid_request', 'Unable to process input image.'));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'Unable to process input image.', retryable: false, status: 400 },
        });
        server.reply(decisionsReply());
        expect(await client().decide({ model: 'gemini-decision-9', input: 'x', questions: { a: NOUL } })).toMatchObject(
            {
                ok: false,
                error: {
                    code: 'http',
                    message: "Model 'gemini-decision-9' not found.",
                    retryable: false,
                    status: 404,
                    model: 'gemini-decision-9',
                },
            },
        );
        expect(await client(server.baseUrl, 'bad-key').decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'API key not valid.', retryable: false, status: 400 },
        });
    });

    it('leaves the document type check to the API and maps its 400 to an http error', async () => {
        const result = await client().decide({
            input: [{ type: 'document', data: 'YQ==', mime_type: 'text/plain' }],
            questions: { a: NOUL },
        });
        expect(result).toMatchObject({
            ok: false,
            error: {
                code: 'http',
                message:
                    "Unsupported mime_type 'text/plain' in document content. Only application/pdf, text/csv, and supported Office/OpenDocument MIME types are supported.",
                status: 400,
                retryable: false,
            },
        });
        expect(server.received).toHaveLength(1);
    });

    it('maps 429 with retry delay, 503 and an empty error body', async () => {
        server.reply(() => apiError(429, '7s'));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'rate_limited', retryable: true, status: 429, retryAfterMs: 7000 },
        });
        server.reply(() => apiError(503));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'fake 503', retryable: true, status: 503 },
        });
        server.reply(() => new Response('', { status: 500 }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'HTTP 500', retryable: true, status: 500 },
        });
    });

    it('maps JSON error bodies without error.message to HTTP <status> and keeps non-JSON bodies', async () => {
        for (const body of ['null', '{}', '[]', '{"error":{"code":500,"status":"INTERNAL"}}', '{"error":null}']) {
            server.reply(() => new Response(body, { status: 500 }));
            expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
                ok: false,
                error: { code: 'http', message: 'HTTP 500', status: 500 },
            });
        }
        server.reply(() => new Response('Bad gateway', { status: 502 }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'Bad gateway', status: 502 },
        });
    });

    it('maps safety error codes and a prompt_feedback block reason to blocked', async () => {
        for (const code of ['safety', 'prohibited_content', 'blocklist', 'image_safety', 'content_blocked']) {
            server.reply(() => decisionError(400, code, 'Input was blocked.'));
            expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toEqual({
                ok: false,
                error: {
                    code: 'blocked',
                    message: `Input blocked (${code}): Input was blocked.`,
                    retryable: false,
                    status: 400,
                    model: 'gemini-decision-1',
                    ms: expect.any(Number),
                },
            });
        }
        server.reply(() =>
            Response.json({ prompt_feedback: { block_reason: 'safety', block_reason_message: 'Unsafe' } }),
        );
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'blocked', message: 'Input blocked (safety): Unsafe', retryable: false },
        });
    });

    it('maps a safety code with an empty message and a block reason without message to blocked', async () => {
        server.reply(() => decisionError(400, 'safety', ''));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'blocked', message: 'Input blocked (safety)', status: 400 },
        });
        server.reply(() => Response.json({ prompt_feedback: { block_reason: 'blocklist' } }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'blocked', message: 'Input blocked (blocklist)', model: 'gemini-decision-1' },
        });
    });

    it('answers unexpected for missing answers, answers of another kind and no JSON object', async () => {
        server.reply(() => Response.json({ answers: {} }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response has no answer for "a"', retryable: false },
        });
        server.reply(() => Response.json({ answers: { a: { type: 'score', score: 1 } } }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The answer for "a" is of type score, not noul' },
        });
        server.reply(() => new Response('not json'));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response is no JSON object' },
        });
        server.reply(() => Response.json({ answers: { a: { type: 'noul', noul: 0.3, confidence: 0.7 } } }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: true,
            data: { usage: { inputTokens: null, outputTokens: null } },
        });
    });

    it('answers unexpected for null answers, inherited keys, array bodies and array answers', async () => {
        server.reply(() => Response.json({ answers: { a: null } }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toEqual({
            ok: false,
            error: {
                code: 'unexpected',
                message: 'The response has no answer for "a"',
                retryable: false,
                model: 'gemini-decision-1',
                ms: expect.any(Number),
            },
        });
        server.reply(() => Response.json({ answers: {} }));
        expect(await client().decide({ input: 'x', questions: { toString: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response has no answer for "toString"' },
        });
        server.reply(() => Response.json([]));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response is no JSON object' },
        });
        server.reply(() => Response.json({ answers: [{ type: 'noul', noul: 0.5, confidence: 0.5 }] }));
        expect(await client().decide({ input: 'x', questions: { 0: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response has no answer for "0"' },
        });
        server.reply(() => Response.json({ answers: { a: 'noul' } }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The answer for "a" is of type undefined, not noul' },
        });
    });

    it('answers questions with Object.prototype names as own keys', async () => {
        const questions = Object.fromEntries([
            ['toString', NOUL],
            ['__proto__', NOUL],
        ]) as Record<string, typeof NOUL>;
        const result = await client().decide({ input: 'x', questions });
        expect(result.ok).toBe(true);
        const answers = result.ok ? result.data.answers : {};
        expect(Object.keys(answers)).toEqual(['toString', '__proto__']);
        expect(Object.hasOwn(answers, '__proto__')).toBe(true);
        expect(Object.getPrototypeOf(answers)).toBe(Object.prototype);
        expect(answers['toString']).toEqual({ type: 'noul', noul: 0.8, confidence: 0.8 });
        expect(Object.keys(server.received[0]?.body?.questions as object)).toEqual(['toString', '__proto__']);
    });

    it('ignores an empty or null prompt_feedback block reason', async () => {
        const answers = { a: { type: 'noul', noul: 0.3, confidence: 0.7 } };
        for (const reason of [null, '']) {
            server.reply(() => Response.json({ prompt_feedback: { block_reason: reason }, answers }));
            expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
                ok: true,
                data: { answers },
            });
        }
        server.reply(() => Response.json({ prompt_feedback: { block_reason: 'safety', block_reason_message: null } }));
        expect(await client().decide({ input: 'x', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'blocked', message: 'Input blocked (safety)' },
        });
    });

    it('maps an aborted call, a timeout and an unreachable server', async () => {
        const controller = new AbortController();
        controller.abort();
        const request: GenaiDecideRequest = { input: 'x', questions: { a: NOUL } };
        expect(await client().decide(request, { signal: controller.signal })).toEqual({
            ok: false,
            error: { code: 'aborted', message: 'Request aborted', retryable: false },
        });
        expect(server.received).toHaveLength(0);
        const running = new AbortController();
        server.reply(async () => {
            running.abort();
            await Bun.sleep(50);
            return Response.json({});
        });
        expect(await client().decide(request, { signal: running.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false, model: 'gemini-decision-1', ms: expect.any(Number) },
        });
        server.reply(async () => {
            await Bun.sleep(200);
            return Response.json({});
        });
        const slow = createGenaiClient({ apiKey: 'k', baseUrl: server.baseUrl, env: {}, timeoutMs: 20 });
        expect(await slow.decide(request)).toMatchObject({
            ok: false,
            error: { code: 'timeout', retryable: true, model: 'gemini-decision-1' },
        });
        expect(await client('http://127.0.0.1:9').decide(request)).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true, model: 'gemini-decision-1', ms: expect.any(Number) },
        });
    });

    it('answers invalid_request for invalid requests and options before any call', async () => {
        const questions = { a: NOUL };
        const many = Object.fromEntries(
            Array.from({ length: MAX_DECIDE_QUESTIONS + 1 }, (_, index) => [`q${index}`, NOUL]),
        );
        const invalid: [unknown, string][] = [
            [null, 'Invalid value for option "request"'],
            [{ input: 'x', questions, temperature: 0 }, 'Option "request.temperature" is not supported'],
            [{ model: '', input: 'x', questions }, 'Invalid value for option "model"'],
            [{ model: '   ', input: 'x', questions }, 'Invalid value for option "model"'],
            [
                { input: [{ type: 'image', data: '   ', mime_type: 'image/png' }], questions },
                'Invalid value for option "input.0.data": a non-empty text',
            ],
            [
                { input: [{ type: 'document', uri: ' ', mime_type: 'application/pdf' }], questions },
                'Invalid value for option "input.0.uri": a non-empty text',
            ],
            [
                { input: [{ type: 'audio', data: 'A', mime_type: '   ' }], questions },
                'Invalid value for option "input.0.mime_type": a non-empty text',
            ],
            [{ input: '', questions }, 'Invalid value for option "input": a non-empty text or 1 or more parts'],
            [{ input: [], questions }, 'Invalid value for option "input": a non-empty text or 1 or more parts'],
            [{ input: 'x', questions: {} }, 'Invalid value for option "questions": 1 to 1000 questions'],
            [{ input: 'x', questions: many }, 'Invalid value for option "questions": 1 to 1000 questions'],
            [{ input: 'x', questions: [NOUL] }, 'Invalid value for option "questions"'],
            [
                { input: 'x', questions: { 'a b': NOUL } },
                'Invalid value for option "questions": key "a b" must be 1 to 64 characters of [A-Za-z0-9_-]',
            ],
            [
                { input: 'x', questions: { [`k${'x'.repeat(64)}`]: NOUL } },
                `Invalid value for option "questions": key "k${'x'.repeat(64)}" must be 1 to 64 characters of [A-Za-z0-9_-]`,
            ],
            [{ input: 'x', questions: { a: { ...NOUL, weight: 1 } } }, 'Option "questions.a.weight" is not supported'],
            [
                { input: 'x', questions: { a: { type: 'rank', instructions: 'i' } } },
                'Invalid value for option "questions.a.type": noul, choice or score',
            ],
            [
                { input: 'x', questions: { a: { type: 'noul', instructions: ' ' } } },
                'Invalid value for option "questions.a.instructions": a non-empty text',
            ],
            [
                { input: 'x', questions: { a: { type: 'noul' } } },
                'Invalid value for option "questions.a.instructions": a non-empty text or text parts',
            ],
            [
                { input: 'x', questions: { a: { type: 'noul', instructions: [] } } },
                'Invalid value for option "questions.a.instructions": a non-empty text or text parts',
            ],
            [
                { input: 'x', questions: { a: { type: 'noul', instructions: [{ type: 'image_url' }] } } },
                'Invalid value for option "questions.a.instructions.0": a non-empty text or text parts',
            ],
            [
                { input: 'x', questions: { a: { ...NOUL, criteria: { yes: 'y' } } } },
                'Option "questions.a.criteria.yes" is not supported',
            ],
            [
                { input: 'x', questions: { a: { ...NOUL, criteria: { true_description: 1 } } } },
                'Invalid value for option "questions.a.criteria.true_description": a text',
            ],
            [
                { input: 'x', questions: { a: { type: 'choice', instructions: 'i', criteria: { only: '' } } } },
                'Invalid value for option "questions.a.criteria": 2 to 255 options',
            ],
            [
                { input: 'x', questions: { a: { type: 'choice', instructions: 'i', criteria: ['x', 'y'] } } },
                'Invalid value for option "questions.a.criteria": an object of option ids to descriptions',
            ],
            [
                { input: 'x', questions: { a: { type: 'choice', instructions: 'i', criteria: { ' x': '', y: '' } } } },
                'Invalid value for option "questions.a.criteria": option id " x" must be non-empty, without outer whitespace or control characters',
            ],
            [
                { input: 'x', questions: { a: { type: 'choice', instructions: 'i', criteria: options(256) } } },
                'Invalid value for option "questions.a.criteria": 2 to 255 options',
            ],
            ...['x ', '', 'a\u0007b'].map((id): [unknown, string] => [
                { input: 'x', questions: { a: { type: 'choice', instructions: 'i', criteria: { [id]: '', y: '' } } } },
                `Invalid value for option "questions.a.criteria": option id "${id}" must be non-empty, without outer whitespace or control characters`,
            ]),
            [
                { input: 'x', questions: { a: { type: 'noul', instructions: { type: 'text', text: '' } } } },
                'Invalid value for option "questions.a.instructions.text": a non-empty text',
            ],
            [
                {
                    input: 'x',
                    questions: { a: { type: 'noul', instructions: [{ type: 'text', text: 'x', extra: 1 }] } },
                },
                'Option "questions.a.instructions.0.extra" is not supported',
            ],
            [
                { input: [{ type: 'image', data: PNG, mime_type: 'image/png', resolution: 'max' }], questions },
                'Invalid value for option "input.0.resolution": low, medium, high, ultra_high',
            ],
            [
                {
                    input: [{ type: 'document', data: 'A', mime_type: 'application/pdf', resolution: 'low' }],
                    questions,
                },
                'Option "input.0.resolution" is not supported',
            ],
            [
                { input: [{ type: 'image', data: '', mime_type: 'image/png' }], questions },
                'Invalid value for option "input.0.data": a non-empty text',
            ],
            [
                { input: [{ type: 'video', uri: '', mime_type: 'video/mp4' }], questions },
                'Invalid value for option "input.0.uri": a non-empty text',
            ],
            [
                { input: 'x', questions: { a: { type: 'choice', instructions: 'i', criteria: { x: 1, y: '' } } } },
                'Invalid value for option "questions.a.criteria.x": a text',
            ],
            [
                { input: 'x', questions: { a: { type: 'score', instructions: 'i', criteria: ['one'] } } },
                'Invalid value for option "questions.a.criteria": 2 to 255 levels',
            ],
            [
                { input: 'x', questions: { a: { type: 'score', instructions: 'i', criteria: Array(256).fill('l') } } },
                'Invalid value for option "questions.a.criteria": 2 to 255 levels',
            ],
            [
                { input: 'x', questions: { a: { type: 'score', instructions: 'i', criteria: ['one', ''] } } },
                'Invalid value for option "questions.a.criteria.1": a non-empty text',
            ],
            [
                { input: new Array(1), questions },
                'Invalid value for option "input.0.type": text, image_url, image, audio, video or document',
            ],
            [
                { input: 'x', questions: { a: { type: 'noul', instructions: new Array(1) } } },
                'Invalid value for option "questions.a.instructions.0": a non-empty text or text parts',
            ],
            [
                { input: [{ text: 'x' }], questions },
                'Invalid value for option "input.0.type": text, image_url, image, audio, video or document',
            ],
            [
                { input: [{ type: 'text', text: '' }], questions },
                'Invalid value for option "input.0.text": a non-empty text',
            ],
            [
                { input: [{ type: 'image_url', image_url: { url: 'https://example.com/a.png' } }], questions },
                'Invalid value for option "input.0.image_url.url": a base64 data URL',
            ],
            [
                {
                    input: [{ type: 'image_url', image_url: { url: `data:image/png;base64,${PNG}`, size: 1 } }],
                    questions,
                },
                'Option "input.0.image_url.size" is not supported',
            ],
            [
                { input: [{ type: 'image', mime_type: 'image/png' }], questions },
                'Invalid value for option "input.0": exactly one of data and uri',
            ],
            [
                { input: [{ type: 'image', data: PNG, uri: 'https://x', mime_type: 'image/png' }], questions },
                'Invalid value for option "input.0": exactly one of data and uri',
            ],
            [
                { input: [{ type: 'image', data: PNG }], questions },
                'Invalid value for option "input.0.mime_type": a non-empty text',
            ],
            [
                { input: [{ type: 'audio', data: 'A', mime_type: 'audio/wav', resolution: 'low' }], questions },
                'Option "input.0.resolution" is not supported',
            ],
            [
                { input: [{ type: 'video', data: 'A', mime_type: 'video/mp4', resolution: 'ultra_high' }], questions },
                'Invalid value for option "input.0.resolution": low, medium, high',
            ],
        ];
        for (const [request, message] of invalid) {
            expect(await client().decide(request as GenaiDecideRequest)).toEqual({
                ok: false,
                error: { code: 'invalid_request', message, retryable: false },
            });
        }
        expect(await client().decide({ input: 'x', questions }, { timeout: 1 } as never)).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Option "timeout" is not supported' },
        });
        expect(server.received).toHaveLength(0);
    });
});

describe('decide on Vertex AI', () => {
    /** Vertex AI client configured for fake Vertex server. */
    function vertex(): GenaiClient {
        return createGenaiClient({
            vertex: { project: 'p', authClient: fakeAuthClient() },
            baseUrl: vertexServer.baseUrl,
            env: {},
        });
    }

    it('answers unsupported without any call, after the request check', async () => {
        expect(await vertex().decide({ input: 'x', questions: { a: NOUL } })).toEqual({
            ok: false,
            error: {
                code: 'unsupported',
                message: 'Decisions API is not supported on Vertex AI: Gemini Developer API only',
                retryable: false,
            },
        });
        expect(await vertex().decide({ input: '', questions: { a: NOUL } })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request' },
        });
        expect(vertexServer.received).toHaveLength(0);
    });

    it('answers aborted before unsupported for an aborted signal', async () => {
        const controller = new AbortController();
        controller.abort();
        expect(await vertex().decide({ input: 'x', questions: { a: NOUL } }, { signal: controller.signal })).toEqual({
            ok: false,
            error: { code: 'aborted', message: 'Request aborted', retryable: false },
        });
        expect(vertexServer.received).toHaveLength(0);
    });
});
