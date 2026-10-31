// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeEach, describe, expect, it } from 'bun:test';
import { createGenaiClient } from '../../../src/core/client.js';
import { MAX_EMBED_INPUTS } from '../../../src/core/embed.js';
import type { GenaiClient, GenaiEmbedRequest } from '../../../src/types.js';
import { apiError, startFakeGemini } from '../fake-gemini.js';
import { fakeAuthClient } from '../fake-vertex.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const server = startFakeGemini();

afterAll(() => {
    server.stop();
});
beforeEach(() => {
    server.received.length = 0;
});

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates Developer API client configured for fake server. */
function client(baseUrl = server.baseUrl): GenaiClient {
    return createGenaiClient({ apiKey: 'test-key', baseUrl, env: {} });
}

/** Creates HTTP response with embedding vectors for batchEmbedContents calls. */
function embedResponse(...vectors: number[][]): Response {
    return Response.json({ embeddings: vectors.map((values) => ({ values })) });
}

/** Extracts request text parts from most recent call. */
function sentTexts(): unknown[] {
    const requests = (server.received.at(-1)?.body?.requests ?? []) as { content: { parts: { text: string }[] } }[];
    return requests.map((request) => request.content.parts[0]?.text);
}

describe('embed on the Developer API', () => {
    it('sends one request per input to batchEmbedContents and returns the vectors in input order', async () => {
        server.reply(() => embedResponse([1, 0], [0, 1]));
        const result = await client().embed({
            model: 'gemini-embedding-001',
            input: ['first', 'second'],
            taskType: 'RETRIEVAL_DOCUMENT',
            dimensions: 2,
        });
        expect(result).toMatchObject({
            ok: true,
            data: {
                embeddings: [
                    [1, 0],
                    [0, 1],
                ],
                model: 'gemini-embedding-001',
            },
        });
        expect(server.received).toHaveLength(1);
        expect(server.received[0]?.path).toBe('/v1beta/models/gemini-embedding-001:batchEmbedContents');
        expect(server.received[0]?.body).toEqual({
            requests: [
                {
                    model: 'models/gemini-embedding-001',
                    content: { role: 'user', parts: [{ text: 'first' }] },
                    taskType: 'RETRIEVAL_DOCUMENT',
                    outputDimensionality: 2,
                },
                {
                    model: 'models/gemini-embedding-001',
                    content: { role: 'user', parts: [{ text: 'second' }] },
                    taskType: 'RETRIEVAL_DOCUMENT',
                    outputDimensionality: 2,
                },
            ],
        });
    });

    it('takes a single string and sends the shared title as title field', async () => {
        server.reply(() => embedResponse([0.5]));
        expect(await client().embed({ model: 'gemini-embedding-001', input: 'only' })).toMatchObject({
            ok: true,
            data: { embeddings: [[0.5]] },
        });
        expect(server.received[0]?.body?.requests).toEqual([
            { model: 'models/gemini-embedding-001', content: { role: 'user', parts: [{ text: 'only' }] } },
        ]);
        server.reply(() => embedResponse([1], [2]));
        const titled: GenaiEmbedRequest = {
            model: 'gemini-embedding-001',
            input: [
                { text: 'a', title: 'Methods' },
                { text: 'b', title: 'Methods' },
            ],
            taskType: 'RETRIEVAL_DOCUMENT',
        };
        expect(await client().embed(titled)).toMatchObject({ ok: true });
        expect(server.received[1]?.body?.requests).toMatchObject([{ title: 'Methods' }, { title: 'Methods' }]);
    });

    it('writes task and titles as text prefixes for gemini-embedding-2 and sends no taskType', async () => {
        server.reply(() => embedResponse([1], [2]));
        const documents = await client().embed({
            model: 'gemini-embedding-2',
            input: [{ text: 'Cells were incubated.', title: 'Methods' }, 'Untitled text.'],
            taskType: 'RETRIEVAL_DOCUMENT',
        });
        expect(documents).toMatchObject({ ok: true });
        expect(sentTexts()).toEqual([
            'title: Methods | text: Cells were incubated.',
            'title: none | text: Untitled text.',
        ]);
        expect(JSON.stringify(server.received[0]?.body)).not.toContain('taskType');
        for (const [taskType, prefix] of [
            ['RETRIEVAL_QUERY', 'task: search result | query: '],
            ['QUESTION_ANSWERING', 'task: question answering | query: '],
            ['FACT_VERIFICATION', 'task: fact checking | query: '],
            ['CODE_RETRIEVAL_QUERY', 'task: code retrieval | query: '],
            ['SEMANTIC_SIMILARITY', 'task: sentence similarity | query: '],
            ['CLASSIFICATION', 'task: classification | query: '],
            ['CLUSTERING', 'task: clustering | query: '],
        ] as const) {
            server.reply(() => embedResponse([1]));
            await client().embed({ model: 'gemini-embedding-2-preview', input: 'q', taskType });
            expect(sentTexts()).toEqual([`${prefix}q`]);
        }
        server.reply(() => embedResponse([1]));
        await client().embed({ model: 'gemini-embedding-2', input: 'plain' });
        expect(sentTexts()).toEqual(['plain']);
        server.reply(() => embedResponse([1]));
        await client().embed({
            model: 'publishers/google/models/gemini-embedding-2',
            input: 'q',
            taskType: 'CLUSTERING',
        });
        expect(sentTexts()).toEqual(['task: clustering | query: q']);
    });

    it('maps HTTP errors, a wrong vector count and empty vectors', async () => {
        server.reply(() => apiError(429, '9s'));
        expect(await client().embed({ model: 'gemini-embedding-001', input: 'x' })).toMatchObject({
            ok: false,
            error: { code: 'rate_limited', retryable: true, status: 429, retryAfterMs: 9000 },
        });
        server.reply(() => apiError(503));
        expect(await client().embed({ model: 'gemini-embedding-001', input: 'x' })).toMatchObject({
            ok: false,
            error: { code: 'http', retryable: true, status: 503 },
        });
        server.reply(() => embedResponse([1]));
        expect(await client().embed({ model: 'gemini-embedding-001', input: ['x', 'y'] })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response has 1 vectors for 2 inputs', retryable: false },
        });
        server.reply(() => embedResponse([1], []));
        expect(await client().embed({ model: 'gemini-embedding-001', input: ['x', 'y'] })).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'The response has an empty vector' },
        });
    });

    it('maps an aborted call and an unreachable server', async () => {
        const controller = new AbortController();
        controller.abort();
        const request: GenaiEmbedRequest = { model: 'gemini-embedding-001', input: 'x' };
        expect(await client().embed(request, { signal: controller.signal })).toMatchObject({
            ok: false,
            error: { code: 'aborted', retryable: false },
        });
        expect(server.received).toHaveLength(0);
        expect(await client('http://127.0.0.1:9').embed(request)).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true },
        });
    });

    it('answers invalid_request for invalid requests and options before any call', async () => {
        const model = 'gemini-embedding-001';
        const invalid: [unknown, string][] = [
            [null, 'Invalid value for option "request"'],
            [{ model, input: 'x', n: 1 }, 'Option "request.n" is not supported'],
            [{ model: '', input: 'x' }, 'Invalid value for option "model"'],
            [{ model, input: [] }, `Invalid value for option "input": 1 to ${MAX_EMBED_INPUTS} texts`],
            [
                { model, input: Array(MAX_EMBED_INPUTS + 1).fill('x') },
                'Invalid value for option "input": 1 to 100 texts',
            ],
            [{ model, input: 7 }, 'Invalid value for option "input": 1 to 100 texts'],
            [{ model, input: '' }, 'Invalid value for option "input.0": a non-empty text'],
            [{ model, input: ['a', { text: '' }] }, 'Invalid value for option "input.1.text": a non-empty text'],
            [{ model, input: [{ text: 'a', name: 'b' }] }, 'Option "input.0.name" is not supported'],
            [
                { model, input: [{ text: 'a', title: '' }], taskType: 'RETRIEVAL_DOCUMENT' },
                'Invalid value for option "input.0.title": a non-empty text',
            ],
            [
                { model, input: [{ text: 'a', title: 'T' }] },
                'Option "input.title" is not supported: only with taskType RETRIEVAL_DOCUMENT',
            ],
            [
                { model, input: [{ text: 'a', title: 'T' }, 'b'], taskType: 'RETRIEVAL_DOCUMENT' },
                'Invalid value for option "input": gemini-embedding-001 takes one title per call',
            ],
            [{ model, input: 'x', taskType: 'SEARCH' }, 'Invalid value for option "taskType"'],
            [{ model, input: 'x', dimensions: 1.5 }, 'Invalid value for option "dimensions": a positive integer'],
        ];
        for (const [request, message] of invalid) {
            expect(await client().embed(request as GenaiEmbedRequest)).toEqual({
                ok: false,
                error: { code: 'invalid_request', message, retryable: false },
            });
        }
        expect(await client().embed({ model, input: 'x' }, { timeout: 1 } as never)).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Option "timeout" is not supported' },
        });
        expect(server.received).toHaveLength(0);
    });
});

describe('embed on Vertex AI', () => {
    /** Vertex AI client configured for fake server. */
    function vertex(): GenaiClient {
        return createGenaiClient({ vertex: { project: 'p', authClient: fakeAuthClient() }, baseUrl: server.baseUrl });
    }

    it('answers unsupported for several texts to a one-content Gemini model without any call', async () => {
        for (const model of ['gemini-embedding-2', 'models/gemini-embedding-001']) {
            expect(await vertex().embed({ model, input: ['a', 'b'] })).toEqual({
                ok: false,
                error: {
                    code: 'unsupported',
                    message: `Vertex AI embeds one text per call for ${model}; got 2`,
                    retryable: false,
                },
            });
        }
        expect(server.received).toHaveLength(0);
    });

    it('answers invalid_request for different titles before unsupported', async () => {
        const input = [
            { text: 'a', title: 'A' },
            { text: 'b', title: 'B' },
        ];
        const taskType = 'RETRIEVAL_DOCUMENT';
        expect(await vertex().embed({ model: 'models/gemini-embedding-001', input, taskType })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request' },
        });
        expect(await vertex().embed({ model: 'text-embedding-005', input, taskType })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request' },
        });
        expect(server.received).toHaveLength(0);
    });

    it('sends several texts to gemini-embedding-001 as predict instances, vectors in input order', async () => {
        server.reply(() =>
            Response.json({ predictions: [{ embeddings: { values: [1, 0] } }, { embeddings: { values: [0, 1] } }] }),
        );
        const result = await vertex().embed({
            model: 'gemini-embedding-001',
            input: ['first', 'second'],
            dimensions: 2,
        });
        expect(result).toMatchObject({
            ok: true,
            data: {
                embeddings: [
                    [1, 0],
                    [0, 1],
                ],
            },
        });
        expect(server.received).toHaveLength(1);
        expect(server.received[0]?.path).toEndWith('/publishers/google/models/gemini-embedding-001:predict');
        expect(server.received[0]?.body).toEqual({
            instances: [{ content: 'first' }, { content: 'second' }],
            parameters: { outputDimensionality: 2 },
        });
    });
});
