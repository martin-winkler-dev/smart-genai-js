// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live (paid): the whole package on Vertex AI. Needs env GOOGLE_CLOUD_PROJECT, VERTEX_BATCH_BUCKET and gcloud
// application-default credentials; without GOOGLE_CLOUD_PROJECT every case is skipped, without VERTEX_BATCH_BUCKET
// the batch cases. Run with `bun run test:live:vertex`; never part of `bun test`.
// 45 generate requests (core cases on three models, breadth on gemini-3.1-flash-lite), 7 embedding requests,
// one batch job of eight items and one job of two batch calls (each 4 to 6 minutes).

import { describe, expect, it } from 'bun:test';
import { GoogleAuth } from 'google-auth-library';
import { createGcsClient } from '../../src/core/batch/gcs.js';
import { createGenaiClient, createGenaiModels, credentialsFile } from '../../src/index.js';
import type { GenaiClient } from '../../src/types.js';
import {
    BATCH_TIMEOUT_MS,
    BREADTH_CASES,
    batchItems,
    CASE_TIMEOUT_MS,
    coreCases,
    expectBatchCallPair,
    expectBatchResults,
    expectEmbeddings,
    expectError,
    expectLocalValidation,
    expectNamedModels,
    expectOk,
    generate,
    LIVE_LIMITER,
    log,
    mask,
    REQUEST_TIMEOUT_MS,
    runBatchJob,
    send,
    sendOnce,
    wordMessages,
} from './shared.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Cheap model: bulk of the cases and the batch jobs. */
const LITE = 'gemini-3.1-flash-lite';

/** Models that run the core cases next to `LITE`. */
const CORE_MODELS = [LITE, 'gemini-3.8-flash', 'gemini-4-hydrogen'] as const;

/** Folder of all batch files in the bucket. */
const BATCH_ROOT = 'smart-genai/';

const HAS_PROJECT = Boolean(process.env.GOOGLE_CLOUD_PROJECT);
const HAS_BUCKET = Boolean(process.env.VERTEX_BATCH_BUCKET);

/** Client created on first use, so a run without Vertex env only skips. */
let cached: GenaiClient | undefined;

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Vertex client from env (bucket included when set). */
function vertex(): GenaiClient {
    cached ??= createGenaiClient({ vertex: true, timeoutMs: REQUEST_TIMEOUT_MS });
    return cached;
}

/** Names of all batch files in the bucket. */
async function batchFiles(): Promise<string[]> {
    const auth = new GoogleAuth({
        keyFilename: credentialsFile(),
        scopes: ['https://www.googleapis.com/auth/cloud-platform'],
    });
    const gcs = createGcsClient({
        bucket: process.env.VERTEX_BATCH_BUCKET as string,
        accessToken: () => auth.getAccessToken(),
        timeoutMs: REQUEST_TIMEOUT_MS,
    });
    return expectOk('bucket list', await gcs.list(BATCH_ROOT));
}

/** Runs `body`, then checks that it left no new batch files in the bucket (`check` cleanup). */
async function expectNoLeftovers(body: () => Promise<void>): Promise<void> {
    const before = new Set(await batchFiles());
    await body();
    const added = (await batchFiles()).filter((name) => !before.has(name));
    log('batch leftovers', added.length === 0 ? 'none' : mask(added.join(', ')));
    expect(added).toEqual([]);
}

describe.skipIf(!HAS_PROJECT)('Vertex AI (live, paid)', () => {
    it('listModels: publisher models with id only', async () => {
        const listed = expectOk(
            'listModels',
            await send(
                'listModels',
                () => vertex().listModels(),
                (data) => `${data.length} models`,
            ),
        );
        const ids = listed.map((model) => model.id);
        log('listModels', CORE_MODELS.map((model) => `${model}=${ids.includes(model)}`).join(' '));
        expect(ids).toContain('gemini-3.8-flash');
        const flash = listed.find((model) => model.id === 'gemini-3.8-flash');
        expect(flash?.actions).toEqual([]);
        expect(flash?.displayName).toBeUndefined();
    });

    it('local validation: invalid_request without a request', async () => {
        await expectLocalValidation(vertex(), LITE);
    });

    it('decide: unsupported without a request, aborted signal first', async () => {
        const request = { input: 'x', questions: { q: { type: 'noul', instructions: 'Is it x?' } } } as const;
        const unsupported = expectError('decide', await vertex().decide(request), 'unsupported');
        expect(unsupported.retryable).toBe(false);
        const controller = new AbortController();
        controller.abort();
        expectError('decide aborted', await vertex().decide(request, { signal: controller.signal }), 'aborted');
    });

    it('named decision model: unsupported without a request, attempts 0', async () => {
        const gemini = createGenaiModels({
            client: { vertex: true, timeoutMs: REQUEST_TIMEOUT_MS },
            models: { triage: { kind: 'decide', modelLimits: { concurrency: 1 } } },
        });
        const request = { input: 'x', questions: { q: { type: 'noul', instructions: 'Is it x?' } } } as const;
        const error = expectError('named decide', await gemini.triage(request), 'unsupported');
        expect(error.attempts).toBe(0);
        expect(error.settings).toEqual({ name: 'triage', model: 'gemini-decision-1' });
    });

    for (const model of CORE_MODELS) {
        describe(model, () => {
            for (const testCase of coreCases(['text', 'image', 'reasoning', 'sampling', 'search', 'tools'])) {
                it(testCase.name, () => testCase.run(vertex(), model), CASE_TIMEOUT_MS);
            }
        });
    }

    describe(`${LITE} breadth`, () => {
        for (const testCase of BREADTH_CASES) {
            it(testCase.name, () => testCase.run(vertex(), LITE), CASE_TIMEOUT_MS);
        }
        it(
            'models/ prefix: HTTP 404 with a message',
            async () => {
                const label = 'models/ prefix';
                const result = await generate(vertex(), label, {
                    model: `models/${LITE}`,
                    messages: [{ role: 'user', content: 'x' }],
                });
                const error = expectError(label, result, 'http', 404);
                expect(error.message.trim().length).toBeGreaterThan(0);
            },
            CASE_TIMEOUT_MS,
        );
        it(
            'publishers/google/models/ prefix',
            async () => {
                const label = 'publisher prefix';
                const result = await generate(vertex(), label, {
                    model: `publishers/google/models/${LITE}`,
                    messages: [{ role: 'user', content: 'Reply with the single word: ok' }],
                    temperature: 0,
                });
                expect(expectOk(label, result).text.toLowerCase()).toContain('ok');
            },
            CASE_TIMEOUT_MS,
        );
        it('named models', () => expectNamedModels({ vertex: true }, LITE), CASE_TIMEOUT_MS);
    });

    describe('embeddings', () => {
        it(
            'gemini-embedding-2: one input per call, the query closest to its answer',
            () => expectEmbeddings(vertex(), 'gemini-embedding-2', true),
            CASE_TIMEOUT_MS,
        );
        it(
            'named gemini-embedding-2: one input, settings and attempts',
            async () => {
                const gemini = createGenaiModels({
                    client: { vertex: true, timeoutMs: REQUEST_TIMEOUT_MS },
                    limiterOptions: LIVE_LIMITER,
                    models: {
                        vectors: {
                            kind: 'embed',
                            model: 'gemini-embedding-2',
                            defaults: { taskType: 'RETRIEVAL_QUERY', dimensions: 256 },
                            modelLimits: { rpm: 30, concurrency: 1 },
                        },
                    },
                });
                const label = 'named embed';
                const data = expectOk(
                    label,
                    await sendOnce(
                        label,
                        () => gemini.vectors('Which organelle makes ATP?'),
                        (output) => `${output.embeddings.length} x ${output.embeddings[0]?.length ?? 0}`,
                    ),
                );
                expect(data.embeddings).toHaveLength(1);
                expect(data.embeddings[0]).toHaveLength(256);
                expect(data.settings).toEqual({ name: 'vectors', model: 'gemini-embedding-2' });
                expect(data.attempts).toBeGreaterThanOrEqual(1);
            },
            CASE_TIMEOUT_MS,
        );
        it(
            'publishers/google/models/gemini-embedding-001: several inputs in one call',
            () => expectEmbeddings(vertex(), 'publishers/google/models/gemini-embedding-001', false),
            CASE_TIMEOUT_MS,
        );
    });

    it.skipIf(!HAS_BUCKET)(
        'batch job: eight items of different shapes, no files left',
        () =>
            expectNoLeftovers(async () => {
                expectBatchResults(await runBatchJob(vertex(), batchItems(LITE)));
            }),
        BATCH_TIMEOUT_MS,
    );

    it.skipIf(!HAS_BUCKET)(
        'batch calls: two calls of a named model share one job',
        () =>
            expectNoLeftovers(async () => {
                const gemini = createGenaiModels({
                    client: { vertex: true, timeoutMs: REQUEST_TIMEOUT_MS, batchWaitSec: 5 },
                    models: { lite: { model: LITE, modelLimits: { concurrency: 1 } } },
                });
                await expectBatchCallPair(
                    (word) => gemini.lite(wordMessages(word), { batch: true, temperature: 0 }),
                    gemini.client,
                );
            }),
        BATCH_TIMEOUT_MS,
    );
});
