// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: generate, listModels, named models and batch on the Gemini Developer API with the key in env GEMINI_API_KEY.
// Part of `bun run test:live:key` (with `embed.live.ts` and `decide.live.ts`); never part of `bun test`.
// Free key: at most 6 requests on gemini-4-hydrogen, 2 on gemini-3.8-flash, 1 on each other model besides
// gemini-3.1-flash-lite (about 26), plus one embedding and one decision request (mixed named models case). Env LIVE_PAID_KEY=1 adds web search and batch (both fail on a free key);
// without it those cases are skipped. Env LIVE_SKIP_MODELS=gemini-3.8-flash skips the cases of that model (often
// overloaded on a free key: HTTP 503/504).

import { describe, expect, it } from 'bun:test';
import { createGenaiClient, createGenaiModels } from '../../src/index.js';
import type { GenaiReasoningEffort, GenaiResponseFormat } from '../../src/types.js';
import {
    BATCH_TIMEOUT_MS,
    BREADTH_CASES,
    batchItems,
    CASE_TIMEOUT_MS,
    CORE_CASES,
    coreCases,
    expectBatchCallPair,
    expectBatchResults,
    expectLocalValidation,
    expectNamedModels,
    expectOk,
    generate,
    IS_PAID_KEY,
    LIVE_LIMITER,
    log,
    REQUEST_TIMEOUT_MS,
    runBatchJob,
    SKIPPED_MODELS,
    send,
    sendOnce,
    wordMessages,
} from './shared.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Model with 500 free requests per day: bulk of the cases. */
const LITE = 'gemini-3.1-flash-lite';

/** Models with 20 free requests per day each. */
const HYDROGEN = 'gemini-4-hydrogen';
const FLASH = 'gemini-3.8-flash';

/** Classification schema for the single-request models. */
const KIND_FORMAT: GenaiResponseFormat = {
    type: 'json_schema',
    json_schema: {
        name: 'answer',
        schema: {
            type: 'object',
            properties: { kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] } },
            required: ['kind'],
        },
    },
};

const client = createGenaiClient({ timeoutMs: REQUEST_TIMEOUT_MS, batchWaitSec: 2 });

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Classifies the invented word "blorf" (a noun) with `reasoning_effort`. */
async function expectKind(model: string, effort: GenaiReasoningEffort): Promise<void> {
    const label = `${model} reasoning ${effort}`;
    const result = await generate(client, label, {
        model,
        messages: [
            { role: 'system', content: 'You classify invented English words by their role in a sentence.' },
            { role: 'user', content: 'Sentence: "The children found a shiny blorf under the bridge."' },
        ],
        response_format: KIND_FORMAT,
        reasoning_effort: effort,
        temperature: 0,
        max_tokens: 4000,
    });
    const data = expectOk(label, result);
    expect(data.json).toEqual({ kind: 'noun' });
    expect(data.settings?.reasoning_effort).toBe(effort);
}

describe('Developer API (live, key)', () => {
    it('listModels: models with actions and token limits', async () => {
        const listed = expectOk(
            'listModels',
            await send(
                'listModels',
                () => client.listModels(),
                (data) => `${data.length} models`,
            ),
        );
        const ids = listed.map((model) => model.id);
        log('listModels', `hydrogen=${ids.includes(HYDROGEN)} flash=${ids.includes(FLASH)}`);
        expect(ids).toContain(LITE);
        expect(ids).toContain(FLASH);
        const lite = listed.find((model) => model.id === LITE);
        expect(lite?.actions).toContain('generateContent');
        expect(lite?.inputTokenLimit ?? 0).toBeGreaterThan(0);
    });

    it('local validation: invalid_request without a request', async () => {
        await expectLocalValidation(client, LITE);
    });

    describe(LITE, () => {
        for (const testCase of coreCases(['text', 'image', 'reasoning', 'sampling', 'tools'])) {
            it(testCase.name, () => testCase.run(client, LITE), CASE_TIMEOUT_MS);
        }
        it.skipIf(!IS_PAID_KEY)(CORE_CASES.search.name, () => CORE_CASES.search.run(client, LITE), CASE_TIMEOUT_MS);
        for (const testCase of BREADTH_CASES) {
            it(testCase.name, () => testCase.run(client, LITE), CASE_TIMEOUT_MS);
        }
        it('named models', () => expectNamedModels({ timeoutMs: REQUEST_TIMEOUT_MS }, LITE), CASE_TIMEOUT_MS);
        it(
            'named models of all kinds under one shared concurrency, started in parallel',
            async () => {
                const gemini = createGenaiModels({
                    // free-tier flash-lite can hang about 70 s before a 503, holding the shared slot
                    client: { timeoutMs: 30_000 },
                    sharedLimits: { concurrency: 1 },
                    limiterOptions: LIVE_LIMITER,
                    models: {
                        lite: { model: LITE, defaults: { reasoning_effort: 'none', temperature: 0, max_tokens: 20 } },
                        vectors: { kind: 'embed', model: 'gemini-embedding-2', defaults: { dimensions: 128 } },
                        triage: { kind: 'decide' },
                    },
                });
                const [word, vectors, decision] = await Promise.all([
                    sendOnce(
                        'mixed generate',
                        () => gemini.lite('Reply with the single word: ok'),
                        (data) => data.text,
                    ),
                    sendOnce(
                        'mixed embed',
                        () => gemini.vectors('A short sentence.'),
                        (data) => `${data.embeddings.length} x ${data.embeddings[0]?.length ?? 0}`,
                    ),
                    sendOnce(
                        'mixed decide',
                        () =>
                            gemini.triage({
                                input: 'The sky is blue.',
                                questions: { color: { type: 'noul', instructions: 'Does the text name a colour?' } },
                            }),
                        (data) => JSON.stringify(data.answers),
                    ),
                ]);
                const wordData = expectOk('mixed generate', word);
                expect(wordData.text.toLowerCase()).toContain('ok');
                expect(wordData.settings).toMatchObject({ name: 'lite', model: LITE });
                const vectorsData = expectOk('mixed embed', vectors);
                expect(vectorsData.embeddings[0]).toHaveLength(128);
                expect(vectorsData.settings).toEqual({ name: 'vectors', model: 'gemini-embedding-2' });
                const decisionData = expectOk('mixed decide', decision);
                expect(decisionData.answers.color.type).toBe('noul');
                expect(decisionData.settings).toEqual({ name: 'triage', model: 'gemini-decision-1' });
                for (const attempts of [wordData.attempts, vectorsData.attempts, decisionData.attempts]) {
                    expect(attempts).toBeGreaterThanOrEqual(1);
                }
            },
            CASE_TIMEOUT_MS,
        );
    });

    describe.skipIf(SKIPPED_MODELS.has(HYDROGEN))(HYDROGEN, () => {
        for (const testCase of coreCases(['image', 'reasoning', 'sampling', 'tools'])) {
            it(testCase.name, () => testCase.run(client, HYDROGEN), CASE_TIMEOUT_MS);
        }
    });

    describe.skipIf(SKIPPED_MODELS.has(FLASH))(FLASH, () => {
        for (const testCase of coreCases(['sampling', 'image'])) {
            it(testCase.name, () => testCase.run(client, FLASH), CASE_TIMEOUT_MS);
        }
    });

    it(
        'gemini-3.5-flash-lite: JSON schema with reasoning minimal',
        () => expectKind('gemini-3.5-flash-lite', 'minimal'),
        CASE_TIMEOUT_MS,
    );
    it('gemma-4-31b-it: JSON schema with reasoning low', () => expectKind('gemma-4-31b-it', 'low'), CASE_TIMEOUT_MS);
    it(
        'gemma-4-26b-a4b-it: JSON schema with reasoning medium',
        () => expectKind('gemma-4-26b-a4b-it', 'medium'),
        CASE_TIMEOUT_MS,
    );

    it.skipIf(!IS_PAID_KEY)(
        'batch job: eight items of different shapes',
        async () => {
            expectBatchResults(await runBatchJob(client, batchItems(LITE)));
        },
        BATCH_TIMEOUT_MS,
    );

    it.skipIf(!IS_PAID_KEY)(
        'batch calls: two calls of a named model share one job',
        async () => {
            const gemini = createGenaiModels({
                client: { timeoutMs: REQUEST_TIMEOUT_MS, batchWaitSec: 5 },
                models: { lite: { model: LITE, modelLimits: { concurrency: 1 } } },
            });
            await expectBatchCallPair(
                (word) => gemini.lite(wordMessages(word), { batch: true, temperature: 0 }),
                gemini.client,
            );
        },
        BATCH_TIMEOUT_MS,
    );
});
