// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: embeddings on the Gemini Developer API with the key in env GEMINI_API_KEY.
// Run with `bun run test:live:embed`, also part of `bun run test:live:key`; never part of `bun test`.
// Sends 6 requests: 4 plain embedding calls, 2 calls of a named embedding model.

import { describe, expect, it } from 'bun:test';
import { createGenaiClient, createGenaiModels } from '../../src/index.js';
import type { GenaiEmbedOutput } from '../../src/types.js';
import { CASE_TIMEOUT_MS, expectEmbeddings, expectOk, LIVE_LIMITER, REQUEST_TIMEOUT_MS, sendOnce } from './shared.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Embedding models tested against Developer API. */
const MODELS = ['gemini-embedding-001', 'gemini-embedding-2'] as const;

const client = createGenaiClient({ timeoutMs: REQUEST_TIMEOUT_MS });

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Compact description of an embedding answer with attempts. */
function describeNamed(data: GenaiEmbedOutput): string {
    return `${data.embeddings.length} x ${data.embeddings[0]?.length ?? 0} attempts=${data.attempts}`;
}

describe('Gemini embeddings (live, key)', () => {
    for (const model of MODELS) {
        it(
            `${model}: one vector per document from one call, the query closest to its answer`,
            () => expectEmbeddings(client, model, false),
            CASE_TIMEOUT_MS,
        );
    }

    it(
        'named embedding model: defaults, per-call dimensions, settings and attempts',
        async () => {
            const model = 'gemini-embedding-2';
            const gemini = createGenaiModels({
                client: { timeoutMs: REQUEST_TIMEOUT_MS },
                limiterOptions: LIVE_LIMITER,
                models: {
                    vectors: {
                        kind: 'embed',
                        model,
                        defaults: { taskType: 'RETRIEVAL_DOCUMENT', dimensions: 256 },
                        modelLimits: { rpm: 20, tpm: 20_000, concurrency: 1 },
                    },
                },
            });
            expect(gemini.vectors.id).toBe(model);
            const documents = expectOk(
                'named embed documents',
                await sendOnce(
                    'named embed documents',
                    () => gemini.vectors(['The bakery sells rye bread.', 'Mitochondria produce ATP.']),
                    describeNamed,
                ),
            );
            expect(documents.embeddings).toHaveLength(2);
            for (const vector of documents.embeddings) {
                expect(vector).toHaveLength(256);
            }
            expect(documents.settings).toEqual({ name: 'vectors', model });
            expect(documents.attempts).toBeGreaterThanOrEqual(1);
            const query = expectOk(
                'named embed query',
                await sendOnce(
                    'named embed query',
                    () =>
                        gemini.vectors('Which organelle makes ATP?', { taskType: 'RETRIEVAL_QUERY', dimensions: 128 }),
                    describeNamed,
                ),
            );
            expect(query.embeddings).toHaveLength(1);
            expect(query.embeddings[0]).toHaveLength(128);
            expect(query.attempts).toBeGreaterThanOrEqual(1);
        },
        CASE_TIMEOUT_MS,
    );
});
