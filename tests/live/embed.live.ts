// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: embeddings on the Gemini Developer API with the free key in env GEMINI_API_KEY.
// Run with `bun run test:live:embed`; never part of `bun test`.

import { describe, expect, it } from 'bun:test';
import { createGenaiClient } from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Embedding models tested against Developer API. */
const MODELS = ['gemini-embedding-001', 'gemini-embedding-2'] as const;

/** Test documents paired with query targeting the second document. */
const DOCUMENTS = [
    { text: 'The bakery opens at six and sells rye bread and croissants.', title: 'Shop' },
    { text: 'Mitochondria produce ATP through oxidative phosphorylation.', title: 'Cell biology' },
    { text: 'The striker scored twice in the second half of the final.', title: 'Sports' },
];
const QUERY = 'Which organelle makes the energy carrier of the cell?';

/** Per-model test timeout. */
const TIMEOUT_MS = 60_000;

const apiKey = process.env.GEMINI_API_KEY;
const client = createGenaiClient({ ...(apiKey === undefined ? {} : { apiKey }), timeoutMs: 30_000 });

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Computes cosine similarity between two equal-length vectors. */
function cosine(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let index = 0; index < a.length; index++) {
        dot += (a[index] ?? 0) * (b[index] ?? 0);
        normA += (a[index] ?? 0) ** 2;
        normB += (b[index] ?? 0) ** 2;
    }
    return dot / Math.sqrt(normA * normB);
}

describe('Gemini embeddings (live, free key)', () => {
    for (const model of MODELS) {
        it(
            `${model}: one vector per document from one call, the query closest to its answer`,
            async () => {
                const started = performance.now();
                // gemini-embedding-001 takes one title per call, so its documents go without titles
                const input = model === 'gemini-embedding-2' ? DOCUMENTS : DOCUMENTS.map((document) => document.text);
                const documents = await client.embed({ model, input, taskType: 'RETRIEVAL_DOCUMENT', dimensions: 768 });
                const query = await client.embed({ model, input: QUERY, taskType: 'RETRIEVAL_QUERY', dimensions: 768 });
                console.log(model, Math.round(performance.now() - started), 'ms');
                expect(documents.ok && query.ok).toBe(true);
                if (!(documents.ok && query.ok)) {
                    console.log(JSON.stringify([documents, query]));
                    return;
                }
                expect(documents.data.embeddings).toHaveLength(DOCUMENTS.length);
                for (const vector of [...documents.data.embeddings, ...query.data.embeddings]) {
                    expect(vector).toHaveLength(768);
                }
                const [queryVector = []] = query.data.embeddings;
                const scores = documents.data.embeddings.map((vector) => cosine(queryVector, vector));
                console.log(model, 'scores', scores.map((score) => score.toFixed(3)).join(' '));
                expect(scores.indexOf(Math.max(...scores))).toBe(1);
            },
            TIMEOUT_MS,
        );
    }
});
