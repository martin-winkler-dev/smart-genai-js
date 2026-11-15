// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: embeddings on the Gemini Developer API with the key in env GEMINI_API_KEY.
// Run with `bun run test:live:embed`, also part of `bun run test:live:key`; never part of `bun test`.

import { describe, it } from 'bun:test';
import { createGenaiClient } from '../../src/index.js';
import { CASE_TIMEOUT_MS, expectEmbeddings, REQUEST_TIMEOUT_MS } from './shared.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Embedding models tested against Developer API. */
const MODELS = ['gemini-embedding-001', 'gemini-embedding-2'] as const;

const client = createGenaiClient({ timeoutMs: REQUEST_TIMEOUT_MS });

describe('Gemini embeddings (live, key)', () => {
    for (const model of MODELS) {
        it(
            `${model}: one vector per document from one call, the query closest to its answer`,
            () => expectEmbeddings(client, model, false),
            CASE_TIMEOUT_MS,
        );
    }
});
