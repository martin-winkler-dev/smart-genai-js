// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: three batch calls (`batch: true`) started together; they go out as one batch job and each call awaits its
// own answer. Developer API: key in env GEMINI_API_KEY (the free key cannot batch: FAILED_PRECONDITION). Vertex AI
// (paid): env GOOGLE_CLOUD_PROJECT, VERTEX_BATCH_BUCKET and gcloud application-default credentials.
// Run with `bun run test:live:batch` (`-t Vertex` or `-t Developer` for one case); never part of `bun test`.

import { describe, expect, it } from 'bun:test';
import { GoogleAuth } from 'google-auth-library';
import { createGcsClient } from '../../src/core/batch/gcs.js';
import { createGenaiClient, credentialsFile } from '../../src/index.js';
import type { GenaiClient, GenaiRequest } from '../../src/types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Maximum timeout for batch job execution. */
const TIMEOUT_MS = 2 * 60 * 60_000;

/** Test prompt words verified in batch responses. */
const WORDS = ['blorf', 'snarf', 'quib'];

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Constructs batch request prompting model to echo specified word. */
function request(model: string, word: string): GenaiRequest {
    return {
        model,
        batch: true,
        messages: [
            { role: 'system', content: 'Answer with the one word asked for, in lower case.' },
            { role: 'user', content: `Repeat this word exactly: ${word}` },
        ],
    };
}

/** Dispatches three parallel batch calls and asserts each received expected word. */
async function runThree(client: GenaiClient, model: string): Promise<void> {
    const startedAt = Date.now();
    const results = await Promise.all(
        WORDS.map(async (word) => {
            const result = await client.generate(request(model, word));
            console.log(word, `${Math.round((Date.now() - startedAt) / 1000)} s`, JSON.stringify(result));
            return result;
        }),
    );
    console.log(`all done after ${Math.round((Date.now() - startedAt) / 1000)} s`);
    results.forEach((result, index) => {
        expect(result).toMatchObject({ ok: true });
        expect(result.ok ? result.data.text.toLowerCase() : '').toContain(WORDS[index] as string);
    });
}

describe('batch calls (live, Developer API)', () => {
    it(
        'three calls in one job, each with its own answer',
        async () => {
            const apiKey = process.env.GEMINI_API_KEY;
            const client = createGenaiClient({
                ...(apiKey === undefined ? {} : { apiKey }),
                timeoutMs: 120_000,
                batchWaitSec: 2,
            });
            await runThree(client, 'gemini-3.1-flash-lite');
        },
        TIMEOUT_MS,
    );
});

describe('batch calls (live, Vertex AI)', () => {
    it(
        'three calls in one job, each with its own answer',
        async () => {
            const bucket = process.env.VERTEX_BATCH_BUCKET;
            const project = process.env.GOOGLE_CLOUD_PROJECT;
            if (!bucket || !project) {
                console.log('Skipping Vertex live test: set env VERTEX_BATCH_BUCKET and GOOGLE_CLOUD_PROJECT');
                return;
            }
            const client = createGenaiClient({ vertex: true, timeoutMs: 120_000, batchWaitSec: 2 });
            try {
                await runThree(client, 'gemini-3.1-flash-lite');
            } finally {
                const auth = new GoogleAuth({
                    keyFilename: credentialsFile(),
                    scopes: ['https://www.googleapis.com/auth/cloud-platform'],
                });
                const gcs = createGcsClient({ bucket, accessToken: () => auth.getAccessToken(), timeoutMs: 120_000 });
                const left = await gcs.list('smart-genai/');
                console.log('leftover batch files:', left.ok ? left.data : left.error);
            }
        },
        TIMEOUT_MS,
    );
});
