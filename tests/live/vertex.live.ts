// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live and paid: one request and one batch of one request to gemini-3.8-flash on Vertex AI.
// Run with `bun run test:live:vertex`; never part of `bun test`. Needs env GOOGLE_CLOUD_PROJECT, VERTEX_BATCH_BUCKET
// and gcloud application-default credentials.

import { describe, expect, it } from 'bun:test';
import { GoogleAuth } from 'google-auth-library';
import { createGcsClient } from '../../src/core/batch/gcs.js';
import { createGenaiClient, credentialsFile } from '../../src/index.js';
import type { GenaiBatchStatus, GenaiRequest } from '../../src/types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Target model evaluated in live Vertex test. */
const MODEL = 'gemini-3.8-flash';

/** Polling interval for batch job. */
const POLL_MS = 60_000;

/** Maximum wait timeout for batch job completion. */
const MAX_WAIT_MS = 40 * 60_000;

const client = createGenaiClient({ vertex: true, timeoutMs: 120_000 });

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Constructs classification request for test word. */
function request(word: string): GenaiRequest {
    return {
        model: MODEL,
        messages: [
            { role: 'system', content: 'You classify invented English words by their role in a sentence.' },
            { role: 'user', content: `Sentence: "The children ${word} the ball across the yard."` },
        ],
        response_format: {
            type: 'json_schema',
            json_schema: {
                name: 'answer',
                schema: {
                    type: 'object',
                    properties: { kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] } },
                    required: ['kind'],
                },
            },
        },
    };
}

describe('Vertex AI (live, paid)', () => {
    it('answers one request', async () => {
        const started = performance.now();
        const result = await client.generate(request('snarbled'));
        console.log('generate', Math.round(performance.now() - started), 'ms', JSON.stringify(result));
        expect(result).toMatchObject({ ok: true });
        if (result.ok) {
            expect(result.data.json).toEqual({ kind: 'verb' });
        }
    });

    it(
        'runs one batch request on Vertex AI and Cloud Storage',
        async () => {
            const bucket = process.env.VERTEX_BATCH_BUCKET;
            if (!bucket || !client.batch) {
                console.log('Skipping Vertex batch test: set env VERTEX_BATCH_BUCKET');
                return;
            }
            const started = performance.now();
            const submitted = await client.batch.submit([{ key: 'a', request: request('glonked') }]);
            console.log('submit', Math.round(performance.now() - started), 'ms', JSON.stringify(submitted));
            expect(submitted).toMatchObject({ ok: true });
            if (!submitted.ok) {
                return;
            }
            const name = submitted.data.name;
            let status: GenaiBatchStatus = submitted.data;
            while (status.phase === 'running') {
                if (performance.now() - started > MAX_WAIT_MS) {
                    console.log(
                        `Vertex batch still running after ${Math.round((performance.now() - started) / 1000)} s`,
                    );
                    return;
                }
                await Bun.sleep(POLL_MS);
                const checked = await client.batch.check(name);
                console.log('check', Math.round(performance.now() - started), 'ms', JSON.stringify(checked));
                expect(checked).toMatchObject({ ok: true });
                if (!checked.ok) {
                    return;
                }
                status = checked.data;
            }
            expect(status.phase).toBe('succeeded');
            const result = status.results?.[0]?.result;
            expect(result).toMatchObject({ ok: true });
            if (result?.ok) {
                expect(result.data.json).toEqual({ kind: 'verb' });
            }
            const auth = new GoogleAuth({
                keyFilename: credentialsFile(),
                scopes: ['https://www.googleapis.com/auth/cloud-platform'],
            });
            const gcs = createGcsClient({ bucket, accessToken: () => auth.getAccessToken(), timeoutMs: 120_000 });
            const left = await gcs.list('smart-genai/');
            console.log('leftover batch files:', left.ok ? left.data : left.error);
        },
        MAX_WAIT_MS + 60_000,
    );
});
