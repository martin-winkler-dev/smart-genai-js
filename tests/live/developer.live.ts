// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: one request per model on the Gemini Developer API with the free key in env GEMINI_API_KEY.
// Run with `bun run test:live`; never part of `bun test`.

import { describe, expect, it } from 'bun:test';
import { createGenaiClient } from '../../src/index.js';
import type { GenaiRequest } from '../../src/types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Models tested against Developer API. */
const MODELS = [
    'gemini-3.8-flash',
    'gemini-3.5-flash-lite',
    'gemini-3.1-flash-lite',
    'gemma-4-31b-it',
    'gemma-4-26b-a4b-it',
] as const;

/** Per-model request timeout. */
const TIMEOUT_MS = 150_000;

const apiKey = process.env.GEMINI_API_KEY;
const client = createGenaiClient({ ...(apiKey === undefined ? {} : { apiKey }), timeoutMs: 120_000 });

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Constructs classification request with JSON Schema. */
function request(model: string): GenaiRequest {
    return {
        model,
        messages: [
            { role: 'system', content: 'You classify invented English words by their role in a sentence.' },
            { role: 'user', content: 'Sentence: "The children found a shiny blorf under the bridge."' },
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

describe('Gemini Developer API (live, free key)', () => {
    for (const model of MODELS) {
        it(
            model,
            async () => {
                const started = performance.now();
                const result = await client.generate(request(model));
                console.log(model, Math.round(performance.now() - started), 'ms', JSON.stringify(result));
                expect(result).toMatchObject({ ok: true });
                if (result.ok) {
                    expect(result.data.json).toEqual({ kind: 'noun' });
                }
            },
            TIMEOUT_MS,
        );
    }
});
