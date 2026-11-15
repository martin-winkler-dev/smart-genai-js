// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: Decisions API (`gemini-decision-1`) on the Gemini Developer API with the free key in env GEMINI_API_KEY.
// Run with `bun run test:live:decide`, also part of `bun run test:live:key`; never part of `bun test`. Sends 9 requests.

import { describe, expect, it } from 'bun:test';
import { createGenaiClient } from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Per-test timeout. */
const TIMEOUT_MS = 60_000;

/** Base64 payload of a 1x1 PNG. */
const PNG = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';

/** Yes/no question whether the product arrived damaged. */
const DAMAGED = {
    type: 'noul',
    instructions: 'Did the item arrive damaged?',
    criteria: {
        true_description: 'The product itself is broken or faulty.',
        false_description: 'Preference, sizing, shipping, or pricing complaint.',
    },
} as const;

const apiKey = process.env.GEMINI_API_KEY;
const client = createGenaiClient({ ...(apiKey === undefined ? {} : { apiKey }), timeoutMs: 30_000 });

describe('Gemini Decisions API (live, free key)', () => {
    it(
        'noul: high probability for a damaged item, low for a color complaint, confidence max(noul, 1 - noul)',
        async () => {
            const damaged = await client.decide({
                input: 'The blue mug I ordered on Monday arrived cracked on Wednesday.',
                questions: { damaged: DAMAGED },
            });
            const disliked = await client.decide({
                input: 'The mug arrived in perfect condition, I just do not like the color.',
                questions: { damaged: DAMAGED },
            });
            console.log(JSON.stringify([damaged, disliked]));
            expect(damaged.ok && disliked.ok).toBe(true);
            if (!(damaged.ok && disliked.ok)) {
                return;
            }
            const high = damaged.data.answers.damaged;
            const low = disliked.data.answers.damaged;
            expect(high.noul).toBeGreaterThan(0.5);
            expect(low.noul).toBeLessThan(0.5);
            expect(low.confidence).toBeCloseTo(Math.max(low.noul, 1 - low.noul), 2);
            expect(damaged.data.usage.inputTokens).toBeGreaterThan(0);
            expect(damaged.data.usage.outputTokens).toBe(0);
            expect(damaged.data.model).toBe('gemini-decision-1');
        },
        TIMEOUT_MS,
    );

    it(
        'mixed kinds in one request: choice among its options, score with legend, probabilities summing to 1',
        async () => {
            const result = await client.decide({
                input: 'This is the third time my order is late. Give me my money back or I will report you.',
                questions: {
                    damaged: DAMAGED,
                    intent: {
                        type: 'choice',
                        instructions: 'What does the customer want?',
                        criteria: { refund: 'Money back', exchange: 'Another item', information: '' },
                    },
                    anger: {
                        type: 'score',
                        instructions: 'How angry is the customer?',
                        criteria: ['calm', 'annoyed', 'angry', 'furious'],
                    },
                },
            });
            console.log(JSON.stringify(result));
            expect(result.ok).toBe(true);
            if (!result.ok) {
                return;
            }
            const { intent, anger } = result.data.answers;
            expect(intent.choice).toBe('refund');
            const sum = Object.values(intent.probabilities).reduce((total, value) => total + value, 0);
            expect(sum).toBeCloseTo(1, 2);
            expect(Object.keys(intent.probabilities).sort()).toEqual(['exchange', 'information', 'refund']);
            expect(anger.legend).toEqual({ 0: 'calm', 1: 'annoyed', 2: 'angry', 3: 'furious' });
            expect(anger.score).toBeGreaterThan(1);
            expect(anger.score).toBeLessThanOrEqual(3);
        },
        TIMEOUT_MS,
    );

    it(
        'chat image part as input, models/ prefix and instructions as text parts',
        async () => {
            const result = await client.decide({
                model: 'models/gemini-decision-1',
                input: [
                    { type: 'text', text: 'A product photo sent by the customer.' },
                    { type: 'image_url', image_url: { url: `data:image/png;base64,${PNG}` } },
                ],
                questions: {
                    photo: { type: 'noul', instructions: [{ type: 'text', text: 'Does the input contain an image?' }] },
                },
            });
            console.log(JSON.stringify(result));
            expect(result.ok).toBe(true);
            if (result.ok) {
                expect(result.data.answers.photo.type).toBe('noul');
                expect(result.data.usage.inputTokens).toBeGreaterThan(0);
            }
        },
        TIMEOUT_MS,
    );

    it(
        'native image block with resolution',
        async () => {
            const result = await client.decide({
                input: [{ type: 'image', data: PNG, mime_type: 'image/png', resolution: 'low' }],
                questions: { bright: { type: 'noul', instructions: 'Is the image mostly bright?' } },
            });
            console.log(JSON.stringify(result));
            expect(result.ok).toBe(true);
        },
        TIMEOUT_MS,
    );

    it(
        'CSV document block, question key toString, numeric option ids as strings',
        async () => {
            const csv = Buffer.from('item,status\nmug,cracked\nplate,fine\n').toString('base64');
            const result = await client.decide({
                input: [{ type: 'document', data: csv, mime_type: 'text/csv' }],
                questions: {
                    toString: { type: 'noul', instructions: 'Does the table list a broken item?' },
                    tier: {
                        type: 'choice',
                        instructions: 'How many rows of items does the table have?',
                        criteria: { 1: 'One row', 2: 'Two rows' },
                    },
                },
            });
            console.log(JSON.stringify(result));
            expect(result.ok).toBe(true);
            if (!result.ok) {
                return;
            }
            const { toString: broken, tier } = result.data.answers;
            expect(broken.noul).toBeGreaterThan(0.5);
            expect(tier.choice).toBe('2');
            expect(Object.keys(tier.probabilities).sort()).toEqual(['1', '2']);
        },
        TIMEOUT_MS,
    );

    it(
        'maps API errors: unknown model 404, broken image 400 invalid_request, invalid key 400',
        async () => {
            const questions = { damaged: DAMAGED };
            const unknown = await client.decide({ model: 'gemini-decision-999', input: 'x', questions });
            const broken = await client.decide({
                input: [{ type: 'image', data: 'AAAA', mime_type: 'image/png' }],
                questions,
            });
            const badKey = await createGenaiClient({ apiKey: 'invalid-key', env: {} }).decide({
                input: 'x',
                questions,
            });
            console.log(JSON.stringify([unknown, broken, badKey]));
            expect(unknown).toMatchObject({ ok: false, error: { code: 'http', status: 404, retryable: false } });
            expect(broken).toMatchObject({ ok: false, error: { code: 'http', status: 400, retryable: false } });
            expect(badKey).toMatchObject({ ok: false, error: { code: 'http', status: 400, retryable: false } });
        },
        TIMEOUT_MS,
    );
});
