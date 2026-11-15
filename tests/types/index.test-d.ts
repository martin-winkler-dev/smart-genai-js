// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenerateContentConfig } from '@google/genai';
import {
    createGenaiClient,
    type DEFAULT_DECIDE_MODEL,
    type GenaiBatchCheckOptions,
    type GenaiBatchJob,
    type GenaiDecideAnswer,
    type GenaiDecideAnswerOf,
    type GenaiDecideChoiceAnswer,
    type GenaiDecideChoiceQuestion,
    type GenaiDecideInputPart,
    type GenaiDecideInstructions,
    type GenaiDecideMediaPart,
    type GenaiDecideNoulAnswer,
    type GenaiDecideNoulQuestion,
    type GenaiDecideOutput,
    type GenaiDecideQuestion,
    type GenaiDecideQuestions,
    type GenaiDecideRequest,
    type GenaiDecideScoreAnswer,
    type GenaiDecideScoreQuestion,
    type GenaiError,
    type GenaiOutput,
    type GenaiRequest,
    type MAX_DECIDE_QUESTIONS,
    mergeGenaiConfig,
    type Result,
    validateGenaiConfig,
} from '../../src/index.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Asserts at compile time that `A` and `B` are the same type. */
type Equal<A, B> = (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2 ? true : false;

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

const request: GenaiRequest = {
    model: 'gemini-x',
    messages: [{ role: 'user', content: 'hi' }],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: { type: 'object' } } },
};
const _pending: Promise<Result<GenaiOutput, unknown>> = createGenaiClient({ apiKey: 'k' }).generate(request);
const _bytes: Result<number, GenaiError> | undefined = createGenaiClient({ apiKey: 'k' }).batch?.requestBytes(request);
const _jobs: Promise<Result<GenaiBatchJob[], GenaiError>> | undefined = createGenaiClient({ apiKey: 'k' }).batch?.list({
    displayName: 'run',
});
const _checkOptions: GenaiBatchCheckOptions = { keepFiles: true };
const _merged: GenerateContentConfig | undefined = mergeGenaiConfig({ temperature: 0 }, undefined, { topK: 1 });
// biome-ignore lint/suspicious/noConfusingVoidType: assert validateGenaiConfig returns void
const _validated: void = validateGenaiConfig({ temperature: 0 }, 'config');

const decided = await createGenaiClient({ apiKey: 'k' }).decide({
    input: [{ type: 'text', text: 'The mug arrived cracked.' }],
    questions: {
        damaged: { type: 'noul', instructions: 'Damaged?' },
        intent: { type: 'choice', instructions: 'Intent?', criteria: { refund: '', exchange: 'Wants another one' } },
        anger: { type: 'score', instructions: 'How angry?', criteria: ['calm', 'furious'] },
    },
});
if (decided.ok) {
    const { damaged, intent, anger } = decided.data.answers;
    const _noul: Equal<typeof damaged, GenaiDecideNoulAnswer> = true;
    const _choice: Equal<typeof intent.choice, 'refund' | 'exchange'> = true;
    const _probabilities: Equal<typeof intent.probabilities, Record<'refund' | 'exchange', number>> = true;
    const _score: Equal<typeof anger, GenaiDecideScoreAnswer> = true;
    const _usage: number | null = decided.data.usage.inputTokens;
    // @ts-expect-error: only the asked question keys exist
    decided.data.answers.other;
}
const wide: GenaiDecideQuestions = { a: { type: 'noul', instructions: 'i' } };
const _wide: Promise<Result<GenaiDecideOutput, GenaiError>> = createGenaiClient({ apiKey: 'k' }).decide({
    input: 'x',
    questions: wide,
});
const wideResult = await _wide;
const _answer: GenaiDecideAnswer | undefined = wideResult.ok ? wideResult.data.answers.a : undefined;
const _request: GenaiDecideRequest = {
    input: 'x',
    // @ts-expect-error: unknown question type
    questions: { a: { type: 'rank', instructions: 'i' } },
};
const tiers = await createGenaiClient({ apiKey: 'k' }).decide({
    input: 'x',
    questions: { tier: { type: 'choice', instructions: 'Tier?', criteria: { 1: 'Tier one', 2: 'Tier two' } } },
});
if (tiers.ok) {
    const _tier: Equal<typeof tiers.data.answers.tier.choice, '1' | '2'> = true;
    const _tierProbabilities: Equal<typeof tiers.data.answers.tier.probabilities, Record<'1' | '2', number>> = true;
}
const _media: GenaiDecideMediaPart[] = [
    { type: 'image', data: 'AAAA', mime_type: 'image/png', resolution: 'ultra_high' },
    { type: 'video', uri: 'https://example.com/a.mp4', mime_type: 'video/mp4', resolution: 'high' },
    { type: 'audio', data: 'AAAA', mime_type: 'audio/wav' },
    { type: 'document', uri: 'https://example.com/a.pdf', mime_type: 'application/pdf' },
];
// @ts-expect-error: no resolution on audio
const _audioResolution: GenaiDecideMediaPart = { type: 'audio', data: 'A', mime_type: 'audio/wav', resolution: 'low' };
const _documentResolution: GenaiDecideMediaPart = {
    type: 'document',
    data: 'A',
    mime_type: 'text/csv',
    // @ts-expect-error: no resolution on documents
    resolution: 'low',
};
// @ts-expect-error: ultra_high only on images
const _videoUltra: GenaiDecideMediaPart = {
    type: 'video',
    data: 'A',
    mime_type: 'video/mp4',
    resolution: 'ultra_high',
};
// @ts-expect-error: neither data nor uri
const _noSource: GenaiDecideMediaPart = { type: 'image', mime_type: 'image/png' };
// @ts-expect-error: both data and uri
const _bothSources: GenaiDecideMediaPart = { type: 'image', data: 'A', uri: 'https://x', mime_type: 'image/png' };
const _defaultModel: Equal<typeof DEFAULT_DECIDE_MODEL, 'gemini-decision-1'> = true;
const _maxQuestions: Equal<typeof MAX_DECIDE_QUESTIONS, 1000> = true;
const _instructions: GenaiDecideInstructions[] = ['i', { type: 'text', text: 'i' }, [{ type: 'text', text: 'i' }]];
// @ts-expect-error: instructions are text only
const _imageInstructions: GenaiDecideInstructions = {
    type: 'image_url',
    image_url: { url: 'data:image/png;base64,A' },
};
const _noulQuestion: GenaiDecideNoulQuestion = { type: 'noul', instructions: 'i', criteria: { true_description: 't' } };
const _choiceQuestion: GenaiDecideChoiceQuestion = { type: 'choice', instructions: 'i', criteria: { a: '', b: '' } };
const _scoreQuestion: GenaiDecideScoreQuestion = { type: 'score', instructions: 'i', criteria: ['low', 'high'] };
const _question: Equal<
    GenaiDecideQuestion,
    GenaiDecideNoulQuestion | GenaiDecideChoiceQuestion | GenaiDecideScoreQuestion
> = true;
const _inputParts: GenaiDecideInputPart[] = [
    { type: 'text', text: 'x' },
    { type: 'image_url', image_url: { url: 'data:image/png;base64,A' } },
    { type: 'audio', uri: 'https://example.com/a.wav', mime_type: 'audio/wav' },
];
const _choiceAnswer: Equal<GenaiDecideAnswerOf<typeof _choiceQuestion>, GenaiDecideChoiceAnswer<string>> = true;
const _literalAnswer: Equal<
    GenaiDecideAnswerOf<{ type: 'choice'; instructions: 'i'; criteria: { a: ''; 7: '' } }>,
    GenaiDecideChoiceAnswer<'a' | '7'>
> = true;
const _noulAnswer: Equal<GenaiDecideAnswerOf<typeof _noulQuestion>, GenaiDecideNoulAnswer> = true;
const _scoreAnswer: Equal<GenaiDecideAnswerOf<typeof _scoreQuestion>, GenaiDecideScoreAnswer> = true;
