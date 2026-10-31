// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenerateContentConfig } from '@google/genai';
import {
    createGenaiClient,
    type GenaiError,
    type GenaiOutput,
    type GenaiRequest,
    mergeGenaiConfig,
    type Result,
    validateGenaiConfig,
} from '../../src/index.js';

const request: GenaiRequest = {
    model: 'gemini-x',
    messages: [{ role: 'user', content: 'hi' }],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: { type: 'object' } } },
};
const _pending: Promise<Result<GenaiOutput, unknown>> = createGenaiClient({ apiKey: 'k' }).generate(request);
const _bytes: Result<number, GenaiError> | undefined = createGenaiClient({ apiKey: 'k' }).batch?.requestBytes(request);
const _merged: GenerateContentConfig | undefined = mergeGenaiConfig({ temperature: 0 }, undefined, { topK: 1 });
// biome-ignore lint/suspicious/noConfusingVoidType: assert validateGenaiConfig returns void
const _validated: void = validateGenaiConfig({ temperature: 0 }, 'config');
