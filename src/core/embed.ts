// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { Content, EmbedContentConfig, EmbedContentResponse, GoogleGenAI } from '@google/genai';
import type {
    GenaiClient,
    GenaiEmbedInput,
    GenaiEmbedOutput,
    GenaiEmbedRequest,
    GenaiEmbedTask,
    GenaiError,
    Result,
} from '../types.js';
import { elapsedMs, guard, toGenaiError } from './answer.js';
import { checkOptionKeys, InvalidRequestError } from './request.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Input with its title read from `GenaiEmbedInput`. */
type EmbedText = { text: string; title?: string };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Maximum inputs allowed per embed call. */
export const MAX_EMBED_INPUTS = 100;

/** Supported field keys of `GenaiEmbedRequest`, of an input object and of the call options. */
const REQUEST_KEYS: ReadonlySet<string> = new Set(['model', 'input', 'taskType', 'dimensions']);
const INPUT_KEYS: ReadonlySet<string> = new Set(['text', 'title']);
const OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/**
 * **Task prefix mapping for `gemini-embedding-2`**: format `task: <name> | query: <text>`.
 *
 * - `RETRIEVAL_DOCUMENT` uses format `title: <title or none> | text: <text>`
 * - symmetric and query task types map to descriptive prefix strings
 */
const TASK_PREFIXES: Readonly<Record<Exclude<GenaiEmbedTask, 'RETRIEVAL_DOCUMENT'>, string>> = {
    RETRIEVAL_QUERY: 'search result',
    QUESTION_ANSWERING: 'question answering',
    FACT_VERIFICATION: 'fact checking',
    CODE_RETRIEVAL_QUERY: 'code retrieval',
    SEMANTIC_SIMILARITY: 'sentence similarity',
    CLASSIFICATION: 'classification',
    CLUSTERING: 'clustering',
};

/** Matches `gemini-embedding-2` ids taking task and title as text prefixes. */
const MATCH_PREFIX_MODEL = /^(?:(?:publishers\/[^/]+\/)?models\/)?gemini-embedding-2/;

/**
 * **Google publisher prefix of Vertex AI model ids**: stripped before the SDK call.
 *
 * - SDK picks `predict` or `embedContent` by exact id (`gemini-embedding-001`) and adds this prefix again
 * - `models/` and other publishers stay unchanged
 */
const MATCH_GOOGLE_PUBLISHER = /^publishers\/google\/models\//;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **`embed` of a client**: one `embedContent` SDK call (Developer API: `batchEmbedContents`) per request.
 *
 * - each input is its own `Content`, so `gemini-embedding-2` returns one vector per input instead of one fused vector
 * - Vertex AI: multi-text input for non-`gemini-embedding-001` Gemini or `maas` models → `unsupported`
 * - invalid requests, more vectors or fewer than inputs and empty vectors never reach the caller as data
 */
export function createEmbed(ai: GoogleGenAI, mode: GenaiClient['mode']): GenaiClient['embed'] {
    return guard(async function embed(request, options = {}) {
        checkOptionKeys(options, OPTION_KEYS);
        const { signal } = options;
        const inputs = checkEmbedRequest(request);
        const model = request.model;
        const sdkModel = mode === 'vertex' ? model.replace(MATCH_GOOGLE_PUBLISHER, '') : model;
        const { contents, config } = toEmbedCall(request, inputs);
        if (signal?.aborted) {
            return { ok: false, error: { code: 'aborted', message: 'Request aborted', retryable: false } };
        }
        if (mode === 'vertex' && inputs.length > 1 && isSingleContentVertexModel(sdkModel)) {
            const message = `Vertex AI embeds one text per call for ${model}; got ${inputs.length}`;
            return { ok: false, error: { code: 'unsupported', message, retryable: false } };
        }
        const startedAt = performance.now();
        let response: EmbedContentResponse;
        try {
            response = await ai.models.embedContent({
                model: sdkModel,
                contents,
                config: { ...config, ...(signal ? { abortSignal: signal } : {}) },
            });
        } catch (error) {
            return { ok: false, error: { ...toGenaiError(error, signal), ms: elapsedMs(startedAt) } };
        }
        return readEmbeddings(response, inputs.length, model, elapsedMs(startedAt));
    });
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Checks if Vertex AI limits model to single content per call (other Gemini/MaaS models route to `embedContent`, `gemini-embedding-001` to `predict`). */
function isSingleContentVertexModel(model: string): boolean {
    return (model.includes('gemini') && model !== 'gemini-embedding-001') || model.includes('maas');
}

/**
 * Validates embedding request and normalizes inputs.
 * @throws {InvalidRequestError} If field, input, task, or dimensions are invalid.
 */
function checkEmbedRequest(request: GenaiEmbedRequest): EmbedText[] {
    checkOptionKeys(request, REQUEST_KEYS, 'request.');
    if (typeof request.model !== 'string' || request.model === '') {
        throw new InvalidRequestError('Invalid value for option "model"');
    }
    const given: readonly GenaiEmbedInput[] = typeof request.input === 'string' ? [request.input] : request.input;
    if (!Array.isArray(given) || given.length === 0 || given.length > MAX_EMBED_INPUTS) {
        throw new InvalidRequestError(`Invalid value for option "input": 1 to ${MAX_EMBED_INPUTS} texts`);
    }
    const inputs = given.map((input, index) => toEmbedText(input, index));
    const { taskType, dimensions } = request;
    if (taskType !== undefined && taskType !== 'RETRIEVAL_DOCUMENT' && !Object.hasOwn(TASK_PREFIXES, taskType)) {
        throw new InvalidRequestError('Invalid value for option "taskType"');
    }
    if (taskType !== 'RETRIEVAL_DOCUMENT' && inputs.some((input) => input.title !== undefined)) {
        throw new InvalidRequestError('Option "input.title" is not supported: only with taskType RETRIEVAL_DOCUMENT');
    }
    if (dimensions !== undefined && (!Number.isInteger(dimensions) || dimensions <= 0)) {
        throw new InvalidRequestError('Invalid value for option "dimensions": a positive integer');
    }
    return inputs;
}

/**
 * Normalizes single embedding input into text and optional title.
 * @throws {InvalidRequestError} If input is not non-empty text or has invalid title.
 */
function toEmbedText(input: GenaiEmbedInput, index: number): EmbedText {
    const option = `input.${index}`;
    if (typeof input === 'string') {
        if (input === '') {
            throw new InvalidRequestError(`Invalid value for option "${option}": a non-empty text`);
        }
        return { text: input };
    }
    checkOptionKeys(input, INPUT_KEYS, `${option}.`);
    if (typeof input.text !== 'string' || input.text === '') {
        throw new InvalidRequestError(`Invalid value for option "${option}.text": a non-empty text`);
    }
    if (input.title !== undefined && (typeof input.title !== 'string' || input.title === '')) {
        throw new InvalidRequestError(`Invalid value for option "${option}.title": a non-empty text`);
    }
    return input.title === undefined ? { text: input.text } : { text: input.text, title: input.title };
}

/**
 * **Contents and config of the SDK call**.
 *
 * - `gemini-embedding-2`: task and titles as text prefixes, no `taskType` or `title` field
 * - other models: `taskType` and the one `title` all inputs share
 * @throws {InvalidRequestError} If inputs of a model with a `title` field have different titles.
 */
function toEmbedCall(
    request: GenaiEmbedRequest,
    inputs: EmbedText[],
): { contents: Content[]; config: EmbedContentConfig } {
    const { taskType, dimensions } = request;
    const config: EmbedContentConfig = dimensions === undefined ? {} : { outputDimensionality: dimensions };
    if (MATCH_PREFIX_MODEL.test(request.model)) {
        return { contents: inputs.map((input) => toContent(prefixedText(input, taskType))), config };
    }
    const titles = new Set(inputs.map((input) => input.title));
    if (titles.size > 1) {
        throw new InvalidRequestError(`Invalid value for option "input": ${request.model} takes one title per call`);
    }
    const [title] = titles;
    return {
        contents: inputs.map((input) => toContent(input.text)),
        config: {
            ...config,
            ...(taskType === undefined ? {} : { taskType }),
            ...(title === undefined ? {} : { title }),
        },
    };
}

/** Prepends `gemini-embedding-2` task/title prefix to input text when `taskType` is set. */
function prefixedText(input: EmbedText, taskType: GenaiEmbedTask | undefined): string {
    if (taskType === undefined) {
        return input.text;
    }
    if (taskType === 'RETRIEVAL_DOCUMENT') {
        return `title: ${input.title ?? 'none'} | text: ${input.text}`;
    }
    return `task: ${TASK_PREFIXES[taskType]} | query: ${input.text}`;
}

/** One user content with one text part. */
function toContent(text: string): Content {
    return { role: 'user', parts: [{ text }] };
}

/** Extracts embedding vectors; returns `unexpected` if vector count mismatches or any vector is empty. */
function readEmbeddings(
    response: EmbedContentResponse,
    count: number,
    model: string,
    ms: number,
): Result<GenaiEmbedOutput, GenaiError> {
    const embeddings = (response.embeddings ?? []).map((embedding) => embedding.values ?? []);
    const message =
        embeddings.length === count
            ? embeddings.some((values) => values.length === 0) && 'The response has an empty vector'
            : `The response has ${embeddings.length} vectors for ${count} inputs`;
    if (message) {
        return { ok: false, error: { code: 'unexpected', message, retryable: false, model, ms } };
    }
    return { ok: true, data: { embeddings, model, ms } };
}
