// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type {
    GenaiClient,
    GenaiDecideAnswer,
    GenaiDecideOutput,
    GenaiDecideQuestion,
    GenaiDecideRequest,
    GenaiError,
    Result,
} from '../types.js';
import { elapsedMs, guard, readRetryDelayMs, statusError, toGenaiError } from './answer.js';
import type { InlineBatchRest } from './batch/inline.js';
import { checkOptionKeys, InvalidRequestError, MATCH_IMAGE_DATA_URL } from './request.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Input block or question in Decisions API form, as sent. */
type ApiBlock = Record<string, unknown>;

/** Request body of `POST /decisions`. */
export type DecideBody = { model: string; input: string | ApiBlock[]; questions: Record<string, GenaiDecideQuestion> };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Default model for `decide`; only model of Decisions API. */
export const DEFAULT_DECIDE_MODEL = 'gemini-decision-1';

/** Maximum questions per decision request. */
export const MAX_DECIDE_QUESTIONS = 1000;

/** Minimum and maximum options of a `choice` question and levels of a `score` question. */
const MIN_CRITERIA = 2;
const MAX_CRITERIA = 255;

/** Supported field keys of `GenaiDecideRequest`, of a question, of `noul` criteria and of the call options. */
const REQUEST_KEYS: ReadonlySet<string> = new Set(['model', 'input', 'questions']);
const QUESTION_KEYS: ReadonlySet<string> = new Set(['type', 'instructions', 'criteria']);
const NOUL_CRITERIA_KEYS: ReadonlySet<string> = new Set(['true_description', 'false_description']);
const OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Supported field keys of a text part, in input and instructions. */
const TEXT_PART_KEYS: ReadonlySet<string> = new Set(['type', 'text']);

/** Supported field keys of input parts by `type`. */
const PART_KEYS: Readonly<Record<string, ReadonlySet<string>>> = {
    text: TEXT_PART_KEYS,
    // biome-ignore lint/style/useNamingConvention: OpenAI part type
    image_url: new Set(['type', 'image_url']),
    image: new Set(['type', 'data', 'uri', 'mime_type', 'resolution']),
    video: new Set(['type', 'data', 'uri', 'mime_type', 'resolution']),
    audio: new Set(['type', 'data', 'uri', 'mime_type']),
    document: new Set(['type', 'data', 'uri', 'mime_type']),
};

/** Supported keys of `image_url` member of a chat image part. */
const IMAGE_URL_KEYS: ReadonlySet<string> = new Set(['url', 'detail']);

/** Resolutions accepted by `image` blocks; `video` blocks take all but `ultra_high`. */
const IMAGE_RESOLUTIONS: ReadonlySet<string> = new Set(['low', 'medium', 'high', 'ultra_high']);
const VIDEO_RESOLUTIONS: ReadonlySet<string> = new Set(['low', 'medium', 'high']);

/** Error codes of Decisions API responses for input blocked by safety filters. */
const BLOCKED_CODES: ReadonlySet<string> = new Set([
    'safety',
    'prohibited_content',
    'blocklist',
    'image_safety',
    'content_blocked',
]);

/** Matches question keys accepted by the API. */
const MATCH_QUESTION_KEY = /^[A-Za-z0-9_-]{1,64}$/;

/** Matches choice option ids: non-empty, no leading or trailing whitespace, no control characters. */
const MATCH_OPTION_ID = /^(?!\s)[^\p{Cc}]+(?<!\s)$/u;

/** Matches trailing slash of base URL. */
const MATCH_TRAILING_SLASH = /\/$/;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **`decide` of a client**: one `POST <baseUrl>/<apiVersion>/decisions` REST call per request (SDK 2.x has no Decisions API).
 *
 * - `rest` missing (Vertex AI): `unsupported` after the request and abort checks, nothing sent
 * - chat image parts (`image_url` data URLs) become `image` blocks; other parts and questions sent as given
 * - error body `{ error: { code, message } }`: safety codes and `prompt_feedback.block_reason` → `blocked`
 * - answers missing or of another kind than their question never reach the caller as data
 */
export function createDecide(rest: InlineBatchRest | undefined): GenaiClient['decide'] {
    const decide = guard(async function decide(
        request: GenaiDecideRequest,
        options: { signal?: AbortSignal } = {},
    ): Promise<Result<GenaiDecideOutput, GenaiError>> {
        checkOptionKeys(options, OPTION_KEYS);
        const { signal } = options;
        const body = toDecideBody(request);
        if (signal?.aborted) {
            return { ok: false, error: { code: 'aborted', message: 'Request aborted', retryable: false } };
        }
        if (rest === undefined) {
            const message = 'Decisions API is not supported on Vertex AI: Gemini Developer API only';
            return { ok: false, error: { code: 'unsupported', message, retryable: false } };
        }
        const url = `${rest.baseUrl.replace(MATCH_TRAILING_SLASH, '')}/${rest.apiVersion}/decisions`;
        const timeout = AbortSignal.timeout(rest.timeoutMs);
        const startedAt = performance.now();
        let status: number;
        let text: string;
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: { 'content-type': 'application/json', 'x-goog-api-key': rest.apiKey },
                body: JSON.stringify(body),
                signal: signal === undefined ? timeout : AbortSignal.any([signal, timeout]),
            });
            status = response.status;
            text = await response.text();
        } catch (error) {
            return {
                ok: false,
                error: { ...toGenaiError(error, signal), model: body.model, ms: elapsedMs(startedAt) },
            };
        }
        const ms = elapsedMs(startedAt);
        if (status < 200 || status >= 300) {
            return { ok: false, error: { ...httpError(status, text), model: body.model, ms } };
        }
        return readDecision(text, body, ms);
    });
    return decide as GenaiClient['decide'];
}

/**
 * Validates decision request and builds request body.
 * @throws {InvalidRequestError} If field, input, or question is invalid.
 */
export function toDecideBody(request: GenaiDecideRequest): DecideBody {
    checkOptionKeys(request, REQUEST_KEYS, 'request.');
    const model = request.model ?? DEFAULT_DECIDE_MODEL;
    if (typeof model !== 'string' || model.trim() === '') {
        throw new InvalidRequestError('Invalid value for option "model"');
    }
    const { questions } = request;
    if (typeof questions !== 'object' || questions === null || Array.isArray(questions)) {
        throw new InvalidRequestError('Invalid value for option "questions"');
    }
    const keys = Object.keys(questions);
    if (keys.length === 0 || keys.length > MAX_DECIDE_QUESTIONS) {
        throw new InvalidRequestError(`Invalid value for option "questions": 1 to ${MAX_DECIDE_QUESTIONS} questions`);
    }
    for (const key of keys) {
        if (!MATCH_QUESTION_KEY.test(key)) {
            throw new InvalidRequestError(
                `Invalid value for option "questions": key "${key}" must be 1 to 64 characters of [A-Za-z0-9_-]`,
            );
        }
        checkQuestion(questions[key], `questions.${key}`);
    }
    return { model, input: toInput(request.input), questions: { ...questions } };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * Converts decision input to string or API block list.
 * @throws {InvalidRequestError} If input is empty or has invalid part.
 */
function toInput(input: unknown): string | ApiBlock[] {
    if (typeof input === 'string') {
        if (input.trim() === '') {
            throw new InvalidRequestError('Invalid value for option "input": a non-empty text or 1 or more parts');
        }
        return input;
    }
    if (!Array.isArray(input) || input.length === 0) {
        throw new InvalidRequestError('Invalid value for option "input": a non-empty text or 1 or more parts');
    }
    return Array.from(input as unknown[], (part, index) => toBlock(part, `input.${index}`));
}

/**
 * **Input part as Decisions API block**.
 *
 * - `text`: non-empty text
 * - `image_url`: base64 data URL of `image/*` → `{ type: 'image', data, mime_type }`; `detail` ignored
 * - media blocks: exactly one of `data` and `uri`, `mime_type`, `resolution` per block type
 * @throws {InvalidRequestError} If the part is of unknown type or has unknown or invalid fields.
 */
function toBlock(part: unknown, option: string): ApiBlock {
    const type = (part as { type?: unknown } | null)?.type;
    const keys = typeof type === 'string' && Object.hasOwn(PART_KEYS, type) ? PART_KEYS[type] : undefined;
    if (keys === undefined) {
        throw new InvalidRequestError(
            `Invalid value for option "${option}.type": text, image_url, image, audio, video or document`,
        );
    }
    checkOptionKeys(part, keys, `${option}.`);
    const fields = part as Record<string, unknown>;
    if (type === 'text') {
        checkText(fields.text, `${option}.text`);
        return { type, text: fields.text };
    }
    if (type === 'image_url') {
        checkOptionKeys(fields.image_url, IMAGE_URL_KEYS, `${option}.image_url.`);
        const url = (fields.image_url as { url?: unknown }).url;
        const match = typeof url === 'string' ? MATCH_IMAGE_DATA_URL.exec(url) : null;
        if (match === null) {
            throw new InvalidRequestError(`Invalid value for option "${option}.image_url.url": a base64 data URL`);
        }
        // biome-ignore lint/style/useNamingConvention: Decisions API field name
        return { type: 'image', data: match[2], mime_type: match[1] };
    }
    const { data, uri, mime_type: mimeType, resolution } = fields;
    if ((data === undefined) === (uri === undefined)) {
        throw new InvalidRequestError(`Invalid value for option "${option}": exactly one of data and uri`);
    }
    for (const [name, value] of [
        ['data', data],
        ['uri', uri],
        ['mime_type', mimeType],
    ] as const) {
        if (value !== undefined || name === 'mime_type') {
            checkText(value, `${option}.${name}`);
        }
    }
    const resolutions = type === 'image' ? IMAGE_RESOLUTIONS : VIDEO_RESOLUTIONS;
    if (resolution !== undefined && (typeof resolution !== 'string' || !resolutions.has(resolution))) {
        throw new InvalidRequestError(
            `Invalid value for option "${option}.resolution": ${[...resolutions].join(', ')}`,
        );
    }
    return { ...fields };
}

/**
 * **Checks one question**.
 *
 * - `noul`: `criteria` optional, only `true_description` and `false_description` texts
 * - `choice`: 2 to 255 option ids (non-empty, no outer whitespace or control characters) with description texts
 * - `score`: 2 to 255 level texts
 * @throws {InvalidRequestError} If the question has unknown fields, an unknown type, no instructions or invalid criteria.
 */
function checkQuestion(question: unknown, option: string): void {
    checkOptionKeys(question, QUESTION_KEYS, `${option}.`);
    const { type, instructions, criteria } = question as Record<string, unknown>;
    if (type !== 'noul' && type !== 'choice' && type !== 'score') {
        throw new InvalidRequestError(`Invalid value for option "${option}.type": noul, choice or score`);
    }
    checkInstructions(instructions, `${option}.instructions`);
    const at = `${option}.criteria`;
    if (type === 'noul') {
        if (criteria !== undefined) {
            checkOptionKeys(criteria, NOUL_CRITERIA_KEYS, `${at}.`);
            for (const [name, value] of Object.entries(criteria as object)) {
                if (value !== undefined && typeof value !== 'string') {
                    throw new InvalidRequestError(`Invalid value for option "${at}.${name}": a text`);
                }
            }
        }
        return;
    }
    if (type === 'choice') {
        if (typeof criteria !== 'object' || criteria === null || Array.isArray(criteria)) {
            throw new InvalidRequestError(`Invalid value for option "${at}": an object of option ids to descriptions`);
        }
        const entries = Object.entries(criteria);
        if (entries.length < MIN_CRITERIA || entries.length > MAX_CRITERIA) {
            throw new InvalidRequestError(
                `Invalid value for option "${at}": ${MIN_CRITERIA} to ${MAX_CRITERIA} options`,
            );
        }
        for (const [id, description] of entries) {
            if (!MATCH_OPTION_ID.test(id)) {
                throw new InvalidRequestError(
                    `Invalid value for option "${at}": option id "${id}" must be non-empty, without outer whitespace or control characters`,
                );
            }
            if (typeof description !== 'string') {
                throw new InvalidRequestError(`Invalid value for option "${at}.${id}": a text`);
            }
        }
        return;
    }
    if (!Array.isArray(criteria) || criteria.length < MIN_CRITERIA || criteria.length > MAX_CRITERIA) {
        throw new InvalidRequestError(`Invalid value for option "${at}": ${MIN_CRITERIA} to ${MAX_CRITERIA} levels`);
    }
    for (const [index, level] of (criteria as unknown[]).entries()) {
        checkText(level, `${at}.${index}`);
    }
}

/**
 * Validates question instructions: non-empty text, text part, or text part list.
 * @throws {InvalidRequestError} If instructions are empty or not text.
 */
function checkInstructions(instructions: unknown, option: string): void {
    if (typeof instructions === 'string') {
        checkText(instructions, option);
        return;
    }
    const parts: unknown[] = Array.isArray(instructions) ? instructions : [instructions];
    if (parts.length === 0) {
        throw new InvalidRequestError(`Invalid value for option "${option}": a non-empty text or text parts`);
    }
    for (const [index, part] of parts.entries()) {
        const at = Array.isArray(instructions) ? `${option}.${index}` : option;
        if ((part as { type?: unknown } | null)?.type !== 'text') {
            throw new InvalidRequestError(`Invalid value for option "${at}": a non-empty text or text parts`);
        }
        checkOptionKeys(part, TEXT_PART_KEYS, `${at}.`);
        checkText((part as { text?: unknown }).text, `${at}.text`);
    }
}

/**
 * Checks non-empty text.
 * @throws {InvalidRequestError} If value is not non-empty text.
 */
function checkText(value: unknown, option: string): void {
    if (typeof value !== 'string' || value.trim() === '') {
        throw new InvalidRequestError(`Invalid value for option "${option}": a non-empty text`);
    }
}

/**
 * **Maps HTTP error response to `GenaiError`**.
 *
 * - Decisions API body `{ error: { code: <text>, message } }`; classic Google body `{ error: { code: <number>, message, status } }`
 * - safety codes (`safety`, `prohibited_content`, …): `blocked`
 * - other statuses via `statusError` with `error.message` as text; 429 with RetryInfo delay
 * - JSON body without text `error.message`: `HTTP <status>`; body no JSON: body text
 */
function httpError(status: number, text: string): GenaiError {
    const parsed = parseJson<{ error?: { code?: unknown; message?: unknown } } | null>(text);
    const error = isPlainObject(parsed) ? parsed.error : undefined;
    const fallback = parsed === undefined ? text : '';
    const message = typeof error?.message === 'string' ? error.message : fallback;
    if (typeof error?.code === 'string' && BLOCKED_CODES.has(error.code)) {
        const detail = typeof error.message === 'string' && error.message !== '' ? `: ${error.message}` : '';
        return { code: 'blocked', message: `Input blocked (${error.code})${detail}`, retryable: false, status };
    }
    return statusError(status, message, status === 429 ? readRetryDelayMs(text) : undefined);
}

/**
 * **Reads decision response**: answers of all questions and token usage.
 *
 * - `prompt_feedback.block_reason`: `blocked`
 * - body no JSON object, answer missing or of another kind than its question: `unexpected`
 */
function readDecision(text: string, body: DecideBody, ms: number): Result<GenaiDecideOutput, GenaiError> {
    const { model } = body;
    const response = parseJson<{
        answers?: unknown;
        // biome-ignore lint/style/useNamingConvention: Decisions API field name
        prompt_feedback?: { block_reason?: unknown; block_reason_message?: unknown };
        // biome-ignore lint/style/useNamingConvention: Decisions API field name
        usage?: { total_input_tokens?: number; total_output_tokens?: number };
    }>(text);
    if (!isPlainObject(response)) {
        return {
            ok: false,
            error: { code: 'unexpected', message: 'The response is no JSON object', retryable: false, model, ms },
        };
    }
    const blockReason = response.prompt_feedback?.block_reason;
    if (typeof blockReason === 'string' && blockReason !== '') {
        const detail = response.prompt_feedback?.block_reason_message;
        const message = `Input blocked (${blockReason})${typeof detail === 'string' && detail !== '' ? `: ${detail}` : ''}`;
        return { ok: false, error: { code: 'blocked', message, retryable: false, model, ms } };
    }
    const rawAnswers = isPlainObject(response.answers) ? response.answers : {};
    const entries: [string, GenaiDecideAnswer][] = [];
    for (const [key, question] of Object.entries(body.questions)) {
        const answer = Object.hasOwn(rawAnswers, key) ? rawAnswers[key] : undefined;
        if (!isPlainObject(answer) || answer.type !== question.type) {
            const message =
                answer === undefined || answer === null
                    ? `The response has no answer for "${key}"`
                    : `The answer for "${key}" is of type ${String((answer as { type?: unknown }).type)}, not ${question.type}`;
            return { ok: false, error: { code: 'unexpected', message, retryable: false, model, ms } };
        }
        entries.push([key, answer as GenaiDecideAnswer]);
    }
    const answers: Record<string, GenaiDecideAnswer> = Object.fromEntries(entries);
    const usage = {
        inputTokens: response.usage?.total_input_tokens ?? null,
        outputTokens: response.usage?.total_output_tokens ?? null,
    };
    return { ok: true, data: { answers, usage, model, ms } };
}

/** Checks whether value is plain non-array object. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Parses JSON text; returns `undefined` on invalid JSON. */
function parseJson<T>(text: string): T | undefined {
    try {
        return JSON.parse(text) as T;
    } catch {
        return undefined;
    }
}
