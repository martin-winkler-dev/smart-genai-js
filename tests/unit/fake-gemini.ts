// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Inline request entry of `batchGenerateContent` body. */
export type GeminiInlinedRequest = {
    metadata: { key: string; format?: string };
    request: Record<string, unknown> & { generationConfig: Record<string, unknown> };
};

/** `batch` member of `batchGenerateContent` body. */
export type GeminiBatchBody = {
    displayName: string;
    inputConfig: { requests: { requests: GeminiInlinedRequest[] } };
};

/** Request received by fake Gemini server; `search` as in URL (`?a=1`, or empty). */
export type GeminiRequest = {
    method: string;
    path: string;
    search: string;
    headers: Headers;
    body: Record<string, unknown> | null;
};

/** Handler returning response for fake Gemini request. */
export type GeminiReply = (request: GeminiRequest) => Response | Promise<Response>;

/** Local fake Gemini Developer API test server. */
export type FakeGemini = {
    baseUrl: string;
    received: GeminiRequest[];
    /** Sets reply sequence; repeats final reply once reached. */
    reply(...replies: GeminiReply[]): void;
    stop(): void;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Models per page when request sets no `pageSize`, as real API does. */
const MODELS_DEFAULT_PAGE_SIZE = 50;

/** Maximum `pageSize` of models list; real API caps larger values, fake rejects them to catch them. */
const MODELS_MAX_PAGE_SIZE = 1000;

/** Top-level fields of a Decisions API request. */
const DECISION_KEYS: ReadonlySet<string> = new Set(['model', 'input', 'questions']);

/** Fields of a Decisions API question and of `noul` criteria. */
const QUESTION_KEYS: ReadonlySet<string> = new Set(['type', 'instructions', 'criteria']);
const NOUL_CRITERIA_KEYS: ReadonlySet<string> = new Set(['true_description', 'false_description']);

/** Fields of Decisions API input blocks by `type`. */
const BLOCK_KEYS: Readonly<Record<string, ReadonlySet<string>>> = {
    text: new Set(['type', 'text']),
    image: new Set(['type', 'data', 'uri', 'mime_type', 'resolution']),
    video: new Set(['type', 'data', 'uri', 'mime_type', 'resolution']),
    audio: new Set(['type', 'data', 'uri', 'mime_type']),
    document: new Set(['type', 'data', 'uri', 'mime_type']),
};

/** Resolutions of `image` and `video` blocks. */
const RESOLUTIONS: Readonly<Record<string, ReadonlySet<string>>> = {
    image: new Set(['low', 'medium', 'high', 'ultra_high']),
    video: new Set(['low', 'medium', 'high']),
};

/** Maximum questions per request and maximum options or levels per question. */
const MAX_QUESTIONS = 1000;
const MAX_CRITERIA = 255;

/** Matches question keys accepted by the Decisions API. */
const MATCH_QUESTION_KEY = /^[A-Za-z0-9_-]{1,64}$/;

/** Matches choice option ids: non-empty, no outer whitespace, no control characters. */
const MATCH_OPTION_ID = /^(?!\s)[^\p{Cc}]+(?<!\s)$/u;

/** Matches `mime_type` of `document` blocks: PDF, CSV, Office and OpenDocument types. */
const MATCH_DOCUMENT_MIME =
    /^(application\/pdf|text\/csv|application\/msword|application\/vnd\.ms-(excel|powerpoint)|application\/vnd\.openxmlformats-officedocument\..+|application\/vnd\.oasis\.opendocument\..+)$/;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/** Starts fake Gemini server on random loopback port. */
export function startFakeGemini(): FakeGemini {
    const received: GeminiRequest[] = [];
    let replies: GeminiReply[] = [() => generateResponse('{}')];
    let index = 0;
    const server = Bun.serve({
        port: 0,
        hostname: '127.0.0.1',
        async fetch(request) {
            const url = new URL(request.url);
            const entry: GeminiRequest = {
                method: request.method,
                path: url.pathname,
                search: url.search,
                headers: request.headers,
                body: request.method === 'GET' ? null : ((await request.json()) as Record<string, unknown>),
            };
            received.push(entry);
            const reply = replies[Math.min(index, replies.length - 1)];
            index++;
            return reply(entry);
        },
    });
    return {
        baseUrl: `http://127.0.0.1:${server.port}`,
        received,
        reply(...next) {
            replies = next;
            index = 0;
        },
        stop() {
            server.stop(true);
        },
    };
}

/**
 * Reads `generationConfig` of received generateContent body; typed as tests read it, not validated.
 * @throws if body has no `generationConfig` object
 */
export function sentConfig(request: GeminiRequest | undefined): Record<string, unknown> {
    return bodyMember(request, 'generationConfig') as Record<string, unknown>;
}

/**
 * Reads `batch` of received `batchGenerateContent` body; typed as tests read it, not validated.
 * @throws if body has no `batch` object
 */
export function sentBatch(request: GeminiRequest | undefined): GeminiBatchBody {
    return bodyMember(request, 'batch') as GeminiBatchBody;
}

/** Constructs wire response body for generateContent calls. */
export function generateBody(
    text: string,
    overrides: { finishReason?: string; thought?: string; blockReason?: string } = {},
): Record<string, unknown> {
    const parts: Record<string, unknown>[] = [];
    if (overrides.thought !== undefined) {
        parts.push({ text: overrides.thought, thought: true });
    }
    parts.push({ text });
    return {
        candidates: [{ content: { role: 'model', parts }, finishReason: overrides.finishReason ?? 'STOP', index: 0 }],
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, thoughtsTokenCount: 3, totalTokenCount: 19 },
        modelVersion: 'gemini-fake-001',
        ...(overrides.blockReason === undefined ? {} : { promptFeedback: { blockReason: overrides.blockReason } }),
    };
}

/** Creates JSON HTTP Response wrapping generateBody. */
export function generateResponse(text: string, overrides: Parameters<typeof generateBody>[1] = {}): Response {
    return Response.json(generateBody(text, overrides));
}

/** Creates JSON HTTP Response representing batch operation; `error` as top-level operation error. */
export function batchResponse(
    name: string,
    state: string,
    responses?: Record<string, unknown>[],
    error?: { code: number; message: string },
): Response {
    return Response.json({
        name,
        ...(error === undefined ? {} : { done: true, error }),
        metadata: {
            '@type': 'type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch',
            name,
            model: 'models/gemini-2.5-flash',
            state,
            ...(responses === undefined ? {} : { output: { inlinedResponses: { inlinedResponses: responses } } }),
        },
    });
}

/** Creates Google API error response with status and optional retry delay. */
export function apiError(status: number, retryDelay?: string): Response {
    const details =
        retryDelay === undefined ? [] : [{ '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay }];
    return Response.json({ error: { code: status, message: `fake ${status}`, status: 'FAKE', details } }, { status });
}

/**
 * **Creates models list reply**: pages of `models` by `pageSize` and `pageToken` of request query.
 *
 * - path other than `GET /v1beta/models`: HTTP 404
 * - no `pageSize`: 50; above 1000 or not a positive integer: HTTP 400 `INVALID_ARGUMENT`
 * - `nextPageToken` (offset of next page) while more models follow
 */
export function modelsReply(models: Record<string, unknown>[]): GeminiReply {
    return (request) => {
        if (request.method !== 'GET' || request.path !== '/v1beta/models') {
            return apiError(404);
        }
        const query = new URLSearchParams(request.search);
        const size = Number(query.get('pageSize') ?? MODELS_DEFAULT_PAGE_SIZE);
        if (!Number.isInteger(size) || size <= 0 || size > MODELS_MAX_PAGE_SIZE) {
            const error = { code: 400, message: 'fake 400: invalid page_size', status: 'INVALID_ARGUMENT' };
            return Response.json({ error }, { status: 400 });
        }
        const offset = Number(query.get('pageToken') ?? '0');
        const next = offset + size < models.length ? { nextPageToken: String(offset + size) } : {};
        return Response.json({ models: models.slice(offset, offset + size), ...next });
    };
}

/**
 * **Creates Decisions API reply**: validates `POST /v1beta/decisions` like the real API, then answers each question.
 *
 * - other path: HTTP 404; key `bad-key`: HTTP 400 in classic Google format (`API_KEY_INVALID`)
 * - `x-goog-api-key` missing or empty: HTTP 403 `PERMISSION_DENIED` in classic Google format
 * - invalid body, input block or question: HTTP 400 `invalid_request`
 * - model other than `gemini-decision-1` (also `models/`): HTTP 404 `not_found`
 * - answers: `noul` 0.8; `choice` first option most probable; `score` weighted index, last level most probable
 * - usage: text 1 token per 4 characters, 258 tokens per media block
 * - messages seen live: HTTP 403 body, `Unknown parameter '<name>'.`, `Unknown parameter '<name>' at 'input[<i>]'.`,
 *   `Invalid input.` (`questions` no object), `Unsupported mime_type …` of documents; all other messages assumed
 */
export function decisionsReply(): GeminiReply {
    return (request) => {
        if (request.method !== 'POST' || request.path !== '/v1beta/decisions') {
            return apiError(404);
        }
        const key = request.headers.get('x-goog-api-key');
        if (key === null || key === '') {
            const error = { code: 403, message: 'Permission denied on the service.', status: 'PERMISSION_DENIED' };
            return Response.json({ error }, { status: 403 });
        }
        if (key === 'bad-key') {
            const details = [{ '@type': 'type.googleapis.com/google.rpc.ErrorInfo', reason: 'API_KEY_INVALID' }];
            const error = { code: 400, message: 'API key not valid.', status: 'INVALID_ARGUMENT', details };
            return Response.json({ error }, { status: 400 });
        }
        const body = request.body ?? {};
        const unknown = Object.keys(body).find((name) => !DECISION_KEYS.has(name));
        if (unknown !== undefined) {
            return decisionError(400, 'invalid_request', `Unknown parameter '${unknown}'.`);
        }
        if (body.model !== 'gemini-decision-1' && body.model !== 'models/gemini-decision-1') {
            return decisionError(404, 'not_found', `Model '${String(body.model)}' not found.`);
        }
        const blocks = toFakeBlocks(body.input);
        if (typeof blocks === 'string') {
            return decisionError(400, 'invalid_request', blocks);
        }
        const problem = questionsProblem(body.questions);
        if (problem !== undefined) {
            return decisionError(400, 'invalid_request', problem);
        }
        const questions = body.questions as Record<string, Record<string, unknown>>;
        const answers = Object.fromEntries(
            Object.entries(questions).map(([key, question]) => [key, fakeAnswer(question)]),
        );
        const text = blocks.map((block) => (typeof block.text === 'string' ? block.text : '')).join('');
        const media = blocks.filter((block) => block.type !== 'text').length;
        const textTokens = Math.ceil(text.length / 4);
        const total = textTokens + 258 * media;
        const byModality = [
            { modality: 'text', tokens: textTokens },
            ...(media === 0 ? [] : [{ modality: 'image', tokens: 258 * media }]),
        ];
        return Response.json({
            answers,
            usage: {
                total_tokens: total,
                total_input_tokens: total,
                input_tokens_by_modality: byModality,
                total_output_tokens: 0,
            },
        });
    };
}

/** Creates Decisions API error response: string `code`, as the real API answers. */
export function decisionError(status: number, code: string, message: string): Response {
    return Response.json({ error: { message, code } }, { status });
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * **Input as blocks, or message of the first invalid block**.
 *
 * - input: non-empty text or 1 or more blocks
 * - block: known `type`, only its fields; `text` non-empty; media exactly one of `data` and `uri`, `mime_type`
 * - `resolution` per block type; `document` only PDF, CSV, Office and OpenDocument types
 */
function toFakeBlocks(input: unknown): Record<string, unknown>[] | string {
    if (typeof input === 'string') {
        return isText(input) ? [{ type: 'text', text: input }] : 'Input must not be empty.';
    }
    if (!Array.isArray(input) || input.length === 0) {
        return 'Invalid input.';
    }
    for (const [index, block] of (input as unknown[]).entries()) {
        const problem = blockProblem(block, `input[${index}]`);
        if (problem !== undefined) {
            return problem;
        }
    }
    return input as Record<string, unknown>[];
}

/** Message of an invalid input block at `at`, or `undefined`. */
function blockProblem(block: unknown, at: string): string | undefined {
    if (!isObject(block) || typeof block.type !== 'string') {
        return `The 'type' parameter is required at '${at}'.`;
    }
    const keys = Object.hasOwn(BLOCK_KEYS, block.type) ? BLOCK_KEYS[block.type] : undefined;
    if (keys === undefined) {
        return `Unknown block type '${block.type}' at '${at}'.`;
    }
    const unknown = Object.keys(block).find((name) => !keys.has(name));
    if (unknown !== undefined) {
        return `Unknown parameter '${unknown}' at '${at}'.`;
    }
    if (block.type === 'text') {
        return isText(block.text) ? undefined : `The 'text' parameter must be a non-empty string at '${at}'.`;
    }
    const sources = [block.data, block.uri].filter((value) => value !== undefined);
    if (sources.length !== 1 || !isText(sources[0])) {
        return `Exactly one of 'data' and 'uri' is required at '${at}'.`;
    }
    if (!isText(block.mime_type)) {
        return `The 'mime_type' parameter is required at '${at}'.`;
    }
    if (block.type === 'document' && !MATCH_DOCUMENT_MIME.test(block.mime_type)) {
        return `Unsupported mime_type '${block.mime_type}' in document content. Only application/pdf, text/csv, and supported Office/OpenDocument MIME types are supported.`;
    }
    const resolutions = RESOLUTIONS[block.type];
    if (block.resolution !== undefined && !resolutions?.has(block.resolution as string)) {
        return `Invalid resolution '${String(block.resolution)}' at '${at}'.`;
    }
    return undefined;
}

/**
 * **Message of the first invalid question, or `undefined`**.
 *
 * - `questions`: object of 1 to 1000 questions; keys `[A-Za-z0-9_-]`, 1 to 64 characters
 * - question: only `type`, `instructions`, `criteria`; instructions non-empty text, text block or text blocks
 * - `noul` criteria: only description texts; `choice`: 2 to 255 option ids with texts; `score`: 2 to 255 level texts
 */
function questionsProblem(questions: unknown): string | undefined {
    if (!isObject(questions)) {
        return 'Invalid input.';
    }
    const entries = Object.entries(questions);
    if (entries.length === 0) {
        return 'At least one question is required.';
    }
    if (entries.length > MAX_QUESTIONS) {
        return `At most ${MAX_QUESTIONS} questions are allowed.`;
    }
    for (const [key, question] of entries) {
        const at = `questions.${key}`;
        if (!MATCH_QUESTION_KEY.test(key)) {
            return `Invalid question key '${key}'.`;
        }
        if (!isObject(question)) {
            return `Invalid question at '${at}'.`;
        }
        const unknown = Object.keys(question).find((name) => !QUESTION_KEYS.has(name));
        if (unknown !== undefined) {
            return `Unknown parameter '${unknown}' at '${at}'.`;
        }
        if (!instructionsValid(question.instructions)) {
            return `The 'instructions' parameter must be a non-empty text or text blocks at '${at}'.`;
        }
        const problem = criteriaProblem(question.type, question.criteria, at);
        if (problem !== undefined) {
            return problem;
        }
    }
    return undefined;
}

/** Message of an invalid question type or criteria at `at`, or `undefined`. */
function criteriaProblem(type: unknown, criteria: unknown, at: string): string | undefined {
    if (type === 'noul') {
        if (criteria === undefined) {
            return undefined;
        }
        const valid =
            isObject(criteria) &&
            Object.entries(criteria).every(
                ([name, value]) => NOUL_CRITERIA_KEYS.has(name) && typeof value === 'string',
            );
        return valid ? undefined : `Invalid noul criteria at '${at}'.`;
    }
    if (type === 'choice') {
        const entries = isObject(criteria) ? Object.entries(criteria) : [];
        const valid =
            entries.length >= 2 &&
            entries.length <= MAX_CRITERIA &&
            entries.every(([id, description]) => MATCH_OPTION_ID.test(id) && typeof description === 'string');
        return valid ? undefined : `A choice question requires 2 to ${MAX_CRITERIA} valid options at '${at}'.`;
    }
    if (type === 'score') {
        const valid =
            Array.isArray(criteria) &&
            criteria.length >= 2 &&
            criteria.length <= MAX_CRITERIA &&
            Array.from(criteria as unknown[]).every(isText);
        return valid ? undefined : `A score question requires 2 to ${MAX_CRITERIA} level texts at '${at}'.`;
    }
    return `Unknown question type '${String(type)}' at '${at}'.`;
}

/** Whether instructions are a non-empty text, a text block or 1 or more text blocks. */
function instructionsValid(instructions: unknown): boolean {
    if (typeof instructions === 'string') {
        return isText(instructions);
    }
    const blocks = Array.isArray(instructions) ? Array.from(instructions as unknown[]) : [instructions];
    return (
        blocks.length > 0 &&
        blocks.every((block) => isObject(block) && block.type === 'text' && blockProblem(block, '') === undefined)
    );
}

/** Whether the value is a text with a non-whitespace character. */
function isText(value: unknown): value is string {
    return typeof value === 'string' && value.trim() !== '';
}

/** Whether the value is an object, not `null` and no array. */
function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Answer of the fake API to one valid question. */
function fakeAnswer(question: Record<string, unknown>): Record<string, unknown> {
    if (question.type === 'noul') {
        return { type: 'noul', noul: 0.8, confidence: 0.8 };
    }
    if (question.type === 'choice') {
        const options = Object.keys(question.criteria as object);
        const other = 0.3 / (options.length - 1);
        const probabilities = Object.fromEntries(options.map((option, index) => [option, index === 0 ? 0.7 : other]));
        return { type: 'choice', choice: options[0], probabilities, confidence: 0.7 };
    }
    const levels = question.criteria as string[];
    const top = levels.length - 1;
    const weights = levels.map((_, index) => (index === top ? 0.75 : 0.25 / top));
    const probabilities = Object.fromEntries(weights.map((weight, index) => [String(index), weight]));
    const score = weights.reduce((sum, weight, index) => sum + index * weight, 0);
    const legend = Object.fromEntries(levels.map((level, index) => [String(index), level]));
    return { type: 'score', score, probabilities, legend, confidence: 0.75 };
}

/**
 * Reads object member of received request body.
 * @throws if request, body or member object missing
 */
function bodyMember(request: GeminiRequest | undefined, name: string): object {
    const value = request?.body?.[name];
    if (typeof value !== 'object' || value === null) {
        throw new Error(`Request body has no "${name}" object`);
    }
    return value;
}
