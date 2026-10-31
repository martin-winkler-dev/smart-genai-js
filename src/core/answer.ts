// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { randomUUID } from 'node:crypto';
import {
    ApiError,
    type GenerateContentResponse,
    type GenerateContentResponseUsageMetadata,
    type Part,
} from '@google/genai';
import type {
    GenaiBatch,
    GenaiBatchStatus,
    GenaiError,
    GenaiOutput,
    GenaiToolCall,
    GenaiUsage,
    Result,
} from '../types.js';
import { InvalidRequestError } from './request.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** HTTP status codes below 500 eligible for retry. */
export const RETRYABLE_STATUSES: ReadonlySet<number> = new Set([408, 409, 425]);

/** Maps Google RPC codes to corresponding HTTP status codes. */
const RPC_HTTP_STATUS: Readonly<Record<number, number>> = {
    1: 499,
    2: 500,
    3: 400,
    4: 504,
    5: 404,
    6: 409,
    7: 403,
    8: 429,
    9: 400,
    10: 409,
    11: 400,
    12: 501,
    13: 500,
    14: 503,
    15: 500,
    16: 401,
};

/** File system error codes treated as non-network errors. */
const FILE_ERROR_CODES: ReadonlySet<string> = new Set(['ENOENT', 'EACCES', 'EPERM', 'EISDIR', 'ENOTDIR']);

/** Matches network and socket failure codes on errors or causes. */
const MATCH_NETWORK_CODE =
    /^(?:ECONN\w+|ENOTFOUND|ETIMEDOUT|EAI_AGAIN|EPIPE|ENETUNREACH|EHOSTUNREACH|UND_ERR_\w+|ConnectionRefused|ConnectionClosed|FailedToOpenSocket)$/;

/** Terminal batch job states containing output results. */
const SUCCEEDED_STATES: ReadonlySet<string> = new Set(['JOB_STATE_SUCCEEDED', 'JOB_STATE_PARTIALLY_SUCCEEDED']);

/** Terminal batch job states without output results. */
const FAILED_STATES: ReadonlySet<string> = new Set(['JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED']);

/** Google RPC RetryInfo detail type URL. */
const RETRY_INFO_TYPE = 'type.googleapis.com/google.rpc.RetryInfo';

/** Matches protobuf duration strings (e.g. `17s`, `1.5s`). */
const MATCH_DURATION = /^(\d+(?:\.\d+)?)s$/;

/** Finish reason codes indicating failed tool executions. */
const TOOL_FINISH_REASONS: ReadonlySet<string> = new Set([
    'MALFORMED_FUNCTION_CALL',
    'UNEXPECTED_TOOL_CALL',
    'TOO_MANY_TOOL_CALLS',
    'MISSING_THOUGHT_SIGNATURE',
]);

/** Matches markdown code fence enclosing JSON response text. */
const MATCH_CODE_FENCE = /^```(?:json)?\s*\n?([\s\S]*?)\n?```$/i;

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * **Parses API response into `GenaiOutput` or `GenaiError`**.
 *
 * - extracts generated text and reasoning thoughts
 * - parses grounding metadata into citations
 * - extracts function call requests into `toolCalls`
 * - validates finish reasons; flags prompt blocks, truncation, and tool errors
 * - parses JSON text when `json` requested
 */
export function readAnswer(
    response: GenerateContentResponse,
    requestedModel: string,
    json: boolean,
    ms: number,
): Result<GenaiOutput, GenaiError> {
    const candidate = response.candidates?.[0];
    const text = (candidate?.content?.parts ?? [])
        .filter((part) => typeof part.text === 'string' && part.thought !== true)
        .map((part) => part.text)
        .join('');
    const reasoning = (candidate?.content?.parts ?? [])
        .filter((part) => typeof part.text === 'string' && part.thought === true)
        .map((part) => part.text)
        .join('');
    const sources = (candidate?.groundingMetadata?.groundingChunks ?? []).flatMap((chunk) =>
        chunk.web?.uri ? [{ url: chunk.web.uri, ...(chunk.web.title ? { title: chunk.web.title } : {}) }] : [],
    );
    const toolCalls = readToolCalls(candidate?.content?.parts ?? []);
    const finishReason = candidate?.finishReason ?? null;
    const output: GenaiOutput = {
        text,
        ...(reasoning === '' ? {} : { reasoning }),
        ...(sources.length === 0 ? {} : { sources }),
        ...(toolCalls.length === 0 ? {} : { toolCalls }),
        model: response.modelVersion || requestedModel,
        usage: readUsage(response.usageMetadata),
        finishReason,
        ms,
    };
    const answered = { text, model: output.model, usage: output.usage, ms, output };
    const blockReason = response.promptFeedback?.blockReason;
    if (blockReason !== undefined) {
        return {
            ok: false,
            error: { code: 'blocked', message: `Prompt blocked: ${blockReason}`, retryable: false, ...answered },
        };
    }
    if (candidate === undefined) {
        const message = 'No candidate in the response';
        return { ok: false, error: { code: 'unexpected', message, retryable: false, ...answered } };
    }
    const continuation = (response as { continuationToken?: unknown }).continuationToken;
    if (finishReason === 'MAX_TOKENS' || finishReason === 'CONTINUATION' || continuation) {
        const message = `Answer truncated, finish reason ${finishReason ?? 'none'}`;
        return { ok: false, error: { code: 'truncated', message, retryable: false, ...answered } };
    }
    if (finishReason !== null && TOOL_FINISH_REASONS.has(finishReason)) {
        const message = `Tool call failed, finish reason ${finishReason}`;
        const retryable = finishReason === 'MALFORMED_FUNCTION_CALL';
        return { ok: false, error: { code: 'unexpected', message, retryable, ...answered } };
    }
    if (finishReason !== null && finishReason !== 'STOP') {
        const message = `Answer stopped, finish reason ${finishReason}`;
        return { ok: false, error: { code: 'blocked', message, retryable: false, ...answered } };
    }
    if (!json || toolCalls.length > 0) {
        return { ok: true, data: output };
    }
    const trimmed = text.trim();
    try {
        return { ok: true, data: { ...output, json: JSON.parse(MATCH_CODE_FENCE.exec(trimmed)?.[1] ?? trimmed) } };
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return { ok: false, error: { code: 'invalid_json', message, retryable: false, ...answered } };
    }
}

/**
 * **Maps thrown SDK error or exception to `GenaiError`**.
 *
 * - maps abort signals to `aborted`
 * - maps API and auth status codes via `statusError`
 * - detects network and socket disconnects as `network`
 * - maps remaining exceptions to `unexpected`
 */
export function toGenaiError(error: unknown, signal: AbortSignal | undefined): GenaiError {
    const message = error instanceof Error ? error.message : String(error);
    if (signal?.aborted) {
        return { code: 'aborted', message, retryable: false };
    }
    const status = error instanceof ApiError ? error.status : authStatus(error);
    if (status !== undefined) {
        return statusError(status, message, status === 429 ? readRetryDelayMs(message) : undefined);
    }
    if (error instanceof Error && (error.name === 'TimeoutError' || error.name === 'AbortError')) {
        return { code: 'timeout', message, retryable: true };
    }
    const code = error instanceof Error && 'code' in error ? error.code : undefined;
    if (typeof code === 'string' && FILE_ERROR_CODES.has(code)) {
        return { code: 'unexpected', message, retryable: false };
    }
    if (isNetworkError(error)) {
        return { code: 'network', message, retryable: true };
    }
    return { code: 'unexpected', message, retryable: false };
}

/** Maps HTTP status code and message to `GenaiError`. */
export function statusError(status: number, message: string, retryAfterMs?: number): GenaiError {
    if (status === 429) {
        return {
            code: 'rate_limited',
            message,
            retryable: true,
            status,
            ...(retryAfterMs === undefined ? {} : { retryAfterMs }),
        };
    }
    return { code: 'http', message, retryable: status >= 500 || RETRYABLE_STATUSES.has(status), status };
}

/** Maps Google RPC error code to `GenaiError` with mapped HTTP status. */
export function rpcError(code: number | undefined, message: string): GenaiError {
    return statusError(code === undefined ? 500 : (RPC_HTTP_STATUS[code] ?? 500), message);
}

/** Maps batch job state string to high-level lifecycle phase (`running`, `succeeded`, or `failed`). */
export function batchPhase(state: string): GenaiBatchStatus['phase'] {
    if (SUCCEEDED_STATES.has(state)) {
        return 'succeeded';
    }
    return FAILED_STATES.has(state) ? 'failed' : 'running';
}

/** Wraps async function to ensure it never throws; maps rejections to `GenaiError`. */
export function guard<A extends unknown[], T>(
    call: (...args: A) => Promise<Result<T, GenaiError>>,
): (...args: A) => Promise<Result<T, GenaiError>> {
    return async (...args) => {
        try {
            return await call(...args);
        } catch (error) {
            return { ok: false, error: thrownError(error) };
        }
    };
}

/** Wraps synchronous function to ensure it never throws; maps thrown exceptions to `GenaiError`. */
export function guardSync<A extends unknown[], T>(
    call: (...args: A) => Result<T, GenaiError>,
): (...args: A) => Result<T, GenaiError> {
    return (...args) => {
        try {
            return call(...args);
        } catch (error) {
            return { ok: false, error: thrownError(error) };
        }
    };
}

/** Wraps all methods of `GenaiBatch` to guarantee they never throw or reject. */
export function guardBatch<
    B extends Omit<GenaiBatch, 'maxBytes' | 'removeFiles'> & { removeFiles?: GenaiBatch['removeFiles'] },
>(batch: B): B {
    return {
        ...batch,
        requestBytes: guardSync(batch.requestBytes),
        submit: guard(batch.submit),
        check: guard(batch.check),
        cancel: guard(batch.cancel),
        ...(batch.removeFiles === undefined ? {} : { removeFiles: guard(batch.removeFiles) }),
    };
}

/** Calculates elapsed milliseconds since start timestamp. */
export function elapsedMs(startedAt: number): number {
    return Math.round(performance.now() - startedAt);
}

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/** Checks whether error represents connection or network socket failure. */
function isNetworkError(error: unknown): boolean {
    if (!(error instanceof Error)) {
        return false;
    }
    const code = 'code' in error ? error.code : undefined;
    const causeCode =
        typeof error.cause === 'object' && error.cause !== null && 'code' in error.cause ? error.cause.code : undefined;
    if (error instanceof TypeError && error.message === 'fetch failed' && typeof causeCode === 'string') {
        return true;
    }
    return [code, causeCode].some((value) => typeof value === 'string' && MATCH_NETWORK_CODE.test(value));
}

/** Maps thrown exception inside guarded call to `invalid_request` or `unexpected`. */
function thrownError(error: unknown): GenaiError {
    const message = error instanceof Error ? error.message : String(error);
    const code = error instanceof InvalidRequestError ? 'invalid_request' : 'unexpected';
    return { code, message, retryable: false };
}

/** Extracts HTTP response status from google-auth-library error if present. */
function authStatus(error: unknown): number | undefined {
    const status = (error as { response?: { status?: unknown } } | null)?.response?.status;
    return error instanceof Error && typeof status === 'number' ? status : undefined;
}

/** Converts response parts into OpenAI-compatible tool calls with thought signatures. */
function readToolCalls(parts: readonly Part[]): GenaiToolCall[] {
    return parts.flatMap((part) => {
        const call = part.functionCall;
        if (call?.name === undefined) {
            return [];
        }
        const toolCall: GenaiToolCall = {
            id: call.id || `call_${randomUUID().replaceAll('-', '')}`,
            type: 'function',
            function: { name: call.name, arguments: JSON.stringify(call.args ?? {}) },
        };
        if (part.thoughtSignature !== undefined) {
            // biome-ignore lint/style/useNamingConvention: OpenAI-compatible field names
            toolCall.extra_content = { google: { thought_signature: part.thoughtSignature } };
        }
        return [toolCall];
    });
}

/** Extracts token usage counts from response metadata. */
function readUsage(usage: GenerateContentResponseUsageMetadata | undefined): GenaiUsage {
    return {
        inputTokens: usage?.promptTokenCount ?? null,
        outputTokens: usage?.candidatesTokenCount ?? null,
        reasoningTokens: usage?.thoughtsTokenCount ?? null,
    };
}

/** Extracts recommended retry delay in milliseconds from RPC RetryInfo detail. */
function readRetryDelayMs(message: string): number | undefined {
    let body: unknown;
    try {
        body = JSON.parse(message);
    } catch {
        return undefined;
    }
    const details = (body as { error?: { details?: unknown } }).error?.details;
    for (const detail of Array.isArray(details) ? details : []) {
        const { '@type': type, retryDelay } = detail as { '@type'?: unknown; retryDelay?: unknown };
        const match =
            type === RETRY_INFO_TYPE && typeof retryDelay === 'string' ? MATCH_DURATION.exec(retryDelay) : null;
        if (match?.[1] !== undefined) {
            return Math.round(Number(match[1]) * 1000);
        }
    }
    return undefined;
}
