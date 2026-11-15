// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { BatchJob, GoogleGenAI, InlinedRequest, InlinedResponse } from '@google/genai';
import type {
    GenaiBatch,
    GenaiBatchResult,
    GenaiBatchStatus,
    GenaiError,
    GenaiOutput,
    GenaiRequest,
    Result,
} from '../../types.js';
import { batchPhase, guardBatch, listBatchJobs, readAnswer, rpcError, toGenaiError } from '../answer.js';
import { batchModel, checkBatchBytes, checkBatchName, checkOptionKeys, toGeminiCall, wantsJson } from '../request.js';
import { finalJobError } from './final.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **REST access for batch operations**: mirrors SDK settings of Developer API client.
 *
 * - `baseUrl`, `apiVersion`: same as SDK client, URL `<baseUrl>/<apiVersion>/batches/<id>`
 * - `timeoutMs`: request timeout, same as SDK client
 */
export type InlineBatchRest = { apiKey: string; baseUrl: string; apiVersion: string; timeoutMs: number };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/**
 * Maximum payload byte budget for inline batch creation; read when batch is created.
 * @internal - exported for testing only
 */
export const inlineBatchLimit = { maxBytes: 19_000_000 };

/** Metadata format flag indicating JSON parsing requested. */
const JSON_FORMAT = 'json';

/** Supported option keys for batch submission. */
const SUBMIT_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'displayName', 'config']);

/** Supported option keys for batch status checks; `keepFiles` accepted for Vertex AI parity, no files here. */
const CHECK_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'keepFiles']);

/** Supported option keys for batch cancellation. */
const CANCEL_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Supported option keys for batch job listing. */
const LIST_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'displayName']);

/** Supported option keys for batch file removal. */
const REMOVE_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'keepSucceededMs']);

/** Matches `models/` prefix on model identifiers. */
const MATCH_MODEL_PREFIX = /^models\//;

/** Matches trailing slash of base URL. */
const MATCH_TRAILING_SLASH = /\/$/;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Batch API implementation for Gemini Developer API**: uses inline requests and responses.
 *
 * - attaches `metadata.key` and format tracking to requests
 * - polls batch job status and parses inline responses
 * - maps individual item errors using RPC status codes
 * - failed job: reads failure reason via REST, as SDK drops operation `error`
 */
export function createInlineBatch(ai: GoogleGenAI, rest: InlineBatchRest): GenaiBatch {
    const { maxBytes } = inlineBatchLimit;
    return guardBatch({
        maxBytes,
        requestBytes(request, config) {
            return { ok: true, data: inlinedBytes(inlinedRequest('', request, config)) };
        },
        async submit(items, options = {}) {
            checkOptionKeys(options, SUBMIT_OPTION_KEYS);
            const model = batchModel(items);
            const src = items.map((item) => inlinedRequest(item.key, item.request, options.config));
            checkBatchBytes(src.map(inlinedBytes), maxBytes);
            try {
                const job = await ai.batches.create({
                    model,
                    src,
                    config: {
                        displayName: options.displayName ?? `smart-genai ${items.length} requests`,
                        ...(options.signal ? { abortSignal: options.signal } : {}),
                    },
                });
                if (!job.name) {
                    return {
                        ok: false,
                        error: { code: 'unexpected', message: 'Batch without name', retryable: false },
                    };
                }
                return readBatch(job, model);
            } catch (error) {
                return { ok: false, error: toGenaiError(error, options.signal) };
            }
        },
        async check(name, options = {}) {
            checkBatchName(name);
            checkOptionKeys(options, CHECK_OPTION_KEYS);
            const { signal } = options;
            try {
                const job = await ai.batches.get({ name, ...(signal ? { config: { abortSignal: signal } } : {}) });
                const status = readBatch({ ...job, name: job.name ?? name }, job.model ?? '');
                if (!status.ok || status.data.phase !== 'failed' || status.data.error !== undefined) {
                    return status;
                }
                const error = await failureReason(rest, status.data.name, signal);
                return error === undefined ? status : { ok: true, data: { ...status.data, error } };
            } catch (error) {
                return { ok: false, error: toGenaiError(error, signal) };
            }
        },
        async cancel(name, options = {}) {
            checkBatchName(name);
            checkOptionKeys(options, CANCEL_OPTION_KEYS);
            const { signal } = options;
            const config = signal ? { config: { abortSignal: signal } } : {};
            try {
                await ai.batches.cancel({ name, ...config });
            } catch (error) {
                if (!(await isSettled(ai, name, config))) {
                    return { ok: false, error: toGenaiError(error, signal) };
                }
            }
            return { ok: true, data: undefined };
        },
        async list(options = {}) {
            checkOptionKeys(options, LIST_OPTION_KEYS);
            return listBatchJobs(ai, options);
        },
        async removeFiles(options = {}) {
            checkOptionKeys(options, REMOVE_OPTION_KEYS);
            const message = 'Developer API batches use no bucket';
            return { ok: false, error: { code: 'unsupported', message, retryable: false } };
        },
    });
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * **Reads failure reason of batch operation via REST**: top-level `error.message`.
 *
 * - SDK `batches.get` drops operation `error` on Developer API
 * - same URL as SDK: `<baseUrl>/<apiVersion>/batches/<id>`
 * - any failure (network, status, body): `undefined`
 */
async function failureReason(
    rest: InlineBatchRest,
    name: string,
    signal: AbortSignal | undefined,
): Promise<string | undefined> {
    const id = name.split('/').pop() ?? '';
    const url = `${rest.baseUrl.replace(MATCH_TRAILING_SLASH, '')}/${rest.apiVersion}/batches/${encodeURIComponent(id)}`;
    const timeout = AbortSignal.timeout(rest.timeoutMs);
    try {
        const response = await fetch(url, {
            method: 'GET',
            headers: { 'x-goog-api-key': rest.apiKey },
            signal: signal === undefined ? timeout : AbortSignal.any([signal, timeout]),
        });
        if (!response.ok) {
            return undefined;
        }
        const body: unknown = await response.json();
        const message = (body as { error?: { message?: unknown } } | null)?.error?.message;
        return typeof message === 'string' && message !== '' ? message : undefined;
    } catch {
        return undefined;
    }
}

/**
 * **Re-reads job after failed cancel**: `true` when cancel is no longer needed.
 *
 * - final job: API rejects cancel, nothing left to do
 * - `JOB_STATE_CANCELLING`: cancel already under way, counts as success
 * - re-read fails or job still running otherwise: `false`
 */
async function isSettled(
    ai: GoogleGenAI,
    name: string,
    config: { config?: { abortSignal: AbortSignal } },
): Promise<boolean> {
    let job: BatchJob;
    try {
        job = await ai.batches.get({ name, ...config });
    } catch {
        return false;
    }
    const state = job.state ?? 'JOB_STATE_UNSPECIFIED';
    return batchPhase(state) !== 'running' || state === 'JOB_STATE_CANCELLING';
}

/** Byte size of inlined request serialized with empty key; measure of `requestBytes` and `submit`. */
function inlinedBytes(inlined: InlinedRequest): number {
    const metadata = { ...inlined.metadata, key: '' };
    return new TextEncoder().encode(JSON.stringify({ ...inlined, metadata })).length;
}

/** Formats request into InlinedRequest with metadata key and format. */
function inlinedRequest(key: string, request: GenaiRequest, config: InlinedRequest['config']): InlinedRequest {
    const call = toGeminiCall(request, config);
    return {
        contents: call.contents,
        config: call.config,
        metadata: { key, ...(wantsJson(request) ? { format: JSON_FORMAT } : {}) },
    };
}

/** Converts BatchJob status into GenaiBatchStatus result. */
function readBatch(job: BatchJob, model: string): Result<GenaiBatchStatus, GenaiError> {
    const state = job.state ?? 'JOB_STATE_UNSPECIFIED';
    const name = job.name ?? '';
    const phase = batchPhase(state);
    if (phase === 'failed') {
        const error = job.error?.message;
        return { ok: true, data: { name, state, phase, ...(error ? { error } : {}) } };
    }
    if (phase === 'running') {
        return { ok: true, data: { name, state, phase } };
    }
    if (job.dest?.inlinedResponses === undefined && job.dest?.fileName !== undefined) {
        const message = `Batch results are in the file ${job.dest.fileName}, not inline`;
        return { ok: false, error: finalJobError({ code: 'unexpected', message, retryable: false }) };
    }
    const results: GenaiBatchResult[] = [];
    for (const response of job.dest?.inlinedResponses ?? []) {
        const key = response.metadata?.key;
        if (key !== undefined) {
            results.push({ key, result: readItem(response, model.replace(MATCH_MODEL_PREFIX, '')) });
        }
    }
    return { ok: true, data: { name, state, phase, results } };
}

/** Converts InlinedResponse into GenaiOutput or GenaiError. */
function readItem(response: InlinedResponse, model: string): Result<GenaiOutput, GenaiError> {
    if (response.error !== undefined) {
        return { ok: false, error: rpcError(response.error.code, response.error.message ?? 'Batch request failed') };
    }
    if (response.response === undefined) {
        return { ok: false, error: { code: 'unexpected', message: 'No response in the batch', retryable: false } };
    }
    return readAnswer(response.response, model, response.metadata?.format === JSON_FORMAT, 0);
}
