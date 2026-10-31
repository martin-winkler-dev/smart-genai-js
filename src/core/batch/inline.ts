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
import { batchPhase, guardBatch, readAnswer, rpcError, toGenaiError } from '../answer.js';
import { batchModel, checkBatchName, checkOptionKeys, toGeminiCall, wantsJson } from '../request.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Maximum payload byte budget for inline batch creation. */
const INLINE_BATCH_MAX_BYTES = 19_000_000;

/** Metadata format flag indicating JSON parsing requested. */
const JSON_FORMAT = 'json';

/** Supported option keys for batch submission. */
const SUBMIT_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'displayName', 'config']);

/** Supported option keys for batch status checks and cancellation. */
const CHECK_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Matches models/ prefix on model identifiers. */
const MATCH_MODEL_PREFIX = /^models\//;

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/**
 * **Batch API implementation for Gemini Developer API**: uses inline requests and responses.
 *
 * - attaches `metadata.key` and format tracking to requests
 * - polls batch job status and parses inline responses
 * - maps individual item errors using RPC status codes
 */
export function createInlineBatch(ai: GoogleGenAI): GenaiBatch {
    return guardBatch({
        maxBytes: INLINE_BATCH_MAX_BYTES,
        requestBytes(request, config) {
            const data = new TextEncoder().encode(JSON.stringify(inlinedRequest('', request, config))).length;
            return { ok: true, data };
        },
        async submit(items, options = {}) {
            checkOptionKeys(options, SUBMIT_OPTION_KEYS);
            const model = batchModel(items);
            const src = items.map((item) => inlinedRequest(item.key, item.request, options.config));
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
                return readBatch({ ...job, name: job.name ?? name }, job.model ?? '');
            } catch (error) {
                return { ok: false, error: toGenaiError(error, signal) };
            }
        },
        async cancel(name, options = {}) {
            checkBatchName(name);
            checkOptionKeys(options, CHECK_OPTION_KEYS);
            const { signal } = options;
            try {
                await ai.batches.cancel({ name, ...(signal ? { config: { abortSignal: signal } } : {}) });
            } catch (error) {
                return { ok: false, error: toGenaiError(error, signal) };
            }
            return { ok: true, data: undefined };
        },
        async removeFiles(options = {}) {
            checkOptionKeys(options, CHECK_OPTION_KEYS);
            const message = 'Developer API batches use no bucket';
            return { ok: false, error: { code: 'unsupported', message, retryable: false } };
        },
    });
}

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

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
        return { ok: false, error: { code: 'unexpected', message, retryable: false } };
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
