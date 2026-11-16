// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import {
    type Bucket,
    createQueue,
    type JobError,
    type JobErrorType,
    type JobOutcome,
    type JobResult,
    type LimiterError,
    type PushOptions,
} from 'smart-limit';
import { guard } from '../core/answer.js';
import type { GenaiError, GenaiLimiterOptions, Result } from '../types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Base of queued job: attempt count and last error for limiter releases. */
export type QueueJob = { attempts: number; lastError?: GenaiError };

/**
 * **Options of `createRunQueue`**: limits plus job-specific call and accounting.
 *
 * - `buckets`: omitted for no rate limits
 * - `execute`: one call; throws mapped to `GenaiError`
 * - `usedTokens`: tokens to settle; omitted for no settlement and no refund
 * - `errorType`: defaults to `jobErrorType`
 * - `countAttempts`: calls one attempt made; defaults to 1
 */
export type RunQueueOptions<TJob extends QueueJob, TOut> = {
    buckets?: Bucket[];
    concurrency: number;
    limiter: GenaiLimiterOptions;
    execute(job: TJob, signal: AbortSignal): Promise<Result<TOut, GenaiError>>;
    usedTokens?(result: Result<TOut, GenaiError>): number | undefined;
    errorType?(error: GenaiError): JobErrorType;
    countAttempts?(result: Result<TOut, GenaiError>): number;
};

/** Smart-limit queue that resolves pushed jobs to `Result`. */
export type RunQueue<TJob extends QueueJob, TOut> = {
    run(job: TJob, push: PushOptions): Promise<Result<TOut, GenaiError>>;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Minimum HTTP status code indicating server error. */
const MIN_SERVER_ERROR_STATUS = 500;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/** Creates smart-limit queue running jobs with retries, token settlement and `Result` mapping. */
export function createRunQueue<TJob extends QueueJob, TOut>(
    options: RunQueueOptions<TJob, TOut>,
): RunQueue<TJob, TOut> {
    const queue = createQueue<TJob, TOut>({
        ...options.limiter,
        concurrency: options.concurrency,
        maxBufferSize: Number.POSITIVE_INFINITY,
        ...(options.buckets === undefined ? {} : { buckets: options.buckets }),
        processJob: (job, context) => attempt(options, job, context.signal),
    });
    return {
        async run(job, push) {
            return toResult(await queue.push(job, push), job);
        },
    };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Maps GenaiError to smart-limit JobErrorType for retry and backoff routing. */
function jobErrorType(error: GenaiError): JobErrorType {
    if (!error.retryable) {
        return 'fatal';
    }
    const isServerError = error.code === 'http' && (error.status ?? 0) >= MIN_SERVER_ERROR_STATUS;
    return error.code === 'rate_limited' || error.code === 'network' || isServerError
        ? 'backoff_global'
        : 'retry_backoff';
}

/** Executes single attempt within limiter queue context. */
async function attempt<TJob extends QueueJob, TOut>(
    options: RunQueueOptions<TJob, TOut>,
    job: TJob,
    signal: AbortSignal,
): Promise<JobResult<TOut>> {
    const execute = guard(options.execute);
    // Counted before the call: an abort resolves the push before execute returns.
    job.attempts += 1;
    const result = await execute(job, signal);
    job.attempts += (options.countAttempts?.(result) ?? 1) - 1;
    const tokens = options.usedTokens?.(result);
    if (result.ok) {
        return tokens === undefined ? result : { ...result, usage: { tokens } };
    }
    const { error } = result;
    job.lastError = error;
    const type = (options.errorType ?? jobErrorType)(error);
    const jobError: JobError = { type, message: error.message, code: error.code, cause: error };
    if (error.retryAfterMs !== undefined) {
        jobError.retryAfterMs = error.retryAfterMs;
    }
    if (tokens !== undefined) {
        jobError.usage = { tokens };
    } else if (error.code === 'rate_limited' && options.usedTokens !== undefined) {
        // a rejected request ran no model: give its token booking back
        jobError.usage = { tokens: 0 };
    }
    return { ok: false, error: jobError };
}

/** Translates smart-limit job outcome into `Result`. */
function toResult<TOut>(outcome: JobOutcome<TOut>, job: QueueJob): Result<TOut, GenaiError> {
    if (outcome.ok) {
        return { ok: true, data: outcome.data };
    }
    if (outcome.error.code === 'ABORTED') {
        return { ok: false, error: { code: 'aborted', message: 'Aborted', retryable: false } };
    }
    const error = fromLimiterError(outcome.error);
    const isRelease = error.limiterCode === 'MAX_WAIT_EXCEEDED' || error.limiterCode === 'STOPPED';
    return { ok: false, error: error.code === 'limiter' && isRelease ? (job.lastError ?? error) : error };
}

/** Converts LimiterError into GenaiError. */
function fromLimiterError(error: LimiterError): GenaiError {
    const cause = error.cause as { cause?: unknown } | undefined;
    const last = cause?.cause as GenaiError | undefined;
    if (last !== undefined && typeof last === 'object' && 'code' in last && 'retryable' in last) {
        return last;
    }
    if (error.code === 'TIMEOUT') {
        return { code: 'timeout', message: error.message, retryable: true };
    }
    return { code: 'limiter', message: error.message, retryable: false, limiterCode: error.code };
}
