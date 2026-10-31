// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenerateContentConfig } from '@google/genai';
import type { GenaiBatch, GenaiBatchStatus, GenaiError, GenaiOutput, GenaiRequest, Result } from '../../types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Result outcome of batch generation call. */
type CallResult = Result<GenaiOutput, GenaiError>;

/** In-flight batch call awaiting settlement. */
type Call = {
    key: string;
    request: GenaiRequest;
    bytes: number;
    settled: boolean;
    settle(result: CallResult): void;
    leave?: () => void;
};

/** Batch calls collected within aggregation window. */
type Group = { calls: Call[]; timer: ReturnType<typeof setTimeout> };

/** Submitted batch job tracking active calls and polling state. */
type Job = {
    calls: Map<string, Call>;
    name?: string;
    done: boolean;
    submittedAt?: number;
    timer?: ReturnType<typeof setTimeout>;
};

/** Batch call manager interface. */
export type BatchCalls = {
    /** Adds request to batch queue and returns promise resolving to individual result. */
    add(request: GenaiRequest, config: GenerateContentConfig | undefined, signal?: AbortSignal): Promise<CallResult>;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/**
 * Timing configuration for batch call polling and aggregation windows.
 * @internal - exported for testing only
 */
export const batchTiming = { pollMs: 30_000, maxPollMs: 72 * 3_600_000, msPerSecond: 1000 };

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Manages batch request collection and polling lifecycle**.
 *
 * - groups calls by model and config
 * - flushes after collection window closes or when chunk reaches `maxBytes`
 * - polls batch job status until completion; gives up (and cancels) after `maxPollMs`
 * - settles individual call promises upon job completion or failure
 */
export function createBatchCalls(batch: GenaiBatch, waitSec: number): BatchCalls {
    const groups = new Map<string, Group>();
    let nextKey = 1;

    /** Submits the group of `id` (several jobs when over `maxBytes`). */
    function flush(id: string, config: GenerateContentConfig | undefined): void {
        const group = groups.get(id);
        groups.delete(id);
        let chunk: Call[] = [];
        let bytes = 0;
        for (const call of group?.calls ?? []) {
            if (chunk.length > 0 && bytes + call.bytes > batch.maxBytes) {
                void submitJob(chunk, config);
                chunk = [];
                bytes = 0;
            }
            chunk.push(call);
            bytes += call.bytes;
        }
        if (chunk.length > 0) {
            void submitJob(chunk, config);
        }
    }

    /** Submits one job and polls it; every outcome settles its calls. */
    async function submitJob(calls: Call[], config: GenerateContentConfig | undefined): Promise<void> {
        const job: Job = { calls: new Map(calls.map((call) => [call.key, call])), done: false };
        for (const call of calls) {
            call.leave = () => onCallLeft(job);
        }
        try {
            const items = calls.map((call) => ({ key: call.key, request: call.request }));
            const submitted = await batch.submit(items, config === undefined ? {} : { config });
            if (!submitted.ok) {
                finish(job, () => submitted);
                return;
            }
            job.name = submitted.data.name;
            job.submittedAt = Date.now();
            if (job.done) {
                await batch.cancel(job.name);
                return;
            }
            schedulePoll(job);
        } catch (error) {
            finish(job, () => unexpected(error));
        }
    }

    /**
     * **Polls the job after `batchTiming.pollMs`** (or a longer `retryAfterMs`).
     *
     * - last poll at `maxPollMs` after the submit, then cancels the job
     */
    function schedulePoll(job: Job, retryAfterMs = 0): void {
        const leftMs = batchTiming.maxPollMs - (Date.now() - (job.submittedAt ?? 0));
        if (leftMs <= 0) {
            const hours = Math.round(batchTiming.maxPollMs / 3_600_000);
            const message = `Batch job ${job.name} not final after ${hours} h; cancelled`;
            finish(job, () => ({ ok: false, error: { code: 'timeout', message, retryable: false } }));
            void batch.cancel(job.name as string).catch(() => undefined);
            return;
        }
        job.timer = setTimeout(
            () => {
                void poll(job);
            },
            Math.min(Math.max(batchTiming.pollMs, retryAfterMs), leftMs),
        );
    }

    /** One poll; settles the calls of a final job. */
    async function poll(job: Job): Promise<void> {
        if (job.done) {
            return;
        }
        try {
            const checked = await batch.check(job.name as string);
            if (job.done) {
                return;
            }
            if (!checked.ok) {
                if (checked.error.retryable) {
                    schedulePoll(job, checked.error.retryAfterMs);
                } else {
                    finish(job, () => checked);
                }
                return;
            }
            readStatus(job, checked.data);
        } catch (error) {
            finish(job, () => unexpected(error));
        }
    }

    /** Settles the calls of a final job, or polls again while it runs. */
    function readStatus(job: Job, status: GenaiBatchStatus): void {
        if (status.phase === 'running') {
            schedulePoll(job);
            return;
        }
        if (status.phase === 'failed') {
            const message = `Batch job ${status.name} ${status.state}${status.error ? `: ${status.error}` : ''}`;
            finish(job, () => ({ ok: false, error: { code: 'unexpected', message, retryable: false } }));
            return;
        }
        const results = new Map((status.results ?? []).map((entry) => [entry.key, entry.result]));
        finish(
            job,
            (call) =>
                results.get(call.key) ?? {
                    ok: false,
                    error: {
                        code: 'unexpected',
                        message: `No result for this call in batch job ${status.name}`,
                        retryable: false,
                    },
                },
        );
    }

    /** Settles every open call of the job and ends its polling. */
    function finish(job: Job, resultOf: (call: Call) => CallResult): void {
        job.done = true;
        clearTimeout(job.timer);
        for (const call of job.calls.values()) {
            call.settle(resultOf(call));
        }
    }

    /** After an abort: a job whose calls are all settled stops polling and is cancelled. */
    function onCallLeft(job: Job): void {
        if (job.done || [...job.calls.values()].some((call) => !call.settled)) {
            return;
        }
        job.done = true;
        clearTimeout(job.timer);
        if (job.name !== undefined) {
            void batch.cancel(job.name);
        }
    }

    return {
        add(request, config, signal) {
            if (signal?.aborted) {
                return Promise.resolve(aborted(0));
            }
            const bytes = batch.requestBytes(request, config);
            if (!bytes.ok) {
                return Promise.resolve(bytes);
            }
            if (bytes.data > batch.maxBytes) {
                const message = `Request of ${bytes.data} bytes exceeds the batch limit of ${batch.maxBytes} bytes`;
                return Promise.resolve({ ok: false, error: { code: 'invalid_request', message, retryable: false } });
            }
            return new Promise<CallResult>((resolve) => {
                const call: Call = {
                    key: `c${nextKey++}`,
                    request,
                    bytes: bytes.data,
                    settled: false,
                    settle(result) {
                        if (call.settled) {
                            return;
                        }
                        call.settled = true;
                        signal?.removeEventListener('abort', onAbort);
                        resolve(result);
                    },
                };
                const id = JSON.stringify([request.model, stableJson(config)]);
                let group = groups.get(id);
                if (group === undefined) {
                    const timer = setTimeout(() => flush(id, config), waitSec * batchTiming.msPerSecond);
                    group = { calls: [], timer };
                    groups.set(id, group);
                }
                const joined = group;
                joined.calls.push(call);
                call.leave = () => {
                    joined.calls.splice(joined.calls.indexOf(call), 1);
                    if (joined.calls.length === 0 && groups.get(id) === joined) {
                        clearTimeout(joined.timer);
                        groups.delete(id);
                    }
                };
                function onAbort(): void {
                    // a call still in its group was never submitted
                    call.settle(groups.get(id) === joined ? aborted(0) : aborted());
                    call.leave?.();
                }
                signal?.addEventListener('abort', onAbort, { once: true });
            });
        },
    };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Returns aborted GenaiError result; `attempts: 0` for call never submitted. */
function aborted(attempts?: number): CallResult {
    const error: GenaiError = { code: 'aborted', message: 'Batch call aborted', retryable: false };
    return { ok: false, error: attempts === undefined ? error : { ...error, attempts } };
}

/** Returns unexpected GenaiError result wrapping caught error. */
function unexpected(error: unknown): CallResult {
    const message = error instanceof Error ? error.message : String(error);
    return { ok: false, error: { code: 'unexpected', message, retryable: false } };
}

/** Serializes object with sorted keys to ensure deterministic representation. */
function stableJson(value: unknown): unknown {
    if (Array.isArray(value)) {
        return value.map(stableJson);
    }
    if (typeof value !== 'object' || value === null) {
        return value;
    }
    return Object.fromEntries(
        Object.keys(value)
            .sort()
            .map((key) => [key, stableJson((value as Record<string, unknown>)[key])]),
    );
}
