// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, it } from 'bun:test';
import { batchTiming, createBatchCalls } from '../../../src/core/batch/calls.js';
import { createGenaiClient } from '../../../src/core/client.js';
import type {
    GenaiBatch,
    GenaiBatchItem,
    GenaiBatchStatus,
    GenaiError,
    GenaiOutput,
    GenaiRequest,
    Result,
} from '../../../src/types.js';
import { apiError, batchResponse, generateBody, startFakeGemini } from '../fake-gemini.js';
import { fakeAuthClient } from '../fake-vertex.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Fake batch API recording submits, checks, and cancels. */
type FakeBatch = GenaiBatch & {
    submits: { name: string; items: GenaiBatchItem[]; config: unknown }[];
    checks: string[];
    cancels: string[];
    submitReply?: Result<GenaiBatchStatus, GenaiError>;
    status(name: string, items: GenaiBatchItem[], round: number): Result<GenaiBatchStatus, GenaiError>;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Original batch timing configuration restored after tests. */
const timing = { ...batchTiming };

beforeAll(() => {
    batchTiming.pollMs = 5;
    batchTiming.msPerSecond = 20;
});
afterAll(() => {
    Object.assign(batchTiming, timing);
});

// -------------------------------------------------------------------
// --- 3. Private Functions
// -------------------------------------------------------------------

/** Creates output echoing last user message of request. */
function echo(request: GenaiRequest): GenaiOutput {
    const last = request.messages.at(-1);
    const text = typeof last?.content === 'string' ? last.content : '';
    return {
        text,
        model: request.model ?? '',
        usage: { inputTokens: 1, outputTokens: 1, reasoningTokens: null },
        finishReason: 'STOP',
        ms: 0,
    };
}

/** Creates fake batch returning echo responses after one poll round. */
function fakeBatch(maxBytes = 1_000_000): FakeBatch {
    const rounds = new Map<string, number>();
    const batch: FakeBatch = {
        maxBytes,
        submits: [],
        checks: [],
        cancels: [],
        requestBytes: (request, config) => ({
            ok: true,
            data: JSON.stringify(request).length + (config === undefined ? 0 : JSON.stringify(config).length),
        }),
        async submit(items, options = {}) {
            if (batch.submitReply !== undefined) {
                return batch.submitReply;
            }
            const name = `batches/${batch.submits.length + 1}`;
            batch.submits.push({ name, items: [...items], config: options.config });
            return { ok: true, data: { name, state: 'JOB_STATE_PENDING', phase: 'running' } };
        },
        async check(name) {
            batch.checks.push(name);
            const round = (rounds.get(name) ?? 0) + 1;
            rounds.set(name, round);
            const items = batch.submits.find((entry) => entry.name === name)?.items ?? [];
            return batch.status(name, items, round);
        },
        async cancel(name) {
            batch.cancels.push(name);
            return { ok: true, data: undefined };
        },
        async removeFiles() {
            return { ok: true, data: { removed: 0, kept: 0 } };
        },
        status(name, items, round) {
            if (round === 1) {
                return { ok: true, data: { name, state: 'JOB_STATE_RUNNING', phase: 'running' } };
            }
            const results = items.map((item) => ({
                key: item.key,
                result: { ok: true as const, data: echo(item.request) },
            }));
            return { ok: true, data: { name, state: 'JOB_STATE_SUCCEEDED', phase: 'succeeded', results } };
        },
    };
    return batch;
}

/** Creates request with user text for specified model. */
function ask(text: string, model = 'gemini-x'): GenaiRequest {
    return { model, messages: [{ role: 'user', content: text }] };
}

describe('batch calls', () => {
    it('collect calls of one model and config into one job and resolve each with its own result', async () => {
        const batch = fakeBatch();
        const calls = createBatchCalls(batch, 1);
        const results = await Promise.all([
            calls.add(ask('a'), undefined),
            calls.add(ask('b'), undefined),
            calls.add(ask('c'), { temperature: 0, stopSequences: ['x', 'y'] }),
            calls.add(ask('d', 'gemini-y'), undefined),
            calls.add(ask('e'), { stopSequences: ['x', 'y'], temperature: 0 }),
        ]);
        expect(results.map((result) => (result.ok ? result.data.text : result.error.code))).toEqual([
            'a',
            'b',
            'c',
            'd',
            'e',
        ]);
        expect(batch.submits.map((entry) => entry.items.map((item) => item.request.messages[0]?.content))).toEqual([
            ['a', 'b'],
            ['c', 'e'],
            ['d'],
        ]);
        expect(batch.submits.map((entry) => entry.config)).toEqual([
            undefined,
            { temperature: 0, stopSequences: ['x', 'y'] },
            undefined,
        ]);
        expect(new Set(batch.submits.flatMap((entry) => entry.items.map((item) => item.key))).size).toBe(5);
        expect(batch.checks).toHaveLength(6);
    });

    it('split a group into several jobs when the sizes exceed maxBytes', async () => {
        const size = JSON.stringify(ask('a')).length;
        const batch = fakeBatch(size * 2);
        const calls = createBatchCalls(batch, 1);
        const results = await Promise.all(['a', 'b', 'c', 'd', 'e'].map((text) => calls.add(ask(text), undefined)));
        expect(results.every((result) => result.ok)).toBe(true);
        expect(batch.submits.map((entry) => entry.items.length)).toEqual([2, 2, 1]);
    });

    it('answer invalid_request at once for a request larger than maxBytes', async () => {
        const size = JSON.stringify(ask('a')).length;
        const batch = fakeBatch(size - 1);
        const calls = createBatchCalls(batch, 1);
        expect(await calls.add(ask('a'), undefined)).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: `Request of ${size} bytes exceeds the batch limit of ${size - 1} bytes`,
                retryable: false,
            },
        });
        expect(batch.submits).toHaveLength(0);
        const badBytes = fakeBatch();
        badBytes.requestBytes = () => ({
            ok: false,
            error: { code: 'invalid_request', message: 'bad', retryable: false },
        });
        expect(await createBatchCalls(badBytes, 1).add(ask('a'), undefined)).toEqual({
            ok: false,
            error: { code: 'invalid_request', message: 'bad', retryable: false },
        });
    });

    it('measure each call with the group config, which every item of the job carries', async () => {
        const size = JSON.stringify(ask('a')).length;
        const batch = fakeBatch(size * 3);
        const calls = createBatchCalls(batch, 1);
        const config = { systemInstruction: 'x'.repeat(size) };
        const results = await Promise.all(['a', 'b', 'c'].map((text) => calls.add(ask(text), config)));
        expect(results.every((result) => result.ok)).toBe(true);
        expect(batch.submits.map((entry) => entry.items.length)).toEqual([1, 1, 1]);
    });

    it('answer unexpected for a key missing from a succeeded job', async () => {
        const batch = fakeBatch();
        batch.status = (name, items) => ({
            ok: true,
            data: {
                name,
                state: 'JOB_STATE_SUCCEEDED',
                phase: 'succeeded',
                results: items
                    .slice(1)
                    .map((item) => ({ key: item.key, result: { ok: true, data: echo(item.request) } })),
            },
        });
        const calls = createBatchCalls(batch, 1);
        const [missing, found] = await Promise.all([calls.add(ask('a'), undefined), calls.add(ask('b'), undefined)]);
        expect(missing).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'No result for this call in batch job batches/1', retryable: false },
        });
        expect(found).toMatchObject({ ok: true, data: { text: 'b' } });
    });

    it('give every call of a failed job, a failed submit or a fatal check the same error', async () => {
        const failed = fakeBatch();
        failed.status = (name) => ({
            ok: true,
            data: { name, state: 'JOB_STATE_EXPIRED', phase: 'failed', error: 'took too long' },
        });
        const failedCalls = createBatchCalls(failed, 1);
        const expired: GenaiError = {
            code: 'unexpected',
            message: 'Batch job batches/1 JOB_STATE_EXPIRED: took too long',
            retryable: false,
        };
        expect(await Promise.all([failedCalls.add(ask('a'), undefined), failedCalls.add(ask('b'), undefined)])).toEqual(
            [
                { ok: false, error: expired },
                { ok: false, error: expired },
            ],
        );

        const refused = fakeBatch();
        const limited: GenaiError = { code: 'rate_limited', message: 'quota', retryable: true, retryAfterMs: 7000 };
        refused.submitReply = { ok: false, error: limited };
        const refusedCalls = createBatchCalls(refused, 1);
        expect(
            await Promise.all([refusedCalls.add(ask('a'), undefined), refusedCalls.add(ask('b'), undefined)]),
        ).toEqual([
            { ok: false, error: limited },
            { ok: false, error: limited },
        ]);

        const flaky = fakeBatch();
        const outage: GenaiError = { code: 'http', message: 'fake 503', retryable: true, status: 503 };
        const gone: GenaiError = { code: 'http', message: 'fake 404', retryable: false, status: 404 };
        flaky.status = (_name, _items, round) => ({ ok: false, error: round < 3 ? outage : gone });
        const flakyCalls = createBatchCalls(flaky, 1);
        expect(await Promise.all([flakyCalls.add(ask('a'), undefined), flakyCalls.add(ask('b'), undefined)])).toEqual([
            { ok: false, error: gone },
            { ok: false, error: gone },
        ]);
        expect(flaky.checks).toHaveLength(3);
    });

    it('poll again after a longer retryAfterMs of a retryable check error', async () => {
        const batch = fakeBatch();
        const checkedAt: number[] = [];
        batch.status = (name, items, round) => {
            checkedAt.push(Date.now());
            if (round === 1) {
                const error = {
                    code: 'rate_limited',
                    message: 'slow down',
                    retryable: true,
                    retryAfterMs: 80,
                } as const;
                return { ok: false, error };
            }
            const results = items.map((item) => ({
                key: item.key,
                result: { ok: true as const, data: echo(item.request) },
            }));
            return { ok: true, data: { name, state: 'JOB_STATE_SUCCEEDED', phase: 'succeeded', results } };
        };
        const calls = createBatchCalls(batch, 1);
        expect(await calls.add(ask('a'), undefined)).toMatchObject({ ok: true, data: { text: 'a' } });
        expect((checkedAt[1] ?? 0) - (checkedAt[0] ?? 0)).toBeGreaterThanOrEqual(70);
    });

    it('cancel a job that is not final after maxPollMs and give its calls a timeout naming the job', async () => {
        const batch = fakeBatch();
        batch.status = (name) => ({ ok: true, data: { name, state: 'JOB_STATE_PAUSED', phase: 'running' } });
        const maxPollMs = batchTiming.maxPollMs;
        batchTiming.maxPollMs = 40;
        try {
            const calls = createBatchCalls(batch, 1);
            expect(await calls.add(ask('a'), undefined)).toEqual({
                ok: false,
                error: {
                    code: 'timeout',
                    message: 'Batch job batches/1 not final after 0 h; cancelled',
                    retryable: false,
                },
            });
            expect(batch.checks.length).toBeGreaterThan(1);
            expect(batch.cancels).toEqual(['batches/1']);
        } finally {
            batchTiming.maxPollMs = maxPollMs;
        }
    });

    it('poll a last time at maxPollMs when the next poll would come later', async () => {
        const batch = fakeBatch();
        const { pollMs, maxPollMs } = batchTiming;
        batchTiming.pollMs = 60;
        batchTiming.maxPollMs = 100;
        try {
            const calls = createBatchCalls(batch, 1);
            expect(await calls.add(ask('a'), undefined)).toMatchObject({ ok: true, data: { text: 'a' } });
            expect(batch.cancels).toEqual([]);
        } finally {
            Object.assign(batchTiming, { pollMs, maxPollMs });
        }
    });

    it('let an aborted call leave its group before the submit', async () => {
        const batch = fakeBatch();
        const calls = createBatchCalls(batch, 1);
        const early = new AbortController();
        early.abort();
        expect(await calls.add(ask('x'), undefined, early.signal)).toMatchObject({
            ok: false,
            error: { code: 'aborted', attempts: 0 },
        });
        const late = new AbortController();
        const leaving = calls.add(ask('a'), undefined, late.signal);
        const staying = calls.add(ask('b'), undefined);
        late.abort();
        expect(await leaving).toEqual({
            ok: false,
            error: { code: 'aborted', message: 'Batch call aborted', retryable: false, attempts: 0 },
        });
        expect(await staying).toMatchObject({ ok: true, data: { text: 'b' } });
        expect(batch.submits.map((entry) => entry.items.length)).toEqual([1]);

        const alone = new AbortController();
        const lonely = calls.add(ask('c'), undefined, alone.signal);
        alone.abort();
        expect(await lonely).toMatchObject({ ok: false, error: { code: 'aborted', attempts: 0 } });
        await Bun.sleep(60);
        expect(batch.submits).toHaveLength(1);
    });

    it('resolve an aborted call at once after the submit and cancel a job whose calls are all aborted', async () => {
        const batch = fakeBatch();
        batch.status = (name) => ({ ok: true, data: { name, state: 'JOB_STATE_RUNNING', phase: 'running' } });
        const calls = createBatchCalls(batch, 1);
        const first = new AbortController();
        const second = new AbortController();
        const pending = [calls.add(ask('a'), undefined, first.signal), calls.add(ask('b'), undefined, second.signal)];
        while (batch.checks.length === 0) {
            await Bun.sleep(5);
        }
        first.abort();
        const submitted = await pending[0];
        expect(submitted).toMatchObject({ ok: false, error: { code: 'aborted' } });
        expect(submitted.ok || submitted.error.attempts).toBeUndefined();
        expect(batch.cancels).toEqual([]);
        second.abort();
        expect(await pending[1]).toMatchObject({ ok: false, error: { code: 'aborted' } });
        expect(batch.cancels).toEqual(['batches/1']);
        const checks = batch.checks.length;
        await Bun.sleep(30);
        expect(batch.checks).toHaveLength(checks);
    });

    it('cancel a job aborted during submit or check and map unexpected batch throws', async () => {
        const duringSubmit = fakeBatch();
        const submitAbort = new AbortController();
        duringSubmit.submit = async (items, options) => {
            submitAbort.abort();
            const name = 'batches/in-flight';
            duringSubmit.submits.push({ name, items: [...items], config: options?.config });
            return { ok: true, data: { name, state: 'JOB_STATE_PENDING', phase: 'running' } };
        };
        const submitCalls = createBatchCalls(duringSubmit, 1);
        expect(await submitCalls.add(ask('a'), undefined, submitAbort.signal)).toMatchObject({
            ok: false,
            error: { code: 'aborted' },
        });
        await Bun.sleep(30);
        expect(duringSubmit.cancels).toEqual(['batches/in-flight']);
        expect(duringSubmit.checks).toEqual([]);

        const duringCheck = fakeBatch();
        const checkAbort = new AbortController();
        duringCheck.status = (name) => {
            checkAbort.abort();
            return { ok: true, data: { name, state: 'JOB_STATE_RUNNING', phase: 'running' } };
        };
        const checkCalls = createBatchCalls(duringCheck, 1);
        expect(await checkCalls.add(ask('a'), undefined, checkAbort.signal)).toMatchObject({
            ok: false,
            error: { code: 'aborted' },
        });
        await Bun.sleep(30);
        expect(duringCheck.cancels).toEqual(['batches/1']);

        const throwingSubmit = fakeBatch();
        throwingSubmit.submit = async () => {
            throw new Error('submit boom');
        };
        expect(await createBatchCalls(throwingSubmit, 1).add(ask('a'), undefined)).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'submit boom', retryable: false },
        });

        const throwingCheck = fakeBatch();
        throwingCheck.check = () => Promise.reject('check boom');
        throwingCheck.cancel = async () => {
            throw new Error('cancel boom');
        };
        expect(await createBatchCalls(throwingCheck, 1).add(ask('a'), undefined)).toEqual({
            ok: false,
            error: { code: 'unexpected', message: 'check boom', retryable: false },
        });
    });
});

describe('batch calls of the client', () => {
    const server = startFakeGemini();
    afterAll(() => {
        server.stop();
    });
    beforeEach(() => {
        server.received.length = 0;
    });
    afterEach(() => {
        server.reply(() => Response.json({}));
    });

    /** Developer client configured for fake server. */
    function client(defaults?: { batch?: boolean }) {
        return createGenaiClient({
            apiKey: 'k',
            baseUrl: server.baseUrl,
            env: {},
            batchWaitSec: 1,
            ...(defaults === undefined ? {} : { defaults }),
        });
    }

    it('send batch: true through one inline job and match the answers by key', async () => {
        let keys: string[] = [];
        server.reply(
            (request) => {
                keys = request.body?.batch.inputConfig.requests.requests.map(
                    (entry: { metadata: { key: string } }) => entry.metadata.key,
                );
                return batchResponse('batches/j1', 'BATCH_STATE_PENDING');
            },
            () =>
                batchResponse(
                    'batches/j1',
                    'BATCH_STATE_SUCCEEDED',
                    keys.map((key, index) => ({ response: generateBody(`answer ${index}`), metadata: { key } })),
                ),
        );
        const studio = client();
        const [first, second] = await Promise.all([
            studio.generate({ ...ask('a'), batch: true }),
            studio.generate({ ...ask('b'), batch: true }),
        ]);
        expect(first).toMatchObject({ ok: true, data: { text: 'answer 0' } });
        expect(second).toMatchObject({ ok: true, data: { text: 'answer 1' } });
        const [submit, check] = server.received;
        expect(submit?.path).toBe('/v1beta/models/gemini-x:batchGenerateContent');
        expect(submit?.body?.batch.inputConfig.requests.requests).toHaveLength(2);
        expect(JSON.stringify(submit?.body)).not.toContain('"batch":true');
        expect(check?.path).toBe('/v1beta/batches/j1');
    });

    it('cancel through the API and answer a submit error to every call', async () => {
        server.reply(() => apiError(400));
        const studio = client();
        const results = await Promise.all([
            studio.generate({ ...ask('a'), batch: true }),
            studio.generate({ ...ask('b'), batch: true }),
        ]);
        expect(results).toMatchObject([
            { ok: false, error: { code: 'http', status: 400 } },
            { ok: false, error: { code: 'http', status: 400 } },
        ]);
        server.received.length = 0;
        server.reply(() => Response.json({}));
        expect(await studio.batch?.cancel('batches/j1')).toEqual({ ok: true, data: undefined });
        expect([server.received[0]?.method, server.received[0]?.path]).toEqual(['POST', '/v1beta/batches/j1:cancel']);
    });

    it('reject batch in submit items but ignore a batch default there; batch: false is a normal call', async () => {
        const studio = client({ batch: true });
        expect(await studio.batch?.submit([{ key: '1', request: { ...ask('a'), batch: true } }])).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Option "batch" is not supported in batch items',
                retryable: false,
            },
        });
        server.reply(() => batchResponse('batches/j2', 'BATCH_STATE_PENDING'));
        expect(await studio.batch?.submit([{ key: '1', request: ask('a') }])).toMatchObject({ ok: true });
        server.reply(() => Response.json(generateBody('now')));
        expect(await studio.generate({ ...ask('a'), batch: false })).toMatchObject({ ok: true, data: { text: 'now' } });
        expect(server.received.at(-1)?.path).toBe('/v1beta/models/gemini-x:generateContent');
        expect(await studio.generate({ ...ask('a'), batch: 'yes' } as never)).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Invalid value for option "batch": true or false' },
        });
    });

    it('answer unsupported on Vertex AI without bucket and validate batchWaitSec', async () => {
        const vertex = createGenaiClient({
            vertex: { project: 'p', authClient: fakeAuthClient() },
            env: {},
            baseUrl: server.baseUrl,
        });
        expect(await vertex.generate({ ...ask('a'), batch: true })).toEqual({
            ok: false,
            error: {
                code: 'unsupported',
                message: 'Batch calls are not supported: Vertex AI without bucket has no batch API',
                retryable: false,
                settings: { model: 'gemini-x', batch: true },
                attempts: 0,
            },
        });
        expect(server.received).toHaveLength(0);
        for (const batchWaitSec of [0, 0.5, 601, Number.NaN, '10']) {
            expect(() => createGenaiClient({ apiKey: 'k', batchWaitSec } as never)).toThrow(
                new TypeError('Invalid value for option "batchWaitSec": 1 to 600 seconds'),
            );
        }
        expect(() => createGenaiClient({ apiKey: 'k', batchWaitSec: 600 })).not.toThrow();
    });
});
