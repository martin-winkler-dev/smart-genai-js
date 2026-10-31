// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type {
    BatchJob,
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    GoogleGenAI,
    Part,
} from '@google/genai';
import type {
    GenaiBatch,
    GenaiBatchResult,
    GenaiBatchStatus,
    GenaiError,
    GenaiOutput,
    GenaiRequest,
    Result,
} from '../../types.js';
import { batchPhase, guardBatch, readAnswer, statusError, toGenaiError } from '../answer.js';
import { batchModel, checkBatchName, checkOptionKeys, toGeminiCall, wantsJson } from '../request.js';
import type { GcsClient } from './gcs.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Formatted JSONL line for Vertex batch input. */
type InputLine = { key: string; request: Record<string, unknown> };

/** Parsed JSONL line from Vertex predictions output. */
type OutputLine = { key?: unknown; response?: GenerateContentResponse; status?: unknown };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Maximum payload byte budget for single Vertex batch input file. */
const VERTEX_BATCH_MAX_BYTES = 100_000_000;

/** Root folder path in bucket for batch job files. */
const BATCH_ROOT = 'smart-genai';

/** Supported option keys for batch submission. */
const SUBMIT_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'displayName', 'config']);

/** Grace period in milliseconds before unreferenced folders can be deleted. */
const REMOVE_GRACE_MS = 3_600_000;

/** Supported option keys for batch status checks and cancellation. */
const CHECK_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Matches line key with format prefix (`j:` for JSON, `t:` for text). */
const MATCH_LINE_KEY = /^([jt]):([\s\S]+)$/;

/** Matches quota status of an output line (answered as `rate_limited`). */
const MATCH_QUOTA_STATUS = /RESOURCE_EXHAUSTED|\b429\b/i;

/** Matches transient failure patterns in job status messages. */
const MATCH_TRANSIENT_STATUS = /RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|DEADLINE_EXCEEDED|\b(?:429|50[0-4])\b/i;

/** SDK config fields placed at top level of REST request payload. */
const REQUEST_FIELDS: ReadonlySet<string> = new Set([
    'safetySettings',
    'tools',
    'toolConfig',
    'cachedContent',
    'labels',
    'serviceTier',
    'modelArmorConfig',
]);

/** Matches publisher resource prefix on Vertex models. */
const MATCH_PUBLISHER_MODEL_PREFIX = /^publishers\/google\/models\//;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Batch API implementation for Vertex AI**: uses Cloud Storage JSONL files.
 *
 * - uploads input requests to `smart-genai/<uuid>/input.jsonl`
 * - polls batch job and reads results from `predictions.jsonl`
 * - cleans up temporary bucket files after completion
 * - parses output line errors and distinguishes transient failures
 */
export function createVertexBatch(ai: GoogleGenAI, gcs: GcsClient): GenaiBatch {
    return guardBatch({
        maxBytes: VERTEX_BATCH_MAX_BYTES,
        requestBytes(request, config) {
            const data = new TextEncoder().encode(`${JSON.stringify(inputLine('', request, config))}\n`).length;
            return { ok: true, data };
        },
        async submit(items, options = {}) {
            checkOptionKeys(options, SUBMIT_OPTION_KEYS);
            const model = batchModel(items);
            const { signal } = options;
            const folder = `${BATCH_ROOT}/${crypto.randomUUID()}`;
            const text = items
                .map((item) => `${JSON.stringify(inputLine(item.key, item.request, options.config))}\n`)
                .join('');
            const uploaded = await gcs.upload(`${folder}/input.jsonl`, text, signal);
            if (!uploaded.ok) {
                return uploaded;
            }
            let job: BatchJob;
            try {
                job = await ai.batches.create({
                    model,
                    src: `gs://${gcs.bucket}/${folder}/input.jsonl`,
                    config: {
                        dest: `gs://${gcs.bucket}/${folder}/output/`,
                        displayName: options.displayName ?? `smart-genai ${items.length} requests`,
                        ...(signal ? { abortSignal: signal } : {}),
                    },
                });
            } catch (error) {
                await gcs.remove(`${folder}/input.jsonl`);
                return { ok: false, error: toGenaiError(error, signal) };
            }
            if (!job.name) {
                await gcs.remove(`${folder}/input.jsonl`);
                return { ok: false, error: { code: 'unexpected', message: 'Batch without name', retryable: false } };
            }
            const state = job.state ?? 'JOB_STATE_UNSPECIFIED';
            return { ok: true, data: { name: job.name, state, phase: batchPhase(state) } };
        },
        async check(name, options = {}) {
            checkBatchName(name);
            checkOptionKeys(options, CHECK_OPTION_KEYS);
            const { signal } = options;
            let job: BatchJob;
            try {
                job = await ai.batches.get({ name, ...(signal ? { config: { abortSignal: signal } } : {}) });
            } catch (error) {
                return { ok: false, error: toGenaiError(error, signal) };
            }
            const state = job.state ?? 'JOB_STATE_UNSPECIFIED';
            const phase = batchPhase(state);
            if (phase === 'running') {
                return { ok: true, data: { name, state, phase } };
            }
            const folder = batchFolder(job, gcs.bucket);
            const jobError = job.error?.message;
            let status: GenaiBatchStatus = { name, state, phase, ...(jobError ? { error: jobError } : {}) };
            if (phase === 'succeeded') {
                if (folder === undefined) {
                    const message = `Batch input is not a smart-genai file in bucket ${gcs.bucket}`;
                    return { ok: false, error: { code: 'unexpected', message, retryable: false } };
                }
                const lines = await readOutput(gcs, `${folder}/output/`, signal);
                if (!lines.ok) {
                    return lines;
                }
                const model = (job.model ?? '').replace(MATCH_PUBLISHER_MODEL_PREFIX, '');
                const results: GenaiBatchResult[] = [...lines.data].map(([key, { json, line }]) => ({
                    key,
                    result: readOutputLine(line, json, model),
                }));
                status = { ...status, results };
            }
            if (folder !== undefined) {
                await removeFolder(gcs, folder);
            }
            return { ok: true, data: status };
        },
        async cancel(name, options = {}) {
            checkBatchName(name);
            checkOptionKeys(options, CHECK_OPTION_KEYS);
            const { signal } = options;
            const config = signal ? { config: { abortSignal: signal } } : {};
            let job: BatchJob;
            try {
                job = await ai.batches.get({ name, ...config });
                // a final job cannot be cancelled: only its files are left to remove
                if (batchPhase(job.state ?? 'JOB_STATE_UNSPECIFIED') === 'running') {
                    await ai.batches.cancel({ name, ...config });
                }
            } catch (error) {
                return { ok: false, error: toGenaiError(error, signal) };
            }
            const folder = batchFolder(job, gcs.bucket);
            if (folder !== undefined) {
                await removeFolder(gcs, folder);
            }
            return { ok: true, data: undefined };
        },
        async removeFiles(options = {}) {
            checkOptionKeys(options, CHECK_OPTION_KEYS);
            const { signal } = options;
            const inUse = await runningFolders(ai, gcs.bucket, signal);
            if (!inUse.ok) {
                return inUse;
            }
            const listed = await gcs.listObjects(`${BATCH_ROOT}/`, signal);
            if (!listed.ok) {
                return listed;
            }
            const folders = new Map<string, string[]>();
            const young = new Set<string>();
            const now = Date.now();
            for (const { name, timeCreated } of listed.data) {
                const folder = rootEntry(name);
                if (folder !== undefined) {
                    folders.set(folder, [...(folders.get(folder) ?? []), name]);
                    // unknown creation time counts as young: never delete what may be in use
                    const created = timeCreated === undefined ? Number.NaN : Date.parse(timeCreated);
                    if (Number.isNaN(created) || now - created < REMOVE_GRACE_MS) {
                        young.add(folder);
                    }
                }
            }
            let removed = 0;
            let kept = 0;
            for (const [folder, objects] of folders) {
                if (inUse.data.has(folder) || young.has(folder)) {
                    kept += 1;
                    continue;
                }
                for (const name of objects) {
                    const deleted = await gcs.remove(name, signal);
                    if (!deleted.ok) {
                        return deleted;
                    }
                }
                if (signal?.aborted) {
                    const message = 'Removing batch files aborted';
                    return { ok: false, error: { code: 'aborted', message, retryable: false } };
                }
                removed += 1;
            }
            return { ok: true, data: { removed, kept } };
        },
    });
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Formats request as single JSONL line for Vertex batch input. */
function inputLine(key: string, request: GenaiRequest, extra: GenerateContentConfig | undefined): InputLine {
    const { contents, config } = toGeminiCall(request, extra);
    const { systemInstruction, ...rest } = config;
    const system = systemContent(systemInstruction);
    const topLevel: Record<string, unknown> = {};
    const generationConfig: Record<string, unknown> = {};
    for (const [field, value] of Object.entries(rest)) {
        (REQUEST_FIELDS.has(field) ? topLevel : generationConfig)[field] = value;
    }
    return {
        key: `${wantsJson(request) ? 'j' : 't'}:${key}`,
        request: {
            contents,
            ...(system === undefined ? {} : { systemInstruction: system }),
            ...topLevel,
            ...(Object.keys(generationConfig).length === 0 ? {} : { generationConfig }),
        },
    };
}

/** Converts system instruction into REST Content structure. */
function systemContent(instruction: GenerateContentConfig['systemInstruction']): Content | undefined {
    if (instruction === undefined) {
        return undefined;
    }
    if (typeof instruction === 'string') {
        return { parts: [{ text: instruction }] };
    }
    if (Array.isArray(instruction)) {
        return { parts: instruction.map((part) => (typeof part === 'string' ? { text: part } : part)) };
    }
    return 'parts' in instruction ? (instruction as Content) : { parts: [instruction as Part] };
}

/** Extracts batch subfolder from input GCS URI; only a direct child of `BATCH_ROOT`. */
function batchFolder(job: BatchJob, bucket: string): string | undefined {
    const prefix = `gs://${bucket}/`;
    const input = job.src?.gcsUri?.[0];
    if (input === undefined || !input.startsWith(prefix) || !input.endsWith('/input.jsonl')) {
        return undefined;
    }
    const folder = input.slice(prefix.length, -'/input.jsonl'.length);
    return rootEntry(folder) === folder ? folder : undefined;
}

/**
 * **Reads predictions JSONL output from Cloud Storage**.
 *
 * - downloads and parses all `.jsonl` files in output folder
 * - extracts results keyed by caller request key
 */
async function readOutput(
    gcs: GcsClient,
    prefix: string,
    signal: AbortSignal | undefined,
): Promise<Result<Map<string, { json: boolean; line: OutputLine }>, GenaiError>> {
    const byKey = new Map<string, { json: boolean; line: OutputLine }>();
    const names = await gcs.list(prefix, signal);
    if (!names.ok) {
        return names;
    }
    const files = names.data.filter((entry) => entry.endsWith('.jsonl'));
    if (files.length === 0) {
        const message = 'Batch output already read or missing';
        return { ok: false, error: { code: 'unexpected', message, retryable: false } };
    }
    for (const name of files) {
        const text = await gcs.download(name, signal);
        if (!text.ok) {
            return text;
        }
        for (const raw of text.data.split('\n')) {
            const line = parseLine(raw);
            const match = typeof line?.key === 'string' ? MATCH_LINE_KEY.exec(line.key) : null;
            const key = match?.[2];
            if (line !== undefined && key !== undefined && !byKey.has(key)) {
                byKey.set(key, { json: match?.[1] === 'j', line });
            }
        }
    }
    return { ok: true, data: byKey };
}

/** Parses raw JSONL line into OutputLine object. */
function parseLine(raw: string): OutputLine | undefined {
    if (raw.trim() === '') {
        return undefined;
    }
    try {
        const value = JSON.parse(raw) as unknown;
        return typeof value === 'object' && value !== null ? (value as OutputLine) : undefined;
    } catch {
        return undefined;
    }
}

/** Converts OutputLine into GenaiOutput or GenaiError. */
function readOutputLine(line: OutputLine, json: boolean, model: string): Result<GenaiOutput, GenaiError> {
    if (typeof line.status === 'string' && line.status !== '') {
        if (MATCH_QUOTA_STATUS.test(line.status)) {
            return { ok: false, error: statusError(429, line.status) };
        }
        const retryable = MATCH_TRANSIENT_STATUS.test(line.status);
        return { ok: false, error: { code: 'http', message: line.status, retryable } };
    }
    if (line.response === undefined) {
        return { ok: false, error: { code: 'unexpected', message: 'No response in the batch', retryable: false } };
    }
    return readAnswer(line.response, model, json, 0);
}

/** Discovers active batch bucket folders currently referenced by running jobs. */
async function runningFolders(
    ai: GoogleGenAI,
    bucket: string,
    signal: AbortSignal | undefined,
): Promise<Result<Set<string>, GenaiError>> {
    const prefix = `gs://${bucket}/`;
    const folders = new Set<string>();
    try {
        const pager = await ai.batches.list({ config: { pageSize: 100, ...(signal ? { abortSignal: signal } : {}) } });
        for await (const job of pager) {
            if (!job || batchPhase(job.state ?? 'JOB_STATE_UNSPECIFIED') !== 'running') {
                continue;
            }
            for (const uri of [...(job.src?.gcsUri ?? []), job.dest?.gcsUri ?? '']) {
                const folder = uri.startsWith(prefix) ? rootEntry(uri.slice(prefix.length)) : undefined;
                if (folder !== undefined) {
                    folders.add(folder);
                }
            }
        }
    } catch (error) {
        return { ok: false, error: toGenaiError(error, signal) };
    }
    return { ok: true, data: folders };
}

/** Resolves root folder path within `BATCH_ROOT` prefix. */
function rootEntry(name: string): string | undefined {
    if (!name.startsWith(`${BATCH_ROOT}/`)) {
        return undefined;
    }
    const entry = name.slice(BATCH_ROOT.length + 1).split('/')[0];
    return entry ? `${BATCH_ROOT}/${entry}` : undefined;
}

/** Deletes all objects within batch folder. */
async function removeFolder(gcs: GcsClient, folder: string): Promise<void> {
    const names = await gcs.list(`${folder}/`);
    for (const name of names.ok ? names.data : []) {
        await gcs.remove(name);
    }
}
