// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { BatchJob, GenerateContentConfig, GenerateContentResponse, GoogleGenAI } from '@google/genai';
import type {
    GenaiBatch,
    GenaiBatchResult,
    GenaiBatchStatus,
    GenaiError,
    GenaiOutput,
    GenaiRequest,
    Result,
} from '../../types.js';
import {
    batchPhase,
    guardBatch,
    listBatchJobs,
    RPC_HTTP_STATUS,
    readAnswer,
    rpcError,
    statusError,
    toGenaiError,
} from '../answer.js';
import {
    batchModel,
    checkBatchBytes,
    checkBatchName,
    checkOptionKeys,
    InvalidRequestError,
    toGeminiCall,
    wantsJson,
} from '../request.js';
import { finalJobError } from './final.js';
import type { GcsClient } from './gcs.js';
import { withoutForeignProperties } from './vertex-foreign.js';
import { vertexRequest } from './vertex-request.js';
import { typedSchemaForms } from './vertex-schema.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Project and location of client; names its batch folders in bucket. */
export type BatchOwner = { project: string; location: string };

/** Formatted JSONL line for Vertex batch input. */
type InputLine = { key: string; request: Record<string, unknown> };

/**
 * **Step on REST request of Vertex batch line**: returns copy, never modifies input.
 *
 * - `key`: item key for error messages (`''` in `requestBytes`)
 * @throws {InvalidRequestError} If the step rejects request.
 */
type LineStep = (request: Record<string, unknown>, key: string) => Record<string, unknown>;

/** Parsed JSONL line from Vertex predictions output; `request`: echoed input request of line. */
type OutputLine = { key?: unknown; request?: unknown; response?: GenerateContentResponse; status?: unknown };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/**
 * Maximum payload byte budget for single Vertex batch input file; read when batch is created.
 * @internal - exported for testing only
 */
export const vertexBatchLimit = { maxBytes: 100_000_000 };

/** Root folder path in bucket for batch job files. */
const BATCH_ROOT = 'smart-genai';

/** Matches folder segment of jobs submitted before owner prefix (plain UUID); cleaned by every client. */
const MATCH_LEGACY_FOLDER = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/;

/** Matches `:` of domain-scoped project ids; written as `~` in folder names. */
const MATCH_PROJECT_COLON = /:/g;

/** Supported option keys for batch submission. */
const SUBMIT_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'displayName', 'config']);

/** Grace period in milliseconds before unreferenced folders can be deleted. */
const REMOVE_GRACE_MS = 3_600_000;

/** Default time folders of succeeded jobs are kept by `removeFiles` after job end (7 days). */
const KEEP_SUCCEEDED_MS = 7 * 24 * 3_600_000;

/** Supported option keys for batch status checks. */
const CHECK_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'keepFiles']);

/** Supported option keys for batch cancellation. */
const CANCEL_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Supported option keys for batch job listing. */
const LIST_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'displayName']);

/** Supported option keys for batch file removal. */
const REMOVE_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'keepSucceededMs']);

/** Matches line key with format prefix (`j:` for JSON, `t:` for text). */
const MATCH_LINE_KEY = /^([jt]):([\s\S]+)$/;

/** Matches quota code name at start of non-JSON item status (answered as `rate_limited`); case-sensitive. */
const MATCH_QUOTA_STATUS = /^RESOURCE_EXHAUSTED\b/;

/** Matches transient code name at start of non-JSON item status; case-sensitive, so "Internal ..." stays permanent. */
const MATCH_TRANSIENT_STATUS = /^(?:RESOURCE_EXHAUSTED|UNAVAILABLE|INTERNAL|DEADLINE_EXCEEDED)\b/;

/** Matches publisher resource prefix on Vertex models. */
const MATCH_PUBLISHER_MODEL_PREFIX = /^publishers\/google\/models\//;

/**
 * **Steps on REST request of Vertex batch line, in order**: final empty-object check runs after the last.
 *
 * - `replaceEmptyObjects`: empty structs the batch import rejects → accepted forms
 * - `stripResponseRequired`: `required` in response schemas fails every item → removed
 * - `typedSchemaForms`: JSON schemas of different shapes fail each other → all schemas (either form) converted to
 *   typed form; empty `properties` of declaration schemas dropped
 * - later steps go after replacements, before final check
 */
const LINE_STEPS: readonly LineStep[] = [replaceEmptyObjects, stripResponseRequired, typedSchemaForms];

/** Response schema fields of `generationConfig`: JSON schema and typed form. */
const RESPONSE_SCHEMA_FIELDS = ['responseJsonSchema', 'responseSchema'] as const;

/** Schema keywords whose value is a single subschema. */
const SCHEMA_SUBSCHEMAS: ReadonlySet<string> = new Set(['items', 'additionalProperties', 'not']);

/** Schema keywords whose value is a list of subschemas. */
const SCHEMA_LISTS: ReadonlySet<string> = new Set(['prefixItems', 'anyOf', 'oneOf', 'allOf']);

/** JSON schema keywords whose value maps names to schemas. */
const SCHEMA_NAME_MAPS: ReadonlySet<string> = new Set([
    'properties',
    'patternProperties',
    '$defs',
    'definitions',
    'dependentSchemas',
]);

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Batch API implementation for Vertex AI**: uses Cloud Storage JSONL files.
 *
 * - uploads input requests to `smart-genai/<project>_<location>_<uuid>/input.jsonl` (see `ownerPrefix`)
 * - polls batch job and reads results from `predictions.jsonl`
 * - cleans up temporary bucket files after completion, unless `check` gets `keepFiles`
 * - `removeFiles`: only own-prefix folders and legacy `smart-genai/<uuid>/` folders
 * - parses output line errors and distinguishes transient failures
 * - removes properties of other lines' schemas from answers, by echoed request of line
 */
export function createVertexBatch(ai: GoogleGenAI, gcs: GcsClient, owner: BatchOwner): GenaiBatch {
    const prefix = ownerPrefix(owner);
    const { maxBytes } = vertexBatchLimit;
    return guardBatch({
        maxBytes,
        requestBytes(request, config) {
            return { ok: true, data: encodeLine('', request, config, owner).bytes };
        },
        async submit(items, options = {}) {
            checkOptionKeys(options, SUBMIT_OPTION_KEYS);
            const model = batchModel(items);
            const lines = items.map((item) => encodeLine(item.key, item.request, options.config, owner));
            checkBatchBytes(
                lines.map((line) => line.bytes),
                maxBytes,
            );
            const { signal } = options;
            const folder = `${BATCH_ROOT}/${prefix}${crypto.randomUUID()}`;
            const text = lines.map((line) => line.text).join('');
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
                    return { ok: false, error: finalJobError({ code: 'unexpected', message, retryable: false }) };
                }
                const lines = await readOutput(gcs, `${folder}/output/`, signal);
                if (!lines.ok) {
                    return { ok: false, error: finalJobError(lines.error) };
                }
                const model = (job.model ?? '').replace(MATCH_PUBLISHER_MODEL_PREFIX, '');
                const results: GenaiBatchResult[] = [...lines.data].map(([key, { json, line }]) => ({
                    key,
                    result: readOutputLine(line, json, model),
                }));
                status = { ...status, results };
            }
            if (folder !== undefined && options.keepFiles !== true) {
                await removeFolder(gcs, folder);
            }
            return { ok: true, data: status };
        },
        async cancel(name, options = {}) {
            checkBatchName(name);
            checkOptionKeys(options, CANCEL_OPTION_KEYS);
            const { signal } = options;
            const config = signal ? { config: { abortSignal: signal } } : {};
            let job: BatchJob;
            try {
                job = await ai.batches.get({ name, ...config });
            } catch (error) {
                return { ok: false, error: toGenaiError(error, signal) };
            }
            // a final job cannot be cancelled: only its files are left to remove
            if (batchPhase(job.state ?? 'JOB_STATE_UNSPECIFIED') === 'running') {
                try {
                    await ai.batches.cancel({ name, ...config });
                } catch (error) {
                    const settled = await settledJob(ai, name, config);
                    if (settled === undefined) {
                        return { ok: false, error: toGenaiError(error, signal) };
                    }
                    job = settled;
                }
            }
            const folder = batchFolder(job, gcs.bucket);
            if (folder !== undefined) {
                await removeFolder(gcs, folder);
            }
            return { ok: true, data: undefined };
        },
        async list(options = {}) {
            checkOptionKeys(options, LIST_OPTION_KEYS);
            return listBatchJobs(ai, options, (job) => batchFolder(job, gcs.bucket) !== undefined);
        },
        async removeFiles(options = {}) {
            checkOptionKeys(options, REMOVE_OPTION_KEYS);
            const { signal, keepSucceededMs = KEEP_SUCCEEDED_MS } = options;
            const inUse = await usedFolders(ai, gcs.bucket, keepSucceededMs, signal);
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
                // folders of other projects or locations sharing the bucket are never touched
                if (folder !== undefined && isOwnFolder(folder, prefix)) {
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

/**
 * **Serializes request once as JSONL line**: text with `key`, size of same line with empty key.
 *
 * - `bytes`: measure of `requestBytes` and `submit` size check (key bytes not counted)
 * @throws {InvalidRequestError} If `inputLine` rejects request.
 */
function encodeLine(
    key: string,
    request: GenaiRequest,
    config: GenerateContentConfig | undefined,
    owner: BatchOwner,
): { text: string; bytes: number } {
    const line = inputLine(key, request, config, owner);
    const body = JSON.stringify(line.request);
    const emptyKey = JSON.stringify(line.key.slice(0, line.key.length - key.length));
    const encoder = new TextEncoder();
    const bytes = encoder.encode(`{"key":${emptyKey},"request":}\n`).length + encoder.encode(body).length;
    return { text: `{"key":${JSON.stringify(line.key)},"request":${body}}\n`, bytes };
}

/**
 * **Formats request as single JSONL line for Vertex batch input**: import rejects empty structs.
 *
 * - pipeline: REST request as SDK builds it (see `vertexRequest`) → `LINE_STEPS` in order → final empty-object check
 * @throws {InvalidRequestError} If config is unknown or unsupported on Vertex, a step rejects request or an empty
 *   object remains after all steps.
 */
function inputLine(
    key: string,
    request: GenaiRequest,
    extra: GenerateContentConfig | undefined,
    owner: BatchOwner,
): InputLine {
    const { contents, config } = toGeminiCall(request, extra);
    const built = vertexRequest(contents, config, owner, key);
    const line = LINE_STEPS.reduce((current, step) => step(current, key), built);
    const empty = emptyObjectPath(line, 'request');
    if (empty !== undefined) {
        const item = key === '' ? '' : ` of item "${key}"`;
        throw new InvalidRequestError(`Empty object "${empty}"${item} is not supported in Vertex batch`);
    }
    return { key: `${wantsJson(request) ? 'j' : 't'}:${key}`, request: line };
}

/**
 * **Replaces empty objects the library or common inputs produce**: copies, caller objects never modified.
 *
 * - `contents`: see `batchContent`
 * - `tools`: see `batchTool`
 */
function replaceEmptyObjects(request: Record<string, unknown>): Record<string, unknown> {
    const { contents, tools } = request;
    return {
        ...request,
        ...(Array.isArray(contents) ? { contents: contents.map(batchContent) } : {}),
        ...(Array.isArray(tools) ? { tools: tools.map(batchTool) } : {}),
    };
}

/**
 * **Copies content in form Vertex batch import accepts**.
 *
 * - empty `functionCall.args` omitted
 * - empty `functionResponse.response` → `{ output: '' }` (model reads: tool returned no data)
 */
function batchContent(content: unknown): unknown {
    if (!isRecord(content) || !Array.isArray(content.parts)) {
        return content;
    }
    const parts = content.parts.map((part: unknown) => {
        if (!isRecord(part)) {
            return part;
        }
        const { functionCall, functionResponse } = part;
        if (isRecord(functionCall) && isEmptyObject(functionCall.args)) {
            const { args: _args, ...call } = functionCall;
            return { ...part, functionCall: call };
        }
        if (isRecord(functionResponse) && isEmptyObject(functionResponse.response)) {
            return { ...part, functionResponse: { ...functionResponse, response: { output: '' } } };
        }
        return part;
    });
    return { ...content, parts };
}

/**
 * **Copies tool in form Vertex batch import accepts**.
 *
 * - empty `googleSearch` → `{ excludeDomains: [] }` (same search behavior)
 * - empty `properties` of function declaration schemas: dropped by `typedSchemaForms`
 * - other tools unchanged
 */
function batchTool(tool: unknown): unknown {
    return isRecord(tool) && isEmptyObject(tool.googleSearch)
        ? { ...tool, googleSearch: { excludeDomains: [] } }
        : tool;
}

/**
 * **Removes `required` from response schemas**: import fails every item with it, in both schema forms.
 *
 * - `generationConfig.responseJsonSchema` and `responseSchema`: see `withoutRequired`
 * - tool parameter schemas unchanged: import accepts `required` there
 * - copies, caller objects never modified
 */
function stripResponseRequired(request: Record<string, unknown>): Record<string, unknown> {
    const { generationConfig } = request;
    if (!isRecord(generationConfig)) {
        return request;
    }
    const copy = { ...generationConfig };
    for (const field of RESPONSE_SCHEMA_FIELDS) {
        if (copy[field] !== undefined) {
            copy[field] = withoutRequired(copy[field]);
        }
    }
    return { ...request, generationConfig: copy };
}

/**
 * **Copies schema without keyword `required`**: on every level, walked structurally.
 *
 * - subschemas: values of name maps (`properties`, `$defs`...), `items`, `additionalProperties`, `not`, list members
 * - names in name maps kept: a property named `required` stays
 * - other values (`enum`, `default`...) copied unchanged
 */
function withoutRequired(schema: unknown): unknown {
    if (!isRecord(schema)) {
        return schema;
    }
    const copy: Record<string, unknown> = {};
    for (const [keyword, value] of Object.entries(schema)) {
        if (keyword === 'required') {
            continue;
        }
        if (SCHEMA_NAME_MAPS.has(keyword) && isRecord(value)) {
            copy[keyword] = Object.fromEntries(
                Object.entries(value).map(([name, entry]) => [name, withoutRequired(entry)]),
            );
        } else if ((SCHEMA_LISTS.has(keyword) || keyword === 'items') && Array.isArray(value)) {
            copy[keyword] = value.map(withoutRequired);
        } else {
            copy[keyword] = SCHEMA_SUBSCHEMAS.has(keyword) ? withoutRequired(value) : value;
        }
    }
    return copy;
}

/** Finds path of first empty object (as JSON serializes it, `undefined` values dropped); `undefined` if none. */
function emptyObjectPath(value: unknown, path: string): string | undefined {
    if (Array.isArray(value)) {
        for (const [index, entry] of value.entries()) {
            const found = emptyObjectPath(entry, `${path}[${index}]`);
            if (found !== undefined) {
                return found;
            }
        }
        return undefined;
    }
    if (!isRecord(value)) {
        return undefined;
    }
    if (isEmptyObject(value)) {
        return path;
    }
    for (const [field, entry] of Object.entries(value)) {
        const found = emptyObjectPath(entry, `${path}.${field}`);
        if (found !== undefined) {
            return found;
        }
    }
    return undefined;
}

/** Checks for plain non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Checks for object without fields as JSON serializes it (`undefined` values dropped). */
function isEmptyObject(value: unknown): boolean {
    return isRecord(value) && Object.values(value).every((entry) => entry === undefined);
}

/**
 * **Builds owner prefix of batch folder segment**: `<project>_<location>_`.
 *
 * - `:` of domain-scoped project ids → `~`; `.` and `-` kept
 * - neither part contains `_`, `~` or `/`: prefix of one owner never starts another owner's
 */
function ownerPrefix(owner: BatchOwner): string {
    return `${owner.project.replace(MATCH_PROJECT_COLON, '~')}_${owner.location}_`;
}

/** Checks root folder for own owner prefix or legacy plain UUID segment. */
function isOwnFolder(folder: string, prefix: string): boolean {
    const segment = folder.slice(BATCH_ROOT.length + 1);
    return segment.startsWith(prefix) || MATCH_LEGACY_FOLDER.test(segment);
}

/**
 * **Extracts batch subfolder from input GCS URI**: only a direct child of `BATCH_ROOT`.
 *
 * - any segment accepted (own prefix or legacy UUID): job comes from client's own project and location
 */
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

/**
 * **Converts OutputLine into GenaiOutput or GenaiError**.
 *
 * - answer: properties of other lines' schemas removed, see `withoutForeignProperties`
 */
function readOutputLine(line: OutputLine, json: boolean, model: string): Result<GenaiOutput, GenaiError> {
    const status = line.status;
    if (typeof status === 'string' && status !== '') {
        return { ok: false, error: statusLineError(status, parseStatus(status)) };
    }
    if (typeof status === 'object' && status !== null && Object.keys(status).length > 0) {
        return { ok: false, error: statusLineError(JSON.stringify(status), status) };
    }
    if (line.response === undefined) {
        return { ok: false, error: { code: 'unexpected', message: 'No response in the batch', retryable: false } };
    }
    const answer = readAnswer(line.response, model, json, 0);
    return answer.ok ? { ok: true, data: withoutForeignProperties(answer.data, line.request) } : answer;
}

/** Parses item status text as JSON; `undefined` when not JSON. */
function parseStatus(text: string): unknown {
    try {
        return JSON.parse(text) as unknown;
    } catch {
        return undefined;
    }
}

/**
 * **Maps item status to `GenaiError`**: same RPC mapping as Developer API items.
 *
 * - parsed object with known RPC `code` (1 to 16): `rpcError`, parsed `message` or raw text
 * - otherwise (not JSON, no or unknown code): raw text, anchored code-name patterns, no `status` unless quota
 */
function statusLineError(text: string, parsed: unknown): GenaiError {
    if (typeof parsed === 'object' && parsed !== null) {
        const { code, message } = parsed as { code?: unknown; message?: unknown };
        if (typeof code === 'number' && Object.hasOwn(RPC_HTTP_STATUS, code)) {
            return rpcError(code, typeof message === 'string' && message !== '' ? message : text);
        }
    }
    if (MATCH_QUOTA_STATUS.test(text)) {
        return statusError(429, text);
    }
    return { code: 'http', message: text, retryable: MATCH_TRANSIENT_STATUS.test(text) };
}

/**
 * **Discovers batch bucket folders still in use**.
 *
 * - folders of running jobs
 * - folders of succeeded jobs ended less than `keepSucceededMs` ago, results maybe not stored yet
 * - unknown end time (`endTime`, else `updateTime`) counts as recent
 */
async function usedFolders(
    ai: GoogleGenAI,
    bucket: string,
    keepSucceededMs: number,
    signal: AbortSignal | undefined,
): Promise<Result<Set<string>, GenaiError>> {
    const prefix = `gs://${bucket}/`;
    const folders = new Set<string>();
    const now = Date.now();
    try {
        const pager = await ai.batches.list({ config: { pageSize: 100, ...(signal ? { abortSignal: signal } : {}) } });
        for await (const job of pager) {
            const phase = job ? batchPhase(job.state ?? 'JOB_STATE_UNSPECIFIED') : 'failed';
            const ended = Date.parse(job?.endTime ?? job?.updateTime ?? '');
            const isRecent = Number.isNaN(ended) || now - ended < keepSucceededMs;
            if (!job || !(phase === 'running' || (phase === 'succeeded' && isRecent))) {
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

/**
 * **Re-reads job after failed cancel**: job when cancel is no longer needed, else `undefined`.
 *
 * - job turned final between `get` and `cancel`: API rejects cancel (`FAILED_PRECONDITION`)
 * - `JOB_STATE_CANCELLING`: cancel already under way, counts as success
 * - re-read fails or job still running otherwise: `undefined`
 */
async function settledJob(
    ai: GoogleGenAI,
    name: string,
    config: { config?: { abortSignal: AbortSignal } },
): Promise<BatchJob | undefined> {
    let job: BatchJob;
    try {
        job = await ai.batches.get({ name, ...config });
    } catch {
        return undefined;
    }
    const state = job.state ?? 'JOB_STATE_UNSPECIFIED';
    return batchPhase(state) !== 'running' || state === 'JOB_STATE_CANCELLING' ? job : undefined;
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
