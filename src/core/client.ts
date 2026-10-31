// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join } from 'node:path';
import { type GenerateContentResponse, GoogleGenAI, type Model } from '@google/genai';
import { GoogleAuth, type GoogleAuthOptions } from 'google-auth-library';
import type {
    GenaiBatch,
    GenaiClient,
    GenaiClientOptions,
    GenaiDefaults,
    GenaiError,
    GenaiModelInfo,
    GenaiOutput,
    GenaiSettings,
    GenaiVertexOptions,
    Result,
} from '../types.js';
import { elapsedMs, guard, guardBatch, readAnswer, toGenaiError } from './answer.js';
import { type BatchCalls, createBatchCalls } from './batch/calls.js';
import { createGcsClient } from './batch/gcs.js';
import { createInlineBatch } from './batch/inline.js';
import { createVertexBatch } from './batch/vertex.js';
import { createEmbed } from './embed.js';
import {
    applyDefaults,
    checkOptionKeys,
    copyPlain,
    InvalidRequestError,
    toGeminiCall,
    toSettings,
    validateDefaults,
    wantsJson,
} from './request.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Vertex configuration target with defaults resolved. */
type ResolvedVertex = Omit<GenaiVertexOptions, 'project' | 'location'> & { project: string; location: string };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Environment variable name for Developer API key. */
export const GEMINI_API_KEY_ENV = 'GEMINI_API_KEY';

/** Environment variable names for Vertex AI configuration. */
export const VERTEX_ENV = {
    project: 'GOOGLE_CLOUD_PROJECT',
    location: 'GOOGLE_CLOUD_LOCATION',
    bucket: 'VERTEX_BATCH_BUCKET',
    credentialsFile: 'GOOGLE_APPLICATION_CREDENTIALS',
} as const;

/** Default Vertex location for serving models. */
const DEFAULT_LOCATION = 'global';

/** Official endpoint URL for Gemini Developer API. */
const DEVELOPER_BASE_URL = 'https://generativelanguage.googleapis.com/';

/** Multi-regional locations supported by Vertex AI. */
const MULTI_REGIONAL_LOCATIONS: ReadonlySet<string> = new Set(['us', 'eu']);

/** Matches Vertex location identifier strings. */
const MATCH_LOCATION = /^[a-z][a-z0-9-]{0,62}$/;

/** Matches Google Cloud project ID format. */
const MATCH_PROJECT = /^[a-z0-9][a-z0-9.:-]{0,127}$/;

/** Default timeout for HTTP requests in milliseconds. */
const DEFAULT_TIMEOUT_MS = 120_000;

/** Required OAuth scope for Vertex AI and Cloud Storage. */
const CLOUD_SCOPE = 'https://www.googleapis.com/auth/cloud-platform';

/** Supported option keys for `createGenaiClient`. */
const OPTION_KEYS: ReadonlySet<string> = new Set([
    'apiKey',
    'vertex',
    'env',
    'timeoutMs',
    'baseUrl',
    'defaults',
    'batchWaitSec',
]);

/** Default batch aggregation window duration in seconds. */
const DEFAULT_BATCH_WAIT_SEC = 10;
/** Minimum batch wait duration in seconds. */
const MIN_BATCH_WAIT_SEC = 1;
/** Maximum batch wait duration in seconds. */
const MAX_BATCH_WAIT_SEC = 600;

/** Supported option keys for `listModels`. */
const LIST_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Maximum page size for model listing queries. */
const LIST_PAGE_SIZE = 1000;

/** Matches API resource prefix on model identifiers. */
const MATCH_MODEL_PREFIX = /^(?:publishers\/[^/]+\/)?models\//;

/** Supported property keys in `vertex` options. */
const VERTEX_KEYS: ReadonlySet<string> = new Set([
    'project',
    'location',
    'bucket',
    'credentialsFile',
    'authClient',
    'storageUrl',
]);

/** Matches valid Cloud Storage bucket naming rules. */
const MATCH_BUCKET_NAME = /^[a-z0-9][a-z0-9._-]{1,61}[a-z0-9]$/;

/** Supported option keys for `generate` calls. */
const CALL_OPTION_KEYS: ReadonlySet<string> = new Set(['signal', 'config']);

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Client for Gemini Developer API or Vertex AI**: creates configured `GenaiClient`.
 *
 * - Developer API: uses `apiKey` or `GEMINI_API_KEY`
 * - Vertex AI: authenticates via credentials file or `authClient`
 * - disables automatic continuation and retries in underlying SDK
 * - never reads `.env` files directly
 * @throws {TypeError} If options are invalid or required credentials missing.
 */
export function createGenaiClient(options: GenaiClientOptions = {}): GenaiClient {
    validateOptions(options);
    const env = options.env ?? process.env;
    const timeout = options.timeoutMs ?? DEFAULT_TIMEOUT_MS;
    if (options.vertex === undefined || options.vertex === false) {
        const apiKey = options.apiKey ?? env[GEMINI_API_KEY_ENV];
        if (!apiKey) {
            throw new TypeError(`Invalid value for option "apiKey": set it or env ${GEMINI_API_KEY_ENV}`);
        }
        const httpOptions = { timeout, baseUrl: options.baseUrl ?? DEVELOPER_BASE_URL };
        const ai = new GoogleGenAI({ apiKey, vertexai: false, httpOptions });
        return buildClient('developer', ai, createInlineBatch(ai), options);
    }
    if (options.apiKey !== undefined) {
        throw new TypeError('Option "apiKey" is not supported with "vertex"');
    }
    const vertex = resolveVertex(options.vertex === true ? {} : options.vertex, env);
    const authOptions: GoogleAuthOptions =
        vertex.authClient === undefined
            ? { keyFilename: vertex.credentialsFile ?? '', projectId: vertex.project, scopes: [CLOUD_SCOPE] }
            : { authClient: vertex.authClient, scopes: [CLOUD_SCOPE] };
    const ai = new GoogleGenAI({
        vertexai: true,
        project: vertex.project,
        location: vertex.location,
        googleAuthOptions: authOptions,
        httpOptions: { timeout, baseUrl: options.baseUrl ?? vertexBaseUrl(vertex.location) },
    });
    if (vertex.bucket === undefined) {
        return buildClient('vertex', ai, undefined, options);
    }
    const auth = new GoogleAuth(authOptions);
    const gcs = createGcsClient({
        bucket: vertex.bucket,
        accessToken: () => auth.getAccessToken(),
        timeoutMs: timeout,
        ...(vertex.storageUrl === undefined ? {} : { baseUrl: vertex.storageUrl }),
    });
    return buildClient('vertex', ai, createVertexBatch(ai, gcs), options);
}

/**
 * **Resolves path to Google credentials file**.
 *
 * - checks `GOOGLE_APPLICATION_CREDENTIALS`
 * - falls back to default gcloud ADC file path by platform
 */
export function credentialsFile(env: Readonly<Record<string, string | undefined>> = process.env): string {
    const fromEnv = env[VERTEX_ENV.credentialsFile];
    if (fromEnv) {
        return fromEnv;
    }
    const configDir =
        process.platform === 'win32'
            ? env.APPDATA || join(homedir(), 'AppData', 'Roaming')
            : join(env.HOME || homedir(), '.config');
    return join(configDir, 'gcloud', 'application_default_credentials.json');
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Assembles client instance for specified operational mode. */
function buildClient(
    mode: GenaiClient['mode'],
    ai: GoogleGenAI,
    batch: GenaiBatch | undefined,
    options: GenaiClientOptions,
): GenaiClient {
    // A copy, so changing the defaults object later changes no request.
    const defaults = copyPlain(options.defaults);
    const calls =
        batch === undefined ? undefined : createBatchCalls(batch, options.batchWaitSec ?? DEFAULT_BATCH_WAIT_SEC);
    return {
        mode,
        ...(batch === undefined ? {} : { batch: defaults === undefined ? batch : withDefaults(batch, defaults) }),
        generate: createGenerate(ai, defaults, calls),
        embed: createEmbed(ai, mode),
        listModels: createListModels(ai),
    };
}

/** Creates `generate` function bound to configured client and defaults. */
function createGenerate(
    ai: GoogleGenAI,
    defaults: GenaiDefaults | undefined,
    calls: BatchCalls | undefined,
): GenaiClient['generate'] {
    return guard(async function generate(given, options = {}) {
        checkOptionKeys(options, CALL_OPTION_KEYS);
        const { signal } = options;
        const { request, config } = applyDefaults(defaults, given, options.config);
        const call = toGeminiCall(request, config);
        const settings = toSettings(request, config, call.model);
        if (signal?.aborted) {
            const error: GenaiError = { code: 'aborted', message: 'Request aborted', retryable: false, attempts: 0 };
            return withSettings({ ok: false, error }, settings);
        }
        if (request.batch === true) {
            if (calls === undefined) {
                const message = 'Batch calls are not supported: Vertex AI without bucket has no batch API';
                return { ok: false, error: { code: 'unsupported', message, retryable: false, settings, attempts: 0 } };
            }
            const { batch: _batch, ...single } = request;
            const result = await calls.add(single, config, signal);
            // too large for a batch: a failed request check, so no settings and no attempts
            const isRefused =
                !result.ok && result.error.code === 'invalid_request' && result.error.attempts === undefined;
            return isRefused ? result : withSettings(result, settings);
        }
        const startedAt = performance.now();
        let response: GenerateContentResponse;
        try {
            response = await ai.models.generateContent({
                model: call.model,
                contents: call.contents,
                config: { ...call.config, automaticContinuation: false, ...(signal ? { abortSignal: signal } : {}) },
            });
        } catch (error) {
            const failed = { ...toGenaiError(error, signal), ms: elapsedMs(startedAt) };
            return withSettings({ ok: false, error: failed }, settings);
        }
        return withSettings(readAnswer(response, call.model, wantsJson(request), elapsedMs(startedAt)), settings);
    });
}

/** Attaches effective settings and attempt counter (default 1) to generation result. */
function withSettings(
    result: Result<GenaiOutput, GenaiError>,
    settings: GenaiSettings,
): Result<GenaiOutput, GenaiError> {
    return result.ok
        ? { ok: true, data: { ...result.data, settings, attempts: result.data.attempts ?? 1 } }
        : { ok: false, error: { ...result.error, settings, attempts: result.error.attempts ?? 1 } };
}

/** Wraps batch interface to inject client-level defaults into requests. */
function withDefaults(batch: GenaiBatch, withBatch: GenaiDefaults): GenaiBatch {
    const { batch: _batch, ...defaults } = withBatch;
    return guardBatch({
        maxBytes: batch.maxBytes,
        requestBytes(request, config) {
            const filled = applyDefaults(defaults, request, config);
            return batch.requestBytes(filled.request, filled.config);
        },
        submit(items, options = {}) {
            const filled = Array.isArray(items)
                ? items.map((item) => ({ ...item, request: applyDefaults(defaults, item?.request, undefined).request }))
                : items;
            if (typeof options !== 'object' || options === null || Array.isArray(options)) {
                throw new InvalidRequestError('Invalid value for option "options"');
            }
            const { config } = applyDefaults(defaults, { messages: [] }, options.config);
            return batch.submit(filled, { ...options, ...(config === undefined ? {} : { config }) });
        },
        check: batch.check,
        cancel: batch.cancel,
        removeFiles: batch.removeFiles,
    });
}

/** Creates `listModels` method fetching all pages of available models. */
function createListModels(ai: GoogleGenAI): GenaiClient['listModels'] {
    return guard(async function listModels(options = {}) {
        checkOptionKeys(options, LIST_OPTION_KEYS);
        const { signal } = options;
        const models: GenaiModelInfo[] = [];
        try {
            const pager = await ai.models.list({
                config: { pageSize: LIST_PAGE_SIZE, ...(signal ? { abortSignal: signal } : {}) },
            });
            for await (const model of pager) {
                const info = toModelInfo(model);
                if (info !== undefined) {
                    models.push(info);
                }
            }
        } catch (error) {
            return { ok: false, error: toGenaiError(error, signal) };
        }
        return { ok: true, data: models };
    });
}

/** Converts SDK Model descriptor into `GenaiModelInfo`. */
function toModelInfo(model: Model | null | undefined): GenaiModelInfo | undefined {
    if (!model?.name) {
        return undefined;
    }
    return {
        id: model.name.replace(MATCH_MODEL_PREFIX, ''),
        ...(model.displayName ? { displayName: model.displayName } : {}),
        ...(model.description ? { description: model.description } : {}),
        ...(model.inputTokenLimit === undefined ? {} : { inputTokenLimit: model.inputTokenLimit }),
        ...(model.outputTokenLimit === undefined ? {} : { outputTokenLimit: model.outputTokenLimit }),
        actions: model.supportedActions ?? [],
        ...(model.thinking === undefined ? {} : { thinking: model.thinking }),
    };
}

/** Resolves endpoint URL for specified Vertex AI location. */
function vertexBaseUrl(location: string): string {
    if (location === 'global') {
        return 'https://aiplatform.googleapis.com/';
    }
    if (MULTI_REGIONAL_LOCATIONS.has(location)) {
        return `https://aiplatform.${location}.rep.googleapis.com/`;
    }
    return `https://${location}-aiplatform.googleapis.com/`;
}

/**
 * Validates client options.
 * @throws {TypeError} If option keys or values are unsupported.
 */
function validateOptions(options: GenaiClientOptions): void {
    if (typeof options !== 'object' || options === null) {
        throw new TypeError('Invalid value for option "options"');
    }
    for (const key of Object.keys(options)) {
        if (!OPTION_KEYS.has(key)) {
            throw new TypeError(`Option "${key}" is not supported`);
        }
    }
    if (options.apiKey !== undefined && (typeof options.apiKey !== 'string' || options.apiKey === '')) {
        throw new TypeError('Invalid value for option "apiKey"');
    }
    const { timeoutMs } = options;
    if (timeoutMs !== undefined && (!Number.isInteger(timeoutMs) || timeoutMs <= 0)) {
        throw new TypeError('Invalid value for option "timeoutMs"');
    }
    if (options.baseUrl !== undefined && !URL.canParse(options.baseUrl)) {
        throw new TypeError('Invalid value for option "baseUrl"');
    }
    if (options.env !== undefined && (typeof options.env !== 'object' || options.env === null)) {
        throw new TypeError('Invalid value for option "env"');
    }
    if (options.defaults !== undefined) {
        validateDefaults(options.defaults);
    }
    const { batchWaitSec } = options;
    if (
        batchWaitSec !== undefined &&
        !(typeof batchWaitSec === 'number' && batchWaitSec >= MIN_BATCH_WAIT_SEC && batchWaitSec <= MAX_BATCH_WAIT_SEC)
    ) {
        throw new TypeError(
            `Invalid value for option "batchWaitSec": ${MIN_BATCH_WAIT_SEC} to ${MAX_BATCH_WAIT_SEC} seconds`,
        );
    }
    const { vertex } = options;
    if (vertex === undefined || typeof vertex === 'boolean') {
        return;
    }
    if (typeof vertex !== 'object' || vertex === null) {
        throw new TypeError('Invalid value for option "vertex"');
    }
    for (const key of Object.keys(vertex)) {
        if (!VERTEX_KEYS.has(key)) {
            throw new TypeError(`Option "vertex.${key}" is not supported`);
        }
    }
    for (const key of ['project', 'location', 'bucket', 'credentialsFile'] as const) {
        const value = vertex[key];
        if (value !== undefined && (typeof value !== 'string' || value === '')) {
            throw new TypeError(`Invalid value for option "vertex.${key}"`);
        }
    }
    if (vertex.storageUrl !== undefined && !URL.canParse(vertex.storageUrl)) {
        throw new TypeError('Invalid value for option "vertex.storageUrl"');
    }
}

/**
 * Resolves Vertex target options with environment defaults.
 * @throws {TypeError} If project is missing/invalid, location or bucket name is invalid, or credentials file missing.
 */
function resolveVertex(vertex: GenaiVertexOptions, env: Readonly<Record<string, string | undefined>>): ResolvedVertex {
    const project = vertex.project ?? env[VERTEX_ENV.project];
    if (!project) {
        throw new TypeError(`Invalid value for option "vertex.project": set it or env ${VERTEX_ENV.project}`);
    }
    if (!MATCH_PROJECT.test(project)) {
        throw new TypeError('Invalid value for option "vertex.project": a Google Cloud project ID');
    }
    const location = vertex.location ?? (env[VERTEX_ENV.location] || DEFAULT_LOCATION);
    if (!MATCH_LOCATION.test(location)) {
        throw new TypeError('Invalid value for option "vertex.location": a Vertex AI location such as global');
    }
    const bucket = vertex.bucket ?? (env[VERTEX_ENV.bucket] || undefined);
    if (bucket !== undefined && !MATCH_BUCKET_NAME.test(bucket)) {
        throw new TypeError('Invalid value for option "vertex.bucket": a bucket name without gs://');
    }
    const resolved: ResolvedVertex = {
        project,
        location,
        ...(bucket === undefined ? {} : { bucket }),
        ...(vertex.storageUrl === undefined ? {} : { storageUrl: vertex.storageUrl }),
    };
    if (vertex.authClient !== undefined) {
        return { ...resolved, authClient: vertex.authClient };
    }
    const file = vertex.credentialsFile ?? credentialsFile(env);
    if (!isNonEmptyFile(file)) {
        throw new TypeError(
            `Invalid value for option "vertex.credentialsFile": ${file} is missing or empty; run "gcloud auth application-default login" or set env ${VERTEX_ENV.credentialsFile}`,
        );
    }
    return { ...resolved, credentialsFile: file };
}

/** Checks whether file exists, is regular file, and is non-empty without throwing. */
function isNonEmptyFile(file: string): boolean {
    try {
        const stats = statSync(file, { throwIfNoEntry: false });
        return stats?.isFile() === true && stats.size > 0;
    } catch {
        return false;
    }
}
