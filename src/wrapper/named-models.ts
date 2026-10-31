// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenerateContentConfig } from '@google/genai';
import {
    type Bucket,
    createConcurrencyBucket,
    createQueue,
    createRPD,
    createRPM,
    createSlidingWindow,
    createTPD,
    createTPM,
    type JobError,
    type JobErrorType,
    type JobOutcome,
    type JobResult,
    type LimiterError,
    type PushOptions,
} from 'smart-limit';
import { guard, guardBatch } from '../core/answer.js';
import { createGenaiClient } from '../core/client.js';
import {
    applyDefaults,
    checkOptionKeys,
    copyPlain,
    InvalidRequestError,
    toGeminiCall,
    toSettings,
    validateDefaults,
} from '../core/request.js';
import type {
    GenaiClient,
    GenaiClientOptions,
    GenaiContentPart,
    GenaiDefaults,
    GenaiError,
    GenaiLimiterOptions,
    GenaiLimits,
    GenaiMessage,
    GenaiModel,
    GenaiModelBatch,
    GenaiModelInput,
    GenaiModelOptions,
    GenaiModelSpec,
    GenaiModels,
    GenaiModelsOptions,
    GenaiOutput,
    GenaiRequest,
    GenaiSettings,
    Result,
} from '../types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Internal execution context for queued generation job. */
type Job = {
    request: GenaiRequest;
    config: GenerateContentConfig | undefined;
    attempts: number;
    lastError?: GenaiError;
};

/** Resolved limits and rate limiting configuration for named model. */
type ModelLimits = {
    buckets: Bucket[];
    concurrency: number;
    limiter: GenaiLimiterOptions;
    charsPerToken: number;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Supported option keys for `createGenaiModels`. */
const OPTION_KEYS: ReadonlySet<string> = new Set([
    'client',
    'models',
    'defaults',
    'sharedLimits',
    'limiterOptions',
    'charsPerToken',
]);

/** Supported property keys in `GenaiModelSpec`. */
const SPEC_KEYS: ReadonlySet<string> = new Set(['model', 'defaults', 'modelLimits', 'charsPerToken']);

/** Supported rate limit property keys in `GenaiLimits`. */
const LIMIT_KEYS: ReadonlySet<string> = new Set([
    'concurrency',
    'rps',
    'rpm',
    'rpd',
    'tpm',
    'tpd',
    'intervalMs',
    'buckets',
]);

/** Supported limiter configuration keys in `GenaiLimiterOptions`. */
const LIMITER_KEYS: ReadonlySet<string> = new Set(['maxRetries', 'retryDelayMs', 'maxRetryDelayMs', 'maxWaitMs']);

/** Property names reserved on `createGenaiModels` result object. */
const RESERVED_NAMES: ReadonlySet<string> = new Set([
    'client',
    'listModels',
    'then',
    'toJSON',
    ...Object.getOwnPropertyNames(Object.prototype),
]);

/** Milliseconds per second. */
const MS_PER_SECOND = 1000;

/** Default unbounded concurrency limit value. */
const UNLIMITED_CONCURRENCY = Number.POSITIVE_INFINITY;

/** Default character-to-token estimation divisor. */
const DEFAULT_CHARS_PER_TOKEN = 4;

/** Fixed token allocation estimate per input image. */
const IMAGE_TOKENS = 1120;

/** Fallback token allocation estimate for unconstrained generation calls. */
const DEFAULT_OUTPUT_TOKENS = 1000;

/** Minimum HTTP status code indicating server error. */
const MIN_SERVER_ERROR_STATUS = 500;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Named models on one client**: wraps client with names, defaults, and smart-limit queues.
 *
 * - binds named functions with configured model ids and merged defaults
 * - applies shared and per-model rate limits
 * - manages request retries and exponential backoff
 * - handles batch calls via separate concurrency tracking
 * @throws {TypeError} If options, models, limits, or defaults are invalid.
 */
export function createGenaiModels<const N extends string>(options: GenaiModelsOptions<N>): GenaiModels<N> {
    checkOptionKeys(options, OPTION_KEYS);
    const client = toClient(options.client);
    const limiter = validateLimiterOptions(options.limiterOptions);
    const charsPerToken = validateCharsPerToken(options.charsPerToken, 'charsPerToken') ?? DEFAULT_CHARS_PER_TOKEN;
    const shared = createBuckets(options.sharedLimits, 'sharedLimits', true);
    if (typeof options.models !== 'object' || options.models === null || Object.keys(options.models).length === 0) {
        throw new TypeError('Invalid value for option "models"');
    }
    if (options.defaults !== undefined) {
        rejectModelDefault(options.defaults, 'defaults');
        validateDefaults(options.defaults);
    }
    const result: Record<string, unknown> = { client, listModels: client.listModels };
    for (const [name, entry] of Object.entries<string | GenaiModelSpec>(options.models)) {
        if (RESERVED_NAMES.has(name)) {
            throw new TypeError(`Invalid value for option "models.${name}": ${name} is reserved`);
        }
        const spec = toSpec(entry, name);
        const defaults: GenaiDefaults = mergeDefaults(options.defaults, spec.defaults, spec.model);
        checkMergedDefaults(defaults, name);
        const own = createBuckets(spec.modelLimits, `models.${name}.modelLimits`, false);
        const model = createModel(client, name, spec.model, defaults, {
            buckets: [...shared, ...own],
            concurrency: spec.modelLimits?.concurrency ?? UNLIMITED_CONCURRENCY,
            limiter,
            charsPerToken: spec.charsPerToken ?? charsPerToken,
        });
        // defineProperty, so every name becomes an own member, whatever setters an object may have
        Object.defineProperty(result, name, { value: model, enumerable: true });
    }
    return result as GenaiModels<N>;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * Creates own client from client options.
 * @throws {TypeError} If option is a client instance or has `defaults`.
 */
function toClient(options: Omit<GenaiClientOptions, 'defaults'> | undefined): GenaiClient {
    if (typeof options === 'object' && options !== null) {
        if (typeof (options as Partial<GenaiClient>).generate === 'function') {
            throw new TypeError('Invalid value for option "client": client options, not a client');
        }
        if ('defaults' in options) {
            throw new TypeError('Option "client.defaults" is not supported: request defaults go into "defaults"');
        }
    }
    return createGenaiClient(options);
}

/**
 * Validates limiter options.
 * @throws {TypeError} If limiter option keys or values are unsupported.
 */
function validateLimiterOptions(limiterOptions: GenaiLimiterOptions | undefined): GenaiLimiterOptions {
    if (limiterOptions === undefined) {
        return {};
    }
    checkOptionKeys(limiterOptions, LIMITER_KEYS, 'limiterOptions.');
    for (const [key, value] of Object.entries(limiterOptions)) {
        if (value !== undefined && (!Number.isInteger(value) || value < 0)) {
            throw new TypeError(`Invalid value for option "limiterOptions.${key}"`);
        }
    }
    return limiterOptions;
}

/**
 * Validates characters-per-token divisor value.
 * @throws {TypeError} If divisor is not a positive finite number.
 */
function validateCharsPerToken(charsPerToken: number | undefined, option: string): number | undefined {
    if (
        charsPerToken !== undefined &&
        (typeof charsPerToken !== 'number' || !(charsPerToken > 0) || !Number.isFinite(charsPerToken))
    ) {
        throw new TypeError(`Invalid value for option "${option}"`);
    }
    return charsPerToken;
}

/**
 * Normalizes string or spec entry into `GenaiModelSpec`.
 * @throws {TypeError} If model spec configuration is invalid.
 */
function toSpec(entry: string | GenaiModelSpec, name: string): GenaiModelSpec {
    if (typeof entry === 'string') {
        if (entry === '') {
            throw new TypeError(`Invalid value for option "models.${name}"`);
        }
        return { model: entry };
    }
    checkOptionKeys(entry, SPEC_KEYS, `models.${name}.`);
    if (typeof entry.model !== 'string' || entry.model === '') {
        throw new TypeError(`Invalid value for option "models.${name}.model"`);
    }
    if (entry.defaults !== undefined) {
        rejectModelDefault(entry.defaults, `models.${name}.defaults`);
        validateDefaults(entry.defaults, `models.${name}.defaults`);
    }
    validateCharsPerToken(entry.charsPerToken, `models.${name}.charsPerToken`);
    return entry;
}

/**
 * Rejects model property in defaults.
 * @throws {TypeError} If defaults attempt to override fixed model id.
 */
function rejectModelDefault(defaults: Omit<GenaiDefaults, 'model'>, option: string): void {
    if (typeof defaults === 'object' && defaults !== null && 'model' in defaults) {
        throw new TypeError(`Option "${option}.model" is not supported`);
    }
}

/** Merges shared and per-model defaults with fixed model identifier. */
function mergeDefaults(
    shared: Omit<GenaiDefaults, 'model'> | undefined,
    own: Omit<GenaiDefaults, 'model'> | undefined,
    model: string,
): GenaiDefaults {
    const { config: ownConfig, ...ownFields } = own ?? {};
    const { request, config } = applyDefaults(shared, { messages: [], ...ownFields }, ownConfig);
    const { messages: _messages, ...fields } = request;
    // A copy, so changing a defaults object later changes no call.
    return copyPlain({ ...fields, model, ...(config === undefined ? {} : { config }) });
}

/**
 * Checks merged defaults against model of named function.
 * @throws {TypeError} If defaults combination fails for that model.
 */
function checkMergedDefaults(defaults: GenaiDefaults, name: string): void {
    const { config, ...fields } = defaults;
    try {
        toGeminiCall({ ...fields, messages: [{ role: 'user', content: 'x' }] } as GenaiRequest, config);
    } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        throw new TypeError(`Invalid value for option "models.${name}.defaults": ${reason}`);
    }
}

/**
 * Instantiates smart-limit bucket instances based on configured limits.
 * @throws {TypeError} If limits configuration is invalid.
 */
function createBuckets(limits: GenaiLimits | undefined, option: string, withConcurrency: boolean): Bucket[] {
    if (limits === undefined) {
        return [];
    }
    checkOptionKeys(limits, LIMIT_KEYS, `${option}.`);
    for (const key of ['concurrency', 'rps', 'rpm', 'rpd', 'tpm', 'tpd', 'intervalMs'] as const) {
        const value = limits[key];
        if (value !== undefined && (!Number.isInteger(value) || value < 1)) {
            throw new TypeError(`Invalid value for option "${option}.${key}"`);
        }
    }
    if (limits.buckets !== undefined && !Array.isArray(limits.buckets)) {
        throw new TypeError(`Invalid value for option "${option}.buckets"`);
    }
    const buckets: Bucket[] = [
        ...(limits.concurrency === undefined || !withConcurrency ? [] : [createConcurrencyBucket(limits.concurrency)]),
        ...(limits.rps === undefined ? [] : [createSlidingWindow({ requests: limits.rps, perMs: MS_PER_SECOND })]),
        ...(limits.rpm === undefined ? [] : [createRPM(limits.rpm)]),
        ...(limits.rpd === undefined ? [] : [createRPD(limits.rpd)]),
        ...(limits.tpm === undefined ? [] : [createTPM(limits.tpm)]),
        ...(limits.tpd === undefined ? [] : [createTPD(limits.tpd)]),
        ...(limits.intervalMs === undefined ? [] : [createSlidingWindow({ requests: 1, perMs: limits.intervalMs })]),
        ...(limits.buckets ?? []),
    ];
    const invalid = buckets.find((bucket) => bucket.configError !== undefined);
    if (invalid !== undefined) {
        throw new TypeError(`Invalid value for option "${option}": ${invalid.configError?.message ?? ''}`);
    }
    return buckets;
}

/** Creates callable model function with attached limiter queues and batch interface. */
function createModel(
    client: GenaiClient,
    name: string,
    id: string,
    defaults: GenaiDefaults,
    limits: ModelLimits,
): GenaiModel {
    const queue = createQueue<Job, GenaiOutput>({
        ...limits.limiter,
        concurrency: limits.concurrency,
        maxBufferSize: Number.POSITIVE_INFINITY,
        buckets: limits.buckets,
        processJob: (job, context) => attempt(client, job, context.signal, false),
    });
    const batchQueue = createQueue<Job, GenaiOutput>({
        ...limits.limiter,
        concurrency: limits.concurrency,
        maxBufferSize: Number.POSITIVE_INFINITY,
        processJob: (job, context) => attempt(client, job, context.signal, true),
    });
    const call = guard(async function call(
        input: GenaiModelInput,
        options: GenaiModelOptions = {},
    ): Promise<Result<GenaiOutput, GenaiError>> {
        if (typeof options !== 'object' || options === null || Array.isArray(options)) {
            throw new InvalidRequestError('Invalid value for the options: an object');
        }
        const { signal, config, ...fields } = options;
        if (signal !== undefined && !(signal instanceof AbortSignal)) {
            throw new InvalidRequestError('Invalid value for option "signal"');
        }
        for (const key of ['model', 'messages']) {
            if (key in fields) {
                throw new InvalidRequestError(`Option "${key}" is not supported: the name and input set it`);
            }
        }
        const request = { ...fields, model: id, messages: toMessages(input) } as GenaiRequest;
        const filled = applyDefaults(defaults, request, config);
        toGeminiCall(filled.request, filled.config);
        const own = toSettings(filled.request, filled.config, id);
        const job: Job = { request: filled.request, config: filled.config, attempts: 0 };
        const isBatch = job.request.batch === true;
        if (isBatch && client.batch !== undefined) {
            // too large for a batch: a failed request check, so no settings and no attempts
            const bytes = client.batch.requestBytes(job.request, job.config);
            if (!bytes.ok) {
                return bytes;
            }
            if (bytes.data > client.batch.maxBytes) {
                const limit = client.batch.maxBytes;
                throw new InvalidRequestError(
                    `Request of ${bytes.data} bytes exceeds the batch limit of ${limit} bytes`,
                );
            }
        }
        const push: PushOptions = {};
        if (!isBatch) {
            push.cost = { tokens: estimateTokens(job.request, limits.charsPerToken) };
        }
        if (signal !== undefined) {
            push.signal = signal;
        }
        const outcome = await (isBatch ? batchQueue : queue).push(job, push);
        return withCall(toResult(outcome, job), name, own, job.attempts);
    });
    const model = call as GenaiModel;
    Object.defineProperty(model, 'id', { value: id, enumerable: true });
    if (client.batch !== undefined) {
        Object.defineProperty(model, 'batch', { value: createModelBatch(client, id, defaults), enumerable: true });
    }
    return model;
}

/**
 * Converts varied model input types into message array.
 * @throws {InvalidRequestError} If input is empty or has invalid structure.
 */
function toMessages(input: GenaiModelInput): GenaiMessage[] {
    if (typeof input === 'string') {
        return [{ role: 'user', content: input }];
    }
    if (Array.isArray(input) && input.length > 0) {
        if (input.every((item) => isObject(item) && 'role' in item)) {
            return [...(input as unknown as readonly GenaiMessage[])];
        }
        if (input.every((item) => isObject(item) && 'type' in item)) {
            return [{ role: 'user', content: [...(input as unknown as readonly GenaiContentPart[])] }];
        }
    }
    throw new InvalidRequestError('Invalid value for the input: a string, content parts or messages');
}

/**
 * Prepares batch item request with fixed model identifier.
 * @throws {InvalidRequestError} If request format is invalid or specifies model.
 */
function toBatchRequest(request: Omit<GenaiRequest, 'model'>, id: string): GenaiRequest {
    if (!isObject(request)) {
        throw new InvalidRequestError('Invalid value for option "request"');
    }
    if ('model' in request) {
        throw new InvalidRequestError('Option "model" is not supported: the model of a name is fixed');
    }
    return { ...request, model: id };
}

/** Checks whether value is non-null object. */
function isObject(value: unknown): value is object {
    return typeof value === 'object' && value !== null;
}

/** Wraps client batch operations with model identifier and defaults. */
function createModelBatch(client: GenaiClient, id: string, withBatch: GenaiDefaults): GenaiModelBatch {
    const batch = client.batch as NonNullable<GenaiClient['batch']>;
    const { batch: _batch, ...defaults } = withBatch;
    return guardBatch({
        maxBytes: batch.maxBytes,
        requestBytes(request, config) {
            const filled = applyDefaults(defaults, toBatchRequest(request, id), config);
            return batch.requestBytes(filled.request, filled.config);
        },
        submit(items, options = {}) {
            if (!Array.isArray(items)) {
                throw new InvalidRequestError('Invalid value for option "items"');
            }
            if (typeof options !== 'object' || options === null || Array.isArray(options)) {
                throw new InvalidRequestError('Invalid value for option "options"');
            }
            const filled = applyDefaults(defaults, { messages: [] }, options.config).config;
            return batch.submit(
                items.map((item) => ({
                    key: item?.key,
                    request: applyDefaults(defaults, toBatchRequest(item?.request, id), undefined).request,
                })),
                { ...options, ...(filled === undefined ? {} : { config: filled }) },
            );
        },
        check: batch.check,
        cancel: batch.cancel,
    });
}

/** Executes single generation attempt within limiter queue context. */
async function attempt(
    client: GenaiClient,
    job: Job,
    signal: AbortSignal,
    isBatch: boolean,
): Promise<JobResult<GenaiOutput>> {
    const generate = guard(() =>
        client.generate(job.request, { signal, ...(job.config === undefined ? {} : { config: job.config }) }),
    );
    // Counted before the call: an abort resolves the push before generate returns.
    job.attempts += 1;
    const result = await generate();
    // without a count: 1 call, or none for a request the client refused (too large for a batch)
    const fallback = !result.ok && result.error.code === 'invalid_request' ? 0 : 1;
    job.attempts += ((result.ok ? result.data.attempts : result.error.attempts) ?? fallback) - 1;
    if (result.ok) {
        const tokens = isBatch ? undefined : usedTokens(result.data.usage);
        return tokens === undefined ? result : { ...result, usage: { tokens } };
    }
    const { error } = result;
    job.lastError = error;
    const type = isBatch ? batchErrorType(error) : jobErrorType(error);
    const jobError: JobError = { type, message: error.message, code: error.code, cause: error };
    if (error.retryAfterMs !== undefined) {
        jobError.retryAfterMs = error.retryAfterMs;
    }
    const tokens = error.usage === undefined || isBatch ? undefined : usedTokens(error.usage);
    if (tokens !== undefined) {
        jobError.usage = { tokens };
    } else if (error.code === 'rate_limited' && !isBatch) {
        // a rejected request ran no model: give its token booking back
        jobError.usage = { tokens: 0 };
    }
    return { ok: false, error: jobError };
}

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

/** Maps batch GenaiError to smart-limit JobErrorType. */
function batchErrorType(error: GenaiError): JobErrorType {
    return error.retryable ? 'retry_backoff' : 'fatal';
}

/** Attaches effective settings and attempt count to result. */
function withCall(
    result: Result<GenaiOutput, GenaiError>,
    name: string,
    own: GenaiSettings,
    attempts: number,
): Result<GenaiOutput, GenaiError> {
    const { name: _, ...reported } = (result.ok ? result.data.settings : result.error.settings) ?? own;
    const settings = { name, ...reported };
    if (result.ok) {
        return { ok: true, data: { ...result.data, settings, attempts } };
    }
    return { ok: false, error: { ...result.error, settings, attempts } };
}

/** Translates smart-limit job outcome into GenaiResult. */
function toResult(outcome: JobOutcome<GenaiOutput>, job: Job): Result<GenaiOutput, GenaiError> {
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

/** Estimates token count from request character length, images, and output limits. */
function estimateTokens(request: GenaiRequest, charsPerToken: number): number {
    let chars = request.tools === undefined ? 0 : JSON.stringify(request.tools).length;
    let images = 0;
    for (const message of request.messages) {
        if (message.role === 'assistant') {
            for (const call of message.tool_calls ?? []) {
                chars += call.function.name.length + call.function.arguments.length;
            }
        }
        if (message.content === null) {
            continue;
        }
        if (typeof message.content === 'string') {
            chars += message.content.length;
            continue;
        }
        for (const part of message.content) {
            if (part.type === 'text') {
                chars += part.text.length;
            } else {
                images += 1;
            }
        }
    }
    const output = request.max_tokens ?? request.max_completion_tokens ?? DEFAULT_OUTPUT_TOKENS;
    return Math.ceil(chars / charsPerToken) + images * IMAGE_TOKENS + output;
}

/** Computes total tokens consumed from usage metadata. */
function usedTokens(usage: GenaiOutput['usage']): number | undefined {
    if (usage.inputTokens === null && usage.outputTokens === null && usage.reasoningTokens === null) {
        return undefined;
    }
    return (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0) + (usage.reasoningTokens ?? 0);
}
