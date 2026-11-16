// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenerateContentConfig } from '@google/genai';
import {
    type Bucket,
    type Cost,
    createConcurrencyBucket,
    createRPD,
    createRPM,
    createSlidingWindow,
    createTPD,
    createTPM,
    type JobErrorType,
    type PushOptions,
} from 'smart-limit';
import { guard, guardBatch } from '../core/answer.js';
import { createGenaiClient } from '../core/client.js';
import { DEFAULT_DECIDE_MODEL, toDecideBody } from '../core/decide.js';
import { checkEmbedRequest, type EmbedText, embedUnsupported, toEmbedCall } from '../core/embed.js';
import {
    applyBatchDefaults,
    applyDefaults,
    checkOptionKeys,
    copyPlain,
    InvalidRequestError,
    toGeminiCall,
    toSettings,
    validateDefaults,
} from '../core/request.js';
import type {
    GenaiAnyModelSpec,
    GenaiClient,
    GenaiClientOptions,
    GenaiContentPart,
    GenaiDecideInstructions,
    GenaiDecideModel,
    GenaiDecideModelSpec,
    GenaiDecideOutput,
    GenaiDecideQuestion,
    GenaiDecideRequest,
    GenaiDefaults,
    GenaiEmbedDefaults,
    GenaiEmbedInput,
    GenaiEmbedModel,
    GenaiEmbedModelOptions,
    GenaiEmbedModelSpec,
    GenaiEmbedOutput,
    GenaiEmbedRequest,
    GenaiError,
    GenaiLimiterOptions,
    GenaiLimits,
    GenaiMessage,
    GenaiModel,
    GenaiModelBatch,
    GenaiModelInput,
    GenaiModelOptions,
    GenaiModels,
    GenaiModelsOptions,
    GenaiOutput,
    GenaiRequest,
    GenaiSettings,
    Result,
} from '../types.js';
import { createRunQueue, type QueueJob } from './queue.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Internal execution context for queued generation job. */
type Job = QueueJob & {
    request: GenaiRequest;
    config: GenerateContentConfig | undefined;
};

/** Internal execution context for queued embedding job. */
type EmbedJob = QueueJob & { request: GenaiEmbedRequest };

/** Internal execution context for queued decision job. */
type DecideJob = QueueJob & { request: GenaiDecideRequest };

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

/** Supported spec property keys per `kind` (missing `kind`: `generate`). */
const SPEC_KEYS: Readonly<Record<NonNullable<GenaiAnyModelSpec['kind']>, ReadonlySet<string>>> = {
    generate: new Set(['kind', 'model', 'defaults', 'modelLimits', 'charsPerToken']),
    embed: new Set(['kind', 'model', 'defaults', 'modelLimits', 'charsPerToken']),
    decide: new Set(['kind', 'model', 'modelLimits', 'charsPerToken']),
};

/** Supported keys of `GenaiDecideModel` call options. */
const DECIDE_OPTION_KEYS: ReadonlySet<string> = new Set(['signal']);

/** Message of a decision on Vertex AI, as from the client. */
const DECIDE_UNSUPPORTED = 'Decisions API is not supported on Vertex AI: Gemini Developer API only';

/** Supported keys of `GenaiEmbedDefaults` and `GenaiEmbedModelOptions`. */
const EMBED_DEFAULT_KEYS: ReadonlySet<string> = new Set(['taskType', 'dimensions']);
const EMBED_OPTION_KEYS: ReadonlySet<string> = new Set([...EMBED_DEFAULT_KEYS, 'signal']);

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

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Named models on one client**: wraps client with names, defaults, and smart-limit queues.
 *
 * - binds named functions with configured model ids and merged defaults
 * - applies shared and per-model rate limits
 * - manages request retries and exponential backoff
 * - batch calls: own queue without concurrency cap or rate limits; `limiterOptions` still apply
 * - `kind: 'embed'` entries: embedding models; own defaults only, same shared buckets
 * - `kind: 'decide'` entries: decision models; no defaults, same shared buckets
 * @throws {TypeError} If options, models, limits, or defaults are invalid.
 */
export function createGenaiModels<const M extends string | Readonly<Record<string, string | GenaiAnyModelSpec>>>(
    options: GenaiModelsOptions<M>,
): GenaiModels<M> {
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
    for (const [name, entry] of Object.entries<string | GenaiAnyModelSpec>(options.models)) {
        if (RESERVED_NAMES.has(name)) {
            throw new TypeError(`Invalid value for option "models.${name}": ${name} is reserved`);
        }
        const spec = toSpec(entry, name);
        const own = createBuckets(spec.modelLimits, `models.${name}.modelLimits`, false);
        const limits: ModelLimits = {
            buckets: [...shared, ...own],
            concurrency: spec.modelLimits?.concurrency ?? UNLIMITED_CONCURRENCY,
            limiter,
            charsPerToken: spec.charsPerToken ?? charsPerToken,
        };
        let model: GenaiModel | GenaiEmbedModel | GenaiDecideModel;
        if (spec.kind === 'embed') {
            model = createEmbedModel(client, name, spec, limits);
        } else if (spec.kind === 'decide') {
            model = createDecideModel(client, name, spec, limits);
        } else {
            const defaults: GenaiDefaults = mergeDefaults(options.defaults, spec.defaults, spec.model);
            checkMergedDefaults(defaults, name);
            model = createModel(client, name, spec.model, defaults, limits);
        }
        // defineProperty, so every name becomes an own member, whatever setters an object may have
        Object.defineProperty(result, name, { value: model, enumerable: true });
    }
    return result as GenaiModels<M>;
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
 * Validates limiter options: non-negative integers; `maxWaitMs` also `Infinity` (no wait limit).
 * @throws {TypeError} If limiter option keys or values are unsupported.
 */
function validateLimiterOptions(limiterOptions: GenaiLimiterOptions | undefined): GenaiLimiterOptions {
    if (limiterOptions === undefined) {
        return {};
    }
    checkOptionKeys(limiterOptions, LIMITER_KEYS, 'limiterOptions.');
    for (const [key, value] of Object.entries(limiterOptions)) {
        if (key === 'maxWaitMs' && value === Number.POSITIVE_INFINITY) {
            continue;
        }
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
 * Normalizes string or spec entry into spec of its `kind`.
 * @throws {TypeError} If model spec configuration is invalid.
 */
function toSpec(entry: string | GenaiAnyModelSpec, name: string): GenaiAnyModelSpec {
    if (typeof entry === 'string') {
        if (entry === '') {
            throw new TypeError(`Invalid value for option "models.${name}"`);
        }
        return { model: entry };
    }
    const kind: unknown = isObject(entry) ? entry.kind : undefined;
    if (kind !== undefined && (typeof kind !== 'string' || !Object.hasOwn(SPEC_KEYS, kind))) {
        throw new TypeError(`Invalid value for option "models.${name}.kind"`);
    }
    checkOptionKeys(entry, SPEC_KEYS[(kind ?? 'generate') as keyof typeof SPEC_KEYS], `models.${name}.`);
    if (entry.kind === 'decide') {
        // optional; if given, as checked by the client
        if (entry.model !== undefined && (typeof entry.model !== 'string' || entry.model.trim() === '')) {
            throw new TypeError(`Invalid value for option "models.${name}.model"`);
        }
    } else if (typeof entry.model !== 'string' || entry.model === '') {
        throw new TypeError(`Invalid value for option "models.${name}.model"`);
    }
    if (entry.kind === 'embed') {
        checkEmbedDefaults(entry, name);
    } else if (entry.kind !== 'decide' && entry.defaults !== undefined) {
        rejectModelDefault(entry.defaults, `models.${name}.defaults`);
        validateDefaults(entry.defaults, `models.${name}.defaults`);
    }
    validateCharsPerToken(entry.charsPerToken, `models.${name}.charsPerToken`);
    return entry;
}

/**
 * Checks keys and values of embedding model defaults with a dummy request.
 * @throws {TypeError} If defaults are no object, have unsupported keys or invalid values.
 */
function checkEmbedDefaults(spec: GenaiEmbedModelSpec, name: string): void {
    if (spec.defaults === undefined) {
        return;
    }
    checkOptionKeys(spec.defaults, EMBED_DEFAULT_KEYS, `models.${name}.defaults.`);
    try {
        checkEmbedRequest({ ...spec.defaults, model: spec.model, input: 'x' });
    } catch (error) {
        const reason = error instanceof Error ? error.message : String(error);
        throw new TypeError(`Invalid value for option "models.${name}.defaults": ${reason}`);
    }
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
        ...(limits.intervalMs === undefined
            ? []
            : [perCall(createSlidingWindow({ requests: 1, perMs: limits.intervalMs }))]),
        ...(limits.buckets ?? []),
    ];
    const invalid = buckets.find((bucket) => bucket.configError !== undefined);
    if (invalid !== undefined) {
        throw new TypeError(`Invalid value for option "${option}": ${invalid.configError?.message ?? ''}`);
    }
    return buckets;
}

/**
 * **Wraps bucket to count one request per call**: for `intervalMs`.
 *
 * - `requests` of booked and settled costs clamped to 1; other units unchanged
 * - keeps multi-input embed calls from `COST_EXCEEDS_LIMIT` on a 1-request window
 */
function perCall(bucket: Bucket): Bucket {
    function once(cost: Cost): Cost {
        return cost.requests === undefined ? cost : { ...cost, requests: Math.min(cost.requests, 1) };
    }
    return {
        ...(bucket.configError === undefined ? {} : { configError: bucket.configError }),
        getWaitMs: (cost, now) => bucket.getWaitMs(once(cost), now),
        reserve: (cost, now) => bucket.reserve(once(cost), now),
        settle: (reservation, actualCost, now) => bucket.settle(reservation, once(actualCost), now),
        ...(bucket.subscribe === undefined ? {} : { subscribe: bucket.subscribe.bind(bucket) }),
    };
}

/** Creates callable model function with attached limiter queues and batch interface. */
function createModel(
    client: GenaiClient,
    name: string,
    id: string,
    defaults: GenaiDefaults,
    limits: ModelLimits,
): GenaiModel {
    function execute(job: Job, signal: AbortSignal): Promise<Result<GenaiOutput, GenaiError>> {
        return client.generate(job.request, { signal, ...(job.config === undefined ? {} : { config: job.config }) });
    }
    const queue = createRunQueue<Job, GenaiOutput>({
        buckets: limits.buckets,
        concurrency: limits.concurrency,
        limiter: limits.limiter,
        execute,
        usedTokens: (result) => usedTokens(result.ok ? result.data.usage : result.error.usage),
        countAttempts,
    });
    // no concurrency cap: a held slot lasts until the job is final, so a cap would split and serialize jobs
    const batchQueue = createRunQueue<Job, GenaiOutput>({
        concurrency: UNLIMITED_CONCURRENCY,
        limiter: limits.limiter,
        execute,
        errorType: batchErrorType,
        countAttempts,
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
        const result = await (isBatch ? batchQueue : queue).run(job, push);
        return withCall(result, name, own, job.attempts);
    });
    const model = call as GenaiModel;
    Object.defineProperty(model, 'id', { value: id, enumerable: true });
    if (client.batch !== undefined) {
        Object.defineProperty(model, 'batch', { value: createModelBatch(client, id, defaults), enumerable: true });
    }
    return model;
}

/**
 * **Callable embedding model with attached limiter queue**: no `batch`.
 *
 * - request checks, abort and Vertex `unsupported` answered before enqueue, without booking
 * - books one request per input text (as API counts them; `intervalMs`, `concurrency` per call)
 * - books estimated input tokens; no settlement (no usage), refund on `rate_limited`
 */
function createEmbedModel(
    client: GenaiClient,
    name: string,
    spec: GenaiEmbedModelSpec,
    limits: ModelLimits,
): GenaiEmbedModel {
    const id = spec.model;
    // a copy, so changing the spec later changes no call
    const defaults: GenaiEmbedDefaults = { ...spec.defaults };
    const queue = createRunQueue<EmbedJob, GenaiEmbedOutput>({
        buckets: limits.buckets,
        concurrency: limits.concurrency,
        limiter: limits.limiter,
        execute: (job, signal) => client.embed(job.request, { signal }),
        // no usage to settle, but keeps the refund of a rate_limited booking
        usedTokens: () => undefined,
    });
    const call = guard(async function embed(
        input: string | readonly GenaiEmbedInput[],
        options: GenaiEmbedModelOptions = {},
    ): Promise<Result<GenaiEmbedOutput, GenaiError>> {
        checkOptionKeys(options, EMBED_OPTION_KEYS);
        const { signal, ...fields } = options;
        const given = Object.fromEntries(Object.entries(fields).filter(([, value]) => value !== undefined));
        const request: GenaiEmbedRequest = { model: id, input, ...defaults, ...given };
        const inputs = checkEmbedRequest(request);
        toEmbedCall(request, inputs);
        if (signal?.aborted) {
            const error: GenaiError = { code: 'aborted', message: 'Request aborted', retryable: false };
            return withModelCall({ ok: false, error }, name, id, 0);
        }
        const message = embedUnsupported(client.mode, id, inputs.length);
        if (message !== undefined) {
            return withModelCall({ ok: false, error: { code: 'unsupported', message, retryable: false } }, name, id, 0);
        }
        const job: EmbedJob = { request, attempts: 0 };
        // the API counts each input text as one request toward its request limits
        const tokens = estimateEmbedTokens(inputs, limits.charsPerToken);
        const push: PushOptions = { cost: { requests: inputs.length, tokens } };
        if (signal !== undefined) {
            push.signal = signal;
        }
        const result = await queue.run(job, push);
        return withModelCall(result, name, id, job.attempts);
    });
    const model = call as GenaiEmbedModel;
    Object.defineProperty(model, 'id', { value: id, enumerable: true });
    return model;
}

/**
 * **Callable decision model with attached limiter queue**: no `batch`.
 *
 * - request checks, abort and Vertex `unsupported` answered before enqueue, without booking
 * - books estimated input tokens; settles against response usage, refund on `rate_limited`
 */
function createDecideModel(
    client: GenaiClient,
    name: string,
    spec: GenaiDecideModelSpec,
    limits: ModelLimits,
): GenaiDecideModel {
    const id = spec.model ?? DEFAULT_DECIDE_MODEL;
    const queue = createRunQueue<DecideJob, GenaiDecideOutput>({
        buckets: limits.buckets,
        concurrency: limits.concurrency,
        limiter: limits.limiter,
        execute: (job, signal) => client.decide(job.request, { signal }),
        // errors carry no usage: only the refund of a rate_limited booking
        usedTokens: (result) => (result.ok ? usedDecideTokens(result.data.usage) : undefined),
    });
    const call = guard(async function decide(
        input: Omit<GenaiDecideRequest, 'model'>,
        options: { signal?: AbortSignal } = {},
    ): Promise<Result<GenaiDecideOutput, GenaiError>> {
        checkOptionKeys(options, DECIDE_OPTION_KEYS);
        if (!isObject(input) || Array.isArray(input)) {
            throw new InvalidRequestError('Invalid value for option "request"');
        }
        if ('model' in input) {
            throw new InvalidRequestError('Option "model" is not supported: the model of a name is fixed');
        }
        const request: GenaiDecideRequest = { ...input, model: id };
        toDecideBody(request);
        const { signal } = options;
        if (signal?.aborted) {
            const error: GenaiError = { code: 'aborted', message: 'Request aborted', retryable: false };
            return withModelCall({ ok: false, error }, name, id, 0);
        }
        if (client.mode === 'vertex') {
            const error: GenaiError = { code: 'unsupported', message: DECIDE_UNSUPPORTED, retryable: false };
            return withModelCall({ ok: false, error }, name, id, 0);
        }
        const job: DecideJob = { request, attempts: 0 };
        const push: PushOptions = { cost: { tokens: estimateDecideTokens(request, limits.charsPerToken) } };
        if (signal !== undefined) {
            push.signal = signal;
        }
        const result = await queue.run(job, push);
        return withModelCall(result, name, id, job.attempts);
    });
    const model = call as GenaiDecideModel;
    Object.defineProperty(model, 'id', { value: id, enumerable: true });
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
            const requests = items.map((item) => ({ key: item?.key, request: toBatchRequest(item?.request, id) }));
            const filled = applyBatchDefaults(defaults, requests, options.config);
            const { config } = filled;
            return batch.submit(filled.items, { ...options, ...(config === undefined ? {} : { config }) });
        },
        check: batch.check,
        cancel: batch.cancel,
    });
}

/** Counts client calls of one generation attempt: client count, else 1, or 0 for refused request. */
function countAttempts(result: Result<GenaiOutput, GenaiError>): number {
    // without a count: 1 call, or none for a request the client refused (too large for a batch)
    const fallback = !result.ok && result.error.code === 'invalid_request' ? 0 : 1;
    return (result.ok ? result.data.attempts : result.error.attempts) ?? fallback;
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

/** Attaches name, model id and attempt count to embedding or decision result. */
function withModelCall<TOut extends GenaiEmbedOutput | GenaiDecideOutput>(
    result: Result<TOut, GenaiError>,
    name: string,
    id: string,
    attempts: number,
): Result<TOut, GenaiError> {
    const settings: GenaiSettings = { name, model: id };
    if (result.ok) {
        return { ok: true, data: { ...result.data, settings, attempts } };
    }
    return { ok: false, error: { ...result.error, settings, attempts } };
}

/** Estimates embedding input tokens from text and title characters. */
function estimateEmbedTokens(inputs: readonly EmbedText[], charsPerToken: number): number {
    const chars = inputs.reduce((sum, input) => sum + input.text.length + (input.title?.length ?? 0), 0);
    return Math.ceil(chars / charsPerToken);
}

/**
 * **Estimates decision input tokens**: no output tokens.
 *
 * - characters of text input, instructions and criteria texts (noul descriptions, choice ids and descriptions, levels)
 * - `IMAGE_TOKENS` per non-text input part
 */
function estimateDecideTokens(request: GenaiDecideRequest, charsPerToken: number): number {
    let chars = 0;
    let media = 0;
    if (typeof request.input === 'string') {
        chars += request.input.length;
    } else {
        for (const part of request.input) {
            if (part.type === 'text') {
                chars += part.text.length;
            } else {
                media += 1;
            }
        }
    }
    for (const question of Object.values<GenaiDecideQuestion>(request.questions)) {
        chars += instructionChars(question.instructions) + criteriaTexts(question).join('').length;
    }
    return Math.ceil(chars / charsPerToken) + media * IMAGE_TOKENS;
}

/** Counts characters of decision instructions. */
function instructionChars(instructions: GenaiDecideInstructions): number {
    if (typeof instructions === 'string') {
        return instructions.length;
    }
    const parts: readonly { text: string }[] = 'text' in instructions ? [instructions] : instructions;
    return parts.reduce((sum, part) => sum + part.text.length, 0);
}

/** Computes decision tokens from response usage; `undefined` without any count. */
function usedDecideTokens(usage: GenaiDecideOutput['usage']): number | undefined {
    if (usage.inputTokens === null && usage.outputTokens === null) {
        return undefined;
    }
    return (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0);
}

/** Lists criteria texts of decision question. */
function criteriaTexts(question: GenaiDecideQuestion): string[] {
    if (question.type === 'noul') {
        return [question.criteria?.true_description ?? '', question.criteria?.false_description ?? ''];
    }
    if (question.type === 'choice') {
        return Object.entries(question.criteria).flat();
    }
    return [...question.criteria];
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
function usedTokens(usage: GenaiOutput['usage'] | undefined): number | undefined {
    if (usage === undefined) {
        return undefined;
    }
    if (usage.inputTokens === null && usage.outputTokens === null && usage.reasoningTokens === null) {
        return undefined;
    }
    return (usage.inputTokens ?? 0) + (usage.outputTokens ?? 0) + (usage.reasoningTokens ?? 0);
}
