// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenerateContentConfig } from '@google/genai';
import type { AuthClient } from 'google-auth-library';
import type { Bucket } from 'smart-limit';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Result of every call; only factory functions throw on invalid options. */
export type Result<T, E> = { ok: true; data: T } | { ok: false; error: E };

/**
 * **Options for `createGenaiClient`**.
 *
 * - without `vertex`: Gemini Developer API (AI Studio); `apiKey` defaults to `GEMINI_API_KEY`
 * - `vertex`: Vertex AI with credentials file; `true` reads configuration from environment
 * - `env`: source for defaults (default `process.env`); never reads `.env` files directly
 * - `timeoutMs`: HTTP timeout per call (default 120 s)
 * - `defaults`: request defaults and Gemini config (see `GenaiDefaults`)
 * - `batchWaitSec`: collection window for batch calls in seconds (default 10, range 1–600)
 */
export type GenaiClientOptions = {
    apiKey?: string;
    vertex?: boolean | GenaiVertexOptions;
    env?: Readonly<Record<string, string | undefined>>;
    timeoutMs?: number;
    baseUrl?: string;
    defaults?: GenaiDefaults;
    batchWaitSec?: number;
};

/**
 * **Request defaults**: any request field except `messages`, plus Gemini `config`.
 *
 * - request fields override defaults
 * - `max_tokens` and `max_completion_tokens` count as one field; either replaces both defaults
 * - `config` merges with call `config` per field; `thinkingConfig` merges per member
 */
export type GenaiDefaults = Omit<GenaiRequest, 'messages'> & { config?: GenerateContentConfig };

/**
 * **Vertex AI configuration**: all values fall back to environment variables.
 *
 * - `project`: Google Cloud project ID (env `GOOGLE_CLOUD_PROJECT`)
 * - `location`: region (env `GOOGLE_CLOUD_LOCATION`, default `global`)
 * - `bucket`: Cloud Storage bucket for batch files (env `VERTEX_BATCH_BUCKET`)
 * - `credentialsFile`: service account or ADC JSON file (env `GOOGLE_APPLICATION_CREDENTIALS`)
 * - `authClient`: custom AuthClient instance
 * - `storageUrl`: custom Cloud Storage endpoint URL
 */
export type GenaiVertexOptions = {
    project?: string;
    location?: string;
    bucket?: string;
    credentialsFile?: string;
    authClient?: AuthClient;
    storageUrl?: string;
};

/** Role of chat message: `system`, `developer`, `user`, `assistant`, or `tool`. */
export type GenaiRole = 'system' | 'developer' | 'user' | 'assistant' | 'tool';

/** Text part of message in OpenAI format. */
export type GenaiTextPart = { type: 'text'; text: string };

/**
 * **Image part of user message in OpenAI format**.
 *
 * - `url`: base64 data URL (e.g. `data:image/png;base64,...`), sent as Gemini `inlineData`
 * - `detail`: accepted for OpenAI compatibility, not sent to Gemini
 */
export type GenaiImagePart = {
    type: 'image_url';
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    image_url: { url: string; detail?: 'auto' | 'low' | 'high' };
};

/** Part of message: text, or image in user message. */
export type GenaiContentPart = GenaiTextPart | GenaiImagePart;

/**
 * **Chat message in OpenAI format**.
 *
 * - `assistant`: previous model turn; `content` may be `null` when `tool_calls` present
 * - `tool`: tool execution result for `tool_call_id`
 */
export type GenaiMessage =
    | { role: 'system' | 'developer' | 'user'; content: string | GenaiContentPart[] }
    | {
          role: 'assistant';
          content: string | GenaiTextPart[] | null;
          // biome-ignore lint/style/useNamingConvention: OpenAI field name
          tool_calls?: GenaiToolCall[];
      }
    | {
          role: 'tool';
          // biome-ignore lint/style/useNamingConvention: OpenAI field name
          tool_call_id: string;
          content: string | GenaiTextPart[];
      };

/**
 * **Function tool declaration in OpenAI format**.
 *
 * - `name`: 1–128 characters (`[A-Za-z0-9_.:-]`)
 * - `parameters`: JSON Schema for function arguments (sent as `parametersJsonSchema`)
 * - `strict`: accepted for OpenAI compatibility, not sent to Gemini
 */
export type GenaiTool = {
    type: 'function';
    function: { name: string; description?: string; parameters?: Record<string, unknown>; strict?: boolean | null };
};

/** Tool selection mode: `auto`, `none`, `required`, or specific function object. */
export type GenaiToolChoice = 'auto' | 'none' | 'required' | { type: 'function'; function: { name: string } };

/**
 * **Tool call in assistant message**.
 *
 * - `id`: Gemini call id or generated `call_...` identifier
 * - `function.arguments`: arguments encoded as JSON string
 * - `extra_content.google.thought_signature`: Gemini 3 thought signature
 */
export type GenaiToolCall = {
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
    // biome-ignore lint/style/useNamingConvention: OpenAI-compatible field name
    extra_content?: { google?: { thought_signature?: string } };
};

/**
 * **Response format specification**.
 *
 * - `json_schema`: JSON constrained by `schema` (sent as `responseJsonSchema`)
 * - `json_object`: unconstrained JSON output (`responseMimeType: application/json`)
 * - `text`: plain text output
 */
export type GenaiResponseFormat =
    | { type: 'text' }
    | { type: 'json_object' }
    | {
          type: 'json_schema';
          // biome-ignore lint/style/useNamingConvention: OpenAI field name
          json_schema: {
              name: string;
              schema?: Record<string, unknown>;
              description?: string;
              strict?: boolean | null;
          };
      };

/**
 * **Chat completion request parameters**.
 *
 * - `model`: required unless client defaults provide it
 * - `max_tokens` / `max_completion_tokens`: maps to `maxOutputTokens`
 * - `reasoning_effort`: thinking level or budget
 * - `web_search`: enables Google Search grounding
 * - `tools` / `tool_choice`: function calling configuration
 * - `batch`: routes call through Batch API
 */
export type GenaiRequest = {
    model?: string;
    messages: GenaiMessage[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    response_format?: GenaiResponseFormat;
    temperature?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    top_p?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    max_tokens?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    max_completion_tokens?: number;
    seed?: number;
    stop?: string | string[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    reasoning_effort?: GenaiReasoningEffort;
    // biome-ignore lint/style/useNamingConvention: same style as the OpenAI fields
    web_search?: boolean;
    tools?: GenaiTool[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    tool_choice?: GenaiToolChoice;
    batch?: boolean;
};

/**
 * **Thinking configuration level**.
 *
 * - Gemini 2.x: sets `thinkingBudget` (0, 512, 1024, 8192, 24576; Gemini 2.5 Pro minimum 128)
 * - other models: sets `thinkingLevel` (`MINIMAL`, `LOW`, `MEDIUM`, `HIGH`)
 * - enables `includeThoughts: true` (reasoning output returned in `reasoning`)
 */
export type GenaiReasoningEffort = 'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max';

/** Call options: `signal` for cancellation and `config` merged into Gemini configuration. */
export type GenaiCallOptions = { signal?: AbortSignal; config?: GenerateContentConfig };

/** Token counts reported by API; `outputTokens` excludes thinking tokens. */
export type GenaiUsage = { inputTokens: number | null; outputTokens: number | null; reasoningTokens: number | null };

/** Snapshot of effective request settings applied to call, excluding messages. */
export type GenaiSettings = {
    name?: string;
    model: string;
    temperature?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    top_p?: number;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    max_tokens?: number;
    seed?: number;
    stop?: string | string[];
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    reasoning_effort?: GenaiReasoningEffort;
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    tool_choice?: GenaiToolChoice;
    // biome-ignore lint/style/useNamingConvention: same style as the OpenAI fields
    web_search?: true;
    batch?: true;
    json?: true;
    tools?: string[];
    config?: GenerateContentConfig;
};

/**
 * **Output of successful generation**.
 *
 * - `text`: generated response text excluding thought parts
 * - `json`: parsed JSON object when `response_format` requests JSON
 * - `reasoning`: thought summaries when present
 * - `sources`: web grounding citations
 * - `toolCalls`: function calls requested by model
 * - `settings` / `attempts`: effective settings snapshot and API call count
 */
export type GenaiOutput = {
    text: string;
    json?: unknown;
    reasoning?: string;
    sources?: GenaiSource[];
    toolCalls?: GenaiToolCall[];
    model: string;
    usage: GenaiUsage;
    finishReason: string | null;
    ms: number;
    settings?: GenaiSettings;
    attempts?: number;
};

/** Web page citation from Google Search grounding. */
export type GenaiSource = { url: string; title?: string };

/**
 * **Error codes returned in `GenaiError`**.
 *
 * - `rate_limited`: HTTP 429 rate limit with optional `retryAfterMs`
 * - `timeout`: request timeout
 * - `network`: connection or network failure
 * - `http`: HTTP error status (408, 409, 425, and 5xx are retryable)
 * - `aborted`: cancelled via AbortSignal
 * - `blocked` / `truncated` / `invalid_json`: response received but unusable
 * - `limiter`: rate limiter capacity exceeded or timed out
 * - `invalid_request`: malformed request or options (non-retryable)
 * - `unsupported`: requested feature unavailable (non-retryable)
 * - `unexpected`: unhandled or internal error
 */
export type GenaiErrorCode =
    | 'rate_limited'
    | 'timeout'
    | 'network'
    | 'http'
    | 'aborted'
    | 'blocked'
    | 'truncated'
    | 'invalid_json'
    | 'limiter'
    | 'invalid_request'
    | 'unsupported'
    | 'unexpected';

/**
 * **Error details for failed call**.
 *
 * - `output`: partial or unusable response data when received from API
 * - `settings` / `attempts`: effective settings and total attempt count
 */
export type GenaiError = {
    code: GenaiErrorCode;
    message: string;
    retryable: boolean;
    status?: number;
    retryAfterMs?: number;
    limiterCode?: string;
    text?: string;
    model?: string;
    usage?: GenaiUsage;
    ms?: number;
    output?: GenaiOutput;
    settings?: GenaiSettings;
    attempts?: number;
};

/** Single batch item with unique identifier and request. */
export type GenaiBatchItem = { key: string; request: GenaiRequest };

/** Result for single batch item mapped by key. */
export type GenaiBatchResult = { key: string; result: Result<GenaiOutput, GenaiError> };

/**
 * **Batch job status**.
 *
 * - `phase`: `running`, `succeeded`, or `failed`
 * - `results`: array of item results in succeeded phase
 * - `error`: failure message when job fails
 */
export type GenaiBatchStatus = {
    name: string;
    state: string;
    phase: 'running' | 'succeeded' | 'failed';
    results?: GenaiBatchResult[];
    error?: string;
};

/** Options for batch job submission. */
export type GenaiBatchOptions = { signal?: AbortSignal; displayName?: string; config?: GenerateContentConfig };

/**
 * **Batch operations interface**.
 *
 * - `maxBytes`: size limit for single batch submission
 * - `requestBytes`: calculates encoded byte size of request
 * - `submit`: creates batch job for multiple items sharing one model
 * - `check`: checks job status and downloads completed results
 * - `cancel`: cancels active batch job
 * - `removeFiles`: removes orphaned batch files from Cloud Storage
 */
export type GenaiBatch = {
    readonly maxBytes: number;
    requestBytes(request: GenaiRequest, config?: GenerateContentConfig): Result<number, GenaiError>;
    submit(
        items: readonly GenaiBatchItem[],
        options?: GenaiBatchOptions,
    ): Promise<Result<GenaiBatchStatus, GenaiError>>;
    check(name: string, options?: { signal?: AbortSignal }): Promise<Result<GenaiBatchStatus, GenaiError>>;
    cancel(name: string, options?: { signal?: AbortSignal }): Promise<Result<void, GenaiError>>;
    /**
     * **Removes orphaned batch folders from Cloud Storage**.
     *
     * - deletes `smart-genai/<id>/` folders not associated with running jobs
     * - preserves folders with recent uploads (grace period)
     * - non-applicable for Developer API (returns `unsupported`)
     */
    removeFiles(options?: { signal?: AbortSignal }): Promise<Result<{ removed: number; kept: number }, GenaiError>>;
};

/**
 * **Client for Gemini Developer API and Vertex AI**.
 *
 * - `generate`: executes single request or batch call (`batch: true`)
 * - `embed`: one vector per input text from one API call (`GenaiEmbedRequest`)
 * - `listModels`: retrieves available models
 * - `batch`: batch API interface
 */
export type GenaiClient = {
    readonly mode: 'developer' | 'vertex';
    readonly batch?: GenaiBatch;
    generate(request: GenaiRequest, options?: GenaiCallOptions): Promise<Result<GenaiOutput, GenaiError>>;
    embed(
        request: GenaiEmbedRequest,
        options?: { signal?: AbortSignal },
    ): Promise<Result<GenaiEmbedOutput, GenaiError>>;
    listModels(options?: { signal?: AbortSignal }): Promise<Result<GenaiModelInfo[], GenaiError>>;
};

/**
 * **Task an embedding is made for** (Gemini `taskType`).
 *
 * - documents to search use `RETRIEVAL_DOCUMENT`; their queries `RETRIEVAL_QUERY`, `QUESTION_ANSWERING`,
 *   `FACT_VERIFICATION` or `CODE_RETRIEVAL_QUERY`
 * - `SEMANTIC_SIMILARITY`, `CLASSIFICATION` and `CLUSTERING` are symmetric: every compared text uses the same task
 */
export type GenaiEmbedTask =
    | 'RETRIEVAL_QUERY'
    | 'RETRIEVAL_DOCUMENT'
    | 'QUESTION_ANSWERING'
    | 'FACT_VERIFICATION'
    | 'CODE_RETRIEVAL_QUERY'
    | 'SEMANTIC_SIMILARITY'
    | 'CLASSIFICATION'
    | 'CLUSTERING';

/** Text to embed; `title` (e.g. the section heading) only with task `RETRIEVAL_DOCUMENT`. */
export type GenaiEmbedInput = string | { text: string; title?: string };

/**
 * **Embedding request**: one vector per input, in input order, from one API call.
 *
 * - `input`: 1 to 100 non-empty texts (`batchEmbedContents`); Vertex AI takes one text per call for Gemini
 *   models other than `gemini-embedding-001` (e.g. `gemini-embedding-2`)
 * - `taskType`: `gemini-embedding-2` ignores the API field, so the task and titles go into the text as the documented
 *   prefixes (`task: search result | query: …`, `title: … | text: …`); other models get `taskType` and `title`
 *   fields, where all inputs of a call share one title
 * - `dimensions`: `outputDimensionality` (Matryoshka truncation); `gemini-embedding-001` normalizes only at 3072
 */
export type GenaiEmbedRequest = {
    model: string;
    input: string | readonly GenaiEmbedInput[];
    taskType?: GenaiEmbedTask;
    dimensions?: number;
};

/** Vectors of an embedding request, in input order. */
export type GenaiEmbedOutput = { embeddings: number[][]; model: string; ms: number };

/** Information about available Gemini model. */
export type GenaiModelInfo = {
    id: string;
    displayName?: string;
    description?: string;
    inputTokenLimit?: number;
    outputTokenLimit?: number;
    actions: string[];
    thinking?: boolean;
};

/**
 * **Rate limiting rules**: configured via smart-limit buckets.
 *
 * - `concurrency`: maximum concurrent requests in flight
 * - `rps` / `rpm` / `rpd`: request counts per second, minute, or day
 * - `tpm` / `tpd`: token limits per minute or day
 * - `intervalMs`: minimum delay between consecutive request starts
 * - `buckets`: custom smart-limit bucket instances
 */
export type GenaiLimits = {
    concurrency?: number;
    rps?: number;
    rpm?: number;
    rpd?: number;
    tpm?: number;
    tpd?: number;
    intervalMs?: number;
    buckets?: readonly Bucket[];
};

/** Rate limiter queue and retry settings. */
export type GenaiLimiterOptions = {
    maxRetries?: number;
    retryDelayMs?: number;
    maxRetryDelayMs?: number;
    maxWaitMs?: number;
};

/** Configuration spec for named model. */
export type GenaiModelSpec = {
    model: string;
    defaults?: Omit<GenaiDefaults, 'model'>;
    modelLimits?: GenaiLimits;
    charsPerToken?: number;
};

/** Configuration options for `createGenaiModels`. */
export type GenaiModelsOptions<N extends string> = {
    client?: Omit<GenaiClientOptions, 'defaults'>;
    models: Readonly<Record<N, string | GenaiModelSpec>>;
    defaults?: Omit<GenaiDefaults, 'model'>;
    sharedLimits?: GenaiLimits;
    limiterOptions?: GenaiLimiterOptions;
    charsPerToken?: number;
};

/** Input accepted by named model: string prompt, content parts, or message list. */
export type GenaiModelInput = string | readonly GenaiContentPart[] | readonly GenaiMessage[];

/** Call options for named model overriding configured defaults. */
export type GenaiModelOptions = {
    [K in keyof (Omit<GenaiRequest, 'model' | 'messages'> & GenaiCallOptions)]?:
        | (Omit<GenaiRequest, 'model' | 'messages'> & GenaiCallOptions)[K]
        | undefined;
};

/**
 * **Named model caller function**.
 *
 * - accepts string prompt or message array, plus per-call options
 * - routes calls through configured limiter queues
 * - provides `id` property and optional `batch` interface
 */
export type GenaiModel = {
    (input: GenaiModelInput, options?: GenaiModelOptions): Promise<Result<GenaiOutput, GenaiError>>;
    readonly id: string;
    readonly batch?: GenaiModelBatch;
};

/** Batch API interface for named model with fixed model identifier. */
export type GenaiModelBatch = {
    readonly maxBytes: number;
    requestBytes(request: Omit<GenaiRequest, 'model'>, config?: GenerateContentConfig): Result<number, GenaiError>;
    submit(
        items: readonly { key: string; request: Omit<GenaiRequest, 'model'> }[],
        options?: GenaiBatchOptions,
    ): Promise<Result<GenaiBatchStatus, GenaiError>>;
    check(name: string, options?: { signal?: AbortSignal }): Promise<Result<GenaiBatchStatus, GenaiError>>;
    cancel(name: string, options?: { signal?: AbortSignal }): Promise<Result<void, GenaiError>>;
};

/** Property names reserved on `createGenaiModels` result object. */
// biome-ignore lint/complexity/noBannedTypes: keyof Object gets the methods of Object.prototype
export type GenaiReservedName = 'client' | 'listModels' | 'then' | 'toJSON' | '__proto__' | keyof Object;

/** Object mapping model names to callable model functions, plus `client` and `listModels`. */
export type GenaiModels<N extends string> = { readonly [K in Exclude<N, GenaiReservedName>]: GenaiModel } & {
    readonly client: GenaiClient;
    listModels: GenaiClient['listModels'];
};
