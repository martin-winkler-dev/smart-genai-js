# smart-genai

[![Test](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-genai-js%2Fmain%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-genai-js%2Fmain%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

One client for Gemini on the **Gemini Developer API** (AI Studio, API key) and **Vertex AI** (Google Cloud
credentials), supporting single requests and batch jobs using the OpenAI chat completion request format (`messages`,
`response_format`, `tools`, `reasoning_effort`).

## Installation

```sh
bun add smart-genai
# or
npm install smart-genai
```

## Client

```ts
import { createGenaiClient } from 'smart-genai';

const studio = createGenaiClient(); // key from env GEMINI_API_KEY, or { apiKey }
const vertex = createGenaiClient({ vertex: true }); // project, location, bucket, credentials from env
```

| Option | Default | Description |
|---|---|---|
| `apiKey` | env `GEMINI_API_KEY` | Developer API key (Developer API only) |
| `vertex.project` | env `GOOGLE_CLOUD_PROJECT` | Google Cloud project ID (required for Vertex AI) |
| `vertex.location` | env `GOOGLE_CLOUD_LOCATION`, else `global` | Vertex AI region or `global` |
| `vertex.bucket` | env `VERTEX_BATCH_BUCKET` | Cloud Storage bucket name (without `gs://`) for Vertex batch files; without a bucket, Vertex has no batch (`client.batch` is `undefined`) |
| `vertex.credentialsFile` | env `GOOGLE_APPLICATION_CREDENTIALS` | Service account or ADC file; falls back to the `gcloud auth application-default login` file |
| `vertex.authClient` | none | Custom `AuthClient` instance (bypasses `credentialsFile` check, e.g. for GCE/Cloud Run metadata server auth) |
| `vertex.storageUrl` | none | Custom Cloud Storage endpoint URL |
| `baseUrl` | official endpoint | Custom API endpoint URL |
| `env` | `process.env` | Source object from which default credentials and settings are read |
| `timeoutMs` | `120000` | HTTP timeout per call in milliseconds (120 s) |
| `defaults` | none | Request defaults and Gemini `config` applied to every request and batch request |
| `batchWaitSec` | `10` | Aggregation window in seconds for [batch calls](#batch-calls) (1 to 600) |

The package never reads `.env` files directly. Bun loads them automatically; on Node.js use `node --env-file=.env` or
`process.loadEnvFile()`, or pass credentials explicitly. On Vertex AI, `createGenaiClient` resolves `credentialsFile`
(`GOOGLE_APPLICATION_CREDENTIALS` or `~/.config/gcloud/application_default_credentials.json`, `%APPDATA%\gcloud\application_default_credentials.json` on Windows), verifies the file exists on disk at client creation, and passes `keyFilename` directly to `google-auth-library` (for both Vertex AI and Cloud Storage), skipping ambient metadata-server discovery (~15 s timeout outside Google Cloud). To use attached service accounts on Cloud Run/GCE without a local key file, pass `vertex: { authClient }`.

Invalid options in `createGenaiClient` and `createGenaiModels` throw a `TypeError`. Methods called on clients and models
never throw or reject: invalid call arguments return an `{ ok: false, error: { code: 'invalid_request', ... } }` result,
and runtime failures return an error result (`Result<T, GenaiError>`).

`@google/genai` environment variables are handled safely:
- `GOOGLE_GEMINI_BASE_URL` and `GOOGLE_VERTEX_BASE_URL` are ignored; the official endpoint or explicit `baseUrl` is always used.
- `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` are always passed explicitly.
- `GEMINI_API_KEY` and `GOOGLE_API_KEY`: the Developer API receives the key explicitly. Vertex AI never sends an API key. Passing `apiKey` together with `vertex` throws a `TypeError`.

## Single request

```ts
const result = await studio.generate({
    model: 'gemini-3.5-flash-lite',
    messages: [
        { role: 'system', content: 'Classify the word.' },
        { role: 'user', content: 'blorf' },
    ],
    response_format: {
        type: 'json_schema',
        json_schema: { name: 'answer', schema: { type: 'object', properties: { kind: { type: 'string' } } } },
    },
});
if (result.ok) {
    result.data.json; // { kind: 'noun' }
}
```

- **Supported fields**: `model`, `messages` (`system`, `developer`, `user`, `assistant`, `tool`; text, and images in
  `user` messages), `response_format` (`text`, `json_object`, `json_schema`), `temperature`, `top_p`, `max_tokens` /
  `max_completion_tokens`, `seed`, `stop`, `reasoning_effort`, `tools`, `tool_choice` (see [Tool calls](#tool-calls)),
  `web_search`, and `batch` (see [Batch calls](#batch-calls)). Any unsupported field returns an `invalid_request` error.
- **`reasoning_effort`** (`none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`): maps to thinking budgets of
  `0` / `512` / `1024` / `8192` / `24576` tokens on Gemini 2.x, and thinking levels `MINIMAL` / `MINIMAL` / `LOW` /
  `MEDIUM` / `HIGH` on newer models (`xhigh` and `max` match `high`). Gemini 3 cannot disable thinking, so `none` sets
  `MINIMAL`. Gemini 2.5 Pro cannot disable thinking either, so `none` sets its minimum budget (`128`). Setting
  `reasoning_effort` automatically enables `includeThoughts: true`, returning thought summaries in `data.reasoning`
  (except on Gemini 2.x with budget `0`); pass `config: { thinkingConfig: { includeThoughts: false } }` to suppress
  them. When `reasoning_effort` is omitted, no thinking configuration is sent.
- **`web_search: true`**: enables Google Search grounding; `false` overrides a configured default. Grounding links return
  in `data.sources` (`{ url, title? }`). Gemini's search tool does not take parameters, so OpenAI `web_search_options`
  is not accepted.
- **Defaults and config merging**: client `defaults` populate any field omitted by the request. `config` merges per field,
  and `thinkingConfig` merges per member, with call-level values taking precedence. `max_tokens` and `max_completion_tokens`
  act as aliases: setting either in a call overrides defaults for both. Defaults objects are copied upon creation, so mutating
  them later has no effect.
- **`json_schema.schema`**: forwarded as Gemini's `responseJsonSchema`. The `name` and `strict` fields are accepted for
  OpenAI compatibility but not sent.
- **Images**: passed using OpenAI syntax `{ type: 'image_url', image_url: { url: 'data:image/png;base64,...' } }` and
  forwarded as Gemini `inlineData`. Only base64 data URLs for `image/*` MIME types are accepted (HTTP and `gs://` URLs
  are rejected). The `detail` property is accepted and ignored.
- **Gemini-specific options**: passed in the second parameter: `generate(request, { config: { ... }, signal })`.
- **Result structure**:
  - `data`: `text`, `json` (for JSON response formats), `model`, `usage` (`inputTokens`, `outputTokens`, `reasoningTokens`),
    `finishReason`, `ms`, plus optional `reasoning`, `sources`, and `toolCalls`.
  - `error`: `code` (`rate_limited`, `timeout`, `network`, `http`, `aborted`, `blocked`, `truncated`, `invalid_json`,
    `invalid_request`, `unsupported`, `unexpected`, `limiter`), `message`, `retryable`, optional `status`, and `retryAfterMs`.
- **`settings` and `attempts`**: present on successful results and valid requests that failed. `settings` reflects the
  effective options used (model, parameters, active tools, and merged `config`). `attempts` reports total API requests
  made (1 for normal calls, 0 if rejected before transmission). An `invalid_request` error detected prior to dispatch
  has neither.
- **Partial output on failure**: when a response is received but unusable (`blocked`, `truncated`, `invalid_json`, or
  unexpected tool finish reasons), the partial response is preserved in `error.output` with its `reasoning`, `usage`,
  and `finishReason`. Network and HTTP transport errors have no `output`.
- **Finish reasons**: any finish reason other than `STOP` is treated as an error even if partial text arrived:
  `MAX_TOKENS` and `CONTINUATION` produce `truncated`, failed tool executions produce `unexpected`, and safety or recitation
  filters produce `blocked`.
- **`retryable`**: marked `true` on HTTP 429, 408, 409, 425, 5xx, timeouts, and network connection failures. Authentication
  errors (`invalid_grant`), missing credentials, and invalid client configurations are non-retryable `unexpected` errors.
- **SDK isolation**: SDK automatic retries and continuation loops are disabled (one HTTP request per call). Call `config`
  must not contain `abortSignal`, `httpOptions`, `automaticContinuation`, or an own `__proto__` property.

## Tool calls

```ts
const tools = [
    {
        type: 'function' as const,
        function: {
            name: 'get_weather',
            description: 'Current weather of a city',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
        },
    },
];
const messages: GenaiMessage[] = [{ role: 'user', content: 'Weather in Rome?' }];
const asked = await studio.generate({ model: 'gemini-3.5-flash', messages, tools });
if (asked.ok && asked.data.toolCalls) {
    messages.push(assistantMessage(asked.data));
    for (const call of asked.data.toolCalls) {
        const args = JSON.parse(call.function.arguments);
        messages.push(toolMessage(call, await getWeather(args.city)));
    }
    const answer = await studio.generate({ model: 'gemini-3.5-flash', messages, tools });
}

// With a named model: tools are options, conversation history is input
const named = await gemini.flash38(messages, { tools });
```

- **Message helpers**: `assistantMessage(output)` converts model output into the next `assistant` message (including
  `tool_calls` with thought signatures; `content` is `null` when the turn only contained tool calls). `toolMessage(call, result)`
  formats a execution result as a `tool` message (strings preserved as-is, objects serialized to JSON).
- **`tools` declarations**: `{ type: 'function', function: { name, description?, parameters?, strict? } }` definitions are
  converted to Gemini `functionDeclarations`. `parameters` is sent as `parametersJsonSchema`; `strict` is accepted and not sent.
  Function names must be 1 to 128 characters matching `^[A-Za-z0-9_.:-]+$` and unique within the request.
- **`tool_choice`**: `'auto'` maps to `AUTO`, `'none'` to `NONE`, `'required'` to `ANY`, and
  `{ type: 'function', function: { name } }` to `ANY` with `allowedFunctionNames: [name]`. When `tools` is omitted,
  `tool_choice` is not sent, allowing default tool choices to sit in client defaults safely. Call-level `config.toolConfig`
  overrides generated settings per field.
- **Handling tool calls**: responses containing function calls succeed with `data.toolCalls` populated as
  `{ id, type: 'function', function: { name, arguments } }` (`arguments` as JSON string). Missing Gemini call IDs are
  assigned generated IDs (`call_<uuid>`). JSON `response_format` validation is bypassed when the model emits tool calls.
- **Tool execution responses**: pass tool calls back in an `assistant` message and supply results as
  `{ role: 'tool', tool_call_id, content }`. Subsequent tool messages are combined into a single Gemini `user` turn of
  `functionResponse` parts. JSON strings representing objects are unpacked into structured JSON objects; plain strings
  are wrapped as `{ output: text }`.
- **Thought signatures**: Gemini 3 signs the first function call of each step and rejects histories missing the signature.
  The signature arrives in `call.extra_content.google.thought_signature` and must accompany the call back to the model.
  Tool calls without signatures (e.g. from other providers) receive Google's `skip_thought_signature_validator` flag on
  the first call of their turn on Gemini 3+; Gemini 2.x receives no validator flag.
- **`web_search` with tools**: combining `web_search: true` with `tools` is supported on Gemini 3 and newer. On Gemini 2.x,
  combining them returns an `invalid_request` error. Server-side search execution parts are omitted (`includeServerSideToolInvocations`
  remains off).
- **Tool execution failures**: finish reasons `MALFORMED_FUNCTION_CALL` (retryable), `UNEXPECTED_TOOL_CALL`,
  `TOO_MANY_TOOL_CALLS`, and `MISSING_THOUGHT_SIGNATURE` return `unexpected` errors.
- **Validation**: malformed tool definitions, invalid `tool_choice`, invalid argument JSON, and tool messages referencing
  unknown `tool_call_id`s return `invalid_request` errors.

## Model list

```ts
const models = await studio.listModels();
// models.data: [{ id: 'gemini-3.5-flash', displayName, inputTokenLimit, outputTokenLimit, actions, thinking }, …]
```

Paginates across all available base models. `id` matches what `model` expects (the `models/` prefix is stripped). Listing
models consumes no tokens. On Vertex AI, Google publisher models are listed.

## Embeddings

```ts
const vectors = await studio.embed({
    model: 'gemini-embedding-2',
    input: [{ text: 'Cells were incubated at 37 °C.', title: 'Methods' }, 'Untitled paragraph.'],
    taskType: 'RETRIEVAL_DOCUMENT',
    dimensions: 768,
});
// vectors.data: { embeddings: [[…768 numbers], […]], model: 'gemini-embedding-2', ms }

const query = await studio.embed({
    model: 'gemini-embedding-2',
    input: 'incubation temperature',
    taskType: 'RETRIEVAL_QUERY',
});
```

- **Input capacity**: embeds 1 to 100 texts (`MAX_EMBED_INPUTS`) in a single API call, returning one embedding vector per
  input in original order. Each text is wrapped in its own `Content` element, preventing `gemini-embedding-2` from fusing
  them into one vector.
- **`taskType` and `title`**: `gemini-embedding-2` ignores native API task fields, so smart-genai prefixes the text
  according to Google's documentation (`title: Methods | text: ...`, `title: none | text: ...`, `task: search result | query: ...`).
  Other models (`gemini-embedding-001`) use standard API fields; all inputs in a single call must share the same title.
- **Validation**: `title` is allowed only with `RETRIEVAL_DOCUMENT`. `dimensions` sets `outputDimensionality` (Matryoshka
  truncation); `gemini-embedding-001` returns normalized vectors only at 3072 dimensions, while `gemini-embedding-2` supports
  all dimensions.
- **Vertex AI limits**: on Vertex AI, `gemini-embedding-001` supports up to 100 texts per request (`predict`); other Gemini
  models (such as `gemini-embedding-2`) support 1 text per call (`embedContent`); supplying more returns `unsupported`.
- **No batch API**: embeddings do not support asynchronous batch jobs; passing `batch` returns `invalid_request`.
- The embedding API does not report token usage metadata.

## Named models

```ts
import { createGenaiModels } from 'smart-genai';

const gemini = createGenaiModels({
    client: { apiKey }, // client options without defaults; gemini.client is built from them
    defaults: { temperature: 0, max_tokens: 2000 }, // shared by all models
    sharedLimits: { concurrency: 4 }, // shared rate limit buckets across all models
    limiterOptions: { maxRetries: 2 }, // retry and backoff options for all calls
    models: {
        flash38: {
            model: 'gemini-3.8-flash',
            defaults: { reasoning_effort: 'low' },
            modelLimits: { rpm: 10, tpm: 250_000 }, // limits for this model only
        },
        lite: { model: 'gemini-3.5-flash-lite', modelLimits: { intervalMs: 3000 } }, // at most one call every 3 s
    },
});

await gemini.flash38('Summarise this abstract: …'); // string input treated as user message
await gemini.lite(messages, { web_search: true, reasoning_effort: 'none' }); // input followed by options
gemini.flash38.id; // 'gemini-3.8-flash'
await gemini.flash38.batch?.submit([{ key: 'a', request: { messages } }]); // model and defaults prefilled
await gemini.listModels();
```

- **Naming**: model names not matching JavaScript identifiers are accessed via brackets (`gemini['flash-lite']`). Property
  names `client`, `listModels`, `then`, `toJSON`, and `Object.prototype` members (`constructor`, `toString`, `__proto__`, etc.)
  are reserved and throw a `TypeError`. Because the underlying model ID is fixed per named function, setting `model` in a
  model's `defaults` throws a `TypeError`, and passing `model` in a call returns an `invalid_request` error.
- **Calling convention**: `gemini.<name>(input, options)`. `input` can be a string prompt, content parts array (text and
  images) for a single user turn, or an array of message objects for full conversations. `options` accepts any request field
  except `model` and `messages`, plus `signal` and Gemini `config`. Options override defaults; passing `undefined` preserves
  the default. The returned function does not depend on `this` and can be passed as a standalone callback:
  `const classify = (text) => gemini.flash38(text, { response_format })`.
- **Defaults hierarchy**: global `defaults` < model-specific `defaults` < per-call `options`. The `client` property takes
  client options without defaults; `createGenaiModels` creates its own client (`gemini.client`) to monitor all calls.
  Specifying `max_tokens` or `max_completion_tokens` at any level overrides both from lower levels.
- **Traceability**: results include `settings` containing `{ name: 'flash38', model: 'gemini-3.8-flash', ... }`, making
  it easy to log which model and parameters produced the result. `attempts` tracks total API calls including retries (0 if
  rejected by the limiter prior to dispatch).
- **Rate limiting (`sharedLimits` and `modelLimits`)**: `createGenaiClient` performs single unthrottled requests without retries;
  `createGenaiModels` wraps each named model in a `smart-limit` queue. Both `sharedLimits` (top-level, shared across all named
  models in the `createGenaiModels` instance for project/provider-wide quotas) and `modelLimits` (per model inside
  `models.<name>.modelLimits`) are optional and unset by default (no rate or concurrency limits apply unless configured;
  `concurrency` defaults to `Infinity`). Neither overrides the other: every synchronous call must acquire capacity from both
  `sharedLimits` buckets and its model's `modelLimits` buckets before starting. Both accept the same limit fields:
  - `concurrency`: max simultaneous in-flight calls (`sharedLimits.concurrency` uses a cross-model `createConcurrencyBucket`;
    `modelLimits.concurrency` caps that model's queue).
  - `rps`, `rpm`, `rpd`: requests per second (1,000 ms), minute (60,000 ms), or day (86,400,000 ms) rolling sliding windows.
  - `tpm`, `tpd`: tokens per minute or day rolling sliding windows.
  - `intervalMs`: minimum spacing in ms (`createSlidingWindow({ requests: 1, perMs: intervalMs })`, e.g. `3000` for one call every 3 s).
  - `buckets`: custom `smart-limit` `Bucket[]` instances (e.g. `createSlidingWindow({ tokens: 50_000, perMs: 10_000 })`).
- **Limiter configuration (`limiterOptions`)**: retry and backoff settings from `smart-limit` applied to all named models
  (always active even when `sharedLimits` and `modelLimits` are omitted):
  - `maxRetries` (default `4`): maximum retries per call for isolated retryable errors (timeouts, HTTP 408, 409, 425, `MALFORMED_FUNCTION_CALL`).
  - `retryDelayMs` (default `2000`): initial exponential backoff delay in milliseconds (doubles each retry).
  - `maxRetryDelayMs` (default `300000`): maximum backoff delay cap (5 minutes).
  - `maxWaitMs` (default `900000`): maximum wait per model queue for bucket capacity, global backoff locks, or retries (15 minutes).
  - HTTP 429, 5xx, and network errors trigger a global queue lock (`backoff_global`) up to `maxWaitMs` and do not consume `maxRetries` (even when `maxRetries: 0`; lower `maxWaitMs` to bound outage waits).
- **Token reservation and `charsPerToken`**: when `tpm` or `tpd` is configured, each call reserves an initial token estimate before
  starting and settles against actual reported usage (`inputTokens + outputTokens + reasoningTokens`) upon completion, returning
  unused tokens to the sliding window. If a call fails with 429 without reported usage, the full reservation is refunded (`0` tokens).
  The estimate is `Math.ceil(chars / charsPerToken) + 1120 * images + (max_tokens ?? max_completion_tokens ?? 1000)`, where `chars`
  counts message text, tool results, tool call names/arguments, and `JSON.stringify(tools)`. `charsPerToken` defaults to `4` (any
  positive finite number, e.g. `3.5`) and can be set at the top level of `createGenaiModels` or overridden per model in
  `models.<name>.charsPerToken`.
- **Queue and backoff behavior**: each named model runs its own FIFO `smart-limit` queue (`createQueue`). Rate limits (429),
  network drops, and 5xx errors lock that model's queue (for `retryAfterMs` if provided, else exponential backoff with a 1,000 ms
  minimum) while other models continue running. Once the lock delay expires, a single probe call executes alone; waiting calls
  resume only after a probe succeeds. If a required bucket wait or lock exceeds `maxWaitMs`, waiting and incoming calls on that
  model fail immediately without sending network requests. Calls rejected by the limiter (e.g. estimate exceeds total `tpm` bucket
  limit, or `maxWaitMs` exceeded before any attempt) return `{ code: 'limiter', limiterCode }` (`COST_EXCEEDS_LIMIT`,
  `MAX_WAIT_EXCEEDED`). Calls that fail after at least one attempt (e.g. `RETRIES_EXHAUSTED` or `MAX_WAIT_EXCEEDED` after a 429)
  return the last provider `GenaiError` (preserving `code: 'rate_limited'` and `retryAfterMs`).
- **Cancellation**: passing an `AbortSignal` terminates queue wait, active retry delays, or in-flight HTTP requests immediately,
  returning `{ ok: false, error: { code: 'aborted', ... } }`.
- Batch jobs (`batch.submit`) and `listModels` bypass rate limiter queues. Batch calls (`batch: true`) wait only for the model's
  own `modelLimits.concurrency`.

## Batch

```ts
const submitted = await vertex.batch?.submit([
    { key: 'a', request },
    { key: 'b', request: other }, // all items must use the same model
]);

// Check status periodically:
const status = await vertex.batch?.check(submitted.data.name);
// status.data.phase: 'running' | 'succeeded' | 'failed'; results: [{ key, result }]
```

- **Vertex AI storage (`VERTEX_BATCH_BUCKET` / `vertex.bucket`)**: Vertex AI Batch Prediction requires a Cloud Storage bucket
  (configured via `vertex.bucket` or env `VERTEX_BATCH_BUCKET`, bucket name without `gs://`; up to 100 MB per submission). Without
  a bucket, `vertex.batch` (and `gemini.<model>.batch`) is `undefined`. On `submit`, requests are uploaded to
  `gs://<bucket>/smart-genai/<uuid>/input.jsonl`, and Vertex AI writes results to `.../output/.../predictions.jsonl`. When `check`
  observes a terminal state (`succeeded` or `failed`), it reads the results (if succeeded) and deletes the job's bucket folder. If
  downloading output fails, the files remain in the bucket so a subsequent `check` can retry.
- **Key encoding on Vertex AI**: keys are prefixed with `j:<key>` (JSON formats) or `t:<key>` (text) to prevent BigQuery and
  JSONL importers from coercing numeric string keys into numbers. Checking an already-cleaned-up job returns `unexpected`.
- **Developer API**: requests and responses are exchanged inline directly within the Batch API payload (< 19 MB per submission; no
  bucket required).
- **Byte limits**: `batch.maxBytes` defines the maximum payload byte budget per submission (19 MB on Developer API, 100 MB on
  Vertex AI). `requestBytes(request, config?)` returns the calculated payload size in bytes. Submissions with duplicate keys,
  mismatched models, empty items, `batch` set inside items, or requests exceeding `maxBytes` return `invalid_request` errors.
- **Result matching**: items are matched by `key`. Item-level errors are mapped to corresponding HTTP errors (Google RPC codes
  are translated on Developer API; transient Vertex AI errors are classified as retryable).
- **Job cancellation**: `cancel(name)` requests job cancellation. On Vertex AI, it also removes temporary bucket files. If a
  job is already finished, `cancel` cleans up remaining files without re-cancelling the job.
- **Orphaned file cleanup (`removeFiles()`)**: deletes leftover batch folders on Vertex AI, returning `{ removed, kept }`.
  All batch files reside in `smart-genai/<uuid>/`. `removeFiles` lists active Vertex batch jobs and keeps folders belonging to
  non-terminal jobs (queued, pending, running, cancelling) and folders with any object younger than one hour (or with an unknown
  creation timestamp). It deletes all other folders under `smart-genai/`—including output of finished jobs that were never read
  via `check`, so call `removeFiles()` only after pending `check` calls for stored jobs have completed, not blindly at startup.
  It never touches files outside `smart-genai/`, is never called automatically, and returns `unsupported` on the Developer API.
  Named model batch interfaces do not expose `removeFiles`; use `client.batch.removeFiles()`.

## Batch calls

```ts
const studio = createGenaiClient({ batchWaitSec: 10 });
const [a, b] = await Promise.all([
    studio.generate({ ...request, batch: true }), // 50% discount, results within 24 hours
    studio.generate({ ...other, batch: true }),
]);
// a and b are Result<GenaiOutput, GenaiError>, exactly like standard generate calls
```

- **Overview**: setting `batch: true` converts a single `generate` call into an asynchronous batch request. It exposes the
  same API and returns an individual `Result<GenaiOutput, GenaiError>`, while routing the request through the Batch API.
  `batch: false` (the default) performs a direct synchronous request. `batch: true` can be configured globally or in model
  defaults:
  ```ts
  models: {
      flash: 'gemini-3.8-flash',
      flashBatch: { model: 'gemini-3.8-flash', defaults: { batch: true } },
  }
  ```
- **Aggregation window**: requests are grouped by model and Gemini `config`. The collection window begins with the first call
  in a group and waits `batchWaitSec` (default `10` s). When the timer expires, the group is submitted as one or more batch
  jobs (split if total payload exceeds `maxBytes`). Any single request exceeding `maxBytes` fails immediately with an
  `invalid_request` error without being submitted.
- **Polling and settlement**: active batch jobs are polled every 30 seconds. Upon completion, each waiting promise resolves
  with its specific result. If a job fails or is cancelled, every call in that job rejects with `{ code: 'unexpected', ... }`.
  If the submission itself fails (e.g. rate limit), all grouped calls receive that error. Retryable polling errors are retried
  automatically (respecting `retryAfterMs`).
- **72-hour timeout**: jobs that fail to complete within 72 hours are automatically cancelled and fail with a `timeout` error.
- **Cancellation**: aborting a call before submission removes it from the pending group. Aborting after submission resolves that
  specific call as `aborted`; if all calls in a batch job are aborted, the batch job itself is cancelled.
- **Prerequisites**: on Vertex AI without a configured bucket, batch calls return `unsupported`. The free Gemini Developer API
  tier does not support batch jobs (`FAILED_PRECONDITION`); batch operations require a paid project with billing enabled.
- **In-memory lifecycle**: batch calls and pollers live in application memory and are not persisted. If the host process exits,
  in-flight call promises are lost while submitted jobs continue running in Google Cloud. For long-running workflows that must
  survive restarts, use `client.batch.submit` and `client.batch.check` with your own persistence layer.
- **Concurrency with named models**: batch calls executed through named models wait only for the model's own
  `modelLimits.concurrency` (tracked separately from synchronous calls). They do not consume shared limits, token buckets,
  or interval limits, and do not trigger synchronous queue locks upon encountering 429 errors.

