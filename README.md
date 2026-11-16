# smart-genai

[![Test](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/test.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/test.yml)
[![Package](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/package.yml/badge.svg)](https://github.com/martin-winkler-dev/smart-genai-js/actions/workflows/package.yml)
[![Bun](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-genai-js%2Fmain%2Fpackage.json&query=%24.engines.bun&label=Bun&logo=bun&color=blue)](https://bun.sh)
[![Node.js](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fraw.githubusercontent.com%2Fmartin-winkler-dev%2Fsmart-genai-js%2Fmain%2Fpackage.json&query=%24.engines.node&label=Node.js&logo=nodedotjs&color=blue)](https://nodejs.org)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Unified client for the **Gemini Developer API** (AI Studio, API key) and **Vertex AI** (Google Cloud credentials),
supporting single requests, embeddings, Decisions (`gemini-decision-1`), named rate-limited models, and batch jobs
using the OpenAI chat completion request shape (`messages`, `response_format`, `tools`, `reasoning_effort`).

## Installation

Requires **Node.js >= 22.12.0** or **Bun >= 1.3.0**. ESM package with bundled TypeScript declarations.

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
| `apiKey` | env `GEMINI_API_KEY` | Developer API key (Developer API only; throws `TypeError` if combined with `vertex`) |
| `vertex` | `false` | `true` or `GenaiVertexOptions` to target Vertex AI |
| `vertex.project` | env `GOOGLE_CLOUD_PROJECT` | Google Cloud project ID (required for Vertex AI) |
| `vertex.location` | env `GOOGLE_CLOUD_LOCATION`, else `'global'` | Vertex AI region, `'us'`, `'eu'`, or `'global'` |
| `vertex.bucket` | env `VERTEX_BATCH_BUCKET` | Cloud Storage bucket name (without `gs://`) for Vertex batch files; without a bucket, `client.batch` is `undefined` |
| `vertex.credentialsFile` | env `GOOGLE_APPLICATION_CREDENTIALS` | Service account or ADC JSON file; falls back to `gcloud auth application-default login` ADC file |
| `vertex.authClient` | none | Custom `AuthClient` instance (bypasses `credentialsFile` disk check, e.g. for GCE/Cloud Run metadata server auth) |
| `vertex.storageUrl` | official GCS endpoint | Custom Cloud Storage endpoint URL |
| `baseUrl` | official endpoint | Custom API endpoint URL |
| `env` | `process.env` | Source object from which default credentials and settings are read |
| `timeoutMs` | `120000` | HTTP timeout per call in milliseconds (positive integer) |
| `defaults` | none | Request defaults and Gemini `config` applied to `generate` and `batch` requests (`embed` and `decide` ignore `defaults`) |
| `batchWaitSec` | `10` | Aggregation window in seconds for [batch calls](#batch-calls) (`1` to `600`) |

- **Environment files**: the package never reads `.env` files directly. Bun loads them automatically; on Node.js use `node --env-file=.env`, `process.loadEnvFile()`, or pass credentials explicitly.
- **Credentials resolution**: on Vertex AI without `vertex.authClient`, `createGenaiClient` resolves `credentialsFile` (`GOOGLE_APPLICATION_CREDENTIALS`, else `$CLOUDSDK_CONFIG/application_default_credentials.json`, else `~/.config/gcloud/application_default_credentials.json` or `%APPDATA%\gcloud\application_default_credentials.json` on Windows), verifies the file exists and is non-empty on disk, and passes `keyFilename` directly to `google-auth-library` (for both Vertex AI and Cloud Storage), avoiding metadata-server timeouts outside Google Cloud.
- **Error contract**: `createGenaiClient`, `createGenaiModels`, and `validateGenaiConfig` throw `TypeError` on invalid options or missing credentials. Methods on clients and named models (including synchronous `batch.requestBytes`) never throw or reject: they return `Result<T, GenaiError>` (`{ ok: true, data } | { ok: false, error }`).
- **Client properties**: `client.mode` is `'developer' | 'vertex'`; `client.batch` is `GenaiBatch | undefined`.
- **`@google/genai` env isolation**: `GOOGLE_GEMINI_BASE_URL` and `GOOGLE_VERTEX_BASE_URL` are ignored (`baseUrl` or the official endpoint is always used); `GOOGLE_GENAI_USE_VERTEXAI`, `GOOGLE_CLOUD_PROJECT`, and `GOOGLE_CLOUD_LOCATION` are passed explicitly; only `GEMINI_API_KEY` (never `GOOGLE_API_KEY`) is read for the Developer API; Vertex AI never uses an API key.

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

- **Supported request fields**: `model` (required unless in `defaults`), `messages` (`system`, `developer`, `user`, `assistant`, `tool`; at least one `user` message required), `response_format` (`text`, `json_object`, `json_schema`), `temperature`, `top_p`, `max_tokens`, `max_completion_tokens`, `seed`, `stop`, `reasoning_effort`, `tools`, `tool_choice` (see [Tool calls](#tool-calls)), `web_search`, and `batch` (see [Batch calls](#batch-calls)). Any unsupported field returns `invalid_request`.
- **Field constraints**:
  - `system` and `developer` messages are joined with `'\n\n'` into `systemInstruction` (overridden if `config.systemInstruction` is set).
  - `temperature` and `top_p` must be finite numbers.
  - `max_tokens` and `max_completion_tokens` must be positive integers and cannot both be set on the same request or defaults object.
  - `seed` must be an int32 integer (`-2147483648` to `2147483647`).
  - `stop` accepts a string or up to 5 strings.
- **Images**: passed as `{ type: 'image_url', image_url: { url: 'data:image/png;base64,...' } }` in `user` messages and forwarded as Gemini `inlineData`. Only base64 `image/*` data URLs with non-empty payloads are accepted (HTTP and `gs://` URLs are rejected). `detail` is accepted and ignored.
- **`response_format`**:
  - `json_object`: sets `responseMimeType: 'application/json'` and parses `data.json`.
  - `json_schema`: forwards `json_schema.schema` as `responseJsonSchema` (`name`, `description`, and `strict` are accepted for OpenAI compatibility and not sent). On Vertex batch, it is converted to typed `responseSchema` (see [Vertex batch request & schema constraints](#vertex-batch-request--schema-constraints)).
  - Strips leading/trailing whitespace and an optional ```` ```json ... ``` ```` code fence before `JSON.parse`. Bypassed when the model returns `toolCalls`.
- **`reasoning_effort`** (`'none' | 'minimal' | 'low' | 'medium' | 'high' | 'xhigh' | 'max'`):
  - Gemini 2.x: sets `thinkingBudget` to `0` / `512` / `1024` / `8192` / `24576` (`xhigh` and `max` match `high`; `gemini-2.5-pro` uses minimum `128` for `none`).
  - Gemini 3+ and other non-Gemma models: sets `thinkingLevel` (`LOW` / `MEDIUM` / `HIGH`, with `xhigh`/`max` as `HIGH`). `none` and `minimal` map to `MINIMAL` only on Gemini 3.0–3.6 Flash and Flash-Lite (`gemini-3-flash-preview`, `gemini-3.1-flash-lite`, etc.), and `LOW` elsewhere.
  - Gemma (`gemma-*`): `none`, `minimal`, and `low` set `MINIMAL`; `medium`, `high`, `xhigh`, and `max` set `HIGH`.
  - Enables `includeThoughts: true` (returning thought summaries in `data.reasoning`), except on Gemini 2.x when budget is `0`. Pass `config: { thinkingConfig: { includeThoughts: false } }` to suppress summaries. When `reasoning_effort` is omitted, no thinking configuration is sent.
- **`web_search: true`**: enables Google Search grounding; `false` overrides a configured default. Grounding web links return in `data.sources` (`{ url, title? }`). OpenAI `web_search_options` is not supported.
- **Defaults and config merging**: client `defaults` populate omitted request fields (deep-copied at creation, so later mutations have no effect). `config` merges per field and `thinkingConfig` merges per member, with call-level values taking precedence. Setting `max_tokens` or `max_completion_tokens` in a call overrides both in defaults.
- **Thinking precedence**: `reasoning_effort`, `config.thinkingConfig.thinkingBudget`, and `config.thinkingConfig.thinkingLevel` act as one setting. If a call sets any of them (request or call `config`), all three are dropped from `defaults` (a dropped default `reasoning_effort` adds no `includeThoughts: true`, while an explicit default `includeThoughts` stays). Within one call, `config` `thinkingBudget` or `thinkingLevel` overrides `reasoning_effort` (while `includeThoughts: true` from `reasoning_effort` stays unless overridden). In `batch.submit`, `options.config` with a budget or level suppresses default `reasoning_effort` across all items; an item's own `reasoning_effort` combined with a budget or level set only by `defaults.config` returns `invalid_request`.
- **Gemini options**: pass `generate(request, { config, signal })`. Call `config` must be a plain object without an own `__proto__` property, `abortSignal`, `httpOptions`, or `automaticContinuation`, and `config.tools` (if set) must be an array or `null`. SDK automatic continuation is disabled and no SDK retries run (one HTTP request per call).
- **Result structure**:
  - `data`: `text`, `json?`, `reasoning?`, `sources?`, `toolCalls?`, `model`, `usage` (`inputTokens`, `outputTokens`, `reasoningTokens`; `outputTokens` excludes thinking tokens), `finishReason`, `ms`, `settings?`, `attempts?`.
  - `error`: `code` (`rate_limited`, `timeout`, `network`, `http`, `aborted`, `blocked`, `truncated`, `invalid_json`, `limiter`, `invalid_request`, `unsupported`, `unexpected`), `message`, `retryable`, `status?`, `retryAfterMs?`, `limiterCode?`, `text?`, `model?`, `usage?`, `ms?`, `output?`, `settings?`, `attempts?`.
- **`settings` and `attempts`**: present on successful results and on valid requests that ran or were aborted after validation. `settings` captures effective options (model, parameters, active tool names, `json`, `batch`, `web_search`, merged `config`). Pre-dispatch `invalid_request` failures have neither `settings` nor `attempts`.
- **Finish reasons and partial output**: any finish reason other than `STOP` (or none) fails even if partial text arrived, preserving the partial `GenaiOutput` in `error.output` (and `text`, `model`, `usage`, `ms` on `error`). `MAX_TOKENS`, `CONTINUATION`, or `continuationToken` return `truncated`; `promptFeedback.blockReason` and safety/recitation stop reasons return `blocked`; missing candidate or tool finish errors (`MALFORMED_FUNCTION_CALL`, `UNEXPECTED_TOOL_CALL`, `TOO_MANY_TOOL_CALLS`, `MISSING_THOUGHT_SIGNATURE`) return `unexpected`.
- **`retryable`**: `true` on HTTP `429` (`rate_limited`), `408`, `409`, `425`, `>= 500`, `timeout`, `network`, and `MALFORMED_FUNCTION_CALL`. Auth failures with an HTTP status return `code: 'http'` with that status and `retryable: false`.

## Tool calls

```ts
import { assistantMessage, toolMessage, type GenaiMessage } from 'smart-genai';

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
```

- **Message helpers**:
  - `assistantMessage(output)` converts `GenaiOutput` into an `assistant` message (preserving `toolCalls` and thought signatures; sets `content: null` when `text === ''` and `toolCalls` is non-empty).
  - `toolMessage(call, result)` formats an execution result as `{ role: 'tool', tool_call_id: call.id, content }`. Strings pass through as-is; other values serialize via `JSON.stringify` (`undefined`, functions, and symbols become `'null'`); non-JSON-serializable values (`BigInt`, cyclic objects) fall back to `String(result)` or `'[unserializable tool result]'`. Never throws.
- **`tools` declarations**: `{ type: 'function', function: { name, description?, parameters?, strict? } }` converts to Gemini `functionDeclarations` (`parameters` sent as `parametersJsonSchema`, or typed `parameters` on Vertex batch; `strict` is accepted and not sent). Names must be unique and 1–128 characters matching `^[A-Za-z0-9_.:-]+$`.
- **`tool_choice`**: `'auto'` -> `AUTO`, `'none'` -> `NONE`, `'required'` -> `ANY`, and `{ type: 'function', function: { name } }` -> `ANY` with `allowedFunctionNames: [name]`. Only sent (and reported in `settings`) when `tools` or `config.tools` (`functionDeclarations`) declares functions; named `tool_choice` must match one of those declarations. `config.toolConfig` overrides generated values per field.
- **Receiving and returning tool calls**: `data.toolCalls` entries have `{ id, type: 'function', function: { name, arguments } }` (`arguments` is a JSON string; missing Gemini IDs get `call_<uuid>`). Consecutive `tool` messages merge into one Gemini `user` turn of `functionResponse` parts; JSON object strings unpack into objects, while other strings wrap as `{ output: text }`.
- **Thought signatures**: Gemini 3+ signs the first function call of each step in `call.extra_content.google.thought_signature` and requires it in history. Tool calls without signatures (e.g. from another provider) receive `skip_thought_signature_validator` on the first call of their assistant turn, except on Gemini 1.x, Gemini 2.x, and non-Google publisher models (`publishers/<name>/models/...`).
- **`web_search` + `tools`**: supported on Gemini 3+; combining them on Gemini 2.x returns `invalid_request`. Server-side tool invocations remain disabled.
- **Validation**: empty `tools`, duplicate/invalid function names, unknown `tool_choice` names, invalid `arguments` JSON, `tool_calls` on non-assistant messages, `content: null` without `tool_calls`, or `tool` messages referencing unknown `tool_call_id`s return `invalid_request`.

## Model list

```ts
const models = await studio.listModels();
// models.data: [{ id: 'gemini-3.5-flash', displayName, inputTokenLimit, outputTokenLimit, actions, thinking }, ...]
```

Paginates across all available base models (consumes no tokens), stripping `models/` and `publishers/<name>/models/` from `id`. On Vertex AI, publisher models report only `id` and `actions: []` (other metadata fields are omitted), so do not filter by `actions` on Vertex AI. Also exposed on named-model objects as `gemini.listModels(options?)`.

## Embeddings

```ts
const vectors = await studio.embed({
    model: 'gemini-embedding-2',
    input: [{ text: 'Cells were incubated at 37 °C.', title: 'Methods' }, 'Untitled paragraph.'],
    taskType: 'RETRIEVAL_DOCUMENT',
    dimensions: 768,
});
// vectors.data: { embeddings: [[...768 numbers], [...]], model: 'gemini-embedding-2', ms }
```

- **Capacity**: 1 to 100 (`MAX_EMBED_INPUTS`) non-empty texts per call, returning one vector per input in order. Each input is wrapped in its own `Content` element so `gemini-embedding-2` does not fuse them into a single vector.
- **Supported fields**: `model`, `input` (`string` or `{ text, title? }[]`), `taskType` (`RETRIEVAL_QUERY`, `RETRIEVAL_DOCUMENT`, `QUESTION_ANSWERING`, `FACT_VERIFICATION`, `CODE_RETRIEVAL_QUERY`, `SEMANTIC_SIMILARITY`, `CLASSIFICATION`, `CLUSTERING`), and `dimensions` (positive integer mapped to `outputDimensionality`). Options: `{ signal }`. Client `defaults` and `batch` do not apply (unknown fields return `invalid_request`).
- **`taskType` and `title`**:
  - `title` (non-empty string) is only allowed with `RETRIEVAL_DOCUMENT`.
  - For `gemini-embedding-2*`, `smart-genai` applies documented text prefixes (`title: <title or none> | text: <text>` for `RETRIEVAL_DOCUMENT`; `task: <name> | query: <text>` for other tasks) instead of API `taskType`/`title` fields.
  - Other models (e.g. `gemini-embedding-001`) receive API `taskType` and `title` fields, and all inputs in one call must share the same `title`. (`gemini-embedding-001` returns normalized vectors only at 3072 dimensions.)
- **Vertex AI**: `publishers/google/models/` prefix is stripped before the SDK call. `gemini-embedding-001` supports up to 100 inputs per call (`predict`); other Gemini models (e.g. `gemini-embedding-2`) and `maas` models support only 1 text per call (`> 1` returns `unsupported`).
- **Output**: `{ embeddings: number[][], model, ms }` (no token usage), plus `settings` and `attempts` only from [named models](#named-models). Returns `unexpected` if vector count mismatches or any vector is empty.
- **Limits & retries**: `client.embed` (also `gemini.client.embed`) sends one request per call without rate limits or retries. For both, use a `kind: 'embed'` entry in [named models](#named-models).

## Decisions (`gemini-decision-1`)

`client.decide` calls Google's Decisions API (`POST /v1beta/decisions`) to score typed questions over a shared input with calibrated probabilities (0 output tokens).

```ts
const decision = await studio.decide({
    input: 'This is the third time my order is late. Give me my money back.',
    questions: {
        damaged: {
            type: 'noul',
            instructions: 'Did the item arrive damaged?',
            criteria: { true_description: 'The product is broken.', false_description: 'Any other complaint.' },
        },
        intent: {
            type: 'choice',
            instructions: 'What does the customer want?',
            criteria: { refund: 'Money back', exchange: 'Another item', information: '' },
        },
        anger: {
            type: 'score',
            instructions: 'How angry is the customer?',
            criteria: ['calm', 'annoyed', 'angry', 'furious'],
        },
    },
});
if (decision.ok) {
    decision.data.answers.damaged; // { type: 'noul', noul: 0.03, confidence: 0.97 }
    decision.data.answers.intent;  // { type: 'choice', choice: 'refund', probabilities: { refund: ..., ... }, confidence: ... }
    decision.data.answers.anger;   // { type: 'score', score: 2.5, probabilities: { 0: ..., 3: ... }, legend: { 0: 'calm', ... }, confidence: ... }
}
```

- **Developer API only**: returns `unsupported` on Vertex AI without sending a request.
- **Model & options**: `model` defaults to `DEFAULT_DECIDE_MODEL` (`'gemini-decision-1'`). Client `defaults` do not apply. Options: `{ signal }`.
- **Limits & retries**: `client.decide` (also `gemini.client.decide`) sends one request per call without rate limits or retries. For both, use a `kind: 'decide'` entry in [named models](#named-models).
- **Questions**: 1 to 1000 (`MAX_DECIDE_QUESTIONS`), keys 1–64 chars matching `[A-Za-z0-9_-]`. `instructions` accepts non-empty string, `{ type: 'text', text }`, or non-empty array of `{ type: 'text', text }`:
  - `noul`: yes/no (`criteria` optional: `{ true_description?, false_description? }`). Answer: `{ type: 'noul', noul, confidence }`.
  - `choice`: `criteria` maps 2–255 non-empty option IDs (no leading/trailing whitespace or control chars) to descriptions (`''` allowed). Answer: `{ type: 'choice', choice, probabilities, confidence }`. With literal `questions`, `choice` and `probabilities` are strictly typed by option IDs (numeric keys as strings).
  - `score`: `criteria` lists 2–255 non-empty level strings, lowest first (2–10 recommended). Answer: `{ type: 'score', score, probabilities, legend, confidence }` keyed by level index (`'0'`, `'1'`, ...).
- **Input**: non-empty (non-whitespace) text or non-empty array of parts:
  - Chat parts: `{ type: 'text', text }` and `{ type: 'image_url', image_url: { url: 'data:image/png;base64,...' } }` (`detail` ignored; converted to `image` block).
  - Media blocks: `image`, `audio`, `video`, `document` with `mime_type` and exactly one of `data` (base64) or `uri`, plus optional `resolution` (`'low' | 'medium' | 'high'` on `image` and `video`; `'ultra_high'` on `image` only).
- **API limits**: 262,144 input tokens per request; inline base64 up to 100 MB (50 MB for PDF); external URIs up to 15 MB; no YouTube or `gs://` URLs; no batch, caching, or tools.
- **Result**: `data` has `answers`, `usage` (`inputTokens`, `outputTokens`), `model`, and `ms`, plus `settings` and `attempts` only from [named models](#named-models). Safety blocks map to `blocked`; missing or mismatched answer types map to `unexpected`.

## Named models

```ts
import { createGenaiModels } from 'smart-genai';

const gemini = createGenaiModels({
    client: { apiKey }, // GenaiClientOptions without defaults; creates gemini.client
    defaults: { temperature: 0, max_tokens: 2000 },
    sharedLimits: { concurrency: 4 },
    limiterOptions: { maxRetries: 2 },
    models: {
        flash38: {
            model: 'gemini-3.8-flash',
            defaults: { reasoning_effort: 'low' },
            modelLimits: { rpm: 10, tpm: 250_000 },
        },
        lite: { model: 'gemini-3.5-flash-lite', modelLimits: { intervalMs: 3000 } },
        flash: 'gemini-3.8-flash', // string shorthand for { model }
        flashBatch: { model: 'gemini-3.8-flash', defaults: { batch: true } },
        docs: { kind: 'embed', model: 'gemini-embedding-2', defaults: { taskType: 'RETRIEVAL_DOCUMENT' } },
        judge: { kind: 'decide' }, // model defaults to 'gemini-decision-1'
    },
});

await gemini.flash38('Summarize this abstract: ...');
await gemini.lite(messages, { web_search: true, reasoning_effort: 'none' });
gemini.flash38.id; // 'gemini-3.8-flash'
await gemini.flash38.batch?.submit([{ key: 'a', request: { messages } }]);
await gemini.docs(['First paragraph.', { text: 'Second paragraph.', title: 'Intro' }], { dimensions: 768 });
await gemini.judge({ input: 'My order is late again.', questions: { angry: { type: 'noul', instructions: 'Is the customer angry?' } } });
await gemini.listModels();
```

- **Entry kinds**: `kind` omitted or `'generate'`: generation model (`{ model, defaults?, modelLimits?, charsPerToken? }` or string shorthand). `'embed'`: embedding model (`{ kind, model, defaults?: { taskType?, dimensions? }, modelLimits?, charsPerToken? }`). `'decide'`: decision model (`{ kind, model?, modelLimits?, charsPerToken? }`; `model` defaults to `'gemini-decision-1'`; no `defaults`). Top-level `defaults` apply to generation entries only. Unknown keys or invalid embed defaults throw `TypeError` at creation.
- **Calling convention**: `gemini.<name>(input, options?)`. `input` is a `string`, non-empty `GenaiContentPart[]` (single user turn), or non-empty `GenaiMessage[]`. `options` accepts any `GenaiRequest` field except `model` and `messages`, plus `signal` and `config` (explicit `undefined` preserves defaults). Returned functions are `this`-independent.
  - Embedding: `gemini.<name>(input, { taskType?, dimensions?, signal? })` with `input` as in [Embeddings](#embeddings); call options override entry `defaults` (explicit `undefined` keeps the default).
  - Decision: `gemini.<name>({ input, questions }, { signal? })`; answers are typed by `questions` as in [Decisions](#decisions-gemini-decision-1). `model` in the request returns `invalid_request`.
- **Properties & naming**: exposes `gemini.client`, `gemini.listModels`, and each model function (with `.id`; `.batch` only on generation models when the client has batch). Model names not valid as JS identifiers use bracket notation (`gemini['flash-lite']`). Reserved names (`client`, `listModels`, `then`, `toJSON`, `__proto__`, and `Object.prototype` keys) throw `TypeError`. Passing a `GenaiClient` instance or `client.defaults` to `createGenaiModels`, setting `model` in defaults, or passing `model`/`messages` in call `options` is rejected (`TypeError` at creation, `invalid_request` at call time).
- **Defaults hierarchy**: top-level `defaults` < `models.<name>.defaults` < per-call `options` (embedding entries: `models.<name>.defaults` < per-call `options`).
- **Traceability**: `settings` includes `name: '<key>'` (embedding and decision results: `settings: { name, model }`); `attempts` reports total API attempts (`0` if rejected before dispatch).
- **Checks before queueing (embedding & decision)**: an invalid request returns `invalid_request` without `settings` and `attempts`. An already aborted `signal` (`aborted`) and unsupported cases (`unsupported`: decision on Vertex AI, more than one embedding input on Vertex models that take 1 text per call) return `settings` and `attempts: 0`. None of these book capacity.
- **Rate limiting (`sharedLimits` & `modelLimits`)**: powered by `smart-limit`. Both accept `concurrency`, `rps`, `rpm`, `rpd`, `tpm`, `tpd`, `intervalMs` (all positive integers `>= 1`), and custom `buckets` (`Bucket[]`). `sharedLimits` applies across all models of all kinds in the instance, so a shared `tpm`/`tpd` is one budget for generation, embedding and decision tokens; use `modelLimits` for per-model quotas. Synchronous calls must acquire capacity from both.
  - Embedding: each input text counts as one request for `rps`, `rpm`, `rpd` and custom `buckets`, as the Gemini API counts them; `intervalMs` and `concurrency` count per call. A call with more inputs than such a limit returns `code: 'limiter'` with `limiterCode: 'COST_EXCEEDS_LIMIT'`.
- **Retry & backoff (`limiterOptions`)**: active even without limits configured. Defaults: `maxRetries: 4`, `retryDelayMs: 2000`, `maxRetryDelayMs: 300000`, `maxWaitMs: 900000` (`Infinity` allowed for `maxWaitMs`).
  - Timeouts, HTTP `408`/`409`/`425`, and `MALFORMED_FUNCTION_CALL` retry per call up to `maxRetries`.
  - HTTP `429`, `5xx`, and `network` errors trigger a queue lock of that model only (`backoff_global`, using `retryAfterMs` or exponential backoff with a `1000` ms minimum) up to `maxWaitMs` without consuming `maxRetries`, followed by a single probe call before resuming queued calls.
  - Calls failing before any attempt return `code: 'limiter'` with `limiterCode` (`COST_EXCEEDS_LIMIT`, `MAX_WAIT_EXCEEDED`); calls failing after at least one attempt return the last provider `GenaiError`.
- **Token reservation & `charsPerToken`**: when `tpm` or `tpd` is configured, each call reserves tokens per kind. `charsPerToken` defaults to `4` (any positive finite number) and can be set globally or per model.
  - Generation: `Math.ceil(chars / charsPerToken) + 1120 * images + (max_tokens ?? max_completion_tokens ?? 1000)`, settled against actual reported usage (`0` if `429` without usage).
  - Embedding: `Math.ceil((text + title chars) / charsPerToken)`; no settlement (no usage reported); refunded on `429`.
  - Decision: `Math.ceil(chars / charsPerToken) + 1120` per non-text input part, with chars of input text, instructions and criteria texts (including choice IDs); settled against reported input + output tokens; refunded on `429`.
- **Batch on named models**: `gemini.<name>.batch` omits `list` and `removeFiles` (use `gemini.client.batch`). Batch calls (`batch: true`) run in a separate queue per model without `concurrency` or rate/token buckets; `limiterOptions` and `signal` still apply. Calls exceeding `maxBytes` fail pre-enqueue with `invalid_request`.

## Batch

```ts
const batch = vertex.batch; // undefined on Vertex AI without a bucket
if (batch) {
    const submitted = await batch.submit([
        { key: 'a', request },
        { key: 'b', request: other }, // all items must share the same model
    ]);
    if (submitted.ok) {
        const status = await batch.check(submitted.data.name);
        if (status.ok && status.data.phase === 'succeeded') {
            status.data.results; // [{ key, result }]
        }
    }
}
```

- **Payload limits**: `batch.maxBytes` is `19_000_000` (19 MB) on the Developer API (inline) and `100_000_000` (100 MB) on Vertex AI (Cloud Storage JSONL). `batch.requestBytes(request, config?)` synchronously returns `Result<number, GenaiError>` for the encoded line excluding the item `key`.
- **Submission (`submit(items, options?)`)**: `items` must be non-empty with unique non-empty `key`s, identical `model`, no `batch` field inside item requests, and total `requestBytes` `<= maxBytes`. `options`: `{ signal?, displayName?, config? }` (`displayName` defaults to `'smart-genai <n> requests'`, which is not unique). To recover a job whose name was lost (e.g. crash right after `submit`), persist a unique `displayName` before calling `submit(items, { displayName })` and look it up via `list({ displayName })`.
- **Status (`check(name, options?)`)**: returns `{ name, state, phase, results?, error? }` where `phase` is `'running' | 'succeeded' | 'failed'`. Item-level RPC errors map to HTTP `GenaiError`s (`RESOURCE_EXHAUSTED`, `UNAVAILABLE`, `INTERNAL`, `DEADLINE_EXCEEDED` marked retryable on Vertex AI).
- **Cancellation (`cancel(name, options?)`)**: cancels an active job (returns `ok` if already finished or cancelling). On Vertex AI, also removes the job's bucket folder, including for already-finished jobs.
- **Listing (`list(options?)`)**: accepts `{ signal?, displayName? }` (exact match filter) and returns `[{ name, displayName?, state, phase, createTime? }]` without modifying jobs or files. On Vertex AI, lists only jobs whose input is directly under `smart-genai/` in the configured bucket; on Developer API, lists all project batch jobs.
- **Crash recovery on Vertex AI (`keepFiles`)**: on Vertex AI, `submit` writes `gs://<bucket>/smart-genai/<project>_<location>_<uuid>/input.jsonl` (`:` in domain-scoped project IDs becomes `~`) and reads results from `.../output/.../*.jsonl`. Terminal `check` deletes the folder unless `keepFiles: true` is passed (a no-op on Developer API). After persisting results, delete the folder with `batch.cancel(name)`. Checking an already-cleaned-up job returns `unexpected`.
- **Orphaned file cleanup (`removeFiles({ keepSucceededMs?, signal? })`)**: Vertex AI only (`unsupported` on Developer API); returns `{ removed, kept }`. Scans own `<project>_<location>_<uuid>` and legacy `<uuid>` folders under `smart-genai/` and retains:
  - folders of non-terminal jobs,
  - folders of succeeded jobs ended less than `keepSucceededMs` ago (default 7 days; unknown end time treated as recent; pass `0` to retain none past the grace period),
  - folders containing any object younger than 1 hour (or with unknown `timeCreated`).

  It deletes all other own and legacy folders, **including output of older succeeded jobs that was never read via `check`**. It never touches files outside `smart-genai/` and is never called automatically.
- **Shared buckets**: different projects/locations sharing one bucket do not touch each other's prefixed folders, but any client's `removeFiles` treats legacy plain-UUID folders as its own and may delete those of another project's or location's jobs (input of running jobs, unread output of succeeded jobs). Let legacy jobs finish and be checked before sharing a bucket, or use one bucket per project and location.

### Vertex batch request & schema constraints

Vertex AI batch import rejects unsupported fields, empty JSON objects, response schema `required` arrays, and mixed schema shapes. `smart-genai` handles these at `submit` and `requestBytes` time:

- **Config translation**: mirrors synchronous SDK placement (e.g. `modelSelectionConfig` -> `generationConfig.modelConfig`, `generationOutputConfig` -> `outputConfig`, short `cachedContent` -> full resource name, `imageConfig.outputMimeType` -> `imageOutputOptions.mimeType`, string `speechConfig` -> prebuilt voice). Unsupported fields (`store`, `enableEnhancedCivicAnswers`, `fileSearch`, `mcpServers`, `toolConfig.includeServerSideToolInvocations`, `partMetadata`, `toolCall`, `toolResponse`, `consentAudio`, `voiceConsentSignature`, or unknown fields) fail with `invalid_request` naming the field and item key. `automaticFunctionCalling` is omitted.
- **Empty objects**: empty tool call `arguments` are omitted, empty tool results (`'{}'`) become `{ output: '' }`, empty `properties` on tool schemas are dropped, and `web_search` uses `{ excludeDomains: [] }`. Any remaining empty object (e.g. `properties: {}` or `items: {}` in `json_schema`, or `{ codeExecution: {} }` in `config.tools`) fails with `invalid_request`.
- **`required` in response schemas**: removed at every depth of `json_schema` / `responseSchema` (tool parameter schemas keep `required`), so always validate parsed JSON answers against required fields.
- **Typed schemas in Vertex batch**: `parametersJsonSchema` and `responseJsonSchema` (and typed schemas in `config`) are converted to Vertex typed form (uppercase `type`, `null` branches/types -> `nullable`, `$schema`/`$id`/`$comment`/`additionalProperties` dropped). Only `anyOf`, `default`, `description`, `enum` (strings only), `example`, `format`, `items`, `maxItems`, `maxLength`, `maxProperties`, `maximum`, `minItems`, `minLength`, `minProperties`, `minimum`, `nullable`, `pattern`, `properties`, `propertyOrdering`, `required`, `title`, and `type` are supported; other keywords (`$ref`, `$defs`, `const`, `oneOf`, `allOf`, `not`), `type` + `anyOf`, `type: 'null'` alone, or non-string `enum` values return `invalid_request`.
- **Mixed schemas**: Vertex AI merges property names across schemas in a single job. `smart-genai` strips foreign properties from parsed `json` (updating `text` to match) and `toolCalls` arguments using the echoed line request, though one job per schema still yields cleaner model output.

## Batch calls

```ts
const studio = createGenaiClient({ batchWaitSec: 10 });
const [a, b] = await Promise.all([
    studio.generate({ ...request, batch: true }),
    studio.generate({ ...other, batch: true }),
]);
```

- **Overview**: `batch: true` (per-call or in `defaults`) routes `generate` through the Batch API (50% cost discount on Gemini Batch API) while returning an individual `Result<GenaiOutput, GenaiError>`.
- **Grouping & splitting**: calls within `batchWaitSec` (default `10` s from the first call in a group) are grouped by `(model, config, response_format, tools)` and submitted as one or more jobs split to `batch.maxBytes`. A single request exceeding `maxBytes` fails immediately with `invalid_request`.
- **Polling & timeout**: active jobs are polled every 30 s for up to 72 hours (then cancelled and settled with `timeout`). Retryable poll errors retry (respecting `retryAfterMs`); non-retryable poll errors settle all calls and cancel the job unless it already reached a terminal state.
- **Job failures**: if a job fails or is cancelled, every call in that job resolves with `{ code: 'unexpected' }`. If the submission itself fails (e.g. rate limit), all grouped calls receive that error.
- **Cancellation**: aborting before submission removes the call from its pending group (`attempts: 0`); aborting after submission settles that call as `aborted`, and cancels the batch job if all calls in it are aborted.
- **Prerequisites & lifecycle**: returns `unsupported` on Vertex AI without a bucket; free-tier Developer API keys reject batch jobs (`FAILED_PRECONDITION`). Batch call groups and pollers live in memory: if the host process exits, in-flight call promises are lost while submitted jobs continue running (and billing) in Google Cloud. For crash-safe persistence across process restarts, use `client.batch.submit` and `client.batch.check`.

## Helpers & Constants

- `credentialsFile(env?)`: resolves the Google credentials path (`GOOGLE_APPLICATION_CREDENTIALS`, else `$CLOUDSDK_CONFIG/application_default_credentials.json`, else platform gcloud ADC path).
- `mergeGenaiConfig(...configs)`: merges `GenerateContentConfig` objects left-to-right (merging `thinkingConfig` per member), skipping `undefined`.
- `validateGenaiConfig(config, optionName)`: validates a `GenerateContentConfig` object; throws `TypeError` on invalid shape, own `__proto__`, `abortSignal`, `httpOptions`, `automaticContinuation`, or invalid `tools`.
- Constants: `GEMINI_API_KEY_ENV` (`'GEMINI_API_KEY'`), `VERTEX_ENV` (`{ project, location, bucket, credentialsFile }`), `MAX_EMBED_INPUTS` (`100`), `DEFAULT_DECIDE_MODEL` (`'gemini-decision-1'`), `MAX_DECIDE_QUESTIONS` (`1000`).
- Types: all `Genai*` types, `Result<T, E>`, and `AuthClient` (from `google-auth-library`) are exported from `'smart-genai'`.
