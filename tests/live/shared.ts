// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// Live: helpers and cases shared by the Developer API suite (`developer.live.ts`, `embed.live.ts`) and the Vertex AI
// suite (`vertex.live.ts`). Each case runs against the client a suite passes in; no requests at import time.
// Logs one line per request; the API key, project id and bucket name are masked in everything logged or thrown.

import { expect } from 'bun:test';
import { assistantMessage, createGenaiClient, createGenaiModels, toolMessage } from '../../src/index.js';
import type {
    GenaiBatchItem,
    GenaiBatchStatus,
    GenaiCallOptions,
    GenaiClient,
    GenaiClientOptions,
    GenaiError,
    GenaiErrorCode,
    GenaiLimiterOptions,
    GenaiMessage,
    GenaiOutput,
    GenaiRequest,
    GenaiTool,
    Result,
} from '../../src/types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Live case run against one client and one model. */
export type LiveCase = { name: string; run(client: GenaiClient, model: string): Promise<void> };

/** Ids of the core cases every important model runs. */
export type CoreCaseId = 'text' | 'image' | 'reasoning' | 'sampling' | 'search' | 'tools';

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Env `LIVE_PAID_KEY=1`: the key may batch and ground with Google Search (both fail on a free key). */
export const IS_PAID_KEY = process.env.LIVE_PAID_KEY === '1';

/** Env `LIVE_SKIP_MODELS`: comma-separated models whose cases are skipped, e.g. one overloaded on a free key. */
export const SKIPPED_MODELS: ReadonlySet<string> = new Set(
    (process.env.LIVE_SKIP_MODELS ?? '').split(',').map((model) => model.trim()),
);

/** Timeout of one synchronous case (retries and up to three requests included). */
export const CASE_TIMEOUT_MS = 240_000;

/** Timeout of one batch case: 40 min wait, cancel and result download. */
export const BATCH_TIMEOUT_MS = 45 * 60_000;

/** Timeout of one HTTP request. */
export const REQUEST_TIMEOUT_MS = 120_000;

/** Data URL of a 32x32 PNG filled with pure red (`#ff0000`). */
export const RED_PNG =
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAIAAAD8GO2jAAAAJ0lEQVR42u3NsQkAAAjAsP7/tF7hIASyp6lTCQQCgUAgEAgEgi/BAjLD/C5w/SM9AAAAAElFTkSuQmCC';

/**
 * **Limiter options of live named models**: retries by smart-limit only, no repeats by `send`.
 *
 * - 429/5xx/network backoff does not consume `maxRetries`; bounded by `maxWaitMs`
 * - `maxWaitMs` must stay below test timeout (`CASE_TIMEOUT_MS`)
 */
export const LIVE_LIMITER: GenaiLimiterOptions = { maxRetries: 3, maxWaitMs: 60_000 };

/** Repeats of a request after a retryable 5xx (overload) or a rate limit (requests per minute of a free key). */
const MAX_RETRIES = 2;

/** Pause before a repeat after a 5xx. */
const RETRY_PAUSE_MS = 5_000;

/** Pause before a repeat after a rate limit without `retryAfterMs`. */
const RATE_PAUSE_MS = 30_000;

/** Longest pause taken from `retryAfterMs`. */
const MAX_RATE_PAUSE_MS = 65_000;

/** No repeat once a request and its repeats took this long, so a case stays below `CASE_TIMEOUT_MS`. */
const RETRY_BUDGET_MS = 60_000;

/** Poll interval of batch jobs. */
const BATCH_POLL_MS = 30_000;

/** Wait for a batch job before it is cancelled. */
const BATCH_MAX_WAIT_MS = 40 * 60_000;

/** Allowed clock difference between this machine and the job `createTime`. */
const JOB_CLOCK_SLACK_MS = 60_000;

/** Values never logged. */
const SECRETS = [process.env.GEMINI_API_KEY, process.env.GOOGLE_CLOUD_PROJECT, process.env.VERTEX_BATCH_BUCKET].filter(
    (value): value is string => value !== undefined && value !== '',
);

/** Matches project resource segments such as `projects/123456` or `projects/my-project`. */
const MATCH_PROJECT_SEGMENT = /projects\/[^/\s"'`]+/g;

/** Matches runs of whitespace, collapsed in log lines. */
const MATCH_WHITESPACE = /\s+/g;

/** Function tools with different parameter shapes. */
const WEATHER: GenaiTool = {
    type: 'function',
    function: {
        name: 'get_weather',
        description: 'Current weather of a city',
        parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
    },
};
const TIME: GenaiTool = {
    type: 'function',
    function: {
        name: 'get_time',
        description: 'Current local time in an IANA time zone',
        parameters: { type: 'object', properties: { zone: { type: 'string' } }, required: ['zone'] },
    },
};
/** Tool without arguments: empty `properties`. */
const UTC_TIME: GenaiTool = {
    type: 'function',
    function: {
        name: 'get_utc_time',
        description: 'Current UTC time',
        parameters: { type: 'object', properties: {} },
    },
};
const TOOLS = [WEATHER, TIME];

/** Weather result returned for every `get_weather` call. */
const WEATHER_RESULT = { city: 'Rome', celsius: 17, sky: 'light rain' };

/** Schema of the colour question about `RED_PNG`. */
const COLOR_SCHEMA = {
    type: 'object',
    properties: {
        color: { type: 'string', enum: ['red', 'green', 'blue', 'yellow', 'black', 'white'] },
        shapes: { type: 'integer', description: 'Number of separate shapes in the image' },
    },
    required: ['color'],
};

/** Schema with `enum`, `anyOf` with `null` and `additionalProperties`. */
const KIND_SCHEMA = {
    type: 'object',
    properties: {
        kind: { type: 'string', enum: ['noun', 'verb', 'adjective'] },
        note: { anyOf: [{ type: 'string' }, { type: 'null' }] },
    },
    required: ['kind', 'note'],
    additionalProperties: false,
};

/** Unsigned tool history, as from another provider. */
const UNSIGNED_HISTORY: GenaiMessage[] = [
    { role: 'user', content: 'What is the weather in Rome right now?' },
    {
        role: 'assistant',
        content: null,
        tool_calls: [
            { id: 'call_1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Rome"}' } },
        ],
    },
    { role: 'tool', tool_call_id: 'call_1', content: JSON.stringify(WEATHER_RESULT) },
];

/** Core cases: every important model runs a subset of these. */
export const CORE_CASES: Record<CoreCaseId, LiveCase> = {
    text: {
        name: 'plain text: answer, finish reason, usage, settings',
        async run(client, model) {
            const label = `${model} text`;
            const result = await generate(client, label, {
                model,
                messages: [{ role: 'user', content: 'Reply with the single word: pong' }],
                temperature: 0,
            });
            const data = expectOk(label, result);
            expect(data.text.toLowerCase()).toContain('pong');
            expect(data.finishReason).toBe('STOP');
            expect(data.usage.inputTokens ?? 0).toBeGreaterThan(0);
            // Vertex AI sometimes reports no output token count next to thinking tokens
            expect(data.usage.outputTokens ?? 1).toBeGreaterThan(0);
            expect(data.settings?.model).toBe(model);
            expect(data.attempts).toBe(1);
            expect(data.ms).toBeGreaterThan(0);
        },
    },
    image: {
        name: 'system message, JSON schema and image',
        async run(client, model) {
            const label = `${model} image`;
            const result = await generate(client, label, {
                model,
                messages: [
                    { role: 'system', content: 'You describe images precisely.' },
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'Which colour fills this image?' },
                            { type: 'image_url', image_url: { url: RED_PNG } },
                        ],
                    },
                ],
                response_format: { type: 'json_schema', json_schema: { name: 'image', schema: COLOR_SCHEMA } },
                temperature: 0,
                max_tokens: 4000,
            });
            const data = expectOk(label, result);
            expect(data.json).toMatchObject({ color: 'red' });
            expect(data.settings?.json).toBe(true);
        },
    },
    reasoning: {
        name: 'json_object with reasoning high',
        async run(client, model) {
            const label = `${model} reasoning`;
            const result = await generate(client, label, {
                model,
                messages: [
                    {
                        role: 'user',
                        content:
                            'Return a JSON object with the number "a" and the string "b": a = 17 * 23, b = the string "zotero" reversed.',
                    },
                ],
                response_format: { type: 'json_object' },
                reasoning_effort: 'high',
                max_tokens: 6000,
            });
            const data = expectOk(label, result);
            const json = data.json as { a?: unknown; b?: unknown };
            expect(Number(json.a)).toBe(391);
            expect(json.b).toBe('oretoz');
            expect(data.usage.reasoningTokens ?? 0).toBeGreaterThan(0);
            expect(data.settings?.reasoning_effort).toBe('high');
        },
    },
    sampling: {
        name: 'reasoning none, temperature, seed and stop',
        async run(client, model) {
            const label = `${model} sampling`;
            const result = await generate(client, label, {
                model,
                messages: [{ role: 'user', content: 'Count from 1 to 10 in digits, comma separated, nothing else.' }],
                reasoning_effort: 'none',
                temperature: 0,
                seed: 7,
                stop: ['6'],
                max_tokens: 2000,
            });
            const data = expectOk(label, result);
            expect(data.text).toContain('5');
            expect(data.text).not.toContain('6');
            expect(data.finishReason).toBe('STOP');
            expect(data.settings).toMatchObject({ reasoning_effort: 'none', temperature: 0, seed: 7, stop: ['6'] });
        },
    },
    search: {
        name: 'web search with sources',
        async run(client, model) {
            const label = `${model} search`;
            const result = await generate(client, label, {
                model,
                messages: [
                    { role: 'user', content: 'Search the web: which organisation develops Zotero? One sentence.' },
                ],
                web_search: true,
                max_tokens: 3000,
            });
            const data = expectOk(label, result);
            expect(data.text.length).toBeGreaterThan(0);
            expect(data.sources?.length ?? 0).toBeGreaterThan(0);
            expect(data.sources?.[0]?.url).toStartWith('http');
            expect(data.settings?.web_search).toBe(true);
        },
    },
    tools: {
        name: 'tool round trip with signed and unsigned history',
        async run(client, model) {
            const messages: GenaiMessage[] = [
                { role: 'user', content: 'What is the weather in Rome right now? Use the tool.' },
            ];
            const asked = expectOk(
                `${model} tool call`,
                await generate(client, `${model} tool call`, {
                    model,
                    messages,
                    tools: TOOLS,
                    temperature: 0,
                    max_tokens: 2000,
                }),
            );
            const [call] = asked.toolCalls ?? [];
            expect(call?.function.name).toBe('get_weather');
            expect(JSON.parse(call?.function.arguments ?? '{}').city).toContain('Rome');
            expect(asked.settings?.tools).toEqual(['get_weather', 'get_time']);
            messages.push(assistantMessage(asked));
            for (const toolCall of asked.toolCalls ?? []) {
                messages.push(toolMessage(toolCall, WEATHER_RESULT));
            }
            const signed = expectOk(
                `${model} tool answer`,
                await generate(client, `${model} tool answer`, {
                    model,
                    messages,
                    tools: TOOLS,
                    temperature: 0,
                    max_tokens: 2000,
                }),
            );
            expect(signed.text).toContain('17');
            const unsigned = expectOk(
                `${model} tool answer, unsigned history`,
                await generate(client, `${model} tool answer, unsigned history`, {
                    model,
                    messages: UNSIGNED_HISTORY,
                    tools: TOOLS,
                    temperature: 0,
                    max_tokens: 2000,
                }),
            );
            expect(unsigned.text).toContain('17');
        },
    },
};

/** Breadth cases on a cheap model with a large quota; one request each. */
export const BREADTH_CASES: LiveCase[] = [
    {
        name: 'multi-turn conversation with developer role',
        async run(client, model) {
            const label = `${model} multi-turn`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [
                        { role: 'developer', content: 'Answer in German, in one short sentence.' },
                        { role: 'user', content: 'My name is Ada.' },
                        { role: 'assistant', content: 'Hallo Ada!' },
                        { role: 'user', content: 'What is my name?' },
                    ],
                    temperature: 0,
                }),
            );
            expect(data.text).toContain('Ada');
        },
    },
    {
        name: 'text parts only',
        async run(client, model) {
            const label = `${model} text parts`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [
                        {
                            role: 'user',
                            content: [
                                { type: 'text', text: 'Say' },
                                { type: 'text', text: 'hello in French, one word.' },
                            ],
                        },
                    ],
                    temperature: 0,
                }),
            );
            expect(data.text.toLowerCase()).toContain('bonjour');
        },
    },
    {
        name: 'max_tokens too small: truncated with partial output',
        async run(client, model) {
            const label = `${model} truncated`;
            const result = await generate(client, label, {
                model,
                messages: [{ role: 'user', content: 'Write 300 words about bibliographies.' }],
                reasoning_effort: 'none',
                max_tokens: 20,
            });
            const error = expectError(label, result, 'truncated');
            expect(error.output?.finishReason).toBe('MAX_TOKENS');
            expect(error.settings?.max_tokens).toBe(20);
        },
    },
    {
        name: 'max_completion_tokens and top_p',
        async run(client, model) {
            const label = `${model} top_p`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [{ role: 'user', content: 'Name three reference managers, comma separated.' }],
                    max_completion_tokens: 500,
                    top_p: 0.5,
                }),
            );
            expect(data.text.length).toBeGreaterThan(0);
            expect(data.settings).toMatchObject({ max_tokens: 500, top_p: 0.5 });
        },
    },
    {
        name: 'tool_choice required',
        async run(client, model) {
            const label = `${model} tool_choice required`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [{ role: 'user', content: 'Hello!' }],
                    tools: TOOLS,
                    tool_choice: 'required',
                    temperature: 0,
                }),
            );
            expect(data.toolCalls?.length ?? 0).toBeGreaterThan(0);
            expect(data.settings?.tool_choice).toBe('required');
        },
    },
    {
        name: 'tool_choice with a named function',
        async run(client, model) {
            const label = `${model} tool_choice named`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [{ role: 'user', content: 'Hello!' }],
                    tools: TOOLS,
                    tool_choice: { type: 'function', function: { name: 'get_time' } },
                    temperature: 0,
                }),
            );
            expect(data.toolCalls?.map((call) => call.function.name)).toEqual(['get_time']);
        },
    },
    {
        name: 'tool_choice none',
        async run(client, model) {
            const label = `${model} tool_choice none`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [{ role: 'user', content: 'What is the weather in Rome?' }],
                    tools: TOOLS,
                    tool_choice: 'none',
                    temperature: 0,
                }),
            );
            expect(data.toolCalls).toBeUndefined();
            expect(data.text.length).toBeGreaterThan(0);
        },
    },
    {
        name: 'tool_choice required with tools from config.tools',
        async run(client, model) {
            const label = `${model} config.tools`;
            const options: GenaiCallOptions = {
                config: {
                    tools: [
                        {
                            functionDeclarations: [
                                {
                                    name: 'get_weather',
                                    description: 'Current weather of a city',
                                    parametersJsonSchema: WEATHER.function.parameters,
                                },
                            ],
                        },
                    ],
                },
            };
            const data = expectOk(
                label,
                await generate(
                    client,
                    label,
                    {
                        model,
                        messages: [{ role: 'user', content: 'Hello!' }],
                        tool_choice: 'required',
                        temperature: 0,
                    },
                    options,
                ),
            );
            expect(data.toolCalls?.[0]?.function.name).toBe('get_weather');
            expect(data.settings?.tool_choice).toBe('required');
        },
    },
    {
        name: 'two parallel tool calls',
        async run(client, model) {
            const label = `${model} parallel tools`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [
                        {
                            role: 'user',
                            content: 'What is the weather in Rome and the local time in Europe/Rome? Call both tools.',
                        },
                    ],
                    tools: TOOLS,
                    temperature: 0,
                }),
            );
            expect(data.toolCalls?.map((call) => call.function.name).sort()).toEqual(['get_time', 'get_weather']);
            expect(new Set(data.toolCalls?.map((call) => call.id)).size).toBe(2);
        },
    },
    {
        name: 'reasoning minimal',
        async run(client, model) {
            const label = `${model} reasoning minimal`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [{ role: 'user', content: 'What is 12 * 12? Answer with the number.' }],
                    reasoning_effort: 'minimal',
                }),
            );
            expect(data.text).toContain('144');
        },
    },
    {
        name: 'reasoning medium, thoughts off via config',
        async run(client, model) {
            const label = `${model} reasoning medium`;
            const data = expectOk(
                label,
                await generate(
                    client,
                    label,
                    {
                        model,
                        messages: [{ role: 'user', content: 'What is 13 * 13? Answer with the number.' }],
                        reasoning_effort: 'medium',
                    },
                    { config: { thinkingConfig: { includeThoughts: false } } },
                ),
            );
            expect(data.text).toContain('169');
            expect(data.reasoning).toBeUndefined();
        },
    },
    {
        name: 'config thinkingLevel wins over reasoning_effort',
        async run(client, model) {
            const label = `${model} thinkingLevel`;
            const data = expectOk(
                label,
                await generate(
                    client,
                    label,
                    {
                        model,
                        messages: [{ role: 'user', content: 'What is 14 * 14? Answer with the number.' }],
                        reasoning_effort: 'high',
                    },
                    { config: { thinkingConfig: { thinkingLevel: 'LOW' as never } } },
                ),
            );
            expect(data.text).toContain('196');
        },
    },
    {
        name: 'client defaults; call thinkingBudget replaces default reasoning_effort',
        async run(client, model) {
            const label = `${model} client defaults`;
            const options: GenaiClientOptions =
                client.mode === 'vertex' ? { vertex: true } : { timeoutMs: REQUEST_TIMEOUT_MS };
            const withDefaults = createGenaiClient({
                ...options,
                defaults: { model, temperature: 0, reasoning_effort: 'low', max_tokens: 800 },
            });
            const data = expectOk(
                label,
                await generate(
                    withDefaults,
                    label,
                    { messages: [{ role: 'user', content: 'Reply with the words: default ok' }] },
                    { config: { thinkingConfig: { thinkingBudget: 512 } } },
                ),
            );
            expect(data.text.toLowerCase()).toContain('default ok');
            expect(data.settings).toMatchObject({ model, temperature: 0, max_tokens: 800 });
            expect(data.settings?.reasoning_effort).toBeUndefined();
        },
    },
    {
        name: 'json_schema with enum, anyOf null and additionalProperties',
        async run(client, model) {
            const label = `${model} anyOf schema`;
            const data = expectOk(
                label,
                await generate(client, label, {
                    model,
                    messages: [{ role: 'user', content: 'Classify the invented word "blorf" in "a shiny blorf".' }],
                    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema: KIND_SCHEMA } },
                    temperature: 0,
                }),
            );
            const json = data.json as { kind?: unknown; note?: unknown };
            expect(json.kind).toBe('noun');
            expect(json.note === null || typeof json.note === 'string').toBe(true);
        },
    },
    {
        name: 'unknown model: HTTP 404 with a message',
        async run(client) {
            const label = 'unknown model';
            const result = await generate(client, label, {
                model: 'gemini-0-nope',
                messages: [{ role: 'user', content: 'x' }],
            });
            const error = expectError(label, result, 'http', 404);
            expect(error.retryable).toBe(false);
            expect(error.message.trim().length).toBeGreaterThan(0);
        },
    },
    {
        name: 'abort after 50 ms',
        async run(client, model) {
            const label = `${model} abort`;
            const controller = new AbortController();
            setTimeout(() => controller.abort(), 50);
            const result = await generate(
                client,
                label,
                { model, messages: [{ role: 'user', content: 'Write 500 words about rivers.' }] },
                { signal: controller.signal },
            );
            expectError(label, result, 'aborted');
        },
    },
];

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/** Masks API key, project id, bucket name and project resource segments; collapses whitespace. */
export function mask(text: string): string {
    let masked = text;
    for (const secret of SECRETS) {
        masked = masked.replaceAll(secret, '<masked>');
    }
    return masked.replace(MATCH_PROJECT_SEGMENT, 'projects/<masked>').replace(MATCH_WHITESPACE, ' ');
}

/** Logs one masked line. */
export function log(label: string, text: string): void {
    console.log(mask(`[live] ${label}: ${text}`));
}

/** Compact masked description of an error. */
export function describeError(error: GenaiError): string {
    const output = error.output === undefined ? '' : ` finish=${error.output.finishReason}`;
    return `${error.code} status=${error.status ?? '-'} retryable=${error.retryable} ${error.message.slice(0, 240)}${output}`;
}

/** Compact description of a generate answer. */
export function describeOutput(data: GenaiOutput): string {
    const usage = `${data.usage.inputTokens}/${data.usage.outputTokens}/${data.usage.reasoningTokens ?? 0}`;
    return [
        `text=${JSON.stringify(data.text.slice(0, 80))}`,
        data.json === undefined ? '' : `json=${JSON.stringify(data.json).slice(0, 120)}`,
        data.toolCalls === undefined
            ? ''
            : `tools=${data.toolCalls.map((call) => `${call.function.name}${call.function.arguments}`).join(',')}`,
        data.reasoning === undefined ? '' : `reasoning=${data.reasoning.length}ch`,
        data.sources === undefined ? '' : `sources=${data.sources.length}`,
        `finish=${data.finishReason} usage=${usage}`,
    ]
        .filter((part) => part !== '')
        .join(' ');
}

/**
 * **Sends one live request and logs it**: label, ms, compact result.
 *
 * - repeats up to `MAX_RETRIES` times on a retryable 5xx (overload) or a rate limit; nothing else
 * - rate limit: waits `retryAfterMs` (at most `MAX_RATE_PAUSE_MS`), for the per-minute limit of a free key
 * - no repeat after `RETRY_BUDGET_MS`, e.g. after a request that ran into its timeout
 */
export async function send<T>(
    label: string,
    request: () => Promise<Result<T, GenaiError>>,
    describe: (data: T) => string,
): Promise<Result<T, GenaiError>> {
    const startedAt = performance.now();
    let result = await request();
    for (let retry = 1; retry <= MAX_RETRIES && !result.ok && isTransient(result.error); retry++) {
        if (performance.now() - startedAt > RETRY_BUDGET_MS) {
            break;
        }
        const pauseMs = retryPauseMs(result.error);
        log(label, `retry ${retry} in ${pauseMs} ms after ${describeError(result.error)}`);
        await Bun.sleep(pauseMs);
        result = await request();
    }
    logResult(label, startedAt, result, describe);
    return result;
}

/** Sends one live request without repeats and logs it like `send`; for calls retried by smart-limit. */
export async function sendOnce<T>(
    label: string,
    request: () => Promise<Result<T, GenaiError>>,
    describe: (data: T) => string,
): Promise<Result<T, GenaiError>> {
    const startedAt = performance.now();
    const result = await request();
    logResult(label, startedAt, result, describe);
    return result;
}

/** `client.generate` through `send`. */
export function generate(
    client: GenaiClient,
    label: string,
    request: GenaiRequest,
    options?: GenaiCallOptions,
): Promise<Result<GenaiOutput, GenaiError>> {
    return send(label, () => client.generate(request, options), describeOutput);
}

/**
 * Returns the data of a successful result.
 * @throws {Error} With the masked error when the result failed.
 */
export function expectOk<T>(label: string, result: Result<T, GenaiError>): T {
    if (!result.ok) {
        throw new Error(mask(`${label}: expected ok, got ${describeError(result.error)}`));
    }
    return result.data;
}

/**
 * Returns the error of a failed result with the expected code (and status).
 * @throws {Error} With a masked description when the result differs.
 */
export function expectError<T>(
    label: string,
    result: Result<T, GenaiError>,
    code: GenaiErrorCode,
    status?: number,
): GenaiError {
    if (result.ok) {
        throw new Error(`${label}: expected ${code}, got ok`);
    }
    const { error } = result;
    if (error.code !== code || (status !== undefined && error.status !== status)) {
        throw new Error(mask(`${label}: expected ${code} ${status ?? ''}, got ${describeError(error)}`));
    }
    return error;
}

/** Core cases by id, in the given order. */
export function coreCases(ids: readonly CoreCaseId[]): LiveCase[] {
    return ids.map((id) => CORE_CASES[id]);
}

/**
 * **Local validation of requests that the API would reject**: no request leaves the client.
 *
 * - unknown field, `seed` beyond int32 and not an integer, more than 5 `stop` sequences, empty image data,
 *   `config.tools` no list, `tool_choice` naming no tool
 */
export async function expectLocalValidation(client: GenaiClient, model: string): Promise<void> {
    const user: GenaiMessage[] = [{ role: 'user', content: 'x' }];
    const invalid: [string, GenaiRequest, GenaiCallOptions?][] = [
        ['unknown field', { model, messages: user, frequency_penalty: 1 } as GenaiRequest],
        ['seed 2^31', { model, messages: user, seed: 2 ** 31 }],
        ['seed 1.5', { model, messages: user, seed: 1.5 }],
        ['six stop sequences', { model, messages: user, stop: ['a', 'b', 'c', 'd', 'e', 'f'] }],
        [
            'empty image data',
            {
                model,
                messages: [
                    { role: 'user', content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,' } }] },
                ],
            },
        ],
        ['config.tools no list', { model, messages: user, tools: TOOLS }, { config: { tools: {} as never } }],
        [
            'tool_choice naming no tool',
            { model, messages: user, tools: TOOLS, tool_choice: { type: 'function', function: { name: 'nope' } } },
        ],
        ['max_tokens with max_completion_tokens', { model, messages: user, max_tokens: 5, max_completion_tokens: 5 }],
    ];
    const startedAt = performance.now();
    for (const [label, request, options] of invalid) {
        const error = expectError(label, await client.generate(request, options), 'invalid_request');
        expect(error.retryable).toBe(false);
    }
    log(
        `${model} local validation`,
        `${invalid.length} requests refused in ${Math.round(performance.now() - startedAt)} ms`,
    );
}

/**
 * **Named models (`createGenaiModels`)**: string prompt, content parts with image, messages with options.
 *
 * - shared and model limits, `limiterOptions.maxWaitMs: Infinity`, defaults per model and per call
 */
export async function expectNamedModels(client: Omit<GenaiClientOptions, 'defaults'>, model: string): Promise<void> {
    const gemini = createGenaiModels({
        client,
        defaults: { temperature: 0, max_tokens: 1500 },
        sharedLimits: { concurrency: 2 },
        limiterOptions: { maxRetries: 1, maxWaitMs: Number.POSITIVE_INFINITY },
        models: { lite: { model, modelLimits: { rpm: 30, tpm: 100_000 }, defaults: { reasoning_effort: 'low' } } },
    });
    expect(gemini.lite.id).toBe(model);
    const [word, image, capital] = await Promise.all([
        send('named string prompt', () => gemini.lite('Reply with the single word: one'), describeOutput),
        send(
            'named content parts',
            () =>
                gemini.lite([
                    { type: 'text', text: 'Which colour fills this image? One word.' },
                    { type: 'image_url', image_url: { url: RED_PNG } },
                ]),
            describeOutput,
        ),
        send(
            'named messages',
            () =>
                gemini.lite(
                    [
                        { role: 'system', content: 'Answer in one word.' },
                        { role: 'user', content: 'Capital of Italy?' },
                    ],
                    { reasoning_effort: 'none' },
                ),
            describeOutput,
        ),
    ]);
    const wordData = expectOk('named string prompt', word);
    expect(wordData.text.toLowerCase()).toContain('one');
    expect(wordData.settings).toMatchObject({ name: 'lite', model, temperature: 0, max_tokens: 1500 });
    expect(wordData.settings?.reasoning_effort).toBe('low');
    expect(expectOk('named content parts', image).text.toLowerCase()).toContain('red');
    const capitalData = expectOk('named messages', capital);
    expect(capitalData.text).toContain('Rom');
    expect(capitalData.settings?.reasoning_effort).toBe('none');
}

/**
 * **Embeddings**: three documents and one query; the query closest to its answer.
 *
 * - `singleInput`: one input per call (Vertex AI, `gemini-embedding-2`)
 * - titles only for `gemini-embedding-2` (other models share one title per call)
 */
export async function expectEmbeddings(client: GenaiClient, model: string, singleInput: boolean): Promise<void> {
    const documents = [
        { text: 'The bakery opens at six and sells rye bread and croissants.', title: 'Shop' },
        { text: 'Mitochondria produce ATP through oxidative phosphorylation.', title: 'Cell biology' },
        { text: 'The striker scored twice in the second half of the final.', title: 'Sports' },
    ];
    const query = 'Which organelle makes the energy carrier of the cell?';
    const input = model.includes('gemini-embedding-2') ? documents : documents.map((document) => document.text);
    const groups = singleInput ? input.map((entry) => [entry]) : [input];
    const vectors: number[][] = [];
    for (const group of groups) {
        const label = `${model} embed documents`;
        const result = await send(
            label,
            () => client.embed({ model, input: group, taskType: 'RETRIEVAL_DOCUMENT', dimensions: 768 }),
            describeEmbedding,
        );
        vectors.push(...expectOk(label, result).embeddings);
    }
    const label = `${model} embed query`;
    const queried = await send(
        label,
        () => client.embed({ model, input: query, taskType: 'RETRIEVAL_QUERY', dimensions: 768 }),
        describeEmbedding,
    );
    const [queryVector = []] = expectOk(label, queried).embeddings;
    expect(vectors).toHaveLength(documents.length);
    for (const vector of [...vectors, queryVector]) {
        expect(vector).toHaveLength(768);
    }
    const scores = vectors.map((vector) => cosine(queryVector, vector));
    log(`${model} embed scores`, scores.map((score) => score.toFixed(3)).join(' '));
    expect(scores.indexOf(Math.max(...scores))).toBe(1);
}

/**
 * **Items of the one batch job per suite**: one model, shapes that once failed Vertex batch jobs.
 *
 * - `schema`: `required` list, `batch: undefined`
 * - `other-schema`: other properties, `anyOf` with `null`
 * - `tools`: three tools of different shapes, one with empty `properties`
 * - `plain`: system message and `temperature` only, next to schema items
 * - `image`: inline image data
 * - `search`: web search
 * - `any-of`: `anyOf` with two object branches of different properties
 * - `history`: tool call with `arguments: '{}'` and tool result `'{}'`
 */
export function batchItems(model: string): GenaiBatchItem[] {
    return [
        {
            key: 'schema',
            request: {
                model,
                messages: userMessages('Classify the invented word "blorf" in "a shiny blorf".'),
                response_format: {
                    type: 'json_schema',
                    json_schema: {
                        name: 'kind',
                        schema: {
                            type: 'object',
                            properties: { kind: { type: 'string', enum: ['noun', 'verb'] } },
                            required: ['kind'],
                        },
                    },
                },
                // explicit undefined, as from the spread of an unset optional field
                ...({ batch: undefined } as object),
            },
        },
        {
            key: 'other-schema',
            request: {
                model,
                messages: userMessages('Give the capital of Italy.'),
                response_format: {
                    type: 'json_schema',
                    json_schema: {
                        name: 'capital',
                        schema: {
                            type: 'object',
                            properties: {
                                capital: { type: 'string' },
                                population: { anyOf: [{ type: 'integer' }, { type: 'null' }] },
                            },
                            required: ['capital'],
                        },
                    },
                },
            },
        },
        {
            key: 'tools',
            request: {
                model,
                messages: userMessages('What is the weather in Rome right now? Use the tool.'),
                tools: [WEATHER, TIME, UTC_TIME],
            },
        },
        {
            key: 'plain',
            request: {
                model,
                messages: [
                    { role: 'system', content: 'Answer in upper case only.' },
                    { role: 'user', content: 'Reply with the word: ok' },
                ],
                temperature: 0,
            },
        },
        {
            key: 'image',
            request: {
                model,
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'Which colour fills this image? One word.' },
                            { type: 'image_url', image_url: { url: RED_PNG } },
                        ],
                    },
                ],
                reasoning_effort: 'low',
            },
        },
        {
            key: 'search',
            request: {
                model,
                messages: userMessages('Search the web: which organisation develops Zotero? One sentence.'),
                web_search: true,
            },
        },
        {
            key: 'any-of',
            request: {
                model,
                messages: userMessages('In which city is the Colosseum? Answer with the city form of "result".'),
                response_format: {
                    type: 'json_schema',
                    json_schema: {
                        name: 'result',
                        schema: {
                            type: 'object',
                            properties: {
                                result: {
                                    anyOf: [
                                        { type: 'object', properties: { city: { type: 'string' } } },
                                        { type: 'object', properties: { error: { type: 'string' } } },
                                    ],
                                },
                            },
                        },
                    },
                },
            },
        },
        {
            key: 'history',
            request: {
                model,
                messages: [
                    { role: 'user', content: 'What time is it in UTC?' },
                    {
                        role: 'assistant',
                        content: null,
                        tool_calls: [
                            { id: 'call_1', type: 'function', function: { name: 'get_utc_time', arguments: '{}' } },
                        ],
                    },
                    { role: 'tool', tool_call_id: 'call_1', content: '{}' },
                ],
                tools: [UTC_TIME],
            },
        },
    ];
}

/** Checks the results of `batchItems` by key. */
export function expectBatchResults(status: GenaiBatchStatus): void {
    expect(status.phase).toBe('succeeded');
    const results = new Map((status.results ?? []).map((entry) => [entry.key, entry.result]));
    for (const [key, result] of results) {
        log(`batch item ${key}`, result.ok ? describeOutput(result.data) : `FAIL ${describeError(result.error)}`);
    }
    function item(key: string): GenaiOutput {
        return expectOk(`batch item ${key}`, results.get(key) ?? missing(`batch item ${key}`));
    }
    expect(results.size).toBe(8);
    expect(item('schema').json).toEqual({ kind: 'noun' });
    const other = item('other-schema').json as { capital?: string };
    expect(other.capital).toContain('Rom');
    // foreign property names of the other items removed
    expect(Object.keys(other).filter((name) => name !== 'capital' && name !== 'population')).toEqual([]);
    // `anyOf` branches get each other's property names: a name one branch declares stays, foreign names go
    const anyOf = item('any-of').json as { result?: { city?: string } };
    expect(Object.keys(anyOf)).toEqual(['result']);
    expect(anyOf.result?.city).toContain('Rom');
    expect(Object.keys(anyOf.result ?? {}).filter((name) => name !== 'city' && name !== 'error')).toEqual([]);
    const [call] = item('tools').toolCalls ?? [];
    expect(call?.function.name).toBe('get_weather');
    expect(JSON.parse(call?.function.arguments ?? '{}').city).toContain('Rome');
    expect(item('plain').text).toContain('OK');
    expect(item('image').text.toLowerCase()).toContain('red');
    expect(item('search').sources?.length ?? 0).toBeGreaterThan(0);
    // the empty tool result makes some models call the tool again: text or another call, the item must not fail
    const history = item('history');
    expect(history.text.length + (history.toolCalls?.length ?? 0)).toBeGreaterThan(0);
}

/**
 * **Submits one batch job and waits for its final status**: polls every 30 s.
 *
 * - still running after 40 min: cancels the job and fails
 */
export async function runBatchJob(client: GenaiClient, items: readonly GenaiBatchItem[]): Promise<GenaiBatchStatus> {
    const batch = client.batch;
    if (batch === undefined) {
        throw new Error('client has no batch API');
    }
    const startedAt = performance.now();
    const submitted = expectOk('batch submit', await batch.submit(items, { displayName: 'smart-genai live' }));
    log('batch submit', `${submitted.name} ${submitted.state}`);
    let status = submitted;
    while (status.phase === 'running') {
        if (performance.now() - startedAt > BATCH_MAX_WAIT_MS) {
            const cancelled = await batch.cancel(submitted.name);
            log('batch cancel', cancelled.ok ? 'ok' : describeError(cancelled.error));
            throw new Error(mask(`batch job ${submitted.name} still running after 40 min, cancelled`));
        }
        await Bun.sleep(BATCH_POLL_MS);
        const checked = await send(
            'batch check',
            () => batch.check(submitted.name),
            (data) => data.state,
        );
        status = expectOk('batch check', checked);
    }
    log('batch final', `${status.state} after ${Math.round((performance.now() - startedAt) / 1000)} s`);
    return status;
}

/**
 * **Two batch calls (`batch: true`) started together**: one job, each call its own answer.
 *
 * - `call`: sends one batch call asking for the given word
 * - `client`: lists the jobs afterwards; exactly one new job `smart-genai 2 requests`
 */
export async function expectBatchCallPair(
    call: (word: string) => Promise<Result<GenaiOutput, GenaiError>>,
    client: GenaiClient,
): Promise<void> {
    const words = ['blorf', 'snarf'];
    const since = Date.now() - JOB_CLOCK_SLACK_MS;
    const startedAt = performance.now();
    const results = await Promise.all(
        words.map((word) => send(`batch call ${word}`, () => call(word), describeOutput)),
    );
    log('batch calls', `both settled after ${Math.round((performance.now() - startedAt) / 1000)} s`);
    results.forEach((result, index) => {
        const word = words[index] as string;
        const data = expectOk(`batch call ${word}`, result);
        expect(data.text.toLowerCase()).toContain(word);
        expect(data.settings?.batch).toBe(true);
    });
    const listed = await send(
        'batch list',
        () => client.batch?.list({ displayName: 'smart-genai 2 requests' }) ?? missing('batch API'),
        (data) => `${data.length} jobs`,
    );
    const jobs = expectOk('batch list', listed).filter((job) => Date.parse(job.createTime ?? '') >= since);
    log('batch calls', `${jobs.length} new jobs: ${jobs.map((job) => job.state).join(', ')}`);
    expect(jobs).toHaveLength(1);
}

/** Messages of one batch call asking to repeat `word`. */
export function wordMessages(word: string): GenaiMessage[] {
    return [
        { role: 'system', content: 'Answer with the one word asked for, in lower case.' },
        { role: 'user', content: `Repeat this word exactly: ${word}` },
    ];
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Logs ms since `startedAt` and compact result of one sent request. */
function logResult<T>(
    label: string,
    startedAt: number,
    result: Result<T, GenaiError>,
    describe: (data: T) => string,
): void {
    const ms = Math.round(performance.now() - startedAt);
    log(label, `${ms} ms ${result.ok ? `ok ${describe(result.data)}` : `FAIL ${describeError(result.error)}`}`);
}

/** Retryable server error such as HTTP 503 overload, or a rate limit. */
function isTransient(error: GenaiError): boolean {
    return error.retryable && ((error.status ?? 0) >= 500 || error.code === 'rate_limited');
}

/** Pause before the repeat of a request that failed with `error`. */
function retryPauseMs(error: GenaiError): number {
    if (error.code !== 'rate_limited') {
        return RETRY_PAUSE_MS;
    }
    return Math.min(error.retryAfterMs ?? RATE_PAUSE_MS, MAX_RATE_PAUSE_MS);
}

/**
 * Fails for something missing, such as a batch item result.
 * @throws {Error} Always.
 */
function missing(what: string): never {
    throw new Error(`${what}: missing`);
}

/** Compact description of an embedding answer. */
function describeEmbedding(data: { embeddings: number[][] }): string {
    return `${data.embeddings.length} x ${data.embeddings[0]?.length ?? 0}`;
}

/** One user message. */
function userMessages(content: string): GenaiMessage[] {
    return [{ role: 'user', content }];
}

/** Cosine similarity of two equal-length vectors. */
function cosine(a: number[], b: number[]): number {
    let dot = 0;
    let normA = 0;
    let normB = 0;
    for (let index = 0; index < a.length; index++) {
        dot += (a[index] ?? 0) * (b[index] ?? 0);
        normA += (a[index] ?? 0) ** 2;
        normB += (b[index] ?? 0) ** 2;
    }
    return dot / Math.sqrt(normA * normB);
}
