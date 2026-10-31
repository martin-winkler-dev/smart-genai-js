// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import {
    type Content,
    FunctionCallingConfigMode,
    type GenerateContentConfig,
    type Part,
    type ThinkingConfig,
    ThinkingLevel,
    type ToolConfig,
} from '@google/genai';
import type {
    GenaiDefaults,
    GenaiImagePart,
    GenaiMessage,
    GenaiReasoningEffort,
    GenaiRequest,
    GenaiSettings,
    GenaiTool,
    GenaiToolCall,
    GenaiToolChoice,
} from '../types.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Prepared request in Gemini SDK form: model, contents, and config. */
export type GeminiCall = { model: string; contents: Content[]; config: GenerateContentConfig };

/** Error indicating invalid request format or options. */
export class InvalidRequestError extends TypeError {}

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Supported field keys in `GenaiRequest`. */
const REQUEST_KEYS: ReadonlySet<string> = new Set([
    'model',
    'messages',
    'response_format',
    'temperature',
    'top_p',
    'max_tokens',
    'max_completion_tokens',
    'seed',
    'stop',
    'reasoning_effort',
    'web_search',
    'tools',
    'tool_choice',
    'batch',
]);

/** Supported field keys in `GenaiDefaults`. */
const DEFAULT_KEYS: ReadonlySet<string> = new Set([...[...REQUEST_KEYS].filter((key) => key !== 'messages'), 'config']);

/** Thinking budget tokens for Gemini 2.x models by reasoning effort. */
const THINKING_BUDGETS: Readonly<Record<GenaiReasoningEffort, number>> = {
    none: 0,
    minimal: 512,
    low: 1024,
    medium: 8192,
    high: 24_576,
    xhigh: 24_576,
    max: 24_576,
};

/** Thinking levels for newer Gemini models by reasoning effort. */
const THINKING_LEVELS: Readonly<Record<GenaiReasoningEffort, ThinkingLevel>> = {
    none: ThinkingLevel.MINIMAL,
    minimal: ThinkingLevel.MINIMAL,
    low: ThinkingLevel.LOW,
    medium: ThinkingLevel.MEDIUM,
    high: ThinkingLevel.HIGH,
    xhigh: ThinkingLevel.HIGH,
    max: ThinkingLevel.HIGH,
};

/** Matches Gemini 2.x model identifiers using token-based thinking budget. */
const MATCH_GEMINI_2 = /^(?:(?:publishers\/[^/]+\/)?models\/)?gemini-2\./;

/** Matches Gemini 2.5 Pro model identifiers requiring minimum thinking budget. */
const MATCH_GEMINI_25_PRO = /^(?:(?:publishers\/[^/]+\/)?models\/)?gemini-2\.5-pro/;

/** Minimum thinking budget for Gemini 2.5 Pro. */
const GEMINI_25_PRO_MIN_BUDGET = 128;

/** Supported message roles. */
const ROLES: ReadonlySet<string> = new Set(['system', 'developer', 'user', 'assistant', 'tool']);

/** Property keys on `GenaiTool`. */
const TOOL_KEYS: ReadonlySet<string> = new Set(['type', 'function']);
/** Property keys on `GenaiTool.function`. */
const TOOL_FUNCTION_KEYS: ReadonlySet<string> = new Set(['name', 'description', 'parameters', 'strict']);
/** Property keys on `GenaiToolCall`. */
const TOOL_CALL_KEYS: ReadonlySet<string> = new Set(['id', 'type', 'function', 'extra_content']);
/** Property keys on `GenaiToolCall.function`. */
const TOOL_CALL_FUNCTION_KEYS: ReadonlySet<string> = new Set(['name', 'arguments']);

/** Matches valid function tool names accepted by Gemini. */
const MATCH_FUNCTION_NAME = /^[A-Za-z0-9_.:-]{1,128}$/;

/** Maps string tool choice values to Gemini FunctionCallingConfigMode. */
const TOOL_CHOICE_MODES: Readonly<Record<string, FunctionCallingConfigMode>> = {
    auto: FunctionCallingConfigMode.AUTO,
    none: FunctionCallingConfigMode.NONE,
    required: FunctionCallingConfigMode.ANY,
};

/** Fallback thought signature used when caller omits initial signature. */
const SKIP_SIGNATURE = 'skip_thought_signature_validator';

/** Matches base64 data URLs for image parts. */
const MATCH_IMAGE_DATA_URL = /^data:(image\/[\w.+-]+);base64,([A-Za-z0-9+/]*={0,2})$/;

/** Supported response format types. */
const FORMAT_TYPES: ReadonlySet<string> = new Set(['text', 'json_object', 'json_schema']);

/** SDK internal configuration fields excluded from request config. */
const SDK_CONFIG_KEYS: ReadonlySet<string> = new Set(['abortSignal', 'httpOptions', 'automaticContinuation']);

/** Numeric request options mapped to Gemini GenerateContentConfig properties. */
const NUMBER_FIELDS: readonly [field: keyof GenaiRequest, target: keyof GenerateContentConfig][] = [
    ['temperature', 'temperature'],
    ['top_p', 'topP'],
    ['max_tokens', 'maxOutputTokens'],
    ['max_completion_tokens', 'maxOutputTokens'],
    ['seed', 'seed'],
];

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Converts OpenAI-form request to Gemini SDK call format**.
 *
 * - joins `system` and `developer` messages into `systemInstruction`
 * - maps `response_format` to `responseMimeType` and `responseJsonSchema`
 * - maps `reasoning_effort` to `thinkingConfig` with `includeThoughts: true`
 * - maps `tools` and `tool_choice` to Gemini function declarations and config
 * - maps `assistant` tool calls and `tool` responses to Gemini turns
 * - merges call `config` last
 * @throws {InvalidRequestError} If request format or options are invalid.
 */
export function toGeminiCall(request: GenaiRequest, config: GenerateContentConfig | undefined): GeminiCall {
    validateRequest(request);
    if (config !== undefined) {
        validateGenaiConfig(config, 'config');
    }
    const model = request.model as string;
    const system: string[] = [];
    const contents: Content[] = [];
    const callNames = new Map<string, string>();
    for (const message of request.messages) {
        if (message.role === 'system' || message.role === 'developer') {
            system.push(textOf(message.content));
        } else if (message.role === 'assistant') {
            for (const call of message.tool_calls ?? []) {
                callNames.set(call.id, call.function.name);
            }
            contents.push({ role: 'model', parts: modelParts(message, !MATCH_GEMINI_2.test(model)) });
        } else if (message.role === 'tool') {
            const part: Part = {
                functionResponse: {
                    id: message.tool_call_id,
                    name: callNames.get(message.tool_call_id) as string,
                    response: toFunctionResponse(textOf(message.content)),
                },
            };
            const last = contents.at(-1);
            if (last?.role === 'user' && last.parts?.every((entry) => entry.functionResponse !== undefined)) {
                last.parts.push(part);
            } else {
                contents.push({ role: 'user', parts: [part] });
            }
        } else {
            contents.push({ role: 'user', parts: toParts(message.content) });
        }
    }
    const built: GenerateContentConfig = system.length === 0 ? {} : { systemInstruction: system.join('\n\n') };
    const format = request.response_format;
    if (format !== undefined && format.type !== 'text') {
        built.responseMimeType = 'application/json';
        if (format.type === 'json_schema' && format.json_schema.schema !== undefined) {
            built.responseJsonSchema = format.json_schema.schema;
        }
    }
    for (const [field, target] of NUMBER_FIELDS) {
        const value = request[field];
        if (typeof value === 'number') {
            (built as Record<string, unknown>)[target] = value;
        }
    }
    if (request.stop !== undefined) {
        built.stopSequences = typeof request.stop === 'string' ? [request.stop] : request.stop;
    }
    const thinking = toThinkingConfig(request.reasoning_effort, model);
    if (thinking !== undefined) {
        built.thinkingConfig = thinking;
    }
    const merged = mergeGenaiConfig(built, config) as GenerateContentConfig;
    const declarations = request.tools?.map(({ function: { name, description, parameters } }) => ({
        name,
        ...(description === undefined ? {} : { description }),
        ...(parameters === undefined ? {} : { parametersJsonSchema: parameters }),
    }));
    const added = [
        ...(declarations === undefined ? [] : [{ functionDeclarations: declarations }]),
        ...(request.web_search === true ? [{ googleSearch: {} }] : []),
    ];
    if (added.length > 0) {
        merged.tools = [...(config?.tools ?? []), ...added];
    }
    if (declarations !== undefined && request.tool_choice !== undefined) {
        const toolConfig: ToolConfig = config?.toolConfig ?? {};
        merged.toolConfig = {
            ...toolConfig,
            functionCallingConfig: { ...toFunctionCalling(request.tool_choice), ...toolConfig.functionCallingConfig },
        };
    }
    return { model, contents, config: merged };
}

/**
 * **Applies request defaults and merges config**.
 *
 * - fills unset request fields from `defaults`
 * - request `max_tokens` or `max_completion_tokens` replaces both defaults
 * - merges `config` with call `config`
 * - checks request keys before the merge, which drops `undefined` values
 * @throws {InvalidRequestError} If the request or `config` contains unsupported fields.
 */
export function applyDefaults(
    defaults: GenaiDefaults | undefined,
    request: GenaiRequest,
    config: GenerateContentConfig | undefined,
): { request: GenaiRequest; config: GenerateContentConfig | undefined } {
    if (defaults === undefined) {
        return { request, config };
    }
    if (typeof request !== 'object' || request === null) {
        return { request, config };
    }
    checkRequestKeys(request);
    if (config !== undefined) {
        validateGenaiConfig(config, 'config');
    }
    const { config: defaultConfig, ...defaultFields } = defaults;
    if (request.max_tokens !== undefined || request.max_completion_tokens !== undefined) {
        delete defaultFields.max_tokens;
        delete defaultFields.max_completion_tokens;
    }
    return {
        request: assignDefined({ ...defaultFields }, request) as GenaiRequest,
        config: mergeGenaiConfig(defaultConfig, config),
    };
}

/**
 * **Merges multiple Gemini configs**: subsequent configs override previous per field.
 *
 * - merges `thinkingConfig` per member
 * - skips `undefined` configs; returns `undefined` if all are undefined
 */
export function mergeGenaiConfig(...configs: (GenerateContentConfig | undefined)[]): GenerateContentConfig | undefined {
    const given = configs.filter((config): config is GenerateContentConfig => config !== undefined);
    if (given.length === 0) {
        return undefined;
    }
    const merged: GenerateContentConfig = assignDefined({}, ...given);
    if (given.some((config) => config.thinkingConfig !== undefined)) {
        merged.thinkingConfig = assignDefined({}, ...given.map((config) => config.thinkingConfig));
    }
    return merged;
}

/**
 * Validates Gemini config options.
 * @throws {InvalidRequestError} If config has `__proto__` key or SDK-only properties (`abortSignal`, `httpOptions`, `automaticContinuation`).
 */
export function validateGenaiConfig(config: unknown, option: string): void {
    if (!isPlainObject(config) || Object.hasOwn(config, '__proto__')) {
        throw new InvalidRequestError(`Invalid value for option "${option}"`);
    }
    for (const key of Object.keys(config)) {
        if (SDK_CONFIG_KEYS.has(key)) {
            throw new InvalidRequestError(`Option "${option}.${key}" is not supported`);
        }
    }
}

/**
 * Validates request defaults.
 * @throws {InvalidRequestError} If unknown fields or unsupported config options are present.
 */
export function validateDefaults(defaults: unknown, option = 'defaults'): void {
    if (typeof defaults !== 'object' || defaults === null || Array.isArray(defaults)) {
        throw new InvalidRequestError(`Invalid value for option "${option}"`);
    }
    for (const key of Object.keys(defaults)) {
        if (!DEFAULT_KEYS.has(key)) {
            throw new InvalidRequestError(`Option "${option}.${key}" is not supported`);
        }
    }
    const { config, ...fields } = defaults as GenaiDefaults;
    toGeminiCall({ model: 'model', ...fields, messages: [{ role: 'user', content: '' }] }, config);
}

/**
 * Validates batch items and extracts their shared model identifier.
 * @throws {InvalidRequestError} If items are empty, invalid, contain `batch`, or use different models.
 */
export function batchModel(items: readonly { key: string; request: GenaiRequest }[]): string {
    if (!Array.isArray(items) || items.length === 0) {
        throw new InvalidRequestError('Invalid value for option "items"');
    }
    const keys = new Set<string>();
    for (const item of items) {
        if (typeof item?.key !== 'string' || item.key === '' || keys.has(item.key)) {
            throw new InvalidRequestError('Invalid value for option "key"');
        }
        keys.add(item.key);
        validateRequest(item.request);
        if ('batch' in item.request) {
            throw new InvalidRequestError('Option "batch" is not supported in batch items');
        }
    }
    const model = items[0].request.model;
    if (items.some((item) => item.request.model !== model)) {
        throw new InvalidRequestError('Invalid value for option "model": a batch takes one model');
    }
    return model;
}

/**
 * Validates option object keys against an allowed set.
 * @throws {InvalidRequestError} If options is not a plain object or contains disallowed keys.
 */
export function checkOptionKeys(options: unknown, allowed: ReadonlySet<string>, prefix = ''): void {
    if (typeof options !== 'object' || options === null || Array.isArray(options)) {
        throw new InvalidRequestError(`Invalid value for option "${prefix.slice(0, -1) || 'options'}"`);
    }
    for (const key of Object.keys(options)) {
        if (!allowed.has(key)) {
            throw new InvalidRequestError(`Option "${prefix}${key}" is not supported`);
        }
    }
    const { signal, displayName } = options as { signal?: unknown; displayName?: unknown };
    if (allowed.has('signal') && signal !== undefined && !(signal instanceof AbortSignal)) {
        throw new InvalidRequestError(`Invalid value for option "${prefix}signal"`);
    }
    if (
        allowed.has('displayName') &&
        displayName !== undefined &&
        (typeof displayName !== 'string' || displayName === '')
    ) {
        throw new InvalidRequestError(`Invalid value for option "${prefix}displayName"`);
    }
}

/**
 * Validates batch job name string.
 * @throws {InvalidRequestError} If name is not a non-empty string.
 */
export function checkBatchName(name: unknown): void {
    if (typeof name !== 'string' || name === '') {
        throw new InvalidRequestError('Invalid value for option "name"');
    }
}

/** Checks whether request specifies JSON output format. */
export function wantsJson(request: GenaiRequest): boolean {
    return request.response_format !== undefined && request.response_format.type !== 'text';
}

/** Extracts effective settings snapshot from validated request and config. */
export function toSettings(
    request: GenaiRequest,
    config: GenerateContentConfig | undefined,
    model: string,
): GenaiSettings {
    const settings: GenaiSettings = { model };
    assignDefined(settings, {
        temperature: request.temperature,
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        top_p: request.top_p,
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        max_tokens: request.max_tokens ?? request.max_completion_tokens,
        seed: request.seed,
        stop: copyPlain(request.stop),
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        reasoning_effort: request.reasoning_effort,
    });
    if (request.web_search === true) {
        settings.web_search = true;
    }
    if (request.batch === true) {
        settings.batch = true;
    }
    if (wantsJson(request)) {
        settings.json = true;
    }
    if (request.tools !== undefined) {
        settings.tools = request.tools.map((tool) => tool.function.name);
        // Only with tools, since it is sent only with them.
        if (request.tool_choice !== undefined) {
            settings.tool_choice = copyPlain(request.tool_choice);
        }
    }
    if (config !== undefined && Object.keys(config).length > 0) {
        settings.config = copyPlain(config);
    }
    return settings;
}

/** Recursively deep-copies plain objects and arrays; preserves class instances and primitives. */
export function copyPlain<T>(value: T): T {
    if (Array.isArray(value)) {
        return value.map(copyPlain) as T;
    }
    if (isPlainObject(value)) {
        const prototype: unknown = Object.getPrototypeOf(value);
        if (prototype === Object.prototype || prototype === null) {
            return Object.fromEntries(Object.entries(value).map(([key, entry]) => [key, copyPlain(entry)])) as T;
        }
    }
    return value;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * Checks that the request has only supported keys.
 * @throws {InvalidRequestError} If a key is not supported.
 */
function checkRequestKeys(request: GenaiRequest): void {
    for (const key of Object.keys(request)) {
        if (!REQUEST_KEYS.has(key)) {
            throw new InvalidRequestError(`Option "${key}" is not supported`);
        }
    }
}

/**
 * Validates full request structure.
 * @throws {InvalidRequestError} If required fields are missing, invalid, or mutually incompatible.
 */
function validateRequest(request: GenaiRequest): void {
    if (typeof request !== 'object' || request === null) {
        throw new InvalidRequestError('Invalid value for option "request"');
    }
    checkRequestKeys(request);
    if (typeof request.model !== 'string' || request.model === '') {
        throw new InvalidRequestError('Invalid value for option "model": set it or defaults.model');
    }
    if (request.reasoning_effort !== undefined && !Object.hasOwn(THINKING_BUDGETS, request.reasoning_effort)) {
        throw new InvalidRequestError('Invalid value for option "reasoning_effort"');
    }
    if (request.batch !== undefined && typeof request.batch !== 'boolean') {
        throw new InvalidRequestError('Invalid value for option "batch": true or false');
    }
    if (request.web_search !== undefined && typeof request.web_search !== 'boolean') {
        throw new InvalidRequestError('Invalid value for option "web_search": true or false');
    }
    if (!Array.isArray(request.messages) || !request.messages.some((message) => message?.role === 'user')) {
        throw new InvalidRequestError('Invalid value for option "messages"');
    }
    validateMessages(request.messages);
    validateTools(request);
    if (request.tools !== undefined && request.web_search === true && MATCH_GEMINI_2.test(request.model)) {
        throw new InvalidRequestError('Option "tools" is not supported together with "web_search" on Gemini 2.x');
    }
    const format = request.response_format;
    if (format !== undefined && (!isPlainObject(format) || !FORMAT_TYPES.has(format.type))) {
        throw new InvalidRequestError('Invalid value for option "response_format"');
    }
    if (format?.type === 'json_schema') {
        const spec: unknown = format.json_schema;
        if (
            !isPlainObject(spec) ||
            typeof spec.name !== 'string' ||
            (spec.schema !== undefined && !isPlainObject(spec.schema))
        ) {
            throw new InvalidRequestError('Invalid value for option "response_format.json_schema"');
        }
    }
    for (const [field] of NUMBER_FIELDS) {
        const value = request[field];
        if (value !== undefined && (typeof value !== 'number' || !Number.isFinite(value))) {
            throw new InvalidRequestError(`Invalid value for option "${field}"`);
        }
    }
    for (const field of ['max_tokens', 'max_completion_tokens'] as const) {
        const value = request[field];
        if (value !== undefined && (!Number.isInteger(value) || value <= 0)) {
            throw new InvalidRequestError(`Invalid value for option "${field}"`);
        }
    }
    if (request.max_tokens !== undefined && request.max_completion_tokens !== undefined) {
        throw new InvalidRequestError('Option "max_tokens" is not supported together with "max_completion_tokens"');
    }
    const { stop } = request;
    if (
        stop !== undefined &&
        typeof stop !== 'string' &&
        !(Array.isArray(stop) && stop.every((entry) => typeof entry === 'string'))
    ) {
        throw new InvalidRequestError('Invalid value for option "stop"');
    }
}

/**
 * Validates chat messages and tool call references.
 * @throws {InvalidRequestError} If message roles, content, or tool call references are invalid.
 */
function validateMessages(messages: readonly GenaiMessage[]): void {
    const callIds = new Set<string>();
    for (const message of messages) {
        if (!ROLES.has(message?.role)) {
            throw new InvalidRequestError('Invalid value for option "messages"');
        }
        if ('tool_calls' in message && message.role !== 'assistant') {
            throw new InvalidRequestError('Invalid value for option "messages": tool_calls only in assistant messages');
        }
        if ('tool_call_id' in message && message.role !== 'tool') {
            throw new InvalidRequestError('Invalid value for option "messages": tool_call_id only in tool messages');
        }
        if (message.role === 'assistant') {
            const calls = message.tool_calls;
            if (calls !== undefined && !Array.isArray(calls)) {
                throw new InvalidRequestError('Invalid value for option "messages.tool_calls"');
            }
            for (const call of calls ?? []) {
                validateToolCall(call);
                callIds.add(call.id);
            }
            if (message.content === null) {
                if (calls === undefined || calls.length === 0) {
                    throw new InvalidRequestError(
                        'Invalid value for option "messages": content null only with tool_calls',
                    );
                }
                continue;
            }
        }
        if (message.role === 'tool' && !callIds.has(message.tool_call_id)) {
            throw new InvalidRequestError(
                'Invalid value for option "messages.tool_call_id": the id of a tool call in an earlier assistant message',
            );
        }
        const parts = toParts(message.content);
        if (message.role !== 'user' && parts.some((part) => part.text === undefined)) {
            throw new InvalidRequestError('Invalid value for option "messages": images only in user messages');
        }
    }
}

/**
 * Validates tool call on assistant message.
 * @throws {InvalidRequestError} If tool call format or arguments JSON is invalid.
 */
function validateToolCall(call: GenaiToolCall): void {
    checkMembers(call, TOOL_CALL_KEYS, 'messages.tool_calls');
    if (typeof call.id !== 'string' || call.id === '') {
        throw new InvalidRequestError('Invalid value for option "messages.tool_calls.id"');
    }
    if (call.type !== 'function') {
        throw new InvalidRequestError('Invalid value for option "messages.tool_calls.type": function');
    }
    checkMembers(call.function, TOOL_CALL_FUNCTION_KEYS, 'messages.tool_calls.function');
    if (typeof call.function.name !== 'string' || call.function.name === '') {
        throw new InvalidRequestError('Invalid value for option "messages.tool_calls.function.name"');
    }
    parseArguments(call.function.arguments);
    const extra = call.extra_content as unknown;
    if (extra === undefined) {
        return;
    }
    const google = (extra as { google?: unknown } | null)?.google;
    const signature = (google as Record<string, unknown> | null | undefined)?.thought_signature;
    if (
        !isPlainObject(extra) ||
        (google !== undefined && !isPlainObject(google)) ||
        (signature !== undefined && typeof signature !== 'string')
    ) {
        throw new InvalidRequestError('Invalid value for option "messages.tool_calls.extra_content"');
    }
}

/**
 * Validates tools declarations and tool_choice.
 * @throws {InvalidRequestError} If tool definitions or tool choice are malformed.
 */
function validateTools(request: GenaiRequest): void {
    const { tools, tool_choice: choice } = request as Record<string, unknown>;
    const names = new Set<string>();
    if (tools !== undefined) {
        if (!Array.isArray(tools) || tools.length === 0) {
            throw new InvalidRequestError('Invalid value for option "tools"');
        }
        for (const tool of tools as GenaiTool[]) {
            checkMembers(tool, TOOL_KEYS, 'tools');
            if (tool.type !== 'function') {
                throw new InvalidRequestError('Invalid value for option "tools.type": function');
            }
            checkMembers(tool.function, TOOL_FUNCTION_KEYS, 'tools.function');
            const { name, description, parameters, strict } = tool.function;
            if (typeof name !== 'string' || !MATCH_FUNCTION_NAME.test(name)) {
                throw new InvalidRequestError(
                    'Invalid value for option "tools.function.name": up to 128 letters, digits, _, ., : and -',
                );
            }
            if (names.has(name)) {
                throw new InvalidRequestError(
                    `Invalid value for option "tools.function.name": ${name} is declared twice`,
                );
            }
            names.add(name);
            if (description !== undefined && typeof description !== 'string') {
                throw new InvalidRequestError('Invalid value for option "tools.function.description"');
            }
            if (parameters !== undefined && !isPlainObject(parameters)) {
                throw new InvalidRequestError(
                    'Invalid value for option "tools.function.parameters": a JSON Schema object',
                );
            }
            if (strict !== undefined && strict !== null && typeof strict !== 'boolean') {
                throw new InvalidRequestError('Invalid value for option "tools.function.strict"');
            }
        }
    }
    if (choice === undefined || (typeof choice === 'string' && Object.hasOwn(TOOL_CHOICE_MODES, choice))) {
        return;
    }
    if (typeof choice === 'string') {
        throw new InvalidRequestError('Invalid value for option "tool_choice": auto, none, required or a function');
    }
    checkMembers(choice, TOOL_KEYS, 'tool_choice');
    const named = choice as Exclude<GenaiToolChoice, string>;
    if (named.type !== 'function') {
        throw new InvalidRequestError('Invalid value for option "tool_choice.type": function');
    }
    checkMembers(named.function, new Set(['name']), 'tool_choice.function');
    const { name } = named.function;
    if (typeof name !== 'string' || (tools !== undefined && !names.has(name))) {
        throw new InvalidRequestError(
            'Invalid value for option "tool_choice.function.name": the name of one of "tools"',
        );
    }
}

/**
 * Checks object members against allowed key set.
 * @throws {InvalidRequestError} If value is not a plain object or has unsupported members.
 */
function checkMembers(value: unknown, allowed: ReadonlySet<string>, option: string): void {
    if (!isPlainObject(value)) {
        throw new InvalidRequestError(`Invalid value for option "${option}"`);
    }
    for (const key of Object.keys(value)) {
        if (!allowed.has(key)) {
            throw new InvalidRequestError(`Option "${option}.${key}" is not supported`);
        }
    }
}

/** Checks whether value is non-array object. */
function isPlainObject(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Parses JSON string arguments into object.
 * @throws {InvalidRequestError} If arguments string does not parse into a plain object.
 */
function parseArguments(text: unknown): Record<string, unknown> {
    if (text === '') {
        return {};
    }
    let value: unknown;
    try {
        value = typeof text === 'string' ? JSON.parse(text) : undefined;
    } catch {
        value = undefined;
    }
    if (!isPlainObject(value)) {
        throw new InvalidRequestError(
            'Invalid value for option "messages.tool_calls.function.arguments": JSON text of an object',
        );
    }
    return value;
}

/** Converts assistant message text and tool calls into Gemini parts. */
function modelParts(message: Extract<GenaiMessage, { role: 'assistant' }>, skipSignature: boolean): Part[] {
    const calls = message.tool_calls ?? [];
    const text = message.content === null ? [] : toParts(message.content);
    const parts = calls.length === 0 ? text : text.filter((part) => part.text !== '');
    calls.forEach((call, index) => {
        const signature =
            call.extra_content?.google?.thought_signature ??
            (index === 0 && skipSignature ? SKIP_SIGNATURE : undefined);
        parts.push({
            functionCall: { id: call.id, name: call.function.name, args: parseArguments(call.function.arguments) },
            ...(signature === undefined ? {} : { thoughtSignature: signature }),
        });
    });
    return parts;
}

/** Maps reasoning effort to Gemini thinkingConfig based on model capabilities. */
function toThinkingConfig(effort: GenaiReasoningEffort | undefined, model: string): ThinkingConfig | undefined {
    if (effort === undefined) {
        return undefined;
    }
    if (!MATCH_GEMINI_2.test(model)) {
        return { thinkingLevel: THINKING_LEVELS[effort], includeThoughts: true };
    }
    const thinkingBudget =
        effort === 'none' && MATCH_GEMINI_25_PRO.test(model) ? GEMINI_25_PRO_MIN_BUDGET : THINKING_BUDGETS[effort];
    return thinkingBudget === 0 ? { thinkingBudget } : { thinkingBudget, includeThoughts: true };
}

/** Parses tool output text into function response object. */
function toFunctionResponse(text: string): Record<string, unknown> {
    try {
        const value: unknown = JSON.parse(text);
        if (isPlainObject(value)) {
            return value;
        }
    } catch {
        // plain text result
    }
    return { output: text };
}

/** Maps tool_choice to Gemini functionCallingConfig. */
function toFunctionCalling(choice: GenaiToolChoice): NonNullable<ToolConfig['functionCallingConfig']> {
    if (typeof choice === 'string') {
        return { mode: TOOL_CHOICE_MODES[choice] };
    }
    return { mode: FunctionCallingConfigMode.ANY, allowedFunctionNames: [choice.function.name] };
}

/** Extracts concatenated text content from message content. */
function textOf(content: unknown): string {
    return toParts(content)
        .map((part) => part.text)
        .join('');
}

/**
 * Converts message content to Gemini parts.
 * @throws {InvalidRequestError} If content format or image data URL is invalid.
 */
function toParts(content: unknown): Part[] {
    if (typeof content === 'string') {
        return [{ text: content }];
    }
    if (!Array.isArray(content)) {
        throw new InvalidRequestError('Invalid value for option "messages"');
    }
    const parts: Part[] = [];
    for (const part of content as unknown[]) {
        const { type, text } = (part ?? {}) as { type?: unknown; text?: unknown };
        const image = (part as Partial<GenaiImagePart> | null)?.image_url as unknown;
        if (type === 'text' && typeof text === 'string') {
            const last = parts.at(-1);
            if (last?.text === undefined) {
                parts.push({ text });
            } else {
                parts[parts.length - 1] = { text: last.text + text };
            }
        } else if (type === 'image_url') {
            const url = (image as { url?: unknown } | null | undefined)?.url;
            const match = typeof url === 'string' ? MATCH_IMAGE_DATA_URL.exec(url) : null;
            if (match === null) {
                throw new InvalidRequestError(
                    'Invalid value for option "messages": an image must be a base64 data URL',
                );
            }
            parts.push({ inlineData: { mimeType: match[1], data: match[2] } });
        } else {
            throw new InvalidRequestError('Invalid value for option "messages"');
        }
    }
    return parts.length === 0 ? [{ text: '' }] : parts;
}

/** Copies defined properties from source objects into target object. */
function assignDefined<T extends object>(target: T, ...sources: (object | undefined)[]): T {
    for (const source of sources) {
        for (const [key, value] of Object.entries(source ?? {})) {
            if (value !== undefined) {
                (target as Record<string, unknown>)[key] = value;
            }
        }
    }
    return target;
}
