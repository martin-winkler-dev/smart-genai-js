// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { FunctionCallingConfigMode, type GenerateContentConfig, ThinkingLevel } from '@google/genai';
import { batchModel, toGeminiCall, wantsJson } from '../../../src/core/request.js';
import { mergeGenaiConfig, validateGenaiConfig } from '../../../src/index.js';
import type { GenaiRequest, GenaiTool } from '../../../src/types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const schema = { type: 'object', properties: { kind: { type: 'string' } }, required: ['kind'] };

const request: GenaiRequest = {
    model: 'gemini-x',
    messages: [
        { role: 'system', content: 'Classify the word.' },
        { role: 'developer', content: [{ type: 'text', text: 'Be brief.' }] },
        { role: 'user', content: 'blorf' },
        { role: 'assistant', content: '{"kind":"noun"}' },
        {
            role: 'user',
            content: [
                { type: 'text', text: 'snarf' },
                { type: 'text', text: '!' },
            ],
        },
    ],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema, strict: true } },
    temperature: 0.2,
    top_p: 0.9,
    max_tokens: 64,
    seed: 7,
    stop: 'END',
};

describe('toGeminiCall', () => {
    it('maps an OpenAI chat request to contents and config', () => {
        expect(toGeminiCall(request, { thinkingConfig: { thinkingBudget: 0 }, temperature: 1 })).toEqual({
            model: 'gemini-x',
            contents: [
                { role: 'user', parts: [{ text: 'blorf' }] },
                { role: 'model', parts: [{ text: '{"kind":"noun"}' }] },
                { role: 'user', parts: [{ text: 'snarf!' }] },
            ],
            config: {
                systemInstruction: 'Classify the word.\n\nBe brief.',
                responseMimeType: 'application/json',
                responseJsonSchema: schema,
                temperature: 1,
                topP: 0.9,
                maxOutputTokens: 64,
                seed: 7,
                stopSequences: ['END'],
                thinkingConfig: { thinkingBudget: 0 },
            },
        });
        expect(wantsJson(request)).toBe(true);
    });

    it('maps image data URLs to inline data next to text parts', () => {
        const call = toGeminiCall(
            {
                model: 'm',
                messages: [
                    {
                        role: 'user',
                        content: [
                            { type: 'text', text: 'What is ' },
                            { type: 'text', text: 'this?' },
                            {
                                type: 'image_url',
                                image_url: { url: 'data:image/png;base64,iVBORw0KGgo=', detail: 'low' },
                            },
                        ],
                    },
                ],
            },
            undefined,
        );
        expect(call.contents).toEqual([
            {
                role: 'user',
                parts: [{ text: 'What is this?' }, { inlineData: { mimeType: 'image/png', data: 'iVBORw0KGgo=' } }],
            },
        ]);
    });

    it('maps json_object, text and max_completion_tokens', () => {
        const base: GenaiRequest = { model: 'm', messages: [{ role: 'user', content: 'hi' }] };
        expect(toGeminiCall({ ...base, response_format: { type: 'json_object' } }, undefined).config).toEqual({
            responseMimeType: 'application/json',
        });
        const text = {
            ...base,
            response_format: { type: 'text' as const },
            max_completion_tokens: 5,
            stop: ['a', 'b'],
        };
        expect(toGeminiCall(text, undefined).config).toEqual({ maxOutputTokens: 5, stopSequences: ['a', 'b'] });
        expect(wantsJson(text)).toBe(false);
        expect(wantsJson(base)).toBe(false);
    });

    it('throws a TypeError for invalid requests', () => {
        const base: GenaiRequest = { model: 'm', messages: [{ role: 'user', content: 'hi' }] };
        const invalid: [unknown, string][] = [
            [{ ...base, n: 2 }, 'Option "n" is not supported'],
            [{ ...base, model: '' }, 'Invalid value for option "model": set it or defaults.model'],
            [{ ...base, messages: [{ role: 'system', content: 'x' }] }, 'Invalid value for option "messages"'],
            [
                { ...base, messages: [{ role: 'user', content: [{ type: 'image_url' }] }] },
                'Invalid value for option "messages": an image must be a base64 data URL',
            ],
            [
                {
                    ...base,
                    messages: [
                        { role: 'user', content: [{ type: 'image_url', image_url: { url: 'https://x.test/a.png' } }] },
                    ],
                },
                'Invalid value for option "messages": an image must be a base64 data URL',
            ],
            [
                {
                    ...base,
                    messages: [
                        {
                            role: 'system',
                            content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }],
                        },
                        { role: 'user', content: 'hi' },
                    ],
                },
                'Invalid value for option "messages": images only in user messages',
            ],
            [
                { ...base, messages: [{ role: 'user', content: [{ type: 'audio' }] }] },
                'Invalid value for option "messages"',
            ],
            [
                {
                    ...base,
                    messages: [
                        { role: 'user', content: 'a' },
                        { role: 'function', content: 'b' },
                    ],
                },
                'Invalid value for option "messages"',
            ],
            [{ ...base, response_format: { type: 'xml' } }, 'Invalid value for option "response_format"'],
            [{ ...base, response_format: [] }, 'Invalid value for option "response_format"'],
            [
                { ...base, response_format: { type: 'json_schema' } },
                'Invalid value for option "response_format.json_schema"',
            ],
            [
                { ...base, response_format: { type: 'json_schema', json_schema: [] } },
                'Invalid value for option "response_format.json_schema"',
            ],
            [
                { ...base, response_format: { type: 'json_schema', json_schema: { schema: {} } } },
                'Invalid value for option "response_format.json_schema"',
            ],
            [
                { ...base, response_format: { type: 'json_schema', json_schema: { name: 'a', schema: [] } } },
                'Invalid value for option "response_format.json_schema"',
            ],
            [{ ...base, temperature: Number.NaN }, 'Invalid value for option "temperature"'],
            [{ ...base, max_tokens: 1.5 }, 'Invalid value for option "max_tokens"'],
            [
                { ...base, max_tokens: 1, max_completion_tokens: 1 },
                'Option "max_tokens" is not supported together with "max_completion_tokens"',
            ],
            [{ ...base, stop: [1] }, 'Invalid value for option "stop"'],
            [null, 'Invalid value for option "request"'],
        ];
        for (const [value, message] of invalid) {
            expect(() => toGeminiCall(value as GenaiRequest, undefined)).toThrow(new TypeError(message));
        }
    });
});

describe('toGeminiCall with tools', () => {
    const weather: GenaiTool = {
        type: 'function',
        function: {
            name: 'get_weather',
            description: 'Weather of a city',
            parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] },
            strict: true,
        },
    };
    const time: GenaiTool = { type: 'function', function: { name: 'get_time' } };
    const history: GenaiRequest['messages'] = [
        { role: 'user', content: 'Weather and time in Rome?' },
        {
            role: 'assistant',
            content: null,
            tool_calls: [
                {
                    id: 'call_1',
                    type: 'function',
                    function: { name: 'get_weather', arguments: '{"city":"Rome"}' },
                    extra_content: { google: { thought_signature: 'sig-1' } },
                },
                { id: 'call_2', type: 'function', function: { name: 'get_time', arguments: '' } },
            ],
        },
        { role: 'tool', tool_call_id: 'call_1', content: '{"celsius":21}' },
        { role: 'tool', tool_call_id: 'call_2', content: [{ type: 'text', text: '12:00' }] },
    ];

    it('maps tools, tool_choice, tool calls and tool results', () => {
        const call = toGeminiCall(
            { model: 'gemini-3-flash', messages: history, tools: [weather, time], tool_choice: 'required' },
            undefined,
        );
        expect(call.contents).toEqual([
            { role: 'user', parts: [{ text: 'Weather and time in Rome?' }] },
            {
                role: 'model',
                parts: [
                    {
                        functionCall: { id: 'call_1', name: 'get_weather', args: { city: 'Rome' } },
                        thoughtSignature: 'sig-1',
                    },
                    { functionCall: { id: 'call_2', name: 'get_time', args: {} } },
                ],
            },
            {
                role: 'user',
                parts: [
                    { functionResponse: { id: 'call_1', name: 'get_weather', response: { celsius: 21 } } },
                    { functionResponse: { id: 'call_2', name: 'get_time', response: { output: '12:00' } } },
                ],
            },
        ]);
        expect(call.config).toEqual({
            tools: [
                {
                    functionDeclarations: [
                        {
                            name: 'get_weather',
                            description: 'Weather of a city',
                            parametersJsonSchema: weather.function.parameters,
                        },
                        { name: 'get_time' },
                    ],
                },
            ],
            toolConfig: { functionCallingConfig: { mode: FunctionCallingConfigMode.ANY } },
        });
    });

    it('maps each tool_choice and lets config.toolConfig win per field', () => {
        const base: GenaiRequest = { model: 'm', messages: [{ role: 'user', content: 'hi' }], tools: [weather] };
        expect(toGeminiCall({ ...base, tool_choice: 'auto' }, undefined).config.toolConfig).toEqual({
            functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO },
        });
        expect(toGeminiCall({ ...base, tool_choice: 'none' }, undefined).config.toolConfig).toEqual({
            functionCallingConfig: { mode: FunctionCallingConfigMode.NONE },
        });
        const named = { type: 'function' as const, function: { name: 'get_weather' } };
        expect(toGeminiCall({ ...base, tool_choice: named }, undefined).config.toolConfig).toEqual({
            functionCallingConfig: { mode: FunctionCallingConfigMode.ANY, allowedFunctionNames: ['get_weather'] },
        });
        expect(toGeminiCall(base, undefined).config.toolConfig).toBeUndefined();
        const overridden = toGeminiCall(
            { ...base, tool_choice: 'required' },
            { toolConfig: { functionCallingConfig: { mode: FunctionCallingConfigMode.VALIDATED } } },
        );
        expect(overridden.config.toolConfig).toEqual({
            functionCallingConfig: { mode: FunctionCallingConfigMode.VALIDATED },
        });
        const noTools = { model: 'm', messages: base.messages, tool_choice: 'required' as const };
        expect(toGeminiCall(noTools, undefined).config).toEqual({});
    });

    it('combines tools with web_search outside Gemini 2.x and adds config tools first', () => {
        const call = toGeminiCall(
            { model: 'gemini-3-flash', messages: [{ role: 'user', content: 'hi' }], tools: [time], web_search: true },
            { tools: [{ codeExecution: {} }] },
        );
        expect(call.config.tools).toEqual([
            { codeExecution: {} },
            { functionDeclarations: [{ name: 'get_time' }] },
            { googleSearch: {} },
        ]);
    });

    it('keeps assistant text beside calls and fills a missing first signature outside Gemini 2.x', () => {
        const messages: GenaiRequest['messages'] = [
            { role: 'user', content: 'Time?' },
            {
                role: 'assistant',
                content: 'Let me check.',
                tool_calls: [{ id: 'c', type: 'function', function: { name: 'get_time', arguments: '{}' } }],
            },
            { role: 'tool', tool_call_id: 'c', content: '[1,2]' },
            { role: 'user', content: 'Thanks' },
        ];
        const gemini3 = toGeminiCall({ model: 'gemini-3-pro', messages, tools: [time] }, undefined);
        expect(gemini3.contents.slice(1)).toEqual([
            {
                role: 'model',
                parts: [
                    { text: 'Let me check.' },
                    {
                        functionCall: { id: 'c', name: 'get_time', args: {} },
                        thoughtSignature: 'skip_thought_signature_validator',
                    },
                ],
            },
            {
                role: 'user',
                parts: [{ functionResponse: { id: 'c', name: 'get_time', response: { output: '[1,2]' } } }],
            },
            { role: 'user', parts: [{ text: 'Thanks' }] },
        ]);
        const gemini2 = toGeminiCall({ model: 'gemini-2.5-flash', messages, tools: [time] }, undefined);
        expect(gemini2.contents[1]?.parts?.[1]).toEqual({ functionCall: { id: 'c', name: 'get_time', args: {} } });
    });

    it('throws a TypeError for invalid tools, tool_choice and tool messages', () => {
        const base: GenaiRequest = { model: 'm', messages: [{ role: 'user', content: 'hi' }] };
        const call = { id: 'c', type: 'function', function: { name: 'get_time', arguments: '{}' } };
        /** Constructs request with modified tool call and optional answering tool message. */
        function withCall(extra: Record<string, unknown>, tool?: Record<string, unknown>): unknown {
            return {
                ...base,
                messages: [
                    { role: 'user', content: 'hi' },
                    { role: 'assistant', content: null, tool_calls: [{ ...call, ...extra }] },
                    ...(tool === undefined ? [] : [{ role: 'tool', tool_call_id: 'c', content: 'x', ...tool }]),
                ],
            };
        }
        const invalid: [unknown, string][] = [
            [{ ...base, tools: [] }, 'Invalid value for option "tools"'],
            [{ ...base, tools: {} }, 'Invalid value for option "tools"'],
            [{ ...base, tools: [{ type: 'web' }] }, 'Invalid value for option "tools.type": function'],
            [{ ...base, tools: [{ ...time, cache: true }] }, 'Option "tools.cache" is not supported'],
            [{ ...base, tools: [{ type: 'function' }] }, 'Invalid value for option "tools.function"'],
            [
                { ...base, tools: [{ type: 'function', function: { name: 'a b' } }] },
                'Invalid value for option "tools.function.name": up to 128 letters, digits, _, ., : and -',
            ],
            [
                { ...base, tools: [time, time] },
                'Invalid value for option "tools.function.name": get_time is declared twice',
            ],
            [
                { ...base, tools: [{ type: 'function', function: { name: 'x', parameters: [] } }] },
                'Invalid value for option "tools.function.parameters": a JSON Schema object',
            ],
            [
                { ...base, tools: [{ type: 'function', function: { name: 'x', examples: [] } }] },
                'Option "tools.function.examples" is not supported',
            ],
            [
                { ...base, tools: [time], tool_choice: 'any' },
                'Invalid value for option "tool_choice": auto, none, required or a function',
            ],
            [
                { ...base, tools: [time], tool_choice: { type: 'function', function: { name: 'other' } } },
                'Invalid value for option "tool_choice.function.name": the name of one of "tools"',
            ],
            [
                { ...base, model: 'gemini-2.5-flash', tools: [time], web_search: true },
                'Option "tools" is not supported together with "web_search" on Gemini 2.x',
            ],
            [
                {
                    ...base,
                    messages: [
                        { role: 'user', content: 'hi' },
                        { role: 'assistant', content: null },
                    ],
                },
                'Invalid value for option "messages": content null only with tool_calls',
            ],
            [
                { ...base, messages: [{ role: 'user', content: 'hi', tool_calls: [] }] },
                'Invalid value for option "messages": tool_calls only in assistant messages',
            ],
            [
                {
                    ...base,
                    messages: [
                        { role: 'user', content: 'hi' },
                        { role: 'tool', tool_call_id: 'nope', content: 'x' },
                    ],
                },
                'Invalid value for option "messages.tool_call_id": the id of a tool call in an earlier assistant message',
            ],
            [withCall({ id: '' }), 'Invalid value for option "messages.tool_calls.id"'],
            [withCall({ type: 'custom' }), 'Invalid value for option "messages.tool_calls.type": function'],
            [withCall({ index: 0 }), 'Option "messages.tool_calls.index" is not supported'],
            [
                withCall({ function: { name: 'get_time', arguments: '[1]' } }),
                'Invalid value for option "messages.tool_calls.function.arguments": JSON text of an object',
            ],
            [
                withCall({ extra_content: { google: { thought_signature: 1 } } }),
                'Invalid value for option "messages.tool_calls.extra_content"',
            ],
            [
                withCall({}, { content: [{ type: 'image_url', image_url: { url: 'data:image/png;base64,AA==' } }] }),
                'Invalid value for option "messages": images only in user messages',
            ],
            [
                { ...base, messages: [{ role: 'user', content: 'hi', tool_call_id: 'c' }] },
                'Invalid value for option "messages": tool_call_id only in tool messages',
            ],
            [
                {
                    ...base,
                    messages: [
                        { role: 'user', content: 'hi' },
                        { role: 'assistant', content: 'hi', tool_calls: 'bad' },
                    ],
                },
                'Invalid value for option "messages.tool_calls"',
            ],
            [
                withCall({ function: { name: '', arguments: '{}' } }),
                'Invalid value for option "messages.tool_calls.function.name"',
            ],
            [
                withCall({ function: { name: 'get_time', arguments: '{bad' } }),
                'Invalid value for option "messages.tool_calls.function.arguments": JSON text of an object',
            ],
            [
                { ...base, tools: [{ type: 'function', function: { name: 'x', description: 1 } }] },
                'Invalid value for option "tools.function.description"',
            ],
            [
                { ...base, tools: [{ type: 'function', function: { name: 'x', strict: 'yes' } }] },
                'Invalid value for option "tools.function.strict"',
            ],
            [
                { ...base, tools: [time], tool_choice: { type: 'other', function: { name: 'get_time' } } },
                'Invalid value for option "tool_choice.type": function',
            ],
            [{ ...base, messages: [{ role: 'user', content: 123 }] }, 'Invalid value for option "messages"'],
        ];
        for (const [value, message] of invalid) {
            expect(() => toGeminiCall(value as GenaiRequest, undefined)).toThrow(new TypeError(message));
        }
    });
});

describe('batchModel', () => {
    it('returns the shared model and rejects empty, duplicate-key and mixed-model batches', () => {
        const base: GenaiRequest = { model: 'm', messages: [{ role: 'user', content: 'hi' }] };
        expect(
            batchModel([
                { key: '1', request: base },
                { key: '2', request: base },
            ]),
        ).toBe('m');
        expect(() => batchModel([])).toThrow(new TypeError('Invalid value for option "items"'));
        expect(() =>
            batchModel([
                { key: '1', request: base },
                { key: '1', request: base },
            ]),
        ).toThrow(new TypeError('Invalid value for option "key"'));
        expect(() =>
            batchModel([
                { key: '1', request: base },
                { key: '2', request: { ...base, model: 'n' } },
            ]),
        ).toThrow(new TypeError('Invalid value for option "model": a batch takes one model'));
    });
});

describe('reasoning_effort', () => {
    const base: GenaiRequest = { model: 'gemini-3-flash', messages: [{ role: 'user', content: 'hi' }] };

    it('asks for thoughts unless Gemini 2.x thinks with budget 0; the config wins per member', () => {
        function thinking(request: GenaiRequest, config?: GenerateContentConfig) {
            return toGeminiCall(request, config).config.thinkingConfig;
        }
        expect(thinking({ ...base, reasoning_effort: 'low' })).toEqual({
            thinkingLevel: ThinkingLevel.LOW,
            includeThoughts: true,
        });
        expect(thinking({ ...base, reasoning_effort: 'none' })).toEqual({
            thinkingLevel: ThinkingLevel.MINIMAL,
            includeThoughts: true,
        });
        const flash2 = { ...base, model: 'gemini-2.5-flash' };
        expect(thinking({ ...flash2, reasoning_effort: 'medium' })).toEqual({
            thinkingBudget: 8192,
            includeThoughts: true,
        });
        expect(thinking({ ...flash2, reasoning_effort: 'none' })).toEqual({ thinkingBudget: 0 });
        expect(thinking({ ...base, model: 'gemini-2.5-pro', reasoning_effort: 'none' })).toEqual({
            thinkingBudget: 128,
            includeThoughts: true,
        });
        expect(
            thinking({ ...base, model: 'publishers/google/models/gemini-2.5-flash', reasoning_effort: 'none' }),
        ).toEqual({ thinkingBudget: 0 });
        expect(
            thinking({ ...base, model: 'publishers/google/models/gemini-2.5-pro', reasoning_effort: 'none' }),
        ).toEqual({ thinkingBudget: 128, includeThoughts: true });
        expect(thinking({ ...base, reasoning_effort: 'high' }, { thinkingConfig: { includeThoughts: false } })).toEqual(
            { thinkingLevel: ThinkingLevel.HIGH, includeThoughts: false },
        );
        expect(thinking(base)).toBeUndefined();
    });
});

describe('mergeGenaiConfig', () => {
    it('lets later configs win per field and merges thinkingConfig per member', () => {
        expect(mergeGenaiConfig()).toBeUndefined();
        expect(mergeGenaiConfig(undefined, undefined)).toBeUndefined();
        const only: GenerateContentConfig = { temperature: 1, thinkingConfig: { includeThoughts: true } };
        const copy = mergeGenaiConfig(undefined, only);
        expect(copy).toEqual(only);
        expect(copy).not.toBe(only);
        expect(copy?.thinkingConfig).not.toBe(only.thinkingConfig);
        expect(
            mergeGenaiConfig(
                { temperature: 1, topK: 3, thinkingConfig: { thinkingBudget: 512, includeThoughts: true } },
                undefined,
                { temperature: 0, thinkingConfig: { includeThoughts: false } },
                { seed: 2 },
            ),
        ).toEqual({
            temperature: 0,
            topK: 3,
            seed: 2,
            thinkingConfig: { thinkingBudget: 512, includeThoughts: false },
        });
        expect(mergeGenaiConfig({ temperature: 1 }, { topK: 2 })).toEqual({ temperature: 1, topK: 2 });
    });
});

describe('validateGenaiConfig', () => {
    it('accepts plain objects and throws a TypeError naming the option otherwise', () => {
        expect(() => validateGenaiConfig({}, 'config')).not.toThrow();
        expect(() => validateGenaiConfig({ temperature: 0, thinkingConfig: {} }, 'config')).not.toThrow();
        const invalid: [unknown, string][] = [
            [undefined, 'Invalid value for option "gemini.config"'],
            [null, 'Invalid value for option "gemini.config"'],
            ['x', 'Invalid value for option "gemini.config"'],
            [[], 'Invalid value for option "gemini.config"'],
            [JSON.parse('{"__proto__":{"topK":1}}'), 'Invalid value for option "gemini.config"'],
            [{ abortSignal: new AbortController().signal }, 'Option "gemini.config.abortSignal" is not supported'],
            [{ httpOptions: {} }, 'Option "gemini.config.httpOptions" is not supported'],
            [{ automaticContinuation: false }, 'Option "gemini.config.automaticContinuation" is not supported'],
        ];
        for (const [config, message] of invalid) {
            expect(() => validateGenaiConfig(config, 'gemini.config')).toThrow(new TypeError(message));
        }
    });
});
