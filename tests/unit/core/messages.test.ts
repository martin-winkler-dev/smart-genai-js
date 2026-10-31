// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { describe, expect, it } from 'bun:test';
import { assistantMessage, toolMessage } from '../../../src/core/messages.js';
import type { GenaiToolCall } from '../../../src/types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const call: GenaiToolCall = {
    id: 'c1',
    type: 'function',
    function: { name: 'f', arguments: '{}' },
    extra_content: { google: { thought_signature: 'c2ln' } },
};

describe('assistantMessage', () => {
    it('keeps the text, and adds the tool calls with their signatures only when the answer has them', () => {
        expect(assistantMessage({ text: 'hi' })).toEqual({ role: 'assistant', content: 'hi' });
        expect(assistantMessage({ text: '', toolCalls: [] })).toEqual({ role: 'assistant', content: '' });
        expect(assistantMessage({ text: '', toolCalls: [call] })).toEqual({
            role: 'assistant',
            content: null,
            tool_calls: [call],
        });
        expect(assistantMessage({ text: 'checking', toolCalls: [call] })).toEqual({
            role: 'assistant',
            content: 'checking',
            tool_calls: [call],
        });
    });
});

describe('toolMessage', () => {
    it('sends a string as it is and anything else as JSON', () => {
        expect(toolMessage(call, 'sunny')).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'sunny' });
        expect(toolMessage(call, { sky: 'sunny' })).toEqual({
            role: 'tool',
            tool_call_id: 'c1',
            content: '{"sky":"sunny"}',
        });
        expect(toolMessage(call, undefined)).toEqual({ role: 'tool', tool_call_id: 'c1', content: 'null' });
    });
});
