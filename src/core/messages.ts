// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenaiMessage, GenaiOutput, GenaiToolCall } from '../types.js';

// -------------------------------------------------------------------
// --- 1. Exported Functions
// -------------------------------------------------------------------

/**
 * **Formats answer as next `assistant` message**: preserves text and tool calls with thought signatures.
 *
 * - sets `content` to `null` if answer has tool calls without text
 */
export function assistantMessage(output: Pick<GenaiOutput, 'text' | 'toolCalls'>): GenaiMessage {
    const toolCalls = output.toolCalls ?? [];
    const content = output.text === '' && toolCalls.length > 0 ? null : output.text;
    return {
        role: 'assistant',
        content,
        // biome-ignore lint/style/useNamingConvention: OpenAI field name
        ...(toolCalls.length === 0 ? {} : { tool_calls: toolCalls }),
    };
}

/** Formats tool call result as `tool` message: string as-is, other values serialized to JSON. */
export function toolMessage(call: Pick<GenaiToolCall, 'id'>, result: unknown): GenaiMessage {
    const content = typeof result === 'string' ? result : (JSON.stringify(result) ?? 'null');
    // biome-ignore lint/style/useNamingConvention: OpenAI field name
    return { role: 'tool', tool_call_id: call.id, content };
}
