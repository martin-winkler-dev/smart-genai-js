// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Request received by fake Gemini server. */
export type GeminiRequest = { method: string; path: string; headers: Headers; body: Record<string, unknown> | null };

/** Handler returning response for fake Gemini request. */
export type GeminiReply = (request: GeminiRequest) => Response | Promise<Response>;

/** Local fake Gemini Developer API test server. */
export type FakeGemini = {
    baseUrl: string;
    received: GeminiRequest[];
    /** Sets reply sequence; repeats final reply once reached. */
    reply(...replies: GeminiReply[]): void;
    stop(): void;
};

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/** Starts fake Gemini server on random loopback port. */
export function startFakeGemini(): FakeGemini {
    const received: GeminiRequest[] = [];
    let replies: GeminiReply[] = [() => generateResponse('{}')];
    let index = 0;
    const server = Bun.serve({
        port: 0,
        hostname: '127.0.0.1',
        async fetch(request) {
            const entry: GeminiRequest = {
                method: request.method,
                path: new URL(request.url).pathname,
                headers: request.headers,
                body: request.method === 'GET' ? null : ((await request.json()) as Record<string, unknown>),
            };
            received.push(entry);
            const reply = replies[Math.min(index, replies.length - 1)];
            index++;
            return reply(entry);
        },
    });
    return {
        baseUrl: `http://127.0.0.1:${server.port}`,
        received,
        reply(...next) {
            replies = next;
            index = 0;
        },
        stop() {
            server.stop(true);
        },
    };
}

/** Constructs wire response body for generateContent calls. */
export function generateBody(
    text: string,
    overrides: { finishReason?: string; thought?: string; blockReason?: string } = {},
): Record<string, unknown> {
    const parts: Record<string, unknown>[] = [];
    if (overrides.thought !== undefined) {
        parts.push({ text: overrides.thought, thought: true });
    }
    parts.push({ text });
    return {
        candidates: [{ content: { role: 'model', parts }, finishReason: overrides.finishReason ?? 'STOP', index: 0 }],
        usageMetadata: { promptTokenCount: 12, candidatesTokenCount: 4, thoughtsTokenCount: 3, totalTokenCount: 19 },
        modelVersion: 'gemini-fake-001',
        ...(overrides.blockReason === undefined ? {} : { promptFeedback: { blockReason: overrides.blockReason } }),
    };
}

/** Creates JSON HTTP Response wrapping generateBody. */
export function generateResponse(text: string, overrides: Parameters<typeof generateBody>[1] = {}): Response {
    return Response.json(generateBody(text, overrides));
}

/** Creates JSON HTTP Response representing batch operation. */
export function batchResponse(name: string, state: string, responses?: Record<string, unknown>[]): Response {
    return Response.json({
        name,
        metadata: {
            '@type': 'type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch',
            name,
            model: 'models/gemini-2.5-flash',
            state,
            ...(responses === undefined ? {} : { output: { inlinedResponses: { inlinedResponses: responses } } }),
        },
    });
}

/** Creates Google API error response with status and optional retry delay. */
export function apiError(status: number, retryDelay?: string): Response {
    const details =
        retryDelay === undefined ? [] : [{ '@type': 'type.googleapis.com/google.rpc.RetryInfo', retryDelay }];
    return Response.json({ error: { code: status, message: `fake ${status}`, status: 'FAKE', details } }, { status });
}
