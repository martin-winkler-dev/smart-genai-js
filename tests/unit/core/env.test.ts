// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { toGenaiError } from '../../../src/core/answer.js';
import { createGenaiClient } from '../../../src/core/client.js';
import type { GenaiRequest } from '../../../src/types.js';
import { generateBody } from '../fake-gemini.js';
import { FAKE_TOKEN, fakeAuthClient } from '../fake-vertex.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Captured fetch request containing URL and request headers. */
type Sent = { url: string; headers: Headers };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Environment variables automatically read by `@google/genai` to verify isolation. */
const SDK_ENV = {
    GOOGLE_GEMINI_BASE_URL: 'http://evil.invalid/',
    GOOGLE_VERTEX_BASE_URL: 'http://evil.invalid/',
    GOOGLE_API_KEY: 'env-google-key',
    GEMINI_API_KEY: 'env-gemini-key',
    GOOGLE_GENAI_USE_VERTEXAI: 'true',
    GOOGLE_CLOUD_PROJECT: 'env-project',
    GOOGLE_CLOUD_LOCATION: 'europe-west4',
};

const request: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };

const realFetch = globalThis.fetch;
const savedEnv = new Map(Object.keys(SDK_ENV).map((key) => [key, process.env[key]]));
let sent: Sent[] = [];

beforeEach(() => {
    sent = [];
    Object.assign(process.env, SDK_ENV);
    globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
        const url = input instanceof Request ? input.url : String(input);
        sent.push({ url, headers: new Headers(input instanceof Request ? input.headers : init?.headers) });
        return Response.json(generateBody('ok'));
    }) as typeof fetch;
});
afterEach(() => {
    globalThis.fetch = realFetch;
    for (const [key, value] of savedEnv) {
        if (value === undefined) {
            delete process.env[key];
        } else {
            process.env[key] = value;
        }
    }
});

describe('env variables read by @google/genai', () => {
    it('cannot change the endpoint or the key on the Developer API', async () => {
        const result = await createGenaiClient({ apiKey: 'option-key', env: {} }).generate(request);
        expect(result.ok).toBe(true);
        expect(sent.map((entry) => new URL(entry.url).origin)).toEqual(['https://generativelanguage.googleapis.com']);
        expect(sent[0]?.headers.get('x-goog-api-key')).toBe('option-key');
        expect(JSON.stringify([...(sent[0]?.headers ?? [])])).not.toContain('env-');
    });

    it('cannot change the endpoint, project or location on Vertex AI, and no key is sent', async () => {
        const vertex = createGenaiClient({
            vertex: { project: 'p', location: 'us-central1', authClient: fakeAuthClient() },
            env: {},
        });
        expect((await vertex.generate(request)).ok).toBe(true);
        const url = new URL(sent[0]?.url ?? '');
        expect(url.origin).toBe('https://us-central1-aiplatform.googleapis.com');
        expect(url.pathname).toBe(
            '/v1beta1/projects/p/locations/us-central1/publishers/google/models/gemini-x:generateContent',
        );
        expect(sent[0]?.headers.get('authorization')).toBe(`Bearer ${FAKE_TOKEN}`);
        expect(sent[0]?.headers.get('x-goog-api-key')).toBeNull();
        expect(url.search).not.toContain('key');
    });

    it('uses the official endpoint of each Vertex location', async () => {
        for (const [location, origin] of [
            ['global', 'https://aiplatform.googleapis.com'],
            ['eu', 'https://aiplatform.eu.rep.googleapis.com'],
        ]) {
            await createGenaiClient({
                vertex: { project: 'p', location, authClient: fakeAuthClient() },
                env: {},
            }).generate(request);
            expect(new URL(sent.at(-1)?.url ?? '').origin).toBe(origin ?? '');
        }
        for (const location of ['evil.com/x', 'us central', '']) {
            expect(() =>
                createGenaiClient({ vertex: { project: 'p', location, authClient: fakeAuthClient() }, env: {} }),
            ).toThrow(TypeError);
        }
        expect(() =>
            createGenaiClient({ vertex: { project: 'p/../q', authClient: fakeAuthClient() }, env: {} }),
        ).toThrow(TypeError);
    });
});

describe('errors outside the API', () => {
    it('classifies sign-in and file errors as fatal, socket errors as network', () => {
        const signIn = Object.assign(new Error('invalid_grant'), { response: { status: 400 } });
        expect(toGenaiError(signIn, undefined)).toMatchObject({ code: 'http', status: 400, retryable: false });
        const file = Object.assign(new Error('no file'), { code: 'ENOENT' });
        expect(toGenaiError(file, undefined)).toMatchObject({ code: 'unexpected', retryable: false });
        const socket = Object.assign(new Error('reset'), { code: 'ECONNRESET' });
        expect(toGenaiError(socket, undefined)).toMatchObject({ code: 'network', retryable: true });
    });

    it('classifies only failed fetches and connections as network, other TypeErrors as unexpected', () => {
        const network = [
            new TypeError('fetch failed', { cause: Object.assign(new Error('x'), { code: 'UND_ERR_SOCKET' }) }),
            Object.assign(new TypeError('Unable to connect'), { code: 'ConnectionRefused' }),
            Object.assign(new TypeError('getaddrinfo ENOTFOUND x'), { code: 'ENOTFOUND' }),
            Object.assign(new Error('timed out'), { code: 'ETIMEDOUT' }),
        ];
        for (const error of network) {
            expect(toGenaiError(error, undefined)).toMatchObject({ code: 'network', retryable: true });
        }
        const local = [
            new TypeError('Do not know how to serialize a BigInt'),
            new TypeError('Converting circular structure to JSON'),
            new TypeError('fetch failed'),
            Object.assign(new TypeError('bad argument'), { code: 'ERR_INVALID_ARG_TYPE' }),
        ];
        for (const error of local) {
            expect(toGenaiError(error, undefined)).toMatchObject({ code: 'unexpected', retryable: false });
        }
    });
});
