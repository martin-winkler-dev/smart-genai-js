// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { AuthClient } from 'google-auth-library';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Request received by fake server. */
export type VertexRequest = { method: string; path: string; search: string; authorization: string; body: string };

/** Local fake Vertex AI and Cloud Storage test server handle. */
export type FakeVertex = {
    baseUrl: string;
    received: VertexRequest[];
    /** Objects stored in fake bucket by name. */
    objects: Map<string, string>;
    /** Object creation timestamps (RFC 3339) by name. */
    created: Map<string, string>;
    /** Batch jobs stored by full resource name. */
    jobs: Map<string, Record<string, unknown>>;
    /** Response body returned for generateContent calls. */
    generateBody: Record<string, unknown>;
    /** Injects HTTP 500 error for job or object listings. */
    failList: 'jobs' | 'objects' | undefined;
    /** Injects HTTP status error for object delete operations. */
    failDelete: number | undefined;
    stop(): void;
};

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Token returned by fake auth client. */
export const FAKE_TOKEN = 'fake-token';

/** Matches Cloud Storage upload URL path. */
const MATCH_UPLOAD_PATH = /^\/upload\/storage\/v1\/b\/[^/]+\/o$/;

/** Matches Cloud Storage object collection and item paths. */
const MATCH_OBJECT_PATH = /^\/storage\/v1\/b\/[^/]+\/o(?:\/(.+))?$/;

/** Matches Vertex batch prediction jobs collection, job item, and cancel endpoints. */
const MATCH_JOBS_PATH = /^\/v1beta1\/(projects\/[^/]+\/locations\/[^/]+\/batchPredictionJobs)(?:\/(\d+)(:cancel)?)?$/;

/** Maximum number of jobs per page in fake listing. */
const JOBS_PAGE_SIZE = 2;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Starts fake Vertex AI and Cloud Storage server**:
 *
 * - storage: handles upload, listing with timestamps, download, and delete
 * - Vertex: handles generateContent and batch jobs lifecycle
 */
export function startFakeVertex(): FakeVertex {
    const received: VertexRequest[] = [];
    const objects = new Map<string, string>();
    const created = new Map<string, string>();
    const jobs = new Map<string, Record<string, unknown>>();
    const fake: Omit<FakeVertex, 'baseUrl' | 'stop'> = {
        received,
        objects,
        created,
        jobs,
        generateBody: {},
        failList: undefined,
        failDelete: undefined,
    };
    let nextJob = 1;
    const server = Bun.serve({
        port: 0,
        hostname: '127.0.0.1',
        async fetch(request) {
            const url = new URL(request.url);
            const body = request.method === 'GET' || request.method === 'DELETE' ? '' : await request.text();
            const authorization = request.headers.get('authorization') ?? '';
            received.push({ method: request.method, path: url.pathname, search: url.search, authorization, body });
            if (MATCH_UPLOAD_PATH.test(url.pathname)) {
                objects.set(url.searchParams.get('name') ?? '', body);
                created.set(url.searchParams.get('name') ?? '', new Date().toISOString());
                return Response.json({ name: url.searchParams.get('name') });
            }
            const object = MATCH_OBJECT_PATH.exec(url.pathname);
            if (object) {
                if (object[1] === undefined && fake.failList === 'objects') {
                    return apiError(500);
                }
                if (request.method === 'DELETE' && fake.failDelete !== undefined) {
                    return apiError(fake.failDelete);
                }
                return storageReply(fake, request.method, object[1], url);
            }
            const job = MATCH_JOBS_PATH.exec(url.pathname);
            if (job?.[3] !== undefined) {
                const stored = jobs.get(`${job[1]}/${job[2]}`);
                if (stored === undefined) {
                    return apiError(404);
                }
                stored.state = 'JOB_STATE_CANCELLED';
                return Response.json({});
            }
            if (job && request.method === 'POST') {
                const name = `${job[1]}/${nextJob++}`;
                const created = { ...(JSON.parse(body) as Record<string, unknown>), name, state: 'JOB_STATE_PENDING' };
                jobs.set(name, created);
                return Response.json(created);
            }
            if (job && job[2] === undefined && request.method === 'GET') {
                return fake.failList === 'jobs' ? apiError(500) : jobsPage([...jobs.values()], url);
            }
            if (job?.[2] !== undefined) {
                const stored = jobs.get(`${job[1]}/${job[2]}`);
                return stored ? Response.json(stored) : apiError(404);
            }
            if (url.pathname.endsWith(':generateContent')) {
                return Response.json(fake.generateBody);
            }
            return apiError(404);
        },
    });
    return Object.assign(fake, {
        baseUrl: `http://127.0.0.1:${server.port}`,
        stop() {
            server.stop(true);
        },
    });
}

/** Creates fake AuthClient returning FAKE_TOKEN without network requests. */
export function fakeAuthClient(): AuthClient {
    const client = {
        getAccessToken: () => Promise.resolve({ token: FAKE_TOKEN }),
        getRequestHeaders: () => Promise.resolve(new Headers({ authorization: `Bearer ${FAKE_TOKEN}` })),
    };
    return client as unknown as AuthClient;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Generates paginated response for batch jobs listing. */
function jobsPage(all: Record<string, unknown>[], url: URL): Response {
    const offset = Number(url.searchParams.get('pageToken') ?? '0');
    const size = JOBS_PAGE_SIZE;
    const next = offset + size < all.length ? { nextPageToken: String(offset + size) } : {};
    return Response.json({ batchPredictionJobs: all.slice(offset, offset + size), ...next });
}

/** Handles storage operations (list, download, delete) for fake bucket. */
function storageReply(
    { objects, created }: Pick<FakeVertex, 'objects' | 'created'>,
    method: string,
    encoded: string | undefined,
    url: URL,
): Response {
    if (encoded === undefined) {
        const prefix = url.searchParams.get('prefix') ?? '';
        const items = [...objects.keys()]
            .filter((name) => name.startsWith(prefix))
            .map((name) => ({ name, timeCreated: created.get(name) ?? '2000-01-01T00:00:00.000Z' }));
        return Response.json(items.length === 0 ? {} : { items });
    }
    const name = decodeURIComponent(encoded);
    const text = objects.get(name);
    if (text === undefined) {
        return apiError(404);
    }
    if (method === 'DELETE') {
        objects.delete(name);
        return new Response(null, { status: 204 });
    }
    return new Response(text);
}

/** Creates JSON error response with specified HTTP status. */
function apiError(status: number): Response {
    return Response.json({ error: { code: status, message: `fake ${status}`, status: 'FAKE' } }, { status });
}
