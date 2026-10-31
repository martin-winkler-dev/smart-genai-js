// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenaiError, Result } from '../../types.js';
import { statusError, toGenaiError } from '../answer.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **Options for `createGcsClient`**.
 *
 * - `bucket`: target bucket name
 * - `accessToken`: async callback resolving OAuth access token
 * - `timeoutMs`: base request timeout in milliseconds
 * - `baseUrl`: custom Cloud Storage endpoint URL
 */
export type GcsClientOptions = {
    bucket: string;
    accessToken: () => Promise<string | null | undefined>;
    timeoutMs: number;
    baseUrl?: string;
};

/** Cloud Storage object metadata item. */
export type GcsObject = { name: string; timeCreated?: string };

/** Cloud Storage client interface for batch file operations. */
export type GcsClient = {
    readonly bucket: string;
    upload(name: string, text: string, signal?: AbortSignal): Promise<Result<void, GenaiError>>;
    list(prefix: string, signal?: AbortSignal): Promise<Result<string[], GenaiError>>;
    listObjects(prefix: string, signal?: AbortSignal): Promise<Result<GcsObject[], GenaiError>>;
    download(name: string, signal?: AbortSignal): Promise<Result<string, GenaiError>>;
    remove(name: string, signal?: AbortSignal): Promise<Result<void, GenaiError>>;
};

/** Single page of object listing response. */
type ListPage = { items?: { name?: string; timeCreated?: string }[]; nextPageToken?: string };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Cloud Storage JSON API base endpoint. */
const GCS_BASE_URL = 'https://storage.googleapis.com';

/** HTTP status code representing missing resource. */
const NOT_FOUND_STATUS = 404;

/** Minimum expected transfer rate in bytes per millisecond (250 KB/s). */
const MIN_BYTES_PER_MS = 250;

/** Maximum planned batch download payload size in bytes. */
const MAX_DOWNLOAD_BYTES = 100_000_000;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Cloud Storage client on the JSON API**: manages bucket operations using fetch.
 *
 * - uploads text content via single `uploadType=media` request
 * - lists object names and creation timestamps with pagination support
 * - downloads and deletes objects
 * - handles timeouts and maps responses to `GenaiError`
 */
export function createGcsClient(options: GcsClientOptions): GcsClient {
    const base = options.baseUrl ?? GCS_BASE_URL;
    const bucket = encodeURIComponent(options.bucket);

    /** JSON API URL of one object. */
    function objectUrl(name: string): string {
        return `${base}/storage/v1/b/${bucket}/o/${encodeURIComponent(name)}`;
    }

    /** One authorized request with timeout; answers the response body; non-2xx and thrown errors become `GenaiError`. */
    async function call(
        url: string,
        init: RequestInit,
        signal: AbortSignal | undefined,
        timeoutMs = options.timeoutMs,
    ): Promise<Result<string, GenaiError>> {
        const timeout = AbortSignal.timeout(timeoutMs);
        try {
            const token = await options.accessToken();
            const response = await fetch(url, {
                ...init,
                headers: { ...init.headers, authorization: `Bearer ${token ?? ''}` },
                signal: signal === undefined ? timeout : AbortSignal.any([signal, timeout]),
            });
            const body = await response.text();
            if (!response.ok) {
                return {
                    ok: false,
                    error: statusError(response.status, `Cloud Storage ${response.status}: ${body.slice(0, 500)}`),
                };
            }
            return { ok: true, data: body };
        } catch (error) {
            const mapped = toGenaiError(error, signal);
            return { ok: false, error: { ...mapped, message: `Cloud Storage: ${mapped.message}` } };
        }
    }

    /** All objects under a prefix with their creation time, following `nextPageToken`. */
    async function listObjects(prefix: string, signal?: AbortSignal): Promise<Result<GcsObject[], GenaiError>> {
        const objects: GcsObject[] = [];
        let pageToken: string | undefined;
        do {
            const query = `prefix=${encodeURIComponent(prefix)}${pageToken ? `&pageToken=${encodeURIComponent(pageToken)}` : ''}`;
            const listed = await call(`${base}/storage/v1/b/${bucket}/o?${query}`, { method: 'GET' }, signal);
            if (!listed.ok) {
                return listed;
            }
            const page = parseListPage(listed.data);
            if (page === undefined) {
                const message = 'Cloud Storage: unreadable object listing';
                return { ok: false, error: { code: 'unexpected', message, retryable: false } };
            }
            for (const { name, timeCreated } of page.items ?? []) {
                if (name) {
                    objects.push(typeof timeCreated === 'string' ? { name, timeCreated } : { name });
                }
            }
            pageToken = page.nextPageToken;
        } while (pageToken);
        return { ok: true, data: objects };
    }

    return {
        bucket: options.bucket,
        async upload(name, text, signal) {
            const url = `${base}/upload/storage/v1/b/${bucket}/o?uploadType=media&name=${encodeURIComponent(name)}`;
            const sent = await call(
                url,
                { method: 'POST', headers: { 'content-type': 'application/jsonl' }, body: text },
                signal,
                options.timeoutMs + Math.ceil(text.length / MIN_BYTES_PER_MS),
            );
            return sent.ok ? { ok: true, data: undefined } : sent;
        },
        listObjects,
        async list(prefix, signal) {
            const listed = await listObjects(prefix, signal);
            return listed.ok ? { ok: true, data: listed.data.map((object) => object.name) } : listed;
        },
        async download(name, signal) {
            const timeoutMs = options.timeoutMs + MAX_DOWNLOAD_BYTES / MIN_BYTES_PER_MS;
            return await call(`${objectUrl(name)}?alt=media`, { method: 'GET' }, signal, timeoutMs);
        },
        async remove(name, signal) {
            const removed = await call(objectUrl(name), { method: 'DELETE' }, signal);
            if (!removed.ok && removed.error.status !== NOT_FOUND_STATUS) {
                return removed;
            }
            return { ok: true, data: undefined };
        },
    };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Parses JSON string into ListPage object. */
function parseListPage(body: string): ListPage | undefined {
    try {
        const page = JSON.parse(body) as unknown;
        return typeof page === 'object' && page !== null ? (page as ListPage) : undefined;
    } catch {
        return undefined;
    }
}
