// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeEach, describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createGenaiClient, credentialsFile } from '../../../src/core/client.js';
import type { GenaiClient, GenaiRequest } from '../../../src/types.js';
import { generateBody } from '../fake-gemini.js';
import { FAKE_TOKEN, fakeAuthClient, startFakeVertex } from '../fake-vertex.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

const server = startFakeVertex();
const tempDir = mkdtempSync(join(tmpdir(), 'smart-genai-'));

afterAll(() => {
    server.stop();
    rmSync(tempDir, { recursive: true, force: true });
});
beforeEach(() => {
    server.received.length = 0;
    server.objects.clear();
    server.created.clear();
    server.jobs.clear();
});

const schema = { type: 'object', properties: { kind: { type: 'string', enum: ['noun', 'verb'] } }, required: ['kind'] };

const request: GenaiRequest = {
    model: 'gemini-x',
    messages: [
        { role: 'system', content: 'Classify the word.' },
        { role: 'user', content: 'blorf' },
    ],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema } },
};

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Creates Vertex AI client configured for fake server; uses bucket unless disabled. */
function client(bucket = true): GenaiClient {
    return createGenaiClient({
        vertex: { authClient: fakeAuthClient(), storageUrl: server.baseUrl },
        env: {
            GOOGLE_CLOUD_PROJECT: 'p',
            GEMINI_API_KEY: 'must-not-be-sent',
            ...(bucket ? { VERTEX_BATCH_BUCKET: 'bkt' } : {}),
        },
        baseUrl: server.baseUrl,
        timeoutMs: 5000,
    });
}

/** Submits two batch requests and returns job name and storage folder. */
async function submitTwo(vertex: GenaiClient): Promise<{ name: string; folder: string }> {
    const submitted = await vertex.batch?.submit([
        { key: '7', request },
        { key: 'Key-8', request: { ...request, messages: [{ role: 'user', content: 'snarf' }], temperature: 0 } },
    ]);
    const name = submitted?.ok ? submitted.data.name : '';
    const input = [...server.objects.keys()][0] ?? '';
    return { name, folder: input.slice(0, -'/input.jsonl'.length) };
}

describe('Vertex AI options', () => {
    it('resolves project, location, bucket and credentials file from env', () => {
        const file = join(tempDir, 'credentials.json');
        writeFileSync(file, '{"type":"authorized_user"}');
        const env = { GOOGLE_CLOUD_PROJECT: 'p', GOOGLE_APPLICATION_CREDENTIALS: file, VERTEX_BATCH_BUCKET: 'bkt' };
        const vertex = createGenaiClient({ vertex: true, env });
        expect(vertex.mode).toBe('vertex');
        expect(vertex.batch).toBeDefined();
        expect(createGenaiClient({ vertex: {}, env: { ...env, VERTEX_BATCH_BUCKET: '' } }).batch).toBeUndefined();
        const empty = join(tempDir, 'empty.json');
        writeFileSync(empty, '');
        expect(() => createGenaiClient({ vertex: { credentialsFile: empty }, env })).toThrow(/is missing or empty/);
        expect(credentialsFile({ GOOGLE_APPLICATION_CREDENTIALS: file })).toBe(file);
        expect(credentialsFile({ APPDATA: 'A', HOME: 'H' })).toMatch(
            /gcloud[\\/]application_default_credentials\.json$/,
        );
    });
});

describe('generate on Vertex AI', () => {
    it('sends single requests to the project and location with the token, never an API key', async () => {
        server.generateBody = generateBody('{"kind":"noun"}');
        const result = await client(false).generate(request);
        expect(result).toMatchObject({ ok: true, data: { json: { kind: 'noun' }, model: 'gemini-fake-001' } });
        const sent = server.received[0];
        expect(sent?.path).toBe(
            '/v1beta1/projects/p/locations/global/publishers/google/models/gemini-x:generateContent',
        );
        expect(sent?.authorization).toBe(`Bearer ${FAKE_TOKEN}`);
        expect(JSON.stringify(sent)).not.toContain('must-not-be-sent');
        expect(client(false).batch).toBeUndefined();
    });
});

describe('batch on Vertex AI', () => {
    it('submits one JSONL file with REST requests and keys, and one job reading it', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        expect(name).toMatch(/batchPredictionJobs\/1$/);
        const [inputName, text] = [...server.objects][0] ?? ['', ''];
        expect(inputName).toMatch(/^smart-genai\/[0-9a-f-]{36}\/input\.jsonl$/);
        const lines = text
            .trimEnd()
            .split('\n')
            .map((line) => JSON.parse(line));
        expect(lines.map((line) => line.key)).toEqual(['j:7', 'j:Key-8']);
        expect(lines[0].request).toEqual({
            contents: [{ role: 'user', parts: [{ text: 'blorf' }] }],
            systemInstruction: { parts: [{ text: 'Classify the word.' }] },
            generationConfig: { responseMimeType: 'application/json', responseJsonSchema: schema },
        });
        expect(lines[1].request.generationConfig.temperature).toBe(0);
        const job = server.received.find((entry) => entry.path.endsWith('/batchPredictionJobs'));
        expect(job?.path).toBe('/v1beta1/projects/p/locations/global/batchPredictionJobs');
        expect(JSON.parse(job?.body ?? '{}')).toMatchObject({
            model: 'publishers/google/models/gemini-x',
            inputConfig: { gcsSource: { uris: [`gs://bkt/${folder}/input.jsonl`] } },
            outputConfig: { gcsDestination: { outputUriPrefix: `gs://bkt/${folder}/output/` } },
        });
        const bytes = vertex.batch?.requestBytes(request);
        expect(bytes?.ok && bytes.data).toBeGreaterThan(100);
        const withConfig = vertex.batch?.requestBytes(request, { systemInstruction: 'x'.repeat(1000) });
        expect(withConfig?.ok && bytes?.ok && withConfig.data - bytes.data).toBeGreaterThan(900);
        const withDefaults = createGenaiClient({
            vertex: { project: 'p', bucket: 'bkt', authClient: fakeAuthClient(), storageUrl: server.baseUrl },
            env: {},
            baseUrl: server.baseUrl,
            defaults: { temperature: 0 },
        });
        const base = withDefaults.batch?.requestBytes(request);
        const extended = withDefaults.batch?.requestBytes(request, { systemInstruction: 'x'.repeat(1000) });
        expect(extended?.ok && base?.ok && extended.data - base.data).toBeGreaterThan(900);
    });

    it('leaves out an empty generationConfig, which the JSONL import rejects', async () => {
        const vertex = client();
        await vertex.batch?.submit([
            { key: '1', request: { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] } },
        ]);
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request).toEqual({ contents: [{ role: 'user', parts: [{ text: 'hi' }] }] });
    });

    it('puts serviceTier and modelArmorConfig top-level, as generate sends them', async () => {
        const modelArmorConfig = { promptTemplateName: 'projects/p/locations/l/templates/t' };
        await client().batch?.submit(
            [{ key: '1', request: { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] } }],
            {
                config: {
                    serviceTier: 'flex' as never,
                    modelArmorConfig,
                    topK: 2,
                    systemInstruction: ['a', { text: 'b' }],
                },
            },
        );
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request).toEqual({
            contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
            systemInstruction: { parts: [{ text: 'a' }, { text: 'b' }] },
            serviceTier: 'flex',
            modelArmorConfig,
            generationConfig: { topK: 2 },
        });
        server.objects.clear();
        await client().batch?.submit(
            [{ key: '1', request: { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] } }],
            { config: { systemInstruction: { text: 'part' } } },
        );
        const partLine = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(partLine.request.systemInstruction).toEqual({ parts: [{ text: 'part' }] });
    });

    it('polls a running job without touching the bucket', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        server.received.length = 0;
        expect(await vertex.batch?.check(name)).toEqual({
            ok: true,
            data: { name, state: 'JOB_STATE_PENDING', phase: 'running' },
        });
        expect(server.received.map((entry) => entry.method)).toEqual(['GET']);
        expect(server.objects.size).toBe(1);
    });

    it('cancels a job and removes its folder', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        server.received.length = 0;
        expect(await vertex.batch?.cancel(name)).toEqual({ ok: true, data: undefined });
        expect(server.received.some((entry) => entry.method === 'POST' && entry.path.endsWith(':cancel'))).toBe(true);
        expect(server.jobs.get(name)?.state).toBe('JOB_STATE_CANCELLED');
        expect(server.objects.size).toBe(0);
        expect(await vertex.batch?.cancel('')).toMatchObject({ ok: false, error: { code: 'invalid_request' } });
    });

    it('removes the folder of a final job without asking the API to cancel it', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        const job = server.jobs.get(name);
        if (job) {
            job.state = 'JOB_STATE_FAILED';
        }
        server.received.length = 0;
        expect(await vertex.batch?.cancel(name)).toEqual({ ok: true, data: undefined });
        expect(server.received.some((entry) => entry.path.endsWith(':cancel'))).toBe(false);
        expect(server.jobs.get(name)?.state).toBe('JOB_STATE_FAILED');
        expect(server.objects.size).toBe(0);
    });

    it('removes no files of a job whose input lies outside the batch root', async () => {
        for (const folder of ['data', 'smart-genai', 'smart-genai/a/b']) {
            const name = 'projects/p/locations/global/batchPredictionJobs/9';
            server.objects.set(`${folder}/input.jsonl`, 'x');
            server.objects.set(`${folder}/keep.txt`, 'x');
            server.jobs.set(name, {
                name,
                state: 'JOB_STATE_RUNNING',
                inputConfig: { gcsSource: { uris: [`gs://bkt/${folder}/input.jsonl`] } },
            });
            expect(await client().batch?.cancel(name)).toEqual({ ok: true, data: undefined });
            expect(server.objects.has(`${folder}/keep.txt`)).toBe(true);
        }
    });

    it('reads results by prefixed key, parses JSON by the prefix, then removes the folder', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_SUCCEEDED' });
        const output = [
            { key: 'j:7', response: generateBody('{"kind":"verb"}'), status: '' },
            { key: 't:Key-8', response: generateBody('plain') },
            { key: 't:9', status: 'INTERNAL: Internal error' },
            { key: 't:10' },
            { key: 't:11', status: 'INVALID_ARGUMENT: bad schema' },
            { key: 't:12', status: 'RESOURCE_EXHAUSTED: Quota exceeded' },
            { key: 7, response: generateBody('unprefixed') },
            'not json',
        ];
        server.objects.set(
            `${folder}/output/prediction-model-1/predictions.jsonl`,
            output.map((line) => (typeof line === 'string' ? line : JSON.stringify(line))).join('\n'),
        );
        const done = await vertex.batch?.check(name);
        expect(done?.ok && done.data.phase).toBe('succeeded');
        const results = done?.ok ? (done.data.results ?? []) : [];
        expect(results.map((item) => item.key)).toEqual(['7', 'Key-8', '9', '10', '11', '12']);
        expect(results[0]?.result).toMatchObject({ ok: true, data: { json: { kind: 'verb' }, ms: 0 } });
        expect(results[1]?.result).toMatchObject({ ok: true, data: { text: 'plain' } });
        expect(results[1]?.result.ok && 'json' in results[1].result.data).toBe(false);
        expect(results[2]?.result).toMatchObject({
            ok: false,
            error: { code: 'http', message: 'INTERNAL: Internal error', retryable: true },
        });
        expect(results[3]?.result).toMatchObject({ ok: false, error: { code: 'unexpected', retryable: false } });
        expect(results[4]?.result).toMatchObject({ ok: false, error: { code: 'http', retryable: false } });
        expect(results[5]?.result).toEqual({
            ok: false,
            error: {
                code: 'rate_limited',
                message: 'RESOURCE_EXHAUSTED: Quota exceeded',
                retryable: true,
                status: 429,
            },
        });
        expect(server.objects.size).toBe(0);
        expect(await vertex.batch?.check(name)).toMatchObject({
            ok: false,
            error: { code: 'unexpected', message: 'Batch output already read or missing' },
        });
    });

    it('keeps digit-only keys as written', async () => {
        const vertex = client();
        const submitted = await vertex.batch?.submit([{ key: '007', request }]);
        const name = submitted?.ok ? submitted.data.name : '';
        const [inputName] = [...server.objects.keys()];
        const folder = (inputName ?? '').slice(0, -'/input.jsonl'.length);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_SUCCEEDED' });
        server.objects.set(
            `${folder}/output/predictions.jsonl`,
            JSON.stringify({ key: 'j:007', response: generateBody('{"kind":"noun"}') }),
        );
        const done = await vertex.batch?.check(name);
        expect(done?.ok && done.data.results?.map((item) => item.key)).toEqual(['007']);
    });

    it('removes the folder of a failed job and keeps the files when a download fails', async () => {
        const failed = await submitTwo(client());
        server.jobs.set(failed.name, { ...server.jobs.get(failed.name), state: 'JOB_STATE_FAILED' });
        expect(await client().batch?.check(failed.name)).toEqual({
            ok: true,
            data: { name: failed.name, state: 'JOB_STATE_FAILED', phase: 'failed' },
        });
        expect(server.objects.size).toBe(0);

        const kept = await submitTwo(client());
        server.jobs.set(kept.name, { ...server.jobs.get(kept.name), state: 'JOB_STATE_SUCCEEDED' });
        const unreachable = createGenaiClient({
            vertex: { project: 'p', bucket: 'bkt', authClient: fakeAuthClient(), storageUrl: 'http://127.0.0.1:1' },
            env: {},
            baseUrl: server.baseUrl,
            timeoutMs: 5000,
        });
        expect(await unreachable.batch?.check(kept.name)).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true },
        });
        expect(server.objects.size).toBe(1);
        expect(await unreachable.batch?.submit([{ key: '1', request }])).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true },
        });

        server.objects.clear();
        const unreachableVertex = createGenaiClient({
            vertex: { project: 'p', bucket: 'bkt', authClient: fakeAuthClient(), storageUrl: server.baseUrl },
            env: {},
            baseUrl: 'http://127.0.0.1:1',
            timeoutMs: 5000,
        });
        expect(await unreachableVertex.batch?.submit([{ key: '1', request }])).toMatchObject({
            ok: false,
            error: { code: 'network', retryable: true },
        });
        expect(server.objects.size).toBe(0);
    });
});

describe('removeFiles on Vertex AI', () => {
    /** Populates fake server with batch prediction job referencing specified folder. */
    function seedJob(id: number, state: string, folder: string, bucket = 'bkt'): void {
        const name = `projects/p/locations/us-central1/batchPredictionJobs/${id}`;
        server.jobs.set(name, {
            name,
            state,
            inputConfig: { gcsSource: { uris: [`gs://${bucket}/${folder}/input.jsonl`] } },
            outputConfig: { gcsDestination: { outputUriPrefix: `gs://${bucket}/${folder}/output/` } },
        });
    }

    /** Populates fake storage bucket with batch folders and external test objects. */
    function seedObjects(): void {
        for (const name of [
            'smart-genai/a/input.jsonl',
            'smart-genai/a/output/prediction-1/predictions.jsonl',
            'smart-genai/b/input.jsonl',
            'smart-genai/c/input.jsonl',
            'smart-genai/d/output/prediction-1/predictions.jsonl',
            'smart-genai/e/input.jsonl',
            'smart-genai/loose.txt',
            'other/x.txt',
            'smart-genaix/y.txt',
        ]) {
            server.objects.set(name, 'x');
        }
    }

    it('removes stale folders, keeps folders of jobs not final yet and leaves objects outside the prefix', async () => {
        seedObjects();
        seedJob(1, 'JOB_STATE_SUCCEEDED', 'smart-genai/a');
        seedJob(2, 'JOB_STATE_RUNNING', 'smart-genai/c');
        seedJob(3, 'JOB_STATE_CANCELLING', 'smart-genai/d');
        seedJob(4, 'JOB_STATE_QUEUED', 'smart-genai/e');
        seedJob(5, 'JOB_STATE_PENDING', 'smart-genai/b', 'other-bucket');
        server.jobs.set('null', null as never);
        expect(await client().batch?.removeFiles()).toEqual({ ok: true, data: { removed: 3, kept: 3 } });
        expect([...server.objects.keys()].sort()).toEqual([
            'other/x.txt',
            'smart-genai/c/input.jsonl',
            'smart-genai/d/output/prediction-1/predictions.jsonl',
            'smart-genai/e/input.jsonl',
            'smart-genaix/y.txt',
        ]);
        const pages = server.received.filter((entry) => entry.method === 'GET' && entry.path.endsWith('Jobs'));
        expect(pages.length).toBe(3);
    });

    it('keeps a folder with an object younger than one hour or of unknown age, e.g. the input of a job being submitted', async () => {
        seedObjects();
        server.created.set('smart-genai/a/output/prediction-1/predictions.jsonl', new Date().toISOString());
        server.created.set('smart-genai/b/input.jsonl', new Date(Date.now() - 3_500_000).toISOString());
        server.created.set('smart-genai/c/input.jsonl', new Date(Date.now() - 3_700_000).toISOString());
        server.created.set('smart-genai/d/output/prediction-1/predictions.jsonl', 'not a time');
        const vertex = client();
        await submitTwo(vertex);
        expect(await vertex.batch?.removeFiles()).toEqual({ ok: true, data: { removed: 3, kept: 4 } });
        expect(server.objects.has('smart-genai/d/output/prediction-1/predictions.jsonl')).toBe(true);
        expect(server.objects.has('smart-genai/a/input.jsonl')).toBe(true);
        expect(server.objects.has('smart-genai/b/input.jsonl')).toBe(true);
        expect(server.objects.has('smart-genai/c/input.jsonl')).toBe(false);
        expect([...server.objects.keys()].filter((name) => name.endsWith('/input.jsonl'))).toHaveLength(3);
    });

    it('reports an empty bucket as nothing removed', async () => {
        expect(await client().batch?.removeFiles()).toEqual({ ok: true, data: { removed: 0, kept: 0 } });
    });

    it('deletes nothing when the job or object listing fails', async () => {
        for (const failList of ['jobs', 'objects'] as const) {
            seedObjects();
            seedJob(1, 'JOB_STATE_SUCCEEDED', 'smart-genai/a');
            server.received.length = 0;
            server.failList = failList;
            try {
                expect(await client().batch?.removeFiles()).toMatchObject({ ok: false, error: { retryable: true } });
            } finally {
                server.failList = undefined;
            }
            expect(server.objects.size).toBe(9);
            expect(server.received.some((entry) => entry.method === 'DELETE')).toBe(false);
        }
    });

    it('answers the error of a failed delete instead of counting the folder as removed', async () => {
        server.objects.set('smart-genai/a/input.jsonl', 'x');
        server.failDelete = 403;
        try {
            expect(await client().batch?.removeFiles()).toMatchObject({
                ok: false,
                error: { code: 'http', status: 403, retryable: false },
            });
        } finally {
            server.failDelete = undefined;
        }
        expect(server.objects.has('smart-genai/a/input.jsonl')).toBe(true);
    });

    it('rejects unknown options as invalid_request', async () => {
        const removed = await client().batch?.removeFiles({ when: 'now' } as never);
        expect(removed).toMatchObject({ ok: false, error: { code: 'invalid_request' } });
    });

    it('rejects a signal that is no AbortSignal as invalid_request', async () => {
        const batch = client().batch;
        const signal = { signal: 'x' } as never;
        const results = [
            await batch?.submit([{ key: 'a', request }], signal),
            await batch?.check('batchPredictionJobs/1', signal),
            await batch?.cancel('batchPredictionJobs/1', signal),
            await batch?.removeFiles(signal),
        ];
        for (const result of results) {
            expect(result).toEqual({
                ok: false,
                error: { code: 'invalid_request', message: 'Invalid value for option "signal"', retryable: false },
            });
        }
    });
});
