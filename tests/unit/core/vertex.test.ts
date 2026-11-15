// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { afterAll, beforeEach, describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { GenerateContentConfig } from '@google/genai';
import { batchTiming } from '../../../src/core/batch/calls.js';
import { vertexBatchLimit } from '../../../src/core/batch/vertex.js';
import { withoutForeignProperties } from '../../../src/core/batch/vertex-foreign.js';
import { createGenaiClient, credentialsFile } from '../../../src/core/client.js';
import type { GenaiClient, GenaiRequest, GenaiTool } from '../../../src/types.js';
import { generateBody } from '../fake-gemini.js';
import { FAKE_TOKEN, fakeAuthClient, PUBLISHER_MODELS, startFakeVertex } from '../fake-vertex.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Config fields left out of `PARITY_CONFIG`: client-side, rejected on Vertex, or schemas the line changes. */
type ParityExcluded =
    | 'httpOptions'
    | 'abortSignal'
    | 'automaticContinuation'
    | 'automaticFunctionCalling'
    | 'enableEnhancedCivicAnswers'
    | 'store'
    | 'responseSchema'
    | 'responseJsonSchema';

// -------------------------------------------------------------------
// --- 2. Constants
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
    server.failCancel = undefined;
    server.stateAfterGet = undefined;
    server.models = [...PUBLISHER_MODELS];
});

/** Own folder prefix of `client()`: project `p`, location `global`. */
const OWN = 'smart-genai/p_global_';

/** Legacy folders (plain UUID segment) written before the owner prefix. */
const LEGACY_A = 'smart-genai/00000000-0000-4000-8000-00000000000a';
const LEGACY_B = 'smart-genai/00000000-0000-4000-8000-00000000000b';

const schema = { type: 'object', properties: { kind: { type: 'string', enum: ['noun', 'verb'] } }, required: ['kind'] };

/**
 * **Config setting every other SDK config field with a simple value**: no schemas, no empty structs.
 *
 * - `Required` fails typecheck when an SDK upgrade adds a config field: re-check `vertex-request.ts` then
 */
const PARITY_CONFIG = {
    serviceTier: 'flex' as never,
    systemInstruction: 'Be brief.',
    temperature: 0.5,
    topP: 0.9,
    topK: 3,
    candidateCount: 1,
    maxOutputTokens: 100,
    stopSequences: ['END'],
    responseLogprobs: true,
    logprobs: 2,
    presencePenalty: 0.1,
    frequencyPenalty: 0.2,
    seed: 7,
    responseMimeType: 'text/plain',
    routingConfig: { autoMode: { modelRoutingPreference: 'BALANCED' } },
    modelSelectionConfig: { featureSelectionPreference: 'BALANCED' as never },
    safetySettings: [{ category: 'HARM_CATEGORY_HARASSMENT' as never, threshold: 'BLOCK_NONE' as never }],
    tools: [
        {
            functionDeclarations: [{ name: 'now', description: 'Current time' }],
            googleSearch: { excludeDomains: ['a.example'] },
            retrieval: { vertexAiSearch: { datastore: 'd' } },
            enterpriseWebSearch: { excludeDomains: ['b.example'] },
        },
    ],
    toolConfig: { functionCallingConfig: { mode: 'AUTO' as never }, retrievalConfig: { languageCode: 'en' } },
    labels: { team: 'a' },
    cachedContent: 'c1',
    responseModalities: ['TEXT'],
    mediaResolution: 'MEDIA_RESOLUTION_LOW' as never,
    speechConfig: 'Kore',
    audioTimestamp: true,
    thinkingConfig: { thinkingBudget: 10 },
    audioTranscriptionConfig: { languageCodes: ['en'] },
    imageConfig: { aspectRatio: '1:1', outputMimeType: 'image/png', outputCompressionQuality: 80 },
    modelArmorConfig: { promptTemplateName: 't' },
    continuationToken: 'tok',
    generationOutputConfig: { outputUriPrefix: 'gs://b/o' } as never,
} satisfies Required<Omit<GenerateContentConfig, ParityExcluded>>;

const request: GenaiRequest = {
    model: 'gemini-x',
    messages: [
        { role: 'system', content: 'Classify the word.' },
        { role: 'user', content: 'blorf' },
    ],
    response_format: { type: 'json_schema', json_schema: { name: 'answer', schema } },
};

// -------------------------------------------------------------------
// --- 3. Private Functions
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

    it('resolves credentials file in CLOUDSDK_CONFIG below GOOGLE_APPLICATION_CREDENTIALS', () => {
        const adc = join('cfg', 'application_default_credentials.json');
        expect(credentialsFile({ CLOUDSDK_CONFIG: 'cfg', APPDATA: 'A', HOME: 'H' })).toBe(adc);
        expect(credentialsFile({ CLOUDSDK_CONFIG: 'cfg', GOOGLE_APPLICATION_CREDENTIALS: 'key.json' })).toBe(
            'key.json',
        );
        expect(credentialsFile({ CLOUDSDK_CONFIG: '', APPDATA: 'A', HOME: 'H' })).toMatch(
            /gcloud[\\/]application_default_credentials\.json$/,
        );
    });

    it('accepts bucket names up to 63 characters, or 222 with dots and components up to 63', () => {
        function create(bucket: string): GenaiClient {
            return createGenaiClient({ vertex: { project: 'p', bucket, authClient: fakeAuthClient() }, env: {} });
        }
        const part = 'a'.repeat(63);
        const dotted = [part, part, part, 'a'.repeat(30)].join('.');
        expect(dotted).toHaveLength(222);
        for (const bucket of ['abc', 'a'.repeat(63), dotted, 'a.b', 'my-bucket_1.example.com']) {
            expect(create(bucket).batch).toBeDefined();
        }
        const message = 'Invalid value for option "vertex.bucket": a bucket name without gs://';
        const invalid = [
            'a'.repeat(64),
            `${dotted}a`,
            `${'a'.repeat(64)}.com`,
            'ab',
            'a..b',
            '-abc',
            'abc.',
            'ABC',
            'a/b',
            'gs://b',
        ];
        for (const bucket of invalid) {
            expect(() => create(bucket)).toThrow(new TypeError(message));
        }
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

describe('listModels on Vertex AI', () => {
    it('lists publisher models with a page size Vertex accepts (at most 300)', async () => {
        const listed = await client(false).listModels();
        expect(listed).toEqual({
            ok: true,
            data: [
                { id: 'gemini-3.8-flash', actions: [] },
                { id: 'gemini-3.8-pro', actions: [] },
            ],
        });
        const sent = server.received[0];
        expect(sent?.path).toBe('/v1beta1/publishers/google/models');
        expect(Number(new URLSearchParams(sent?.search).get('pageSize'))).toBeLessThanOrEqual(300);
    });

    it('fake rejects a page size that is no integer from 0 to 300, as Vertex does', async () => {
        for (const pageSize of ['abc', '1.5', '-1', '301']) {
            const answer = await fetch(`${server.baseUrl}/v1beta1/publishers/google/models?pageSize=${pageSize}`);
            expect(answer.status).toBe(400);
        }
        const answer = await fetch(`${server.baseUrl}/v1beta1/publishers/google/models?pageSize=300`);
        expect(answer.status).toBe(200);
    });

    it('reads every page and reports only the id without the publishers/google/models/ prefix', async () => {
        server.models = Array.from({ length: 250 }, (_, index) => ({
            name: `publishers/google/models/m-${index}`,
            version: 'default',
            tunedModelInfo: {},
        }));
        const listed = await client(false).listModels();
        expect(listed.ok && listed.data.length).toBe(250);
        expect(listed.ok && listed.data.slice(0, 2)).toEqual([
            { id: 'm-0', actions: [] },
            { id: 'm-1', actions: [] },
        ]);
        expect(listed.ok && listed.data.at(-1)).toEqual({ id: 'm-249', actions: [] });
        const queries = server.received.map((entry) => new URLSearchParams(entry.search));
        expect(server.received.every((entry) => entry.path === '/v1beta1/publishers/google/models')).toBe(true);
        expect(queries.map((query) => query.get('pageSize'))).toEqual(['100', '100', '100']);
        expect(queries.map((query) => query.get('pageToken'))).toEqual([null, '100', '200']);
    });
});

describe('batch on Vertex AI', () => {
    it('submits one JSONL file with REST requests and keys, and one job reading it', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        expect(name).toMatch(/batchPredictionJobs\/1$/);
        const [inputName, text] = [...server.objects][0] ?? ['', ''];
        expect(inputName).toMatch(/^smart-genai\/p_global_[0-9a-f-]{36}\/input\.jsonl$/);
        const lines = text
            .trimEnd()
            .split('\n')
            .map((line) => JSON.parse(line));
        expect(lines.map((line) => line.key)).toEqual(['j:7', 'j:Key-8']);
        expect(lines[0].request).toEqual({
            contents: [{ role: 'user', parts: [{ text: 'blorf' }] }],
            systemInstruction: { role: 'user', parts: [{ text: 'Classify the word.' }] },
            generationConfig: {
                responseMimeType: 'application/json',
                responseSchema: { type: 'OBJECT', properties: { kind: { type: 'STRING', enum: ['noun', 'verb'] } } },
            },
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

    it('accepts an item with batch undefined as not set and rejects batch true, without defaults', async () => {
        const vertex = client();
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const submitted = await vertex.batch?.submit([{ key: '1', request: { ...hi, batch: undefined } as never }]);
        expect(submitted).toMatchObject({ ok: true });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request).toEqual({ contents: [{ role: 'user', parts: [{ text: 'hi' }] }] });
        expect(await vertex.batch?.submit([{ key: '1', request: { ...hi, batch: true } }])).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Option "batch" is not supported in batch items',
                retryable: false,
            },
        });
        expect(server.objects.size).toBe(1);
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
            systemInstruction: { role: 'user', parts: [{ text: 'a' }, { text: 'b' }] },
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
        expect(partLine.request.systemInstruction).toEqual({ role: 'user', parts: [{ text: 'part' }] });
    });

    it('places every config field as the SDK places it in a synchronous Vertex request (guards SDK upgrades)', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        await client().batch?.submit([{ key: '1', request: hi }], { config: PARITY_CONFIG });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        server.generateBody = generateBody('ok');
        expect(await client(false).generate(hi, { config: PARITY_CONFIG })).toMatchObject({ ok: true });
        const sent = server.received.find((entry) => entry.path.endsWith(':generateContent'));
        expect(line.request).toEqual(JSON.parse(sent?.body ?? '{}'));
        expect(Object.keys(line.request)).toHaveLength(12);
        expect(Object.keys(line.request.generationConfig)).toHaveLength(21);
    });

    it('renames and converts nested config fields as the SDK does on Vertex, drops automaticFunctionCalling', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const config: GenerateContentConfig = {
            modelSelectionConfig: { featureSelectionPreference: 'BALANCED' as never },
            generationOutputConfig: { outputUriPrefix: 'gs://b/o' } as never,
            continuationToken: 'tok',
            cachedContent: 'c1',
            imageConfig: { aspectRatio: '1:1', outputMimeType: 'image/png', outputCompressionQuality: 80 },
            speechConfig: 'Kore',
            responseSchema: { type: 'object', properties: { a: { type: 'string' } }, additionalProperties: false },
            automaticFunctionCalling: { disable: true },
        };
        await client().batch?.submit([{ key: '1', request: hi }], { config });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request).toEqual({
            contents: [{ role: 'user', parts: [{ text: 'hi' }] }],
            outputConfig: { outputUriPrefix: 'gs://b/o' },
            continuationToken: 'tok',
            cachedContent: 'projects/p/locations/global/cachedContents/c1',
            generationConfig: {
                modelConfig: { featureSelectionPreference: 'BALANCED' },
                imageConfig: {
                    aspectRatio: '1:1',
                    imageOutputOptions: { mimeType: 'image/png', compressionQuality: 80 },
                },
                speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                responseSchema: { type: 'OBJECT', properties: { a: { type: 'STRING' } } },
            },
        });
        const names = [
            'cachedContents/c1',
            'locations/l/cachedContents/c1',
            'projects/q/locations/l/cachedContents/c1',
        ];
        const expected = [
            'projects/p/locations/global/cachedContents/c1',
            'projects/p/locations/l/cachedContents/c1',
            'projects/q/locations/l/cachedContents/c1',
        ];
        for (const [index, cachedContent] of names.entries()) {
            server.objects.clear();
            await client().batch?.submit([{ key: '1', request: hi }], { config: { cachedContent } });
            const named = JSON.parse([...server.objects.values()][0] ?? '{}');
            expect(named.request.cachedContent).toBe(expected[index] as string);
        }
    });

    it('rejects config fields unsupported on Vertex or unknown at requestBytes and submit, naming the field', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const rejected: [GenerateContentConfig, string][] = [
            [{ enableEnhancedCivicAnswers: true }, 'config.enableEnhancedCivicAnswers'],
            [{ store: false }, 'config.store'],
            [{ tools: [{ fileSearch: { fileSearchStoreNames: ['s'] } }] }, 'config.tools[0].fileSearch'],
            [
                { toolConfig: { includeServerSideToolInvocations: true } },
                'config.toolConfig.includeServerSideToolInvocations',
            ],
            [
                { systemInstruction: { parts: [{ text: 'a', partMetadata: { a: 1 } }] } },
                'config.systemInstruction.parts[0].partMetadata',
            ],
            [{ tools: [{ mcpServers: [{ name: 'm' }] }] }, 'config.tools[0].mcpServers'],
            [
                { speechConfig: { voiceConfig: { replicatedVoiceConfig: { consentAudio: 'x' } } } as never },
                'config.speechConfig.voiceConfig.replicatedVoiceConfig.consentAudio',
            ],
        ];
        // the SDK sends every MCP server as {} on Vertex: rejected in batch, accepted by generate
        const batchOnly: [GenerateContentConfig, string][] = [
            [{ tools: [{ mcpServers: [{}] }] }, 'config.tools[0].mcpServers'],
            [{ tools: [{ mcpServers: [] }] }, 'config.tools[0].mcpServers'],
        ];
        const unknown: [GenerateContentConfig, string][] = [[{ unknownField: 1 } as never, 'config.unknownField']];
        for (const [config, path] of [...rejected, ...batchOnly, ...unknown]) {
            const message = `Field "${path}" is not supported in Vertex batch`;
            expect(client().batch?.requestBytes(hi, config)).toEqual({
                ok: false,
                error: { code: 'invalid_request', message, retryable: false },
            });
            expect(await client().batch?.submit([{ key: 'k', request: hi }], { config })).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `Field "${path}" of item "k" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
        }
        expect(server.objects.size).toBe(0);
        for (const [config] of rejected) {
            expect(await client(false).generate(hi, { config })).toMatchObject({ ok: false });
        }
        expect(server.received.some((entry) => entry.path.endsWith(':generateContent'))).toBe(false);
    });

    it('rejects config values the SDK converter cannot convert at requestBytes and submit, naming the field', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const cases: [Record<string, unknown>, string][] = [
            [{ cachedContent: 1 }, 'config.cachedContent'],
            [{ speechConfig: 1 }, 'config.speechConfig'],
            [{ systemInstruction: [] }, 'config.systemInstruction'],
            [{ systemInstruction: ['a', 1] }, 'config.systemInstruction'],
            [
                { imageConfig: { outputMimeType: 'image/png', imageOutputOptions: 'x' } },
                'config.imageConfig.imageOutputOptions',
            ],
        ];
        for (const [config, path] of cases) {
            expect(client().batch?.requestBytes(hi, config as never)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `Value of field "${path}" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
            expect(await client().batch?.submit([{ key: 'k', request: hi }], { config: config as never })).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `Value of field "${path}" of item "k" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
        }
        expect(server.objects.size).toBe(0);
    });

    it('converts multi-speaker speech config and given image output options as generate sends them', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const config = {
            speechConfig: {
                languageCode: 'en-US',
                multiSpeakerVoiceConfig: {
                    speakerVoiceConfigs: [
                        { speaker: 'A', voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                        { speaker: 'B', voiceConfig: { replicatedVoiceConfig: { mimeType: 'audio/wav' } } },
                    ],
                },
            },
            imageConfig: {
                outputMimeType: 'image/png',
                outputCompressionQuality: 80,
                imageOutputOptions: { compressionQuality: 50 },
            },
        } as GenerateContentConfig;
        await client().batch?.submit([{ key: '1', request: hi }], { config });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.generationConfig).toEqual({
            speechConfig: {
                languageCode: 'en-US',
                multiSpeakerVoiceConfig: {
                    speakerVoiceConfigs: [
                        { speaker: 'A', voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } } },
                        { speaker: 'B', voiceConfig: { replicatedVoiceConfig: { mimeType: 'audio/wav' } } },
                    ],
                },
            },
            imageConfig: { imageOutputOptions: { mimeType: 'image/png', compressionQuality: 50 } },
        });
        server.generateBody = generateBody('ok');
        expect(await client(false).generate(hi, { config })).toMatchObject({ ok: true });
        const sent = server.received.find((entry) => entry.path.endsWith(':generateContent'));
        expect(line.request).toEqual(JSON.parse(sent?.body ?? '{}'));
    });

    it('writes web_search as googleSearch with empty excludeDomains, since the import rejects empty structs', async () => {
        const vertex = client();
        const search: GenaiRequest = {
            model: 'gemini-x',
            messages: [{ role: 'user', content: 'news' }],
            web_search: true,
        };
        const submitted = await vertex.batch?.submit([{ key: '1', request: search }]);
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.tools).toEqual([{ googleSearch: { excludeDomains: [] } }]);
        const name = submitted?.ok ? submitted.data.name : '';
        expect(await vertex.batch?.check(name)).toMatchObject({ ok: true, data: { phase: 'running' } });
        server.generateBody = generateBody('ok');
        await client(false).generate(search);
        const sent = server.received.find((entry) => entry.path.endsWith(':generateContent'));
        expect(JSON.parse(sent?.body ?? '{}').tools).toEqual([{ googleSearch: {} }]);
    });

    it('replaces an empty googleSearch from config.tools on the line copy only', async () => {
        const tools = [{ googleSearch: {} }, { googleSearch: { excludeDomains: ['a.example'] } }];
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        await client().batch?.submit([{ key: '1', request: hi }], { config: { tools } });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.tools).toEqual([
            { googleSearch: { excludeDomains: [] } },
            { googleSearch: { excludeDomains: ['a.example'] } },
        ]);
        expect(tools[0]).toEqual({ googleSearch: {} });
    });

    it('omits empty functionCall args and writes an empty functionResponse as output text on the line copy only', async () => {
        const vertex = client();
        const roundTrip: GenaiRequest = {
            model: 'gemini-x',
            messages: [
                { role: 'user', content: 'time?' },
                {
                    role: 'assistant',
                    content: null,
                    tool_calls: [
                        { id: 'a', type: 'function', function: { name: 'now', arguments: '{}' } },
                        { id: 'b', type: 'function', function: { name: 'now', arguments: '' } },
                    ],
                },
                { role: 'tool', tool_call_id: 'a', content: '{}' },
                { role: 'tool', tool_call_id: 'b', content: '{"time":"noon"}' },
            ],
        };
        const submitted = await vertex.batch?.submit([{ key: '1', request: roundTrip }]);
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        const [, model, results] = line.request.contents;
        expect(model.parts.map((part: { functionCall: unknown }) => part.functionCall)).toEqual([
            { id: 'a', name: 'now' },
            { id: 'b', name: 'now' },
        ]);
        expect(results.parts).toEqual([
            { functionResponse: { id: 'a', name: 'now', response: { output: '' } } },
            { functionResponse: { id: 'b', name: 'now', response: { time: 'noon' } } },
        ]);
        const name = submitted?.ok ? submitted.data.name : '';
        expect(await vertex.batch?.check(name)).toMatchObject({ ok: true, data: { phase: 'running' } });
        server.generateBody = generateBody('ok');
        await client(false).generate(roundTrip);
        const sent = JSON.parse(server.received.find((entry) => entry.path.endsWith(':generateContent'))?.body ?? '{}');
        expect(sent.contents[1].parts[0].functionCall.args).toEqual({});
        expect(sent.contents[2].parts[0].functionResponse.response).toEqual({});
    });

    it('strips empty properties from tool parameter schemas on every level, on the line copy only', async () => {
        const vertex = client();
        const parameters = {
            type: 'object',
            properties: {
                filter: { type: 'object', properties: {} },
                properties: { type: 'array', items: { type: 'object', properties: {} } },
            },
        };
        const declared = [
            { functionDeclarations: [{ name: 'cfg', parametersJsonSchema: { type: 'object', properties: {} } }] },
        ];
        const withTools: GenaiRequest = {
            model: 'gemini-x',
            messages: [{ role: 'user', content: 'hi' }],
            tools: [
                { type: 'function', function: { name: 'none', parameters: { type: 'object', properties: {} } } },
                { type: 'function', function: { name: 'nested', parameters } },
            ],
        };
        const submitted = await vertex.batch?.submit([{ key: '1', request: withTools }], {
            config: { tools: declared },
        });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.tools).toEqual([
            { functionDeclarations: [{ name: 'cfg', parameters: { type: 'OBJECT' } }] },
            {
                functionDeclarations: [
                    { name: 'none', parameters: { type: 'OBJECT' } },
                    {
                        name: 'nested',
                        parameters: {
                            type: 'OBJECT',
                            properties: {
                                filter: { type: 'OBJECT' },
                                properties: { type: 'ARRAY', items: { type: 'OBJECT' } },
                            },
                        },
                    },
                ],
            },
        ]);
        expect(parameters.properties.filter).toEqual({ type: 'object', properties: {} });
        expect(declared[0]?.functionDeclarations[0]?.parametersJsonSchema).toEqual({ type: 'object', properties: {} });
        const name = submitted?.ok ? submitted.data.name : '';
        expect(await vertex.batch?.check(name)).toMatchObject({ ok: true, data: { phase: 'running' } });
    });

    it('removes the keyword required at every depth of the response schema on the line copy only', async () => {
        const nested = {
            type: 'object',
            properties: {
                required: { type: 'string' },
                list: {
                    type: 'array',
                    items: { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] },
                },
                choice: {
                    anyOf: [
                        { type: 'object', properties: { c: { type: 'string' } }, required: ['c'] },
                        { type: 'object', properties: { d: { type: 'string' } }, required: ['d'] },
                    ],
                },
            },
            required: ['required', 'list'],
        };
        const original = structuredClone(nested);
        const vertex = client();
        const given: GenaiRequest = {
            ...request,
            response_format: { type: 'json_schema', json_schema: { name: 'a', schema: nested } },
        };
        const submitted = await vertex.batch?.submit([{ key: '1', request: given }]);
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.generationConfig.responseSchema).toEqual({
            type: 'OBJECT',
            properties: {
                required: { type: 'STRING' },
                list: { type: 'ARRAY', items: { type: 'OBJECT', properties: { a: { type: 'STRING' } } } },
                choice: {
                    anyOf: [
                        { type: 'OBJECT', properties: { c: { type: 'STRING' } } },
                        { type: 'OBJECT', properties: { d: { type: 'STRING' } } },
                    ],
                },
            },
        });
        expect(nested).toEqual(original);
        const name = submitted?.ok ? submitted.data.name : '';
        expect([...server.objects.keys()].filter((key) => key.includes('/output/'))).toEqual([]);
        expect(await vertex.batch?.check(name)).toMatchObject({ ok: true, data: { phase: 'running' } });
        server.generateBody = generateBody('{"kind":"noun"}');
        await client(false).generate(given);
        const generated = server.received.find((entry) => entry.path.endsWith(':generateContent'));
        expect(JSON.parse(generated?.body ?? '{}').generationConfig.responseJsonSchema).toEqual(original);
    });

    it('removes required from a typed responseSchema in config, keeps it in tool parameters', async () => {
        const responseSchema = {
            type: 'OBJECT',
            properties: { n: { type: 'OBJECT', properties: { m: { type: 'INTEGER' } }, required: ['m'] } },
            required: ['n'],
        } as const;
        const parameters = { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] };
        const typed = { type: 'OBJECT', properties: { city: { type: 'STRING' } }, required: ['city'] } as const;
        const config = {
            responseMimeType: 'application/json',
            responseSchema,
            tools: [{ functionDeclarations: [{ name: 'typed', parameters: typed }] }],
        } as never;
        const withTools: GenaiRequest = {
            model: 'gemini-x',
            messages: [{ role: 'user', content: 'hi' }],
            tools: [{ type: 'function', function: { name: 'weather', parameters } }],
        };
        await client().batch?.submit([{ key: '1', request: withTools }], { config });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.generationConfig.responseSchema).toEqual({
            type: 'OBJECT',
            properties: { n: { type: 'OBJECT', properties: { m: { type: 'INTEGER' } } } },
        });
        expect(line.request.tools).toEqual([
            { functionDeclarations: [{ name: 'typed', parameters: typed }] },
            {
                functionDeclarations: [
                    {
                        name: 'weather',
                        parameters: { type: 'OBJECT', properties: { city: { type: 'STRING' } }, required: ['city'] },
                    },
                ],
            },
        ]);
        expect(responseSchema.required).toEqual(['n']);
        expect(responseSchema.properties.n.required).toEqual(['m']);
    });

    it('fails an item whose response schema holds required at any depth, with the real item errors', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        const job = server.jobs.get(name) ?? {};
        const lines = [
            { key: 't:1', request: { generationConfig: { responseJsonSchema: schema } } },
            {
                key: 't:2',
                request: {
                    generationConfig: {
                        responseSchema: { type: 'OBJECT', properties: { a: { type: 'OBJECT', required: ['b'] } } },
                    },
                },
            },
            {
                key: 't:3',
                request: {
                    generationConfig: {
                        responseSchema: { type: 'OBJECT', properties: { required: { type: 'STRING' } } },
                    },
                    tools: [{ functionDeclarations: [{ name: 'f', parameters: { ...schema, type: 'OBJECT' } }] }],
                },
            },
        ];
        server.objects.clear();
        server.objects.set(`${folder}/input.jsonl`, lines.map((line) => `${JSON.stringify(line)}\n`).join(''));
        const created = await fetch(`${server.baseUrl}/v1beta1/projects/p/locations/global/batchPredictionJobs`, {
            method: 'POST',
            body: JSON.stringify({ inputConfig: job.inputConfig, outputConfig: job.outputConfig }),
        });
        const failed = (await created.json()) as { name: string };
        server.jobs.set(failed.name, { ...server.jobs.get(failed.name), state: 'JOB_STATE_SUCCEEDED' });
        const checked = await vertex.batch?.check(failed.name);
        expect(checked?.ok && checked.data.results).toEqual([
            {
                key: '1',
                result: {
                    ok: false,
                    error: {
                        code: 'http',
                        message: 'Invalid response_json_schema: value at top-level must be a string',
                        retryable: false,
                    },
                },
            },
            {
                key: '2',
                result: {
                    ok: false,
                    error: {
                        code: 'http',
                        message: expect.stringContaining('generationConfig.responseSchema.properties.a.required[0]'),
                        retryable: false,
                    },
                },
            },
        ]);
    });

    it('rejects an item with an empty object left after the replacements, naming the item key and the path', async () => {
        const vertex = client();
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const cases: [GenaiRequest, Record<string, unknown> | undefined, string][] = [
            [hi, { tools: [{ codeExecution: {} }] }, 'request.tools[0].codeExecution'],
            [hi, { tools: [{ googleSearch: {} }, { urlContext: {} }] }, 'request.tools[1].urlContext'],
            [
                {
                    ...hi,
                    response_format: {
                        type: 'json_schema',
                        json_schema: { name: 'a', schema: { type: 'array', items: {} } },
                    },
                },
                undefined,
                'request.generationConfig.responseSchema.items',
            ],
            [
                {
                    ...hi,
                    response_format: {
                        type: 'json_schema',
                        json_schema: { name: 'a', schema: { type: 'object', properties: {} } },
                    },
                },
                undefined,
                'request.generationConfig.responseSchema.properties',
            ],
            [
                {
                    ...hi,
                    messages: [
                        ...hi.messages,
                        {
                            role: 'assistant',
                            content: null,
                            tool_calls: [{ id: 'a', type: 'function', function: { name: 'f', arguments: '{"x":{}}' } }],
                        },
                    ],
                },
                undefined,
                'request.contents[1].parts[0].functionCall.args.x',
            ],
        ];
        for (const [given, config, path] of cases) {
            const submitted = await vertex.batch?.submit(
                // job config applies to every item: the first one is named
                config === undefined
                    ? [
                          { key: 'ok', request: hi },
                          { key: 'bad', request: given },
                      ]
                    : [{ key: 'bad', request: given }],
                config === undefined ? {} : { config },
            );
            expect(submitted).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `Empty object "${path}" of item "bad" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
            expect(vertex.batch?.requestBytes(given, config)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `Empty object "${path}" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
        }
        expect(server.objects.size).toBe(0);
        expect(server.jobs.size).toBe(0);
    });

    it('converts tool parameter and response schemas to the typed form on the line copy only', async () => {
        const parameters = {
            $schema: 'https://json-schema.org/draft/2020-12/schema',
            $id: 'weather',
            $comment: 'generated',
            type: 'object',
            additionalProperties: false,
            properties: {
                const: { type: ['string', 'null'], description: 'name const', format: 'date', pattern: '^a' },
                $ref: { type: ['integer', 'number'], minimum: 0, maximum: 9, default: 1, example: 2 },
                list: { type: 'array', items: { type: 'boolean' }, minItems: 1, maxItems: 3 },
                map: { type: 'object', additionalProperties: { type: 'string' }, minProperties: 1 },
            },
            required: ['const'],
            propertyOrdering: ['const', '$ref'],
        };
        const response = {
            title: 'answer',
            type: 'object',
            properties: {
                one: { description: 'outer', anyOf: [{ type: 'null' }, { type: 'string', maxLength: 5 }] },
                many: {
                    anyOf: [
                        { type: 'string' },
                        { type: 'null' },
                        { type: 'object', properties: { x: { type: 'string' } } },
                    ],
                },
                kind: { type: 'string', enum: ['a', 'b'], nullable: true },
            },
            required: ['one'],
        };
        const typedParameters = {
            type: 'OBJECT',
            properties: {
                const: { type: 'STRING', nullable: true, description: 'name const', format: 'date', pattern: '^a' },
                $ref: {
                    anyOf: [{ type: 'INTEGER' }, { type: 'NUMBER' }],
                    minimum: 0,
                    maximum: 9,
                    default: 1,
                    example: 2,
                },
                list: { type: 'ARRAY', items: { type: 'BOOLEAN' }, minItems: 1, maxItems: 3 },
                map: { type: 'OBJECT', minProperties: 1 },
            },
            required: ['const'],
            propertyOrdering: ['const', '$ref'],
        };
        const typedResponse = {
            title: 'answer',
            type: 'OBJECT',
            properties: {
                one: { type: 'STRING', maxLength: 5, description: 'outer', nullable: true },
                many: {
                    nullable: true,
                    anyOf: [{ type: 'STRING' }, { type: 'OBJECT', properties: { x: { type: 'STRING' } } }],
                },
                kind: { type: 'STRING', enum: ['a', 'b'], nullable: true },
            },
        };
        const given: GenaiRequest = {
            model: 'gemini-x',
            messages: [{ role: 'user', content: 'hi' }],
            response_format: { type: 'json_schema', json_schema: { name: 'a', schema: response } },
            tools: [
                { type: 'function', function: { name: 'weather', parameters } },
                { type: 'function', function: { name: 'empty', parameters: {} } },
                {
                    type: 'function',
                    function: { name: 'loose', parameters: { $schema: 'x', additionalProperties: true } },
                },
            ],
        };
        const originals = structuredClone({ parameters, response });
        const vertex = client();
        const submitted = await vertex.batch?.submit([{ key: '1', request: given }]);
        expect(submitted).toMatchObject({ ok: true });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.generationConfig).toEqual({
            responseMimeType: 'application/json',
            responseSchema: typedResponse,
        });
        expect(line.request.tools).toEqual([
            {
                functionDeclarations: [
                    { name: 'weather', parameters: typedParameters },
                    { name: 'empty' },
                    { name: 'loose' },
                ],
            },
        ]);
        expect({ parameters, response }).toEqual(originals);
        const name = submitted?.ok ? submitted.data.name : '';
        expect([...server.objects.keys()].filter((key) => key.includes('/output/'))).toEqual([]);
        expect(await vertex.batch?.check(name)).toMatchObject({ ok: true, data: { phase: 'running' } });
        server.generateBody = generateBody('{"one":"a"}');
        await client(false).generate(given);
        const sent = JSON.parse(server.received.find((entry) => entry.path.endsWith(':generateContent'))?.body ?? '{}');
        expect(sent.generationConfig.responseJsonSchema).toEqual(originals.response);
        expect(sent.tools[0].functionDeclarations[0].parametersJsonSchema).toEqual(originals.parameters);
    });

    it('converts JSON schemas a caller config carries, keeps typed ones unchanged', async () => {
        const responseJsonSchema = { type: 'object', properties: { n: { type: 'integer' } }, required: ['n'] };
        const declarations = [
            {
                name: 'json',
                parametersJsonSchema: { type: 'object', properties: { a: { type: 'string' } }, required: ['a'] },
                responseJsonSchema: { type: 'object', properties: { b: { type: 'number' } } },
            },
            { name: 'typed', parameters: { type: 'OBJECT', properties: { c: { type: 'STRING' } } } },
            { name: 'none', responseJsonSchema: { $schema: 'x' } },
        ];
        const config = {
            responseMimeType: 'application/json',
            responseJsonSchema,
            tools: [{ functionDeclarations: declarations }],
        };
        const original = structuredClone(config);
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        await client().batch?.submit([{ key: '1', request: hi }], { config: config as never });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.generationConfig).toEqual({
            responseMimeType: 'application/json',
            responseSchema: { type: 'OBJECT', properties: { n: { type: 'INTEGER' } } },
        });
        expect(line.request.tools).toEqual([
            {
                functionDeclarations: [
                    {
                        name: 'json',
                        parameters: { type: 'OBJECT', properties: { a: { type: 'STRING' } }, required: ['a'] },
                        response: { type: 'OBJECT', properties: { b: { type: 'NUMBER' } } },
                    },
                    { name: 'typed', parameters: { type: 'OBJECT', properties: { c: { type: 'STRING' } } } },
                    { name: 'none' },
                ],
            },
        ]);
        expect(config).toEqual(original);
    });

    it('applies the JSON schema rules to schemas a caller config gives in typed form', async () => {
        const responseSchema = {
            $schema: 'x',
            type: 'object',
            additionalProperties: false,
            properties: { a: { description: 'outer', anyOf: [{ type: 'null' }, { type: 'string' }] } },
        };
        const declarations = [
            { name: 'none', parameters: { type: 'object', properties: {} } },
            { name: 'nested', response: { type: 'OBJECT', properties: { x: { type: 'OBJECT', properties: {} } } } },
            { name: 'empty', parameters: { additionalProperties: false }, response: { $comment: 'c' } },
        ];
        const config = {
            responseMimeType: 'application/json',
            responseSchema,
            tools: [{ functionDeclarations: declarations }],
        };
        const original = structuredClone(config);
        const vertex = client();
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const submitted = await vertex.batch?.submit([{ key: '1', request: hi }], { config: config as never });
        expect(submitted).toMatchObject({ ok: true });
        const line = JSON.parse([...server.objects.values()][0] ?? '{}');
        expect(line.request.generationConfig).toEqual({
            responseMimeType: 'application/json',
            responseSchema: {
                type: 'OBJECT',
                properties: { a: { type: 'STRING', description: 'outer', nullable: true } },
            },
        });
        expect(line.request.tools).toEqual([
            {
                functionDeclarations: [
                    { name: 'none', parameters: { type: 'OBJECT' } },
                    { name: 'nested', response: { type: 'OBJECT', properties: { x: { type: 'OBJECT' } } } },
                    { name: 'empty' },
                ],
            },
        ]);
        expect(config).toEqual(original);
        const name = submitted?.ok ? submitted.data.name : '';
        expect(await vertex.batch?.check(name)).toMatchObject({ ok: true, data: { phase: 'running' } });
    });

    it('rejects typed-form schemas of a caller config with the JSON schema errors, naming the path', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const at = 'request.generationConfig';
        const tool = 'request.tools[0].functionDeclarations[0]';
        function declared(declaration: Record<string, unknown>): Record<string, unknown> {
            return { tools: [{ functionDeclarations: [{ name: 'f', ...declaration }] }] };
        }
        const cases: [Record<string, unknown>, string][] = [
            [{ responseSchema: { $ref: '#/a' } }, `Schema keyword "$ref" at "${at}.responseSchema"`],
            [{ responseSchema: { type: 'date' } }, `Schema keyword "type" with value "date" at "${at}.responseSchema"`],
            [
                { responseSchema: { type: 'OBJECT', properties: { a: { type: 'NULL' } } } },
                `Schema keyword "type" with only "null" at "${at}.responseSchema.properties.a"`,
            ],
            [
                { responseSchema: { anyOf: [{ type: 'NULL' }] } },
                `Schema keyword "anyOf" with only "null" branches at "${at}.responseSchema"`,
            ],
            [
                { responseSchema: { type: 'STRING', enum: [1] } },
                `Schema keyword "enum" with non-string values at "${at}.responseSchema"`,
            ],
            [{ responseSchema: { type: [] } }, `Schema keyword "type" with an empty list at "${at}.responseSchema"`],
            [
                { responseSchema: { type: 'OBJECT' }, responseJsonSchema: { type: 'object' } },
                `Schema fields "responseJsonSchema" and "responseSchema" together at "${at}"`,
            ],
            [declared({ parameters: { oneOf: [] } }), `Schema keyword "oneOf" at "${tool}.parameters"`],
            [declared({ response: { const: 1 } }), `Schema keyword "const" at "${tool}.response"`],
            [
                declared({ response: { type: 'OBJECT' }, responseJsonSchema: { type: 'object' } }),
                `Schema fields "responseJsonSchema" and "response" together at "${tool}"`,
            ],
        ];
        for (const [config, prefix] of cases) {
            expect(client().batch?.requestBytes(hi, config as never)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `${prefix} is not supported in Vertex batch`,
                    retryable: false,
                },
            });
            expect(await client().batch?.submit([{ key: 'k', request: hi }], { config: config as never })).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `${prefix} of item "k" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
        }
        expect(server.objects.size).toBe(0);
    });

    it('converts the typed output of a schema again to an equal schema', async () => {
        const schema = {
            type: 'object',
            properties: {
                maybe: { type: ['string', 'null'], description: 'd', enum: ['a', 'b'] },
                number: { type: ['integer', 'number', 'null'], minimum: 0 },
                choice: {
                    description: 'outer',
                    anyOf: [{ type: 'null' }, { type: 'array', items: { type: 'boolean' } }],
                },
                many: { anyOf: [{ type: 'string' }, { type: 'object', properties: { x: { type: 'string' } } }] },
                data: { type: 'object', default: { properties: { a: 1 } }, example: { b: 2 } },
            },
            required: ['maybe'],
            propertyOrdering: ['maybe', 'number'],
        };
        const first: GenaiRequest = {
            model: 'gemini-x',
            messages: [{ role: 'user', content: 'hi' }],
            response_format: { type: 'json_schema', json_schema: { name: 'a', schema } },
            tools: [{ type: 'function', function: { name: 'f', parameters: schema } }],
        };
        await client().batch?.submit([{ key: '1', request: first }]);
        const typed = JSON.parse([...server.objects.values()][0] ?? '{}').request;
        const parameters = typed.tools[0].functionDeclarations[0].parameters;
        const { responseSchema } = typed.generationConfig;
        expect(parameters.properties.number).toEqual({
            nullable: true,
            anyOf: [{ type: 'INTEGER' }, { type: 'NUMBER' }],
            minimum: 0,
        });
        server.objects.clear();
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const config = {
            responseMimeType: 'application/json',
            responseSchema,
            tools: [{ functionDeclarations: [{ name: 'f', parameters }] }],
        };
        await client().batch?.submit([{ key: '1', request: hi }], { config: config as never });
        const again = JSON.parse([...server.objects.values()][0] ?? '{}').request;
        expect(again.generationConfig.responseSchema).toEqual(responseSchema);
        expect(again.tools[0].functionDeclarations[0].parameters).toEqual(parameters);
    });

    it('drops empty properties of declaration schemas only where a subschema stands, never in data values', async () => {
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const parameters = { type: 'object', properties: { p: { type: 'object', default: { properties: {}, x: 1 } } } };
        const config = { tools: [{ functionDeclarations: [{ name: 'f', parameters }] }] };
        const path = 'request.tools[0].functionDeclarations[0].parameters.properties.p.default.properties';
        expect(client().batch?.requestBytes(hi, config as never)).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: `Empty object "${path}" is not supported in Vertex batch`,
                retryable: false,
            },
        });
    });

    it('rejects schema parts the typed form cannot express, naming the keyword, the item key and the path', async () => {
        const vertex = client();
        const hi: GenaiRequest = { model: 'gemini-x', messages: [{ role: 'user', content: 'hi' }] };
        const at = 'request.generationConfig.responseJsonSchema';
        const tool = 'request.tools[0].functionDeclarations[0]';
        const cases: [Record<string, unknown>, Record<string, unknown> | undefined, string][] = [
            [{ $ref: '#/$defs/a' }, undefined, `Schema keyword "$ref" at "${at}"`],
            [{ $defs: { a: { type: 'string' } } }, undefined, `Schema keyword "$defs" at "${at}"`],
            [{ definitions: {} }, undefined, `Schema keyword "definitions" at "${at}"`],
            [{ properties: { a: { const: 1 } } }, undefined, `Schema keyword "const" at "${at}.properties.a"`],
            [{ oneOf: [{ type: 'string' }] }, undefined, `Schema keyword "oneOf" at "${at}"`],
            [{ items: { allOf: [{ type: 'string' }] } }, undefined, `Schema keyword "allOf" at "${at}.items"`],
            [{ anyOf: [{ not: { type: 'string' } }] }, undefined, `Schema keyword "not" at "${at}.anyOf[0]"`],
            [{ type: 'array', prefixItems: [] }, undefined, `Schema keyword "prefixItems" at "${at}"`],
            [{ enum: ['a', 1] }, undefined, `Schema keyword "enum" with non-string values at "${at}"`],
            [
                { type: 'string', anyOf: [{ type: 'string' }] },
                undefined,
                `Schema keyword "type" together with "anyOf" at "${at}"`,
            ],
            [{ type: 'null' }, undefined, `Schema keyword "type" with only "null" at "${at}"`],
            [{ type: ['null'] }, undefined, `Schema keyword "type" with only "null" at "${at}"`],
            [{ type: [] }, undefined, `Schema keyword "type" with an empty list at "${at}"`],
            [{ type: 'date' }, undefined, `Schema keyword "type" with value "date" at "${at}"`],
            [{ anyOf: [{ type: 'null' }] }, undefined, `Schema keyword "anyOf" with only "null" branches at "${at}"`],
            [{ properties: { a: true } }, undefined, `Non-object schema at "${at}.properties.a"`],
            [
                { type: 'object' },
                { responseSchema: { type: 'OBJECT' } },
                'Schema fields "responseJsonSchema" and "responseSchema" together at "request.generationConfig"',
            ],
            [
                { type: 'object' },
                {
                    tools: [
                        {
                            functionDeclarations: [
                                { name: 'f', parametersJsonSchema: { type: 'object', patternProperties: {} } },
                            ],
                        },
                    ],
                },
                `Schema keyword "patternProperties" at "${tool}.parametersJsonSchema"`,
            ],
            [
                { type: 'object' },
                {
                    tools: [
                        {
                            functionDeclarations: [
                                { name: 'f', parametersJsonSchema: { type: 'object' }, parameters: { type: 'OBJECT' } },
                            ],
                        },
                    ],
                },
                `Schema fields "parametersJsonSchema" and "parameters" together at "${tool}"`,
            ],
        ];
        for (const [schema, config, prefix] of cases) {
            const given: GenaiRequest = {
                ...hi,
                response_format: { type: 'json_schema', json_schema: { name: 'a', schema } },
            };
            const submitted = await vertex.batch?.submit(
                [
                    { key: 'ok', request: hi },
                    { key: 'bad', request: given },
                ],
                config === undefined ? {} : { config: config as never },
            );
            // tools of the job config apply to every item: the first one is named
            const key = config?.tools === undefined ? 'bad' : 'ok';
            expect(submitted).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `${prefix} of item "${key}" is not supported in Vertex batch`,
                    retryable: false,
                },
            });
            expect(vertex.batch?.requestBytes(given, config as never)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `${prefix} is not supported in Vertex batch`,
                    retryable: false,
                },
            });
        }
        expect(server.objects.size).toBe(0);
        expect(server.jobs.size).toBe(0);
    });

    it('fails an item whose line holds a JSON schema form, with the real item errors', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        const job = server.jobs.get(name) ?? {};
        const object = { type: 'object', properties: { city: { type: 'string' } } };
        const lines = [
            { key: 't:1', request: { generationConfig: { responseJsonSchema: object } } },
            {
                key: 't:2',
                request: { tools: [{ functionDeclarations: [{ name: 'f', parametersJsonSchema: object }] }] },
            },
            {
                key: 't:3',
                request: {
                    generationConfig: { responseSchema: { type: 'OBJECT' } },
                    tools: [{ functionDeclarations: [{ name: 'g', parameters: { type: 'OBJECT' } }] }],
                },
            },
            {
                key: 't:4',
                request: { tools: [{ functionDeclarations: [{ name: 'h', responseJsonSchema: object }] }] },
            },
        ];
        server.objects.clear();
        server.objects.set(`${folder}/input.jsonl`, lines.map((line) => `${JSON.stringify(line)}\n`).join(''));
        const created = await fetch(`${server.baseUrl}/v1beta1/projects/p/locations/global/batchPredictionJobs`, {
            method: 'POST',
            body: JSON.stringify({ inputConfig: job.inputConfig, outputConfig: job.outputConfig }),
        });
        const failed = (await created.json()) as { name: string };
        server.jobs.set(failed.name, { ...server.jobs.get(failed.name), state: 'JOB_STATE_SUCCEEDED' });
        const checked = await vertex.batch?.check(failed.name);
        const message = 'schema at top-level must be a boolean or an object';
        expect(checked?.ok && checked.data.results).toEqual([
            {
                key: '1',
                result: {
                    ok: false,
                    error: { code: 'http', message: `Invalid response_json_schema: ${message}`, retryable: false },
                },
            },
            {
                key: '2',
                result: {
                    ok: false,
                    error: {
                        code: 'http',
                        message: `Failed to parse parameters_json_schema for function 'f': ${message}`,
                        retryable: false,
                    },
                },
            },
            {
                key: '4',
                result: {
                    ok: false,
                    error: {
                        code: 'http',
                        message: `Failed to parse response_json_schema for function 'h': ${message}`,
                        retryable: false,
                    },
                },
            },
        ]);
    });

    it('fails a job whose input holds any empty struct, with the path without array indices as the real import does', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        const job = server.jobs.get(name) ?? {};
        const inputs: [Record<string, unknown>, string][] = [
            [{ contents: [], tools: [{ googleSearch: {} }] }, 'request.tools.googleSearch'],
            [
                { contents: [{ role: 'user', parts: [{ text: 'a' }, { functionCall: { name: 'f', args: {} } }] }] },
                'request.contents.parts.functionCall.args',
            ],
        ];
        for (const [request, field] of inputs) {
            server.objects.set(`${folder}/input.jsonl`, `${JSON.stringify({ key: 't:1', request })}\n`);
            const created = await fetch(`${server.baseUrl}/v1beta1/projects/p/locations/global/batchPredictionJobs`, {
                method: 'POST',
                body: JSON.stringify({ inputConfig: job.inputConfig }),
            });
            const failed = (await created.json()) as { name: string };
            expect(server.jobs.get(failed.name)).toMatchObject({
                state: 'JOB_STATE_FAILED',
                error: { message: expect.stringContaining(`Cannot store struct '${field}' with no fields`) },
            });
        }
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

    it('removes the folder of a job that turned final between get and cancel', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        server.stateAfterGet = 'JOB_STATE_SUCCEEDED';
        server.received.length = 0;
        expect(await vertex.batch?.cancel(name)).toEqual({ ok: true, data: undefined });
        expect(
            server.received.filter((entry) => entry.path === `/v1beta1/${name}`).map((entry) => entry.method),
        ).toEqual(['GET', 'GET']);
        expect(server.received.filter((entry) => entry.path.endsWith(':cancel'))).toHaveLength(1);
        expect(server.jobs.get(name)?.state).toBe('JOB_STATE_SUCCEEDED');
        expect(server.objects.size).toBe(0);
    });
    it('returns the cancel error and keeps the folder when the job is still running', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        server.failCancel = 'JOB_STATE_RUNNING';
        expect(await vertex.batch?.cancel(name)).toMatchObject({ ok: false, error: { code: 'http', status: 400 } });
        expect(server.objects.size).toBe(1);
    });

    it('counts a failed cancel of a cancelling job as success and removes its folder', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        server.failCancel = 'JOB_STATE_CANCELLING';
        expect(await vertex.batch?.cancel(name)).toEqual({ ok: true, data: undefined });
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

    it('reads and cancels jobs of legacy folders and of owner-prefixed folders alike', async () => {
        for (const [folder, other] of [
            [LEGACY_A, LEGACY_B],
            [`${OWN}1`, `${OWN}2`],
        ] as const) {
            server.objects.clear();
            server.jobs.clear();
            const [done, running] = [1, 2].map((id) => `projects/p/locations/global/batchPredictionJobs/${id}`);
            for (const [name, state, uri] of [
                [done, 'JOB_STATE_SUCCEEDED', folder],
                [running, 'JOB_STATE_RUNNING', other],
            ] as const) {
                server.jobs.set(name, {
                    name,
                    state,
                    inputConfig: { gcsSource: { uris: [`gs://bkt/${uri}/input.jsonl`] } },
                });
            }
            server.objects.set(`${folder}/input.jsonl`, 'x');
            server.objects.set(
                `${folder}/output/predictions.jsonl`,
                JSON.stringify({ key: 'j:7', response: generateBody('{"kind":"noun"}') }),
            );
            server.objects.set(`${other}/input.jsonl`, 'x');
            const vertex = client();
            const listed = await vertex.batch?.list();
            expect(listed?.ok && listed.data.map((job) => job.phase)).toEqual(['succeeded', 'running']);
            const checked = await vertex.batch?.check(done ?? '');
            expect(checked?.ok && checked.data.results?.map((item) => item.key)).toEqual(['7']);
            expect(server.objects.has(`${folder}/input.jsonl`)).toBe(false);
            expect(await vertex.batch?.cancel(running ?? '')).toEqual({ ok: true, data: undefined });
            expect(server.jobs.get(running ?? '')?.state).toBe('JOB_STATE_CANCELLED');
            expect(server.objects.size).toBe(0);
        }
    });

    it('writes a domain-scoped project id with ~ for : in the folder name', async () => {
        const scoped = createGenaiClient({
            vertex: {
                project: 'example.com:p',
                bucket: 'bkt',
                authClient: fakeAuthClient(),
                storageUrl: server.baseUrl,
            },
            env: {},
            baseUrl: server.baseUrl,
        });
        await scoped.batch?.submit([{ key: '1', request }]);
        expect([...server.objects.keys()][0]).toMatch(
            /^smart-genai\/example\.com~p_global_[0-9a-f-]{36}\/input\.jsonl$/,
        );
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

    it('removes property names of other items schemas from JSON answers, nested too', async () => {
        const vertex = client();
        function object(properties: Record<string, unknown>): Record<string, unknown> {
            return { type: 'object', properties };
        }
        const items = [
            { key: 's1', schema: schema },
            { key: 's2', schema: object({ capital: { type: 'string' }, population: { type: ['integer', 'null'] } }) },
            { key: 's3', schema: object({ place: object({ city: { type: 'string' } }) }) },
            { key: 's4', schema: object({ place: object({ country: { type: 'string' } }) }) },
        ].map(({ key, schema: itemSchema }) => ({
            key,
            request: {
                ...request,
                response_format: { type: 'json_schema' as const, json_schema: { name: 'a', schema: itemSchema } },
            },
        }));
        const submitted = await vertex.batch?.submit(items);
        const name = submitted?.ok ? submitted.data.name : '';
        server.succeed(name);
        const output = [...server.objects].find(([path]) => path.includes('prediction-model'))?.[1] ?? '';
        // fake mirrors live import: foreign names as null in echo, answered by the model
        const first = JSON.parse(output.split('\n')[0] ?? '{}');
        expect(first.request.generationConfig.responseSchema.properties).toMatchObject({ capital: null, place: null });
        expect(first.response.candidates[0].content.parts[0].text).toContain('"capital":"foreign"');
        const done = await vertex.batch?.check(name);
        const byKey = new Map(done?.ok ? (done.data.results ?? []).map((item) => [item.key, item.result]) : []);
        const answers = {
            s1: { kind: 'noun' },
            s2: { capital: 'x', population: 1 },
            s3: { place: { city: 'x' } },
            s4: { place: { country: 'x' } },
        };
        for (const [key, json] of Object.entries(answers)) {
            const result = byKey.get(key);
            expect(result?.ok && [result.data.json, result.data.text]).toEqual([json, JSON.stringify(json)]);
        }
    });

    it('removes parameter names of other declarations from tool call arguments', async () => {
        const vertex = client();
        function tool(name: string, parameter: string): GenaiTool {
            return {
                type: 'function',
                function: {
                    name,
                    parameters: {
                        type: 'object',
                        properties: { [parameter]: { type: 'string' } },
                        required: [parameter],
                    },
                },
            };
        }
        const submitted = await vertex.batch?.submit([
            {
                key: 't1',
                request: {
                    model: 'gemini-x',
                    messages: [{ role: 'user', content: 'Weather and time in Rome?' }],
                    tools: [tool('get_weather', 'city'), tool('get_time', 'zone')],
                },
            },
        ]);
        const name = submitted?.ok ? submitted.data.name : '';
        server.succeed(name);
        const done = await vertex.batch?.check(name);
        const result = done?.ok ? done.data.results?.[0]?.result : undefined;
        const calls = result?.ok ? (result.data.toolCalls ?? []) : [];
        expect(calls.map((call) => [call.function.name, call.function.arguments])).toEqual([
            ['get_weather', '{"city":"x"}'],
            ['get_time', '{"zone":"x"}'],
        ]);
    });

    it('removes a key of an anyOf schema only when no branch declares it', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_SUCCEEDED' });
        const responseSchema = {
            anyOf: [
                { type: 'OBJECT', properties: { a: { type: 'STRING' }, b: null, c: null } },
                { type: 'OBJECT', properties: { b: { type: 'STRING' }, c: null } },
            ],
        };
        const items = {
            type: 'ARRAY',
            items: { type: 'OBJECT', properties: { kept: { type: 'STRING' }, dropped: null } },
        };
        const lines = [
            { key: 'j:1', schema: responseSchema, answer: { a: '1', b: '2', c: '3', d: '4' } },
            { key: 'j:2', schema: items, answer: [{ kept: 'k', dropped: 'd' }, 'other'] },
            { key: 'j:3', schema: { anyOf: [null, 'bad'] }, answer: { a: '1' } },
            { key: 'j:4', schema: { type: 'ARRAY' }, answer: [{ a: '1' }] },
        ];
        server.objects.set(
            `${folder}/output/predictions.jsonl`,
            lines
                .map(({ key, schema: echoed, answer }) =>
                    JSON.stringify({
                        key,
                        request: { generationConfig: { responseSchema: echoed } },
                        response: generateBody(JSON.stringify(answer)),
                    }),
                )
                .join('\n'),
        );
        const done = await vertex.batch?.check(name);
        const byKey = new Map(done?.ok ? (done.data.results ?? []).map((item) => [item.key, item.result]) : []);
        function answer(key: string): unknown {
            const result = byKey.get(key);
            return result?.ok && [result.data.json, result.data.text];
        }
        expect(answer('1')).toEqual([{ a: '1', b: '2', d: '4' }, '{"a":"1","b":"2","d":"4"}']);
        expect(answer('2')).toEqual([[{ kept: 'k' }, 'other'], '[{"kept":"k"},"other"]']);
        expect(answer('3')).toEqual([{ a: '1' }, '{"a":"1"}']);
        expect(answer('4')).toEqual([[{ a: '1' }], '[{"a":"1"}]']);
    });

    it('keeps answers unchanged when nothing is foreign, without echo, or with unreadable arguments', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_SUCCEEDED' });
        const own = {
            generationConfig: { responseSchema: { type: 'OBJECT', properties: { kind: { type: 'STRING' } } } },
        };
        const foreign = { type: 'OBJECT', properties: { kind: { type: 'STRING' }, capital: null } };
        function call(args: unknown): Record<string, unknown> {
            const content = { role: 'model', parts: [{ functionCall: { name: 'f', args } }] };
            return { candidates: [{ content, finishReason: 'STOP' }] };
        }
        const declared = {
            tools: [{ functionDeclarations: [{ name: 'f', parameters: foreign }, { name: 'g' }, 'odd'] }, 'odd'],
        };
        const output = [
            { key: 'j:1', request: own, response: generateBody('{ "kind": "noun" }') },
            { key: 'j:2', response: generateBody('{ "kind": "noun", "capital": "Rome" }') },
            { key: 'j:3', request: 'odd', response: generateBody('{ "capital": "Rome" }') },
            { key: 't:4', request: declared, response: call({ kind: 'k' }) },
            { key: 't:5', request: { tools: 'odd' }, response: call({ capital: 'Rome' }) },
            { key: 't:6', request: declared, response: call({ kind: 'k', capital: 'Rome' }) },
        ];
        server.objects.set(`${folder}/output/predictions.jsonl`, output.map((line) => JSON.stringify(line)).join('\n'));
        const done = await vertex.batch?.check(name);
        const byKey = new Map(done?.ok ? (done.data.results ?? []).map((item) => [item.key, item.result]) : []);
        expect(byKey.get('1')).toMatchObject({ ok: true, data: { text: '{ "kind": "noun" }' } });
        expect(byKey.get('2')).toMatchObject({
            ok: true,
            data: { text: '{ "kind": "noun", "capital": "Rome" }', json: { kind: 'noun', capital: 'Rome' } },
        });
        expect(byKey.get('3')).toMatchObject({ ok: true, data: { text: '{ "capital": "Rome" }' } });
        function args(key: string): string[] | undefined {
            const result = byKey.get(key);
            return result?.ok ? result.data.toolCalls?.map((entry) => entry.function.arguments) : undefined;
        }
        expect(args('4')).toEqual(['{"kind":"k"}']);
        expect(args('5')).toEqual(['{"capital":"Rome"}']);
        expect(args('6')).toEqual(['{"kind":"k"}']);
        const unreadable = byKey.get('6');
        if (unreadable?.ok) {
            const toolCalls = (unreadable.data.toolCalls ?? []).map((entry) => ({
                ...entry,
                function: { ...entry.function, arguments: 'not json' },
            }));
            const data = { ...unreadable.data, toolCalls };
            expect(withoutForeignProperties(data, declared)).toBe(data);
        }
    });

    it('classifies item statuses by JSON RPC code, else by code name at the start of the text', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_SUCCEEDED' });
        const invalid = '{"code":3,"space":"generic","message":"Invalid response_json_schema: x"}';
        const output = [
            { key: 't:1', status: invalid },
            { key: 't:2', status: '{"code":8,"message":"Quota exceeded"}' },
            { key: 't:3', status: '{"code":14,"message":""}' },
            { key: 't:4', status: '{"code":99,"message":"INTERNAL: odd"}' },
            { key: 't:5', status: '{"message":"no code"}' },
            { key: 't:6', status: { code: 13, message: 'Internal error' } },
            { key: 't:7', status: 'Internal error while reading 429 tokens' },
            { key: 't:8', status: 'Request of 503 bytes failed: INTERNAL' },
            { key: 't:9', status: 'UNAVAILABLE: try later' },
            { key: 't:10', status: 'RESOURCE_EXHAUSTED: Quota exceeded' },
        ];
        server.objects.set(`${folder}/output/predictions.jsonl`, output.map((line) => JSON.stringify(line)).join('\n'));
        const done = await vertex.batch?.check(name);
        const byKey = new Map(done?.ok ? (done.data.results ?? []).map((item) => [item.key, item.result]) : []);
        expect(byKey.get('1')).toEqual({
            ok: false,
            error: { code: 'http', message: 'Invalid response_json_schema: x', retryable: false, status: 400 },
        });
        expect(byKey.get('2')).toEqual({
            ok: false,
            error: { code: 'rate_limited', message: 'Quota exceeded', retryable: true, status: 429 },
        });
        expect(byKey.get('3')).toEqual({
            ok: false,
            error: { code: 'http', message: '{"code":14,"message":""}', retryable: true, status: 503 },
        });
        expect(byKey.get('4')).toEqual({
            ok: false,
            error: { code: 'http', message: '{"code":99,"message":"INTERNAL: odd"}', retryable: false },
        });
        expect(byKey.get('5')).toEqual({
            ok: false,
            error: { code: 'http', message: '{"message":"no code"}', retryable: false },
        });
        expect(byKey.get('6')).toEqual({
            ok: false,
            error: { code: 'http', message: 'Internal error', retryable: true, status: 500 },
        });
        expect(byKey.get('7')).toMatchObject({ ok: false, error: { code: 'http', retryable: false } });
        expect(byKey.get('8')).toMatchObject({ ok: false, error: { code: 'http', retryable: false } });
        expect(byKey.get('9')).toEqual({
            ok: false,
            error: { code: 'http', message: 'UNAVAILABLE: try later', retryable: true },
        });
        expect(byKey.get('10')).toMatchObject({ ok: false, error: { code: 'rate_limited', status: 429 } });
    });

    it('keeps the files with keepFiles, so check can be repeated, and cancel removes them afterwards', async () => {
        const vertex = client();
        const { name, folder } = await submitTwo(vertex);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_SUCCEEDED' });
        server.objects.set(
            `${folder}/output/predictions.jsonl`,
            JSON.stringify({ key: 'j:7', response: generateBody('{"kind":"noun"}') }),
        );
        for (let round = 0; round < 2; round++) {
            const kept = await vertex.batch?.check(name, { keepFiles: true });
            expect(kept?.ok && kept.data.results?.map((item) => item.key)).toEqual(['7']);
            expect(server.objects.size).toBe(2);
        }
        const defaults = createGenaiClient({
            vertex: { project: 'p', bucket: 'bkt', authClient: fakeAuthClient(), storageUrl: server.baseUrl },
            env: {},
            baseUrl: server.baseUrl,
            defaults: { temperature: 0 },
        });
        expect((await defaults.batch?.check(name, { keepFiles: true }))?.ok).toBe(true);
        expect(server.objects.size).toBe(2);
        server.received.length = 0;
        expect(await vertex.batch?.cancel(name)).toEqual({ ok: true, data: undefined });
        expect(server.received.some((entry) => entry.path.endsWith(':cancel'))).toBe(false);
        expect(server.objects.size).toBe(0);
    });

    it('removes the files of a final job with keepFiles false, and rejects keepFiles that is no boolean', async () => {
        const vertex = client();
        const { name } = await submitTwo(vertex);
        server.jobs.set(name, { ...server.jobs.get(name), state: 'JOB_STATE_FAILED' });
        expect(await vertex.batch?.check(name, { keepFiles: 'yes' } as never)).toEqual({
            ok: false,
            error: {
                code: 'invalid_request',
                message: 'Invalid value for option "keepFiles": true or false',
                retryable: false,
            },
        });
        expect(await vertex.batch?.cancel(name, { keepFiles: true } as never)).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Option "keepFiles" is not supported' },
        });
        expect(server.objects.size).toBe(1);
        expect((await vertex.batch?.check(name, { keepFiles: false }))?.ok).toBe(true);
        expect(server.objects.size).toBe(0);
    });

    it('batch calls: keeps the files of a final job whose output cannot be read, and cancels a missing job', async () => {
        const { pollMs, msPerSecond } = batchTiming;
        Object.assign(batchTiming, { pollMs: 5, msPerSecond: 20 });
        try {
            const vertex = createGenaiClient({
                vertex: { project: 'p', bucket: 'bkt', authClient: fakeAuthClient(), storageUrl: server.baseUrl },
                env: {},
                baseUrl: server.baseUrl,
                batchWaitSec: 1,
            });
            const unread = vertex.generate({ ...request, batch: true });
            while (server.jobs.size === 0) {
                await Bun.sleep(5);
            }
            const [name, job] = [...server.jobs][0] ?? ['', {}];
            server.jobs.set(name, { ...job, state: 'JOB_STATE_SUCCEEDED' });
            expect(await unread).toMatchObject({
                ok: false,
                error: { code: 'unexpected', message: 'Batch output already read or missing' },
            });
            await Bun.sleep(30);
            expect(server.received.some((entry) => entry.path.endsWith(':cancel'))).toBe(false);
            expect(server.objects.size).toBe(1);

            server.jobs.clear();
            const missing = vertex.generate({ ...request, batch: true });
            while (server.jobs.size === 0) {
                await Bun.sleep(5);
            }
            const [gone = ''] = [...server.jobs.keys()];
            server.jobs.clear();
            expect(await missing).toMatchObject({ ok: false, error: { code: 'http', status: 404 } });
            await Bun.sleep(30);
            // one GET of the poll, one of `cancel` (which answers 404 before its cancel request)
            const reads = server.received.filter((entry) => entry.method === 'GET' && entry.path.endsWith(gone));
            expect(reads).toHaveLength(2);
        } finally {
            Object.assign(batchTiming, { pollMs, msPerSecond });
        }
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

    it('rejects a submission above maxBytes by requestBytes before the upload and accepts one at the limit', async () => {
        const items = [
            { key: 'Key-1', request },
            { key: 'Key-2', request: { ...request, messages: [{ role: 'user' as const, content: 'snarf' }] } },
        ];
        const sizes = items.map((item) => {
            const bytes = client().batch?.requestBytes(item.request);
            return bytes?.ok ? bytes.data : 0;
        });
        const limit = (sizes[0] ?? 0) + (sizes[1] ?? 0);
        const { maxBytes } = vertexBatchLimit;
        try {
            vertexBatchLimit.maxBytes = limit - 1;
            expect(await client().batch?.submit(items)).toEqual({
                ok: false,
                error: {
                    code: 'invalid_request',
                    message: `Batch of ${limit} bytes exceeds the batch limit of ${limit - 1} bytes`,
                    retryable: false,
                },
            });
            expect(server.received).toHaveLength(0);
            expect(server.objects.size).toBe(0);
            vertexBatchLimit.maxBytes = limit;
            expect(await client().batch?.submit(items)).toMatchObject({ ok: true });
            // uploaded lines carry their keys: measure is each line with empty key (format prefix only)
            const [text = ''] = [...server.objects.values()];
            expect(new TextEncoder().encode(text).length).toBeGreaterThan(limit);
            const measured = text
                .trimEnd()
                .split('\n')
                .map((raw) => {
                    const line = JSON.parse(raw);
                    const keyless = `${JSON.stringify({ ...line, key: line.key.slice(0, 2) })}\n`;
                    return new TextEncoder().encode(keyless).length;
                });
            expect(measured).toEqual(sizes);
        } finally {
            vertexBatchLimit.maxBytes = maxBytes;
        }
    });

    it('batch calls: submits calls packed exactly to maxBytes as one job', async () => {
        const second: GenaiRequest = { ...request, messages: [{ role: 'user', content: 'snarf' }] };
        const sizes = [request, second].map((entry) => {
            const bytes = client().batch?.requestBytes(entry);
            return bytes?.ok ? bytes.data : 0;
        });
        const { pollMs, msPerSecond } = batchTiming;
        const { maxBytes } = vertexBatchLimit;
        Object.assign(batchTiming, { pollMs: 5, msPerSecond: 20 });
        vertexBatchLimit.maxBytes = (sizes[0] ?? 0) + (sizes[1] ?? 0);
        try {
            const vertex = createGenaiClient({
                vertex: { project: 'p', bucket: 'bkt', authClient: fakeAuthClient(), storageUrl: server.baseUrl },
                env: {},
                baseUrl: server.baseUrl,
                batchWaitSec: 1,
            });
            const calls = Promise.all([
                vertex.generate({ ...request, batch: true }),
                vertex.generate({ ...second, batch: true }),
            ]);
            while (server.jobs.size === 0) {
                await Bun.sleep(5);
            }
            expect(server.jobs.size).toBe(1);
            const [name, job] = [...server.jobs][0] ?? ['', {}];
            const [inputName = '', text = ''] = [...server.objects][0] ?? [];
            const keys = text
                .trimEnd()
                .split('\n')
                .map((raw) => JSON.parse(raw).key);
            expect(keys).toHaveLength(2);
            const folder = inputName.slice(0, -'/input.jsonl'.length);
            server.objects.set(
                `${folder}/output/predictions.jsonl`,
                keys.map((key) => JSON.stringify({ key, response: generateBody('{"kind":"noun"}') })).join('\n'),
            );
            server.jobs.set(name, { ...job, state: 'JOB_STATE_SUCCEEDED' });
            expect(await calls).toMatchObject([{ ok: true }, { ok: true }]);
        } finally {
            Object.assign(batchTiming, { pollMs, msPerSecond });
            vertexBatchLimit.maxBytes = maxBytes;
        }
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

/** Populates fake server with batch prediction job referencing specified folder. */
function seedJob(id: number, state: string, folder: string, bucket = 'bkt', extra: Record<string, unknown> = {}): void {
    const name = `projects/p/locations/us-central1/batchPredictionJobs/${id}`;
    server.jobs.set(name, {
        name,
        state,
        inputConfig: { gcsSource: { uris: [`gs://${bucket}/${folder}/input.jsonl`] } },
        outputConfig: { gcsDestination: { outputUriPrefix: `gs://${bucket}/${folder}/output/` } },
        ...extra,
    });
}

/** Time `ms` milliseconds ago as RFC 3339. */
function ago(ms: number): string {
    return new Date(Date.now() - ms).toISOString();
}

describe('list on Vertex AI', () => {
    it('lists jobs of smart-genai folders in the bucket over all pages, and removes nothing', async () => {
        seedJob(1, 'JOB_STATE_SUCCEEDED', 'smart-genai/a', 'bkt', {
            displayName: 'run-1',
            createTime: '2026-01-01T00:00:00Z',
        });
        seedJob(2, 'JOB_STATE_RUNNING', 'smart-genai/b', 'bkt', { displayName: 'run-2' });
        seedJob(3, 'JOB_STATE_FAILED', 'smart-genai/c');
        seedJob(4, 'JOB_STATE_RUNNING', 'smart-genai/d', 'other-bucket', { displayName: 'run-1' });
        seedJob(5, 'JOB_STATE_RUNNING', 'data', 'bkt', { displayName: 'run-1' });
        server.jobs.set('null', null as never);
        server.objects.set('smart-genai/a/input.jsonl', 'x');
        const listed = await client().batch?.list();
        expect(listed).toEqual({
            ok: true,
            data: [
                {
                    name: 'projects/p/locations/us-central1/batchPredictionJobs/1',
                    displayName: 'run-1',
                    state: 'JOB_STATE_SUCCEEDED',
                    phase: 'succeeded',
                    createTime: '2026-01-01T00:00:00Z',
                },
                {
                    name: 'projects/p/locations/us-central1/batchPredictionJobs/2',
                    displayName: 'run-2',
                    state: 'JOB_STATE_RUNNING',
                    phase: 'running',
                },
                {
                    name: 'projects/p/locations/us-central1/batchPredictionJobs/3',
                    state: 'JOB_STATE_FAILED',
                    phase: 'failed',
                },
            ],
        });
        const named = await client().batch?.list({ displayName: 'run-1' });
        expect(named?.ok && named.data.map((job) => job.name)).toEqual([
            'projects/p/locations/us-central1/batchPredictionJobs/1',
        ]);
        expect(server.objects.size).toBe(1);
        expect(server.received.some((entry) => entry.method !== 'GET')).toBe(false);
    });

    it('finds a job submitted with a display name, the way to recover a lost job name', async () => {
        const vertex = client();
        const submitted = await vertex.batch?.submit([{ key: '1', request }], { displayName: 'doc-42' });
        const listed = await vertex.batch?.list({ displayName: 'doc-42' });
        expect(listed?.ok && listed.data.map((job) => job.name)).toEqual([submitted?.ok ? submitted.data.name : '']);
    });

    it('answers listing errors and invalid options', async () => {
        server.failList = 'jobs';
        try {
            expect(await client().batch?.list()).toMatchObject({ ok: false, error: { code: 'http', retryable: true } });
        } finally {
            server.failList = undefined;
        }
        expect(await client().batch?.list({ displayName: '' })).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Invalid value for option "displayName"' },
        });
        expect(await client().batch?.list({ keepFiles: true } as never)).toMatchObject({
            ok: false,
            error: { code: 'invalid_request', message: 'Option "keepFiles" is not supported' },
        });
    });
});

describe('removeFiles on Vertex AI', () => {
    /** Populates fake storage bucket with batch folders and external test objects. */
    function seedObjects(): void {
        for (const name of [
            `${OWN}a/input.jsonl`,
            `${OWN}a/output/prediction-1/predictions.jsonl`,
            `${OWN}b/input.jsonl`,
            `${OWN}c/input.jsonl`,
            `${OWN}d/output/prediction-1/predictions.jsonl`,
            `${OWN}e/input.jsonl`,
            'smart-genai/loose.txt',
            'other/x.txt',
            'smart-genaix/y.txt',
        ]) {
            server.objects.set(name, 'x');
        }
    }

    it('removes stale folders, keeps folders of jobs not final yet and leaves objects outside the prefix', async () => {
        seedObjects();
        seedJob(1, 'JOB_STATE_SUCCEEDED', `${OWN}a`, 'bkt', { endTime: ago(8 * 24 * 3_600_000) });
        seedJob(2, 'JOB_STATE_RUNNING', `${OWN}c`);
        seedJob(3, 'JOB_STATE_CANCELLING', `${OWN}d`);
        seedJob(4, 'JOB_STATE_QUEUED', `${OWN}e`);
        seedJob(5, 'JOB_STATE_PENDING', `${OWN}b`, 'other-bucket');
        server.jobs.set('null', null as never);
        expect(await client().batch?.removeFiles()).toEqual({ ok: true, data: { removed: 2, kept: 3 } });
        expect([...server.objects.keys()].sort()).toEqual([
            'other/x.txt',
            'smart-genai/loose.txt',
            `${OWN}c/input.jsonl`,
            `${OWN}d/output/prediction-1/predictions.jsonl`,
            `${OWN}e/input.jsonl`,
            'smart-genaix/y.txt',
        ]);
        const pages = server.received.filter((entry) => entry.method === 'GET' && entry.path.endsWith('Jobs'));
        expect(pages.length).toBe(3);
        const listings = server.received
            .map((entry) => new URLSearchParams(entry.search))
            .filter((query) => query.get('prefix') === 'smart-genai/');
        expect(listings.map((query) => query.get('pageToken'))).toEqual([null, '2', '4', '6']);
    });

    it('keeps a folder with an object younger than one hour or of unknown age, e.g. the input of a job being submitted', async () => {
        seedObjects();
        server.created.set(`${OWN}a/output/prediction-1/predictions.jsonl`, new Date().toISOString());
        server.created.set(`${OWN}b/input.jsonl`, new Date(Date.now() - 3_500_000).toISOString());
        server.created.set(`${OWN}c/input.jsonl`, new Date(Date.now() - 3_700_000).toISOString());
        server.created.set(`${OWN}d/output/prediction-1/predictions.jsonl`, 'not a time');
        const vertex = client();
        await submitTwo(vertex);
        expect(await vertex.batch?.removeFiles()).toEqual({ ok: true, data: { removed: 2, kept: 4 } });
        expect(server.objects.has(`${OWN}d/output/prediction-1/predictions.jsonl`)).toBe(true);
        expect(server.objects.has(`${OWN}a/input.jsonl`)).toBe(true);
        expect(server.objects.has(`${OWN}b/input.jsonl`)).toBe(true);
        expect(server.objects.has(`${OWN}c/input.jsonl`)).toBe(false);
        expect([...server.objects.keys()].filter((name) => name.endsWith('/input.jsonl'))).toHaveLength(3);
    });

    it('keeps folders of succeeded jobs within keepSucceededMs after job end, or of unknown end', async () => {
        for (const folder of ['a', 'b', 'c', 'd', 'e']) {
            server.objects.set(`${OWN}${folder}/input.jsonl`, 'x');
        }
        seedJob(1, 'JOB_STATE_SUCCEEDED', `${OWN}a`, 'bkt', { endTime: ago(6 * 24 * 3_600_000) });
        seedJob(2, 'JOB_STATE_SUCCEEDED', `${OWN}b`, 'bkt', { updateTime: ago(3_600_000) });
        seedJob(3, 'JOB_STATE_SUCCEEDED', `${OWN}c`);
        seedJob(4, 'JOB_STATE_SUCCEEDED', `${OWN}d`, 'bkt', { endTime: ago(8 * 24 * 3_600_000) });
        seedJob(5, 'JOB_STATE_FAILED', `${OWN}e`, 'bkt', { endTime: ago(3_600_000) });
        expect(await client().batch?.removeFiles()).toEqual({ ok: true, data: { removed: 2, kept: 3 } });
        expect([...server.objects.keys()].sort()).toEqual([
            `${OWN}a/input.jsonl`,
            `${OWN}b/input.jsonl`,
            `${OWN}c/input.jsonl`,
        ]);
        expect(await client().batch?.removeFiles({ keepSucceededMs: 2 * 3_600_000 })).toEqual({
            ok: true,
            data: { removed: 1, kept: 2 },
        });
        expect(server.objects.has(`${OWN}a/input.jsonl`)).toBe(false);
        expect(await client().batch?.removeFiles({ keepSucceededMs: 0 })).toEqual({
            ok: true,
            data: { removed: 1, kept: 1 },
        });
        expect([...server.objects.keys()]).toEqual([`${OWN}c/input.jsonl`]);
        for (const keepSucceededMs of [-1, Number.NaN, '1']) {
            expect(await client().batch?.removeFiles({ keepSucceededMs } as never)).toMatchObject({
                ok: false,
                error: { code: 'invalid_request', message: 'Invalid value for option "keepSucceededMs"' },
            });
        }
    });

    it('removes own and legacy orphans only, never folders of other projects or locations', async () => {
        const foreign = [
            'smart-genai/q_global_1',
            'smart-genai/p_europe-west4_1',
            'smart-genai/p_globalx_1',
            'smart-genai/example.com~p_global_1',
            'smart-genai/a',
        ];
        for (const folder of [...foreign, `${OWN}1`, LEGACY_A, LEGACY_B]) {
            server.objects.set(`${folder}/input.jsonl`, 'x');
        }
        seedJob(1, 'JOB_STATE_RUNNING', LEGACY_B);
        expect(await client().batch?.removeFiles()).toEqual({ ok: true, data: { removed: 2, kept: 1 } });
        expect([...server.objects.keys()].sort()).toEqual(
            [...foreign, LEGACY_B].map((folder) => `${folder}/input.jsonl`).sort(),
        );
    });

    it('reports an empty bucket as nothing removed', async () => {
        expect(await client().batch?.removeFiles()).toEqual({ ok: true, data: { removed: 0, kept: 0 } });
    });

    it('deletes nothing when the job or object listing fails', async () => {
        for (const failList of ['jobs', 'objects'] as const) {
            seedObjects();
            seedJob(1, 'JOB_STATE_SUCCEEDED', `${OWN}a`);
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
        server.objects.set(`${OWN}a/input.jsonl`, 'x');
        server.failDelete = 403;
        try {
            expect(await client().batch?.removeFiles()).toMatchObject({
                ok: false,
                error: { code: 'http', status: 403, retryable: false },
            });
        } finally {
            server.failDelete = undefined;
        }
        expect(server.objects.has(`${OWN}a/input.jsonl`)).toBe(true);
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
            await batch?.list(signal),
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
