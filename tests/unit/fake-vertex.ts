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
    /** Job state set on cancel, which then answers HTTP 400 `FAILED_PRECONDITION`. */
    failCancel: string | undefined;
    /** Job state set after next job read, once: job turns final between `get` and `cancel`. */
    stateAfterGet: string | undefined;
    /** Publisher models listed by models route, in real Vertex shape. */
    models: Record<string, unknown>[];
    /** Answers every line of job in live batch form (see `writeAnswers`) and sets job succeeded. */
    succeed(name: string): void;
    stop(): void;
};

/** Property names found in schemas of one job: name → names of nested properties. */
type NameTree = Map<string, NameTree>;

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

/** Matches Vertex publisher models list endpoint (no project or location prefix). */
const MATCH_MODELS_PATH = /^\/v1beta1\/publishers\/google\/models$/;

/** Matches Vertex embedding endpoints of Google publisher models: model id and method. */
const MATCH_EMBED_PATH =
    /^\/v1beta1\/projects\/[^/]+\/locations\/[^/]+\/publishers\/google\/models\/([^/:]+):(predict|embedContent)$/;

/** Matches `gs://<bucket>/` prefix of Cloud Storage URI. */
const MATCH_GCS_BUCKET_PREFIX = /^gs:\/\/[^/]+\//;

/** Maximum number of jobs per page in fake listing. */
const JOBS_PAGE_SIZE = 2;

/** Maximum number of objects per page in fake Cloud Storage listing. */
const OBJECTS_PAGE_SIZE = 2;

/** Maximum `pageSize` accepted by Vertex publisher models list. */
const MODELS_MAX_PAGE_SIZE = 300;

/** Models per page when request sets no `pageSize` (fake choice). */
const MODELS_DEFAULT_PAGE_SIZE = 50;

/** Vector length when embed request sets no `outputDimensionality` (fake choice). */
const EMBED_DEFAULT_DIMENSIONS = 2;

/** Final Vertex job states: cancel answers HTTP 400 `FAILED_PRECONDITION`. */
const FINAL_JOB_STATES: ReadonlySet<string> = new Set([
    'JOB_STATE_SUCCEEDED',
    'JOB_STATE_FAILED',
    'JOB_STATE_CANCELLED',
    'JOB_STATE_EXPIRED',
    'JOB_STATE_PARTIALLY_SUCCEEDED',
]);

/** Publisher models in real Vertex shape: only `name`, `version`, `tunedModelInfo`. */
export const PUBLISHER_MODELS: readonly Record<string, unknown>[] = [
    { name: 'publishers/google/models/gemini-3.8-flash', version: 'default', tunedModelInfo: {} },
    { name: 'publishers/google/models/gemini-3.8-pro', version: 'default', tunedModelInfo: {} },
];

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Starts fake Vertex AI and Cloud Storage server**:
 *
 * - storage: handles upload, listing with timestamps (paged), download, and delete
 * - Vertex: handles generateContent, embeddings, publisher models list (paged), and batch jobs lifecycle
 * - cancel of a final job: HTTP 400 `FAILED_PRECONDITION`, as real API does
 * - batch input: fails job on any empty struct in a line's request, as real import does
 * - batch input: fails item whose response schema contains `required`, as real import does
 * - batch input: fails item with JSON schema forms (`responseJsonSchema`, `parametersJsonSchema`), see `writeItemErrors`
 * - batch output (`succeed`): property names of all schemas in job mixed into each line, see `writeAnswers`
 */
export function startFakeVertex(): FakeVertex {
    const received: VertexRequest[] = [];
    const objects = new Map<string, string>();
    const created = new Map<string, string>();
    const jobs = new Map<string, Record<string, unknown>>();
    const fake: Omit<FakeVertex, 'baseUrl' | 'succeed' | 'stop'> = {
        received,
        objects,
        created,
        jobs,
        generateBody: {},
        failList: undefined,
        failDelete: undefined,
        failCancel: undefined,
        stateAfterGet: undefined,
        models: [...PUBLISHER_MODELS],
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
                if (fake.failCancel !== undefined || FINAL_JOB_STATES.has(String(stored.state))) {
                    stored.state = fake.failCancel ?? stored.state;
                    const error = { code: 400, message: 'fake 400', status: 'FAILED_PRECONDITION' };
                    return Response.json({ error }, { status: 400 });
                }
                stored.state = 'JOB_STATE_CANCELLED';
                return Response.json({});
            }
            if (job && request.method === 'POST') {
                const name = `${job[1]}/${nextJob++}`;
                const created = { ...(JSON.parse(body) as Record<string, unknown>), name, state: 'JOB_STATE_PENDING' };
                const importError = inputError(objects, created);
                jobs.set(name, importError ? { ...created, state: 'JOB_STATE_FAILED', error: importError } : created);
                writeItemErrors(objects, created);
                return Response.json(created);
            }
            if (job && job[2] === undefined && request.method === 'GET') {
                return fake.failList === 'jobs' ? apiError(500) : jobsPage([...jobs.values()], url);
            }
            if (job?.[2] !== undefined) {
                const stored = jobs.get(`${job[1]}/${job[2]}`);
                if (stored === undefined) {
                    return apiError(404);
                }
                const answer = Response.json(stored);
                if (fake.stateAfterGet !== undefined) {
                    stored.state = fake.stateAfterGet;
                    fake.stateAfterGet = undefined;
                }
                return answer;
            }
            if (MATCH_MODELS_PATH.test(url.pathname) && request.method === 'GET') {
                return modelsPage(fake.models, url);
            }
            const embed = MATCH_EMBED_PATH.exec(url.pathname);
            if (embed && request.method === 'POST') {
                return embedReply(embed[2] === 'predict', body);
            }
            if (url.pathname.endsWith(':generateContent')) {
                return Response.json(fake.generateBody);
            }
            return apiError(404);
        },
    });
    return Object.assign(fake, {
        baseUrl: `http://127.0.0.1:${server.port}`,
        succeed(name: string) {
            const stored = jobs.get(name) ?? {};
            writeAnswers(objects, stored);
            jobs.set(name, { ...stored, state: 'JOB_STATE_SUCCEEDED' });
        },
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

/**
 * **Answers publisher models list**: pages of `pageSize` models, `nextPageToken` while more follow.
 *
 * - `pageSize` above 300, negative or not an integer: HTTP 400 with real Vertex message
 * - `pageToken`: offset written by previous page
 */
function modelsPage(models: Record<string, unknown>[], url: URL): Response {
    const pageSize = Number(url.searchParams.get('pageSize') ?? '0');
    if (!Number.isInteger(pageSize) || pageSize < 0 || pageSize > MODELS_MAX_PAGE_SIZE) {
        return invalidArgument(
            'Field: page_size; Message: Page size should be non-negative and the maximum size is 300.',
        );
    }
    const size = pageSize === 0 ? MODELS_DEFAULT_PAGE_SIZE : pageSize;
    const offset = Number(url.searchParams.get('pageToken') ?? '0');
    const next = offset + size < models.length ? { nextPageToken: String(offset + size) } : {};
    return Response.json({ publisherModels: models.slice(offset, offset + size), ...next });
}

/**
 * **Answers Vertex embed call in real wire shape**: one-hot vectors, `1` at input index modulo length.
 *
 * - `:predict`: needs non-empty `instances` with text `content`; answers `predictions[].embeddings.values`
 * - `:embedContent`: needs one `content` with text parts; answers `embedding.values`
 * - length: `outputDimensionality` of `parameters` or `embedContentConfig`, else 2
 * - malformed body: HTTP 400 `INVALID_ARGUMENT`
 */
function embedReply(predict: boolean, body: string): Response {
    const sent = JSON.parse(body) as {
        instances?: { content?: unknown }[];
        parameters?: { outputDimensionality?: number };
        content?: { parts?: { text?: unknown }[] };
        embedContentConfig?: { outputDimensionality?: number };
    };
    const length =
        (predict ? sent.parameters : sent.embedContentConfig)?.outputDimensionality ?? EMBED_DEFAULT_DIMENSIONS;
    function vector(index: number): number[] {
        return Array.from({ length }, (_, at) => (at === index % length ? 1 : 0));
    }
    if (predict) {
        const instances = Array.isArray(sent.instances) ? sent.instances : [];
        if (instances.length === 0 || instances.some((instance) => typeof instance?.content !== 'string')) {
            return invalidArgument('instances: non-empty list of text contents expected');
        }
        return Response.json({ predictions: instances.map((_, index) => ({ embeddings: { values: vector(index) } })) });
    }
    const parts = sent.content?.parts;
    if (!Array.isArray(parts) || parts.length === 0 || parts.some((part) => typeof part?.text !== 'string')) {
        return invalidArgument('content: one content with text parts expected');
    }
    return Response.json({ embedding: { values: vector(0) }, usageMetadata: { promptTokenCount: 1 } });
}

/**
 * **Imports job input as real Vertex batch does**: job fails later, creation still answers pending.
 *
 * - rejects any empty struct in a line's `request` with real import message
 * - struct path as in real message: no array indices (`request.tools.googleSearch`)
 * - input outside fake bucket or unparsable: no error
 */
function inputError(objects: Map<string, string>, job: Record<string, unknown>): { message: string } | undefined {
    for (const line of inputLines(objects, job)) {
        const field = emptyStruct(line.request, 'request');
        if (field !== undefined) {
            return { message: `Failed to import data. Query error: Cannot store struct '${field}' with no fields` };
        }
    }
    return undefined;
}

/**
 * **Writes item errors of job input as real Vertex batch does**: to `<output>/prediction-fake/predictions.jsonl`.
 *
 * - item whose `generationConfig.responseJsonSchema` or `responseSchema` holds `required` at any depth: `status` line
 * - item with `generationConfig.responseJsonSchema` or a declaration with `parametersJsonSchema` or
 *   `responseJsonSchema`: `status` line;
 *   real import fails these when schemas of a job differ in shape, fake fails every one
 * - written at job creation, before any output a test stores; first line per key wins on read
 * - no failing item: no file
 */
function writeItemErrors(objects: Map<string, string>, job: Record<string, unknown>): void {
    const errors: string[] = [];
    for (const line of inputLines(objects, job)) {
        const config = (line.request as { generationConfig?: Record<string, unknown> } | undefined)?.generationConfig;
        const path = requiredPath(config?.responseSchema, 'generationConfig.responseSchema');
        const declaration = jsonSchemaDeclaration(line.request);
        const shape = 'schema at top-level must be a boolean or an object';
        let status: string | undefined;
        if (requiredPath(config?.responseJsonSchema, '') !== undefined) {
            status = 'Invalid response_json_schema: value at top-level must be a string';
        } else if (path !== undefined) {
            status = `Failed to parse JSON into proto: invalid value at '${path}[0]'`;
        } else if (config?.responseJsonSchema !== undefined) {
            status = `Invalid response_json_schema: ${shape}`;
        } else if (declaration !== undefined) {
            status = `Failed to parse ${declaration.field} for function '${declaration.name}': ${shape}`;
        }
        if (status !== undefined) {
            errors.push(`${JSON.stringify({ key: line.key, request: line.request, status })}\n`);
        }
    }
    const prefix = (job.outputConfig as { gcsDestination?: { outputUriPrefix?: string } } | undefined)?.gcsDestination
        ?.outputUriPrefix;
    if (errors.length > 0 && prefix !== undefined) {
        objects.set(`${prefix.replace(MATCH_GCS_BUCKET_PREFIX, '')}prediction-fake/predictions.jsonl`, errors.join(''));
    }
}

/**
 * **Writes answers of job input as live Vertex batch does**: to `<output>/prediction-model/predictions.jsonl`.
 *
 * - mirrors live behavior: import gives all lines of job one structure
 * - names of `generationConfig.responseSchema.properties` (nested too) of all lines: missing ones added as `null`
 * - names of declaration `parameters.properties` of all declarations and lines: missing ones added as `null`
 * - names sorted, as live echo shows them; each line echoes its request in this form
 * - `j:` line with response schema: JSON answer with every listed name, foreign ones as `"foreign"`
 * - other line: one function call per declaration with every listed parameter name, else text `fake`
 */
function writeAnswers(objects: Map<string, string>, job: Record<string, unknown>): void {
    const lines = inputLines(objects, job).map((line) => (isRecord(line.request) ? line : { ...line, request: {} }));
    const schemas: NameTree = new Map();
    const parameters: NameTree = new Map();
    for (const line of lines) {
        collectNames(responseSchema(line.request), schemas);
        for (const declaration of declarations(line.request)) {
            collectNames(declaration.parameters, parameters);
        }
    }
    const output = lines.map((line) => {
        const request = line.request as Record<string, unknown>;
        const config = request.generationConfig;
        const schema = filled(responseSchema(request), schemas);
        const tools = Array.isArray(request.tools)
            ? request.tools.map((tool: unknown) =>
                  isRecord(tool) && Array.isArray(tool.functionDeclarations)
                      ? {
                            ...tool,
                            functionDeclarations: tool.functionDeclarations.map((declaration: unknown) =>
                                isRecord(declaration) && declaration.parameters !== undefined
                                    ? { ...declaration, parameters: filled(declaration.parameters, parameters) }
                                    : declaration,
                            ),
                        }
                      : tool,
              )
            : undefined;
        const echoed = {
            ...request,
            ...(isRecord(config) && schema !== undefined
                ? { generationConfig: { ...config, responseSchema: schema } }
                : {}),
            ...(tools === undefined ? {} : { tools }),
        };
        const calls = declarations(echoed).map((declaration) => ({
            functionCall: { name: declaration.name, args: sample(declaration.parameters) ?? {} },
        }));
        const json = String(line.key).startsWith('j:') && schema !== undefined;
        const parts = json ? [{ text: JSON.stringify(sample(schema)) }] : calls.length > 0 ? calls : [{ text: 'fake' }];
        const response = {
            candidates: [{ content: { role: 'model', parts }, finishReason: 'STOP', index: 0 }],
            modelVersion: 'gemini-fake-001',
        };
        return `${JSON.stringify({ key: line.key, request: echoed, response })}\n`;
    });
    const prefix = (job.outputConfig as { gcsDestination?: { outputUriPrefix?: string } } | undefined)?.gcsDestination
        ?.outputUriPrefix;
    if (prefix !== undefined) {
        objects.set(
            `${prefix.replace(MATCH_GCS_BUCKET_PREFIX, '')}prediction-model/predictions.jsonl`,
            output.join(''),
        );
    }
}

/** Reads `generationConfig.responseSchema` of line request. */
function responseSchema(request: unknown): unknown {
    const config = isRecord(request) ? request.generationConfig : undefined;
    return isRecord(config) ? config.responseSchema : undefined;
}

/** Lists function declarations of all tools of line request. */
function declarations(request: unknown): Record<string, unknown>[] {
    const tools = isRecord(request) && Array.isArray(request.tools) ? request.tools : [];
    return tools.flatMap((tool: unknown) =>
        isRecord(tool) && Array.isArray(tool.functionDeclarations) ? tool.functionDeclarations.filter(isRecord) : [],
    );
}

/** Adds property names of schema (nested `properties`) to tree. */
function collectNames(schema: unknown, tree: NameTree): void {
    if (!isRecord(schema) || !isRecord(schema.properties)) {
        return;
    }
    for (const [name, entry] of Object.entries(schema.properties)) {
        const child = tree.get(name) ?? new Map();
        tree.set(name, child);
        collectNames(entry, child);
    }
}

/** Copies schema with every name of tree in its `properties`: missing names as `null`, sorted. */
function filled(schema: unknown, tree: NameTree): unknown {
    if (!isRecord(schema) || !isRecord(schema.properties)) {
        return schema;
    }
    const own = schema.properties;
    const names = [...new Set([...Object.keys(own), ...tree.keys()])].sort();
    const properties = Object.fromEntries(
        names.map((name) => [name, own[name] === undefined ? null : filled(own[name], tree.get(name) ?? new Map())]),
    );
    return { ...schema, properties };
}

/** Builds value of typed schema: first enum value or `x`, `1`, `true`, lists of one; `null` entry → `"foreign"`. */
function sample(schema: unknown): unknown {
    if (!isRecord(schema)) {
        return undefined;
    }
    if (isRecord(schema.properties)) {
        return Object.fromEntries(
            Object.entries(schema.properties).map(([name, entry]) => [
                name,
                entry === null ? 'foreign' : sample(entry),
            ]),
        );
    }
    switch (schema.type) {
        case 'STRING':
            return Array.isArray(schema.enum) ? schema.enum[0] : 'x';
        case 'INTEGER':
        case 'NUMBER':
            return 1;
        case 'BOOLEAN':
            return true;
        case 'ARRAY':
            return [sample(schema.items)];
        default:
            return null;
    }
}

/** Checks for plain non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/** Parses lines of job input file in fake bucket; input outside bucket or unparsable lines skipped. */
function inputLines(
    objects: Map<string, string>,
    job: Record<string, unknown>,
): { key?: unknown; request?: unknown }[] {
    const uri = (job.inputConfig as { gcsSource?: { uris?: string[] } } | undefined)?.gcsSource?.uris?.[0] ?? '';
    const text = objects.get(uri.replace(MATCH_GCS_BUCKET_PREFIX, '')) ?? '';
    return text
        .split('\n')
        .filter((raw) => raw !== '')
        .flatMap((raw) => {
            try {
                return [JSON.parse(raw) as { key?: unknown; request?: unknown }];
            } catch {
                return [];
            }
        });
}

/**
 * **Finds first function declaration with JSON schema form in line request**: `undefined` if none.
 *
 * - `field`: `parameters_json_schema` or `response_json_schema`, as real message names it
 */
function jsonSchemaDeclaration(request: unknown): { name: string; field: string } | undefined {
    const tools = (request as { tools?: { functionDeclarations?: Record<string, unknown>[] }[] } | undefined)?.tools;
    const declaration = (Array.isArray(tools) ? tools : [])
        .flatMap((tool) => (Array.isArray(tool?.functionDeclarations) ? tool.functionDeclarations : []))
        .find((entry) => entry?.parametersJsonSchema !== undefined || entry?.responseJsonSchema !== undefined);
    if (declaration === undefined) {
        return undefined;
    }
    const field = declaration.parametersJsonSchema === undefined ? 'response_json_schema' : 'parameters_json_schema';
    return { name: String(declaration.name), field };
}

/**
 * **Finds path of first schema keyword `required`**: walked structurally, as real import reads schemas.
 *
 * - subschemas: `properties`, `$defs`, `definitions` values, `items`, `prefixItems`, `anyOf`/`oneOf`/`allOf`, `not`,
 *   object `additionalProperties`
 * - property named `required` is a name, not the keyword
 */
function requiredPath(schema: unknown, path: string): string | undefined {
    if (typeof schema !== 'object' || schema === null || Array.isArray(schema)) {
        return undefined;
    }
    const record = schema as Record<string, unknown>;
    if (record.required !== undefined) {
        return `${path}.required`;
    }
    const children: [unknown, string][] = [];
    for (const map of ['properties', '$defs', 'definitions']) {
        const value = record[map];
        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            children.push(
                ...Object.entries(value).map(([name, entry]): [unknown, string] => [entry, `${path}.${map}.${name}`]),
            );
        }
    }
    for (const list of ['items', 'prefixItems', 'anyOf', 'oneOf', 'allOf']) {
        const value = record[list];
        if (Array.isArray(value)) {
            children.push(...value.map((entry, index): [unknown, string] => [entry, `${path}.${list}[${index}]`]));
        }
    }
    for (const single of ['items', 'additionalProperties', 'not']) {
        children.push([record[single], `${path}.${single}`]);
    }
    return children.map(([entry, at]) => requiredPath(entry, at)).find((found) => found !== undefined);
}

/** Finds path of first empty struct in parsed JSON value; array entries keep parent path. */
function emptyStruct(value: unknown, path: string): string | undefined {
    if (Array.isArray(value)) {
        return value.map((entry) => emptyStruct(entry, path)).find((found) => found !== undefined);
    }
    if (typeof value !== 'object' || value === null) {
        return undefined;
    }
    const entries = Object.entries(value);
    if (entries.length === 0) {
        return path;
    }
    return entries.map(([key, entry]) => emptyStruct(entry, `${path}.${key}`)).find((found) => found !== undefined);
}

/** Handles storage operations (list paged by `OBJECTS_PAGE_SIZE`, download, delete) for fake bucket. */
function storageReply(
    { objects, created }: Pick<FakeVertex, 'objects' | 'created'>,
    method: string,
    encoded: string | undefined,
    url: URL,
): Response {
    if (encoded === undefined) {
        const prefix = url.searchParams.get('prefix') ?? '';
        const offset = Number(url.searchParams.get('pageToken') ?? '0');
        const all = [...objects.keys()].filter((name) => name.startsWith(prefix));
        const items = all
            .slice(offset, offset + OBJECTS_PAGE_SIZE)
            .map((name) => ({ name, timeCreated: created.get(name) ?? '2000-01-01T00:00:00.000Z' }));
        const end = offset + OBJECTS_PAGE_SIZE;
        const next = end < all.length ? { nextPageToken: String(end) } : {};
        return Response.json(items.length === 0 ? next : { items, ...next });
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

/** Creates HTTP 400 `INVALID_ARGUMENT` response with message. */
function invalidArgument(message: string): Response {
    return Response.json({ error: { code: 400, message, status: 'INVALID_ARGUMENT' } }, { status: 400 });
}

/** Creates JSON error response with specified HTTP status. */
function apiError(status: number): Response {
    return Response.json({ error: { code: status, message: `fake ${status}`, status: 'FAKE' } }, { status });
}
