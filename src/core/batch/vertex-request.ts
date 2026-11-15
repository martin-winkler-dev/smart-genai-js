// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { Content, GenerateContentConfig } from '@google/genai';
import { InvalidRequestError } from '../request.js';
import type { BatchOwner } from './vertex.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/** Batch owner and item text (` of item "<key>"`, `''` in `requestBytes`) of conversion. */
type Context = { owner: BatchOwner; item: string };

/** Converts config value at path; result `undefined` leaves field out. */
type Convert = (value: unknown, path: string, context: Context) => unknown;

/**
 * **Placement of config field in REST request**.
 *
 * - `top`: top level of request; `generation`: inside `generationConfig`
 * - `name`: REST name when SDK renames field
 * - `convert`: nested conversion; value copied unchanged without
 */
type FieldRule = { place: 'top' | 'generation'; name?: string; convert?: Convert };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/**
 * **Config fields of SDK converter `generateContentConfigToVertex`** (`@google/genai` 2.31.0).
 *
 * - mirror of SDK, which exports no converter: re-check this file on every `@google/genai` upgrade
 * - parity test in `tests/unit/core/vertex.test.ts` compares placement with synchronous `generate`
 * - nested conversions (`toolToVertex`, `toolConfigToVertex`, `speechConfigToVertex`, `imageConfigToVertex`,
 *   `tCachedContentName`, `tContent`, `partToVertex`) mirrored below; unknown nested fields dropped, as SDK does
 * - schemas (`responseSchema`, `responseJsonSchema`, declaration schemas) copied in given form, not as SDK `tSchema`:
 *   `typedSchemaForms` (`vertex-schema.ts`) converts both forms with same rules
 */
const CONFIG_FIELDS: ReadonlyMap<string, FieldRule> = new Map<string, FieldRule>([
    ['serviceTier', { place: 'top' }],
    ['systemInstruction', { place: 'top', convert: systemContent }],
    ['temperature', { place: 'generation' }],
    ['topP', { place: 'generation' }],
    ['topK', { place: 'generation' }],
    ['candidateCount', { place: 'generation' }],
    ['maxOutputTokens', { place: 'generation' }],
    ['stopSequences', { place: 'generation' }],
    ['responseLogprobs', { place: 'generation' }],
    ['logprobs', { place: 'generation' }],
    ['presencePenalty', { place: 'generation' }],
    ['frequencyPenalty', { place: 'generation' }],
    ['seed', { place: 'generation' }],
    ['responseMimeType', { place: 'generation' }],
    ['responseSchema', { place: 'generation' }],
    ['responseJsonSchema', { place: 'generation' }],
    ['routingConfig', { place: 'generation' }],
    ['modelSelectionConfig', { place: 'generation', name: 'modelConfig' }],
    ['safetySettings', { place: 'top' }],
    ['tools', { place: 'top', convert: vertexTools }],
    ['toolConfig', { place: 'top', convert: vertexToolConfig }],
    ['labels', { place: 'top' }],
    ['cachedContent', { place: 'top', convert: cachedContentName }],
    ['responseModalities', { place: 'generation' }],
    ['mediaResolution', { place: 'generation' }],
    ['speechConfig', { place: 'generation', convert: vertexSpeechConfig }],
    ['audioTimestamp', { place: 'generation' }],
    ['thinkingConfig', { place: 'generation' }],
    ['audioTranscriptionConfig', { place: 'generation' }],
    ['imageConfig', { place: 'generation', convert: vertexImageConfig }],
    ['modelArmorConfig', { place: 'top' }],
    ['continuationToken', { place: 'top' }],
    ['generationOutputConfig', { place: 'top', name: 'outputConfig' }],
]);

/** Config fields SDK rejects on Vertex. */
const REJECTED_CONFIG_FIELDS: ReadonlySet<string> = new Set(['enableEnhancedCivicAnswers', 'store']);

/** Client-side config fields SDK leaves out of REST request. */
const CLIENT_CONFIG_FIELDS: ReadonlySet<string> = new Set(['automaticFunctionCalling']);

/** Part fields copied by SDK `partToVertex`. */
const PART_FIELDS = [
    'mediaResolution',
    'audioTranscription',
    'codeExecutionResult',
    'executableCode',
    'fileData',
    'functionCall',
    'functionResponse',
    'inlineData',
    'text',
    'thought',
    'thoughtSignature',
    'videoMetadata',
    'mediaProcessing',
    'speechMetadata',
] as const;

/** Part fields SDK rejects on Vertex. */
const REJECTED_PART_FIELDS = ['toolCall', 'toolResponse', 'partMetadata'] as const;

/** Tool fields copied by SDK `toolToVertex`; `functionDeclarations` copied as given. */
const TOOL_FIELDS = [
    'retrieval',
    'googleMaps',
    'codeExecution',
    'computerUse',
    'enterpriseWebSearch',
    'exaAiSearch',
    'functionDeclarations',
    'googleSearch',
    'googleSearchRetrieval',
    'parallelAiSearch',
    'urlContext',
] as const;

/**
 * **Tool fields rejected in Vertex batch**.
 *
 * - `fileSearch`: SDK rejects on Vertex
 * - `mcpServers`: SDK `mcpServerToVertex` sends each server as empty object on Vertex
 */
const REJECTED_TOOL_FIELDS = ['fileSearch', 'mcpServers'] as const;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Builds Vertex REST request of batch line as SDK builds it for `generateContent`**: mirror of SDK converter.
 *
 * - placement and renames: see `CONFIG_FIELDS`; empty `generationConfig` omitted
 * - schemas copied in given form; typed form built later by `typedSchemaForms`
 * - `null`/`undefined` config values skipped (as SDK)
 * - client-side fields (`automaticFunctionCalling`) left out
 * - copies, caller objects never modified
 * @throws {InvalidRequestError} If a field is unknown or not supported on Vertex, or a value cannot be converted.
 */
export function vertexRequest(
    contents: readonly Content[],
    config: GenerateContentConfig,
    owner: BatchOwner,
    key: string,
): Record<string, unknown> {
    const context: Context = { owner, item: key === '' ? '' : ` of item "${key}"` };
    const request: Record<string, unknown> = {
        contents: contents.map((content, index) => vertexContent(content, `contents[${index}]`, context)),
    };
    const generationConfig: Record<string, unknown> = {};
    for (const [field, value] of Object.entries(config)) {
        const path = `config.${field}`;
        if (REJECTED_CONFIG_FIELDS.has(field)) {
            throw unsupported(path, context);
        }
        if (value === undefined || value === null || CLIENT_CONFIG_FIELDS.has(field)) {
            continue;
        }
        const rule = CONFIG_FIELDS.get(field);
        if (rule === undefined) {
            throw unsupported(path, context);
        }
        const converted = rule.convert === undefined ? value : rule.convert(value, path, context);
        if (converted !== undefined) {
            (rule.place === 'top' ? request : generationConfig)[rule.name ?? field] = converted;
        }
    }
    return Object.keys(generationConfig).length === 0 ? request : { ...request, generationConfig };
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/**
 * **Converts system instruction as SDK `tContent`**: Content kept, other forms → `role: 'user'` Content.
 *
 * - string → text part; list → parts (strings → text parts); part object → single part
 */
function systemContent(value: unknown, path: string, context: Context): unknown {
    if (isRecord(value) && Array.isArray(value.parts)) {
        return vertexContent({ parts: value.parts, role: value.role }, path, context);
    }
    const parts = Array.isArray(value) ? value : [value];
    if (parts.length === 0) {
        throw invalid(path, context);
    }
    const converted = parts.map((part) => {
        if (typeof part === 'string') {
            return { text: part };
        }
        if (typeof part !== 'object' || part === null) {
            throw invalid(path, context);
        }
        return part;
    });
    return vertexContent({ role: 'user', parts: converted }, path, context);
}

/** Copies Content as SDK `contentToVertex`: only `parts` (see `vertexPart`) and `role`; parts always a list here. */
function vertexContent(
    content: { parts?: readonly unknown[]; role?: unknown },
    path: string,
    context: Context,
): Record<string, unknown> {
    const { parts = [], role } = content;
    return {
        parts: parts.map((part, index) => vertexPart(part, `${path}.parts[${index}]`, context)),
        ...(role == null ? {} : { role }),
    };
}

/** Copies part as SDK `partToVertex`: known fields only. */
function vertexPart(part: unknown, path: string, context: Context): Record<string, unknown> {
    rejectFields(part, REJECTED_PART_FIELDS, path, context);
    return pick(part, PART_FIELDS);
}

/** Converts tool list as SDK `tTools` and `toolToVertex`. */
function vertexTools(tools: unknown, path: string, context: Context): unknown {
    if (!Array.isArray(tools)) {
        throw invalid(path, context);
    }
    return tools.map((tool: unknown, index) => vertexTool(tool, `${path}[${index}]`, context));
}

/** Copies tool as SDK `toolToVertex`: known fields only, `REJECTED_TOOL_FIELDS` rejected. */
function vertexTool(tool: unknown, path: string, context: Context): Record<string, unknown> {
    rejectFields(tool, REJECTED_TOOL_FIELDS, path, context);
    return pick(tool, TOOL_FIELDS);
}

/** Copies tool config as SDK `toolConfigToVertex`: `includeServerSideToolInvocations` rejected. */
function vertexToolConfig(toolConfig: unknown, path: string, context: Context): Record<string, unknown> {
    rejectFields(toolConfig, ['includeServerSideToolInvocations'], path, context);
    return pick(toolConfig, ['functionCallingConfig', 'retrievalConfig']);
}

/**
 * **Builds full cached content name as SDK `tCachedContentName` on Vertex**.
 *
 * - `projects/...` kept; `locations/...` → project prepended
 * - `cachedContents/<id>` or bare id → `projects/<project>/locations/<location>/cachedContents/<id>`
 * - other names kept
 */
function cachedContentName(name: unknown, path: string, context: Context): string {
    if (typeof name !== 'string') {
        throw invalid(path, context);
    }
    const { project, location } = context.owner;
    if (name.startsWith('projects/')) {
        return name;
    }
    if (name.startsWith('locations/')) {
        return `projects/${project}/${name}`;
    }
    if (name.startsWith('cachedContents/')) {
        return `projects/${project}/locations/${location}/${name}`;
    }
    return name.split('/').length === 1 ? `projects/${project}/locations/${location}/cachedContents/${name}` : name;
}

/**
 * **Converts speech config as SDK `tSpeechConfig` and `speechConfigToVertex`**: known fields only.
 *
 * - string → `voiceConfig.prebuiltVoiceConfig.voiceName`
 * - `voiceConfig`, `multiSpeakerVoiceConfig.speakerVoiceConfigs[].voiceConfig`: see `vertexVoiceConfig`
 */
function vertexSpeechConfig(value: unknown, path: string, context: Context): Record<string, unknown> {
    if (typeof value !== 'string' && typeof value !== 'object') {
        throw invalid(path, context);
    }
    const config = typeof value === 'string' ? { voiceConfig: { prebuiltVoiceConfig: { voiceName: value } } } : value;
    const copy = pick(config, ['languageCode']);
    const { voiceConfig, multiSpeakerVoiceConfig } = isRecord(config) ? config : {};
    if (voiceConfig != null) {
        copy.voiceConfig = vertexVoiceConfig(voiceConfig, `${path}.voiceConfig`, context);
    }
    if (multiSpeakerVoiceConfig != null) {
        const at = `${path}.multiSpeakerVoiceConfig`;
        const speakers = isRecord(multiSpeakerVoiceConfig) ? multiSpeakerVoiceConfig.speakerVoiceConfigs : undefined;
        copy.multiSpeakerVoiceConfig =
            speakers == null
                ? {}
                : {
                      speakerVoiceConfigs: Array.isArray(speakers)
                          ? speakers.map((speaker: unknown, index) =>
                                vertexSpeaker(speaker, `${at}.speakerVoiceConfigs[${index}]`, context),
                            )
                          : speakers,
                  };
    }
    return copy;
}

/** Copies speaker voice config as SDK `speakerVoiceConfigToVertex`. */
function vertexSpeaker(speaker: unknown, path: string, context: Context): Record<string, unknown> {
    const copy = pick(speaker, ['speaker']);
    const voiceConfig = isRecord(speaker) ? speaker.voiceConfig : undefined;
    if (voiceConfig != null) {
        copy.voiceConfig = vertexVoiceConfig(voiceConfig, `${path}.voiceConfig`, context);
    }
    return copy;
}

/**
 * **Copies voice config as SDK `voiceConfigToVertex`**: known fields only.
 *
 * - `replicatedVoiceConfig`: `mimeType`, `voiceSampleAudio`; `consentAudio`, `voiceConsentSignature` rejected
 */
function vertexVoiceConfig(voiceConfig: unknown, path: string, context: Context): Record<string, unknown> {
    const copy = pick(voiceConfig, ['prebuiltVoiceConfig', 'voice']);
    const replicated = isRecord(voiceConfig) ? voiceConfig.replicatedVoiceConfig : undefined;
    if (replicated != null) {
        const at = `${path}.replicatedVoiceConfig`;
        rejectFields(replicated, ['consentAudio', 'voiceConsentSignature'], at, context);
        copy.replicatedVoiceConfig = pick(replicated, ['mimeType', 'voiceSampleAudio']);
    }
    return copy;
}

/**
 * **Converts image config as SDK `imageConfigToVertex`**: known fields only.
 *
 * - `outputMimeType` → `imageOutputOptions.mimeType`
 * - `outputCompressionQuality` → `imageOutputOptions.compressionQuality`
 * - given `imageOutputOptions` merged over these, own fields win
 */
function vertexImageConfig(imageConfig: unknown, path: string, context: Context): Record<string, unknown> {
    const copy = pick(imageConfig, ['aspectRatio', 'imageSize', 'personGeneration', 'prominentPeople']);
    const { outputMimeType, outputCompressionQuality, imageOutputOptions } = isRecord(imageConfig) ? imageConfig : {};
    const derived = {
        ...(outputMimeType == null ? {} : { mimeType: outputMimeType }),
        ...(outputCompressionQuality == null ? {} : { compressionQuality: outputCompressionQuality }),
    };
    if (imageOutputOptions != null) {
        if (Object.keys(derived).length > 0 && !isRecord(imageOutputOptions)) {
            throw invalid(`${path}.imageOutputOptions`, context);
        }
        copy.imageOutputOptions = isRecord(imageOutputOptions)
            ? { ...derived, ...imageOutputOptions }
            : imageOutputOptions;
    } else if (Object.keys(derived).length > 0) {
        copy.imageOutputOptions = derived;
    }
    return copy;
}

/** Copies listed non-nullish fields of object; non-objects return empty object (as SDK). */
function pick(value: unknown, fields: readonly string[]): Record<string, unknown> {
    const copy: Record<string, unknown> = {};
    if (!isRecord(value)) {
        return copy;
    }
    for (const field of fields) {
        if (value[field] != null) {
            copy[field] = value[field];
        }
    }
    return copy;
}

/**
 * Rejects fields SDK rejects on Vertex; set as any value but `undefined` (as SDK).
 * @throws {InvalidRequestError} If one of the fields is set.
 */
function rejectFields(value: unknown, fields: readonly string[], path: string, context: Context): void {
    const found = isRecord(value) ? fields.find((field) => value[field] !== undefined) : undefined;
    if (found !== undefined) {
        throw unsupported(`${path}.${found}`, context);
    }
}

/** Builds error for field unknown to or rejected by SDK on Vertex. */
function unsupported(path: string, context: Context): InvalidRequestError {
    return new InvalidRequestError(`Field "${path}"${context.item} is not supported in Vertex batch`);
}

/** Builds error for value SDK converter cannot convert. */
function invalid(path: string, context: Context): InvalidRequestError {
    return new InvalidRequestError(`Value of field "${path}"${context.item} is not supported in Vertex batch`);
}

/** Checks for plain non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
