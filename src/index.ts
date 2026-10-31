// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

export type { AuthClient } from 'google-auth-library';
export { createGenaiClient, credentialsFile, GEMINI_API_KEY_ENV, VERTEX_ENV } from './core/client.js';
export { MAX_EMBED_INPUTS } from './core/embed.js';
export { assistantMessage, toolMessage } from './core/messages.js';
export { mergeGenaiConfig, validateGenaiConfig } from './core/request.js';
export type {
    GenaiBatch,
    GenaiBatchItem,
    GenaiBatchOptions,
    GenaiBatchResult,
    GenaiBatchStatus,
    GenaiCallOptions,
    GenaiClient,
    GenaiClientOptions,
    GenaiContentPart,
    GenaiDefaults,
    GenaiEmbedInput,
    GenaiEmbedOutput,
    GenaiEmbedRequest,
    GenaiEmbedTask,
    GenaiError,
    GenaiErrorCode,
    GenaiImagePart,
    GenaiLimiterOptions,
    GenaiLimits,
    GenaiMessage,
    GenaiModel,
    GenaiModelBatch,
    GenaiModelInfo,
    GenaiModelInput,
    GenaiModelOptions,
    GenaiModelSpec,
    GenaiModels,
    GenaiModelsOptions,
    GenaiOutput,
    GenaiReasoningEffort,
    GenaiRequest,
    GenaiReservedName,
    GenaiResponseFormat,
    GenaiRole,
    GenaiSettings,
    GenaiSource,
    GenaiTextPart,
    GenaiTool,
    GenaiToolCall,
    GenaiToolChoice,
    GenaiUsage,
    GenaiVertexOptions,
    Result,
} from './types.js';
export { createGenaiModels } from './wrapper/named-models.js';
