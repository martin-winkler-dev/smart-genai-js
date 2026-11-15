// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenaiOutput, GenaiToolCall } from '../../types.js';

// -------------------------------------------------------------------
// --- 1. Exported Functions
// -------------------------------------------------------------------

/**
 * **Removes foreign properties from answer of Vertex batch line**: import mixes property names of all schemas in job.
 *
 * - echoed `request` of line: own property schema is object, foreign one is `null`
 * - `json`: cleaned with echoed `generationConfig.responseSchema`; `text` then `JSON.stringify` of cleaned value
 * - `toolCalls`: arguments cleaned with echoed `parameters` of declaration of same name
 * - nothing removed, no echo or unknown shapes: output unchanged; never throws
 */
export function withoutForeignProperties(output: GenaiOutput, request: unknown): GenaiOutput {
    if (!isRecord(request)) {
        return output;
    }
    let cleaned = output;
    const config = request.generationConfig;
    if ('json' in output && isRecord(config) && isRecord(config.responseSchema)) {
        const json = withoutForeign(output.json, config.responseSchema);
        if (json !== output.json) {
            cleaned = { ...cleaned, json, text: JSON.stringify(json) };
        }
    }
    if (output.toolCalls !== undefined) {
        const parameters = declarationParameters(request.tools);
        const toolCalls = output.toolCalls.map((call) => cleanCall(call, parameters.get(call.function.name)));
        if (toolCalls.some((call, index) => call !== output.toolCalls?.[index])) {
            cleaned = { ...cleaned, toolCalls };
        }
    }
    return cleaned;
}

// -------------------------------------------------------------------
// --- 2. Private Functions
// -------------------------------------------------------------------

/** Maps declaration names of echoed `tools` to their `parameters` schemas. */
function declarationParameters(tools: unknown): Map<string, unknown> {
    const parameters = new Map<string, unknown>();
    for (const tool of Array.isArray(tools) ? tools : []) {
        const declarations = isRecord(tool) ? tool.functionDeclarations : undefined;
        for (const declaration of Array.isArray(declarations) ? declarations : []) {
            if (isRecord(declaration) && typeof declaration.name === 'string') {
                parameters.set(declaration.name, declaration.parameters);
            }
        }
    }
    return parameters;
}

/** Cleans JSON arguments of tool call; same call when nothing removed or arguments not JSON. */
function cleanCall(call: GenaiToolCall, schema: unknown): GenaiToolCall {
    if (!isRecord(schema)) {
        return call;
    }
    let args: unknown;
    try {
        args = JSON.parse(call.function.arguments) as unknown;
    } catch {
        return call;
    }
    const cleaned = withoutForeign(args, schema);
    return cleaned === args ? call : { ...call, function: { ...call.function, arguments: JSON.stringify(cleaned) } };
}

/**
 * **Copies value without keys the schema lists as `null`**: same value when nothing removed.
 *
 * - branches: schema itself and its `anyOf` members (nested `anyOf` flattened)
 * - object key: dropped when no branch declares it with non-null schema and at least one lists it as `null`
 * - declared keys: recursed with declaring schema (several: as `anyOf`); unlisted keys kept
 * - arrays: elements recursed with `items` of branches
 */
function withoutForeign(value: unknown, schema: Record<string, unknown>): unknown {
    const branches = schemaBranches(schema);
    if (Array.isArray(value)) {
        const items = branches.flatMap((branch) => (isRecord(branch.items) ? [branch.items] : []));
        if (items.length === 0) {
            return value;
        }
        const copy = value.map((entry) => withoutForeign(entry, joined(items)));
        return copy.some((entry, index) => entry !== value[index]) ? copy : value;
    }
    if (!isRecord(value)) {
        return value;
    }
    let changed = false;
    const copy: Record<string, unknown> = {};
    for (const [key, entry] of Object.entries(value)) {
        const listed = branches.flatMap((branch) =>
            isRecord(branch.properties) && Object.hasOwn(branch.properties, key) ? [branch.properties[key]] : [],
        );
        const declared = listed.filter(isRecord);
        if (declared.length === 0 && listed.includes(null)) {
            changed = true;
            continue;
        }
        const cleaned = declared.length === 0 ? entry : withoutForeign(entry, joined(declared));
        changed ||= cleaned !== entry;
        copy[key] = cleaned;
    }
    return changed ? copy : value;
}

/** Lists schema and its `anyOf` members, nested `anyOf` flattened. */
function schemaBranches(schema: Record<string, unknown>): Record<string, unknown>[] {
    const members = Array.isArray(schema.anyOf) ? schema.anyOf.filter(isRecord) : [];
    return [schema, ...members.flatMap(schemaBranches)];
}

/** Single schema, or several as `anyOf`. */
function joined(schemas: Record<string, unknown>[]): Record<string, unknown> {
    return schemas.length === 1 ? (schemas[0] as Record<string, unknown>) : { anyOf: schemas };
}

/** Checks for plain non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
