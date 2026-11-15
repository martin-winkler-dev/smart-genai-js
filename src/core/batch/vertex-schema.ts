// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import { InvalidRequestError } from '../request.js';

// -------------------------------------------------------------------
// --- 1. Types
// -------------------------------------------------------------------

/**
 * **Where schema is converted**.
 *
 * - `item`: item text of errors (` of item "<key>"`, `''` in `requestBytes`)
 * - `declaration`: function declaration schema; empty `properties` dropped on every level, empty schema omitted
 */
type Where = { item: string; declaration: boolean };

// -------------------------------------------------------------------
// --- 2. Constants
// -------------------------------------------------------------------

/** Fields of SDK `Schema` type: only ones the typed schema form takes. */
const TYPED_SCHEMA_FIELDS: ReadonlySet<string> = new Set([
    'anyOf',
    'default',
    'description',
    'enum',
    'example',
    'format',
    'items',
    'maxItems',
    'maxLength',
    'maxProperties',
    'maximum',
    'minItems',
    'minLength',
    'minProperties',
    'minimum',
    'nullable',
    'pattern',
    'properties',
    'propertyOrdering',
    'required',
    'title',
    'type',
]);

/** JSON schema keywords dropped without error: annotations, or no typed counterpart (as SDK converter). */
const DROPPED_SCHEMA_KEYWORDS: ReadonlySet<string> = new Set(['$schema', '$id', '$comment', 'additionalProperties']);

/** Type values of typed schema form, uppercase. */
const TYPED_TYPES: ReadonlySet<string> = new Set(['OBJECT', 'STRING', 'INTEGER', 'NUMBER', 'BOOLEAN', 'ARRAY']);

/** Schema field pairs of function declarations: JSON schema form, typed form. */
const DECLARATION_SCHEMA_FIELDS = [
    ['parametersJsonSchema', 'parameters'],
    ['responseJsonSchema', 'response'],
] as const;

// -------------------------------------------------------------------
// --- 3. Exported Functions
// -------------------------------------------------------------------

/**
 * **Converts schemas of Vertex batch line to typed form**: import mixes JSON schemas of different shapes.
 *
 * - only converter of batch schemas: `vertexRequest` copies them in given form
 * - `generationConfig.responseJsonSchema` or `responseSchema` → `responseSchema`
 * - `tools[].functionDeclarations[]`: `parametersJsonSchema` or `parameters` → `parameters`,
 *   `responseJsonSchema` or `response` → `response`
 * - both forms through `typedSchema`: typed input gives same result as its JSON schema form
 * - declaration schemas: empty `properties` dropped on every level; schema empty after conversion omitted
 * - copies, caller objects never modified
 * @throws {InvalidRequestError} If a schema cannot be expressed in typed form, or both forms are set.
 */
export function typedSchemaForms(request: Record<string, unknown>, key: string): Record<string, unknown> {
    const item = key === '' ? '' : ` of item "${key}"`;
    const { tools, generationConfig } = request;
    const copy = { ...request };
    if (Array.isArray(tools)) {
        copy.tools = tools.map((tool: unknown, index) => typedTool(tool, `request.tools[${index}]`, item));
    }
    if (isRecord(generationConfig)) {
        const where: Where = { item, declaration: false };
        copy.generationConfig = typedField(
            generationConfig,
            ['responseJsonSchema', 'responseSchema'],
            'request.generationConfig',
            where,
        );
    }
    return copy;
}

// -------------------------------------------------------------------
// --- 4. Private Functions
// -------------------------------------------------------------------

/** Copies tool with function declaration schemas in typed form; other tools unchanged. */
function typedTool(tool: unknown, path: string, item: string): unknown {
    if (!isRecord(tool) || !Array.isArray(tool.functionDeclarations)) {
        return tool;
    }
    const where: Where = { item, declaration: true };
    const functionDeclarations = tool.functionDeclarations.map((declaration: unknown, index) => {
        if (!isRecord(declaration)) {
            return declaration;
        }
        const at = `${path}.functionDeclarations[${index}]`;
        return DECLARATION_SCHEMA_FIELDS.reduce((copy, fields) => typedField(copy, fields, at, where), declaration);
    });
    return { ...tool, functionDeclarations };
}

/**
 * **Copies object with schema of either form as typed field**: see `typedSchema`.
 *
 * - neither form set: object unchanged
 * - declaration schema empty after conversion: typed field omitted
 * @throws {InvalidRequestError} If both forms are set, or `typedSchema` rejects schema.
 */
function typedField(
    object: Record<string, unknown>,
    [json, typed]: readonly [string, string],
    path: string,
    where: Where,
): Record<string, unknown> {
    const { [json]: jsonForm, [typed]: typedForm, ...rest } = object;
    if (jsonForm === undefined && typedForm === undefined) {
        return object;
    }
    if (jsonForm !== undefined && typedForm !== undefined) {
        throw unsupported(`Schema fields "${json}" and "${typed}" together`, path, where.item);
    }
    const [field, schema] = jsonForm === undefined ? [typed, typedForm] : [json, jsonForm];
    const converted = typedSchema(schema, `${path}.${field}`, where);
    return where.declaration && Object.keys(converted).length === 0 ? rest : { ...rest, [typed]: converted };
}

/**
 * **Converts JSON schema or typed schema to typed schema form**: mirrors SDK converter `processJsonSchema`.
 *
 * - keywords checked against `TYPED_SCHEMA_FIELDS`; `DROPPED_SCHEMA_KEYWORDS` skipped; any other rejected
 * - `null`/`undefined` keyword values skipped (as SDK)
 * - `type`: see `typedType`; `type` together with `anyOf` rejected
 * - `anyOf`: see `typedAnyOf`; single remaining branch merged, own keywords win
 * - subschemas: `properties` values, `items`, `anyOf` members; property names are data, not keywords
 * - other values (`enum`, `default`, `example`...) copied unchanged; `enum`: string values only
 * - typed input (uppercase `type`, `nullable`, `propertyOrdering`) kept: own output converts to equal schema
 * - empty `properties` of declaration schema dropped (see `Where`)
 * @throws {InvalidRequestError} If the schema holds an unsupported keyword or value.
 */
function typedSchema(schema: unknown, path: string, where: Where): Record<string, unknown> {
    const { item } = where;
    if (!isRecord(schema)) {
        throw unsupported('Non-object schema', path, item);
    }
    const entries = Object.entries(schema).filter(([, value]) => value !== undefined && value !== null);
    for (const [keyword, value] of entries) {
        if (!TYPED_SCHEMA_FIELDS.has(keyword) && !DROPPED_SCHEMA_KEYWORDS.has(keyword)) {
            throw unsupported(`Schema keyword "${keyword}"`, path, item);
        }
        if (keyword === 'enum' && !(Array.isArray(value) && value.every((entry) => typeof entry === 'string'))) {
            throw unsupported('Schema keyword "enum" with non-string values', path, item);
        }
    }
    if (entries.some(([keyword]) => keyword === 'type') && entries.some(([keyword]) => keyword === 'anyOf')) {
        throw unsupported('Schema keyword "type" together with "anyOf"', path, item);
    }
    const typed: Record<string, unknown> = {};
    let branch: Record<string, unknown> | undefined;
    for (const [keyword, value] of entries) {
        if (DROPPED_SCHEMA_KEYWORDS.has(keyword)) {
            continue;
        }
        if (keyword === 'type') {
            Object.assign(typed, typedType(value, path, item));
        } else if (keyword === 'items') {
            typed.items = typedSchema(value, `${path}.items`, where);
        } else if (keyword === 'properties') {
            if (!isRecord(value)) {
                throw unsupported('Schema keyword "properties" with a non-object value', path, item);
            }
            const properties = Object.entries(value).filter(([, entry]) => entry !== undefined);
            if (properties.length > 0 || !where.declaration) {
                typed.properties = Object.fromEntries(
                    properties.map(([name, entry]) => [name, typedSchema(entry, `${path}.properties.${name}`, where)]),
                );
            }
        } else if (keyword === 'anyOf') {
            const { nullable, members } = typedAnyOf(value, path, where);
            if (nullable) {
                typed.nullable = true;
            }
            if (members.length === 1) {
                branch = members[0];
            } else {
                typed.anyOf = members;
            }
        } else {
            typed[keyword] = value;
        }
    }
    return branch === undefined ? typed : { ...branch, ...typed };
}

/**
 * **Converts `type` value to typed form**: mirrors SDK `flattenTypeArrayToAnyOf`.
 *
 * - type name → uppercase; unknown names rejected
 * - list: `null` entry → `nullable: true`; one other type → `type`; several → `anyOf` of `{ type }`
 * - `null` alone (string or list) and empty list rejected
 * @throws {InvalidRequestError} If the value is unknown, only `null` or an empty list.
 */
function typedType(value: unknown, path: string, item: string): Record<string, unknown> {
    const names = Array.isArray(value) ? value : [value];
    if (names.length === 0) {
        throw unsupported('Schema keyword "type" with an empty list', path, item);
    }
    const typed: Record<string, unknown> = names.some(isNullType) ? { nullable: true } : {};
    const types = names
        .filter((name) => !isNullType(name))
        .map((name) => {
            const upper = typeof name === 'string' ? name.toUpperCase() : '';
            if (!TYPED_TYPES.has(upper)) {
                throw unsupported(`Schema keyword "type" with value ${JSON.stringify(name)}`, path, item);
            }
            return upper;
        });
    if (types.length === 0) {
        throw unsupported('Schema keyword "type" with only "null"', path, item);
    }
    return types.length === 1 ? { ...typed, type: types[0] } : { ...typed, anyOf: types.map((type) => ({ type })) };
}

/**
 * **Converts `anyOf` members**: branches with `type: 'null'` → `nullable`, as SDK.
 *
 * - remaining members converted; none remaining rejected
 * @throws {InvalidRequestError} If the value is no list or holds only `null` branches.
 */
function typedAnyOf(
    value: unknown,
    path: string,
    where: Where,
): { nullable: boolean; members: Record<string, unknown>[] } {
    if (!Array.isArray(value)) {
        throw unsupported('Schema keyword "anyOf" with a non-list value', path, where.item);
    }
    let nullable = false;
    const members: Record<string, unknown>[] = [];
    for (const [index, member] of value.entries()) {
        if (isRecord(member) && isNullType(member.type)) {
            nullable = true;
        } else {
            members.push(typedSchema(member, `${path}.anyOf[${index}]`, where));
        }
    }
    if (members.length === 0) {
        throw unsupported('Schema keyword "anyOf" with only "null" branches', path, where.item);
    }
    return { nullable, members };
}

/** Checks for type name `null` in JSON schema or typed form (`NULL`). */
function isNullType(name: unknown): boolean {
    return name === 'null' || name === 'NULL';
}

/** Builds error for schema part the typed form cannot express. */
function unsupported(what: string, path: string, item: string): InvalidRequestError {
    return new InvalidRequestError(`${what} at "${path}"${item} is not supported in Vertex batch`);
}

/** Checks for plain non-array object. */
function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}
