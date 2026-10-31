// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const pkg = require('smart-genai');

assert.equal(typeof pkg.createGenaiClient, 'function');
assert.equal(typeof pkg.credentialsFile, 'function');
assert.equal(typeof pkg.GEMINI_API_KEY_ENV, 'string');
assert.equal(typeof pkg.VERTEX_ENV, 'object');
assert.equal(typeof pkg.MAX_EMBED_INPUTS, 'number');
assert.equal(typeof pkg.assistantMessage, 'function');
assert.equal(typeof pkg.toolMessage, 'function');
assert.equal(typeof pkg.mergeGenaiConfig, 'function');
assert.equal(typeof pkg.validateGenaiConfig, 'function');
assert.equal(typeof pkg.createGenaiModels, 'function');
assert.equal(pkg.createGenaiClient({ apiKey: 'k' }).mode, 'developer');
