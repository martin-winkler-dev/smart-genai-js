// Copyright 2026 Martin Winkler
// SPDX-License-Identifier: Apache-2.0

import type { GenaiError } from '../../types.js';

// -------------------------------------------------------------------
// --- 1. Constants
// -------------------------------------------------------------------

/** Check errors of jobs found final; batch calls never cancel these jobs. */
const FINAL_JOB_ERRORS = new WeakSet<GenaiError>();

// -------------------------------------------------------------------
// --- 2. Exported Functions
// -------------------------------------------------------------------

/** Marks `error` as a `check` error of a final job (e.g. result read failure); returns `error`. */
export function finalJobError(error: GenaiError): GenaiError {
    FINAL_JOB_ERRORS.add(error);
    return error;
}

/** Whether `error` was marked by `finalJobError`. */
export function isFinalJobError(error: GenaiError): boolean {
    return FINAL_JOB_ERRORS.has(error);
}
