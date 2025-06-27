/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { DEFAULT_GEMINI_FLASH_MODEL } from '../config/models.js';

export function getFlashModel(): string {
  return process.env.FLASH_MODEL || DEFAULT_GEMINI_FLASH_MODEL;
}