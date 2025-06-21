/**
 * Pre-compiled regex patterns for optimal performance
 * These patterns are compiled once and reused throughout the application
 */

// Text validation patterns
export const SENTENCE_ENDING_SPACE = /[.!?]\s+$/;
export const SENTENCE_PATTERN = /[.!?]\s+([a-zA-Z]+)$/;
export const VOWEL_PATTERN = /[aeiouAEIOU]/;
export const PUNCTUATION_PATTERN = /[\s,;:]/;
export const WORD_BOUNDARY_PATTERN = /\s+/;
export const ALPHABETIC_ONLY = /^[a-zA-Z]+$/;

// Capitalization patterns
export const SENTENCE_CAPITALIZATION = /([.!?]\s+)([a-z])/g;
export const STANDALONE_I = /\b(i)\b/g;
export const STANDALONE_I_SPACE = /\b(i)\s/g;
export const PROPER_NOUNS = /\b(we|us|our)\b/gi;
export const AFTER_SENTENCE_PATTERN = /[.!?]\s+[a-zA-Z]+\s*$/;
export const ENDS_WITH_SENTENCE = /[.!?]\s*$/;

// Text processing patterns
export const WORD_SPLIT_PATTERN = /(\s+|[.,!?;:])/;
export const CORRECTION_TRIGGER = /[\s.,!?;:]$/;

// Cleaning patterns for AI responses
export const STARTING_CLEANUP = /^["'`\.\s*→←↑↓▲▼►◄]*/;
export const ENDING_CLEANUP = /["'`\.\s*→←↑↓▲▼►◄]*$/;
export const LINE_BREAK_CLEANUP = /\n.*$/g;
export const PUNCTUATION_CLEANUP = /[.!?;:,'""`*→←↑↓▲▼►◄]/g;
export const MULTI_SPACE_CLEANUP = /\s+/g;
export const NON_ALPHABETIC_CLEANUP = /[^a-zA-Z]/g;

// Performance constants
export const MIN_TEXT_LENGTH = 10;
export const MIN_WORD_LENGTH = 3;
export const MIN_NEW_CHARS = 2;
export const MIN_TEXTAREA_HEIGHT = 96;
export const DEBOUNCE_DELAY = 1000;
