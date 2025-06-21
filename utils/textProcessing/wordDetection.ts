import {
  SENTENCE_ENDING_SPACE,
  SENTENCE_PATTERN,
  VOWEL_PATTERN,
  PUNCTUATION_PATTERN,
  WORD_BOUNDARY_PATTERN,
  MIN_WORD_LENGTH,
} from "../constants/regexPatterns";

/**
 * Memoized word detection cache
 * Stores already computed results to avoid recomputation
 */
const wordDetectionCache = new Map<string, boolean>();

/**
 * Check if text is ready for suggestions (not in middle of typing a word)
 * Optimized with memoization for performance
 * @param text - The text to analyze
 * @returns True if ready for suggestions
 */
export const isReadyForSuggestions = (text: string): boolean => {
  if (!text) return false;

  // Check cache first
  const cached = wordDetectionCache.get(text);
  if (cached !== undefined) {
    return cached;
  }

  let result = false;

  // Check if we're at the end of a sentence (period/question/exclamation + space)
  if (SENTENCE_ENDING_SPACE.test(text)) {
    // Don't suggest immediately after sentence endings - wait for user to start next sentence
    result = false;
  } else {
    // Check if we just finished a sentence and user has started typing a new word
    const sentenceMatch = text.match(SENTENCE_PATTERN);
    if (sentenceMatch) {
      // User has started typing after a sentence ending, check if word is complete enough
      const newWord = sentenceMatch[1];
      result = newWord.length >= MIN_WORD_LENGTH && VOWEL_PATTERN.test(newWord);
    } else {
      // For non-sentence-ending cases, check normal word completion
      const lastChar = text[text.length - 1];
      if (PUNCTUATION_PATTERN.test(lastChar)) {
        // Ready after comma, semicolon, colon, or regular space (but not after sentence endings)
        result = true;
      } else {
        // If text doesn't end with space/punctuation, check if last "word" is reasonable length
        const words = text.trim().split(WORD_BOUNDARY_PATTERN);
        const lastWord = words[words.length - 1];

        // Allow suggestions if the last word is at least 3 characters and seems complete
        result =
          lastWord.length >= MIN_WORD_LENGTH && VOWEL_PATTERN.test(lastWord);
      }
    }
  }

  // Cache the result (limit cache size to prevent memory leaks)
  if (wordDetectionCache.size < 1000) {
    wordDetectionCache.set(text, result);
  }

  return result;
};

/**
 * Extract the last word from text for analysis
 * @param text - The text to analyze
 * @returns The last word or empty string
 */
export const getLastWord = (text: string): string => {
  if (!text) return "";

  const words = text.trim().split(WORD_BOUNDARY_PATTERN);
  return words[words.length - 1] || "";
};

/**
 * Check if text ends with a complete word
 * @param text - The text to analyze
 * @returns True if ends with a complete word
 */
export const endsWithCompleteWord = (text: string): boolean => {
  if (!text) return false;

  const lastChar = text[text.length - 1];
  return PUNCTUATION_PATTERN.test(lastChar);
};

/**
 * Count words in text
 * @param text - The text to count
 * @returns Number of words
 */
export const countWords = (text: string): number => {
  if (!text) return 0;

  return text
    .trim()
    .split(WORD_BOUNDARY_PATTERN)
    .filter((word) => word.length > 0).length;
};

/**
 * Clear the word detection cache
 * Useful for testing or memory management
 */
export const clearWordDetectionCache = (): void => {
  wordDetectionCache.clear();
};

/**
 * Get word detection cache stats
 */
export const getWordDetectionStats = () => ({
  cacheSize: wordDetectionCache.size,
  maxCacheSize: 1000,
});
