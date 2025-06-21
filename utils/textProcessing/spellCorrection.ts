import { SPELL_CORRECTIONS } from "../constants/spellCorrections";
import {
  WORD_SPLIT_PATTERN,
  ALPHABETIC_ONLY,
} from "../constants/regexPatterns";

/**
 * Memoized spell correction cache
 * Stores already processed text to avoid recomputation
 */
const correctionCache = new Map<string, string>();

/**
 * Auto-correct misspelled words with memoization for performance
 * @param text - The text to correct
 * @returns Corrected text with preserved capitalization
 */
export const autoCorrectText = (text: string): string => {
  if (!text) return text;

  // Check cache first
  const cached = correctionCache.get(text);
  if (cached !== undefined) {
    return cached;
  }

  // Split into words while preserving spaces and punctuation
  const words = text.split(WORD_SPLIT_PATTERN);
  let hasChanges = false;

  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    if (word && ALPHABETIC_ONLY.test(word)) {
      // Only check actual words
      const lowerWord = word.toLowerCase();
      const correction = SPELL_CORRECTIONS[lowerWord];

      if (correction) {
        // Preserve original capitalization pattern
        if (word[0] === word[0].toUpperCase()) {
          words[i] = correction.charAt(0).toUpperCase() + correction.slice(1);
        } else {
          words[i] = correction;
        }
        hasChanges = true;
      }
    }
  }

  const result = hasChanges ? words.join("") : text;

  // Cache the result (limit cache size to prevent memory leaks)
  if (correctionCache.size < 1000) {
    correctionCache.set(text, result);
  }

  return result;
};

/**
 * Clear the spell correction cache
 * Useful for testing or memory management
 */
export const clearSpellCorrectionCache = (): void => {
  correctionCache.clear();
};

/**
 * Get spell correction cache stats
 */
export const getSpellCorrectionStats = () => ({
  cacheSize: correctionCache.size,
  maxCacheSize: 1000,
});
