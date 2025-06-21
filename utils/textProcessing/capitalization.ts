import {
  SENTENCE_CAPITALIZATION,
  STANDALONE_I,
  STANDALONE_I_SPACE,
  PROPER_NOUNS,
  AFTER_SENTENCE_PATTERN,
  ENDS_WITH_SENTENCE,
} from "../constants/regexPatterns";

/**
 * Memoized capitalization cache
 * Stores already processed text to avoid recomputation
 */
const capitalizationCache = new Map<string, string>();

/**
 * Auto-capitalize user input text based on sentence context with memoization
 * @param text - The text to capitalize
 * @returns Text with proper capitalization
 */
export const autoCapitalizeText = (text: string): string => {
  if (!text) return text;

  // Check cache first
  const cached = capitalizationCache.get(text);
  if (cached !== undefined) {
    return cached;
  }

  let result = text;

  // Capitalize first character of the entire text
  if (result.length > 0) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  // Capitalize after sentence-ending punctuation followed by one or more spaces
  result = result.replace(
    SENTENCE_CAPITALIZATION,
    (match, punctuation, letter) => {
      return punctuation + letter.toUpperCase();
    }
  );

  // Capitalize "I" when it's a standalone word (but be more specific)
  result = result.replace(STANDALONE_I, "I");

  // Capitalize proper nouns and common words that should be capitalized
  result = result.replace(PROPER_NOUNS, (match) => {
    // Only capitalize if it's at start of sentence or after punctuation
    const beforeMatch = result.substring(0, result.indexOf(match));
    if (beforeMatch === "" || /[.!?]\s*$/.test(beforeMatch)) {
      return match.charAt(0).toUpperCase() + match.slice(1).toLowerCase();
    }
    return match.toLowerCase();
  });

  // Cache the result (limit cache size to prevent memory leaks)
  if (capitalizationCache.size < 1000) {
    capitalizationCache.set(text, result);
  }

  return result;
};

/**
 * Apply only basic capitalization while user is typing with memoization
 * @param text - The text to capitalize
 * @returns Text with basic capitalization
 */
export const applyBasicCapitalization = (text: string): string => {
  if (!text) return text;

  // Use a separate cache key for basic capitalization
  const cacheKey = `basic:${text}`;
  const cached = capitalizationCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  let result = text;

  // Capitalize first character of the entire text
  if (result.length > 0) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  // Capitalize after sentence-ending punctuation followed by one or more spaces
  result = result.replace(
    SENTENCE_CAPITALIZATION,
    (match, punctuation, letter) => {
      return punctuation + letter.toUpperCase();
    }
  );

  // Only capitalize standalone "I" if it's followed by a space (completed word)
  result = result.replace(STANDALONE_I_SPACE, "I ");

  // Cache the result
  if (capitalizationCache.size < 1000) {
    capitalizationCache.set(cacheKey, result);
  }

  return result;
};

/**
 * Adjust suggestion capitalization based on sentence context with memoization
 * @param userText - The user's current text
 * @param suggestion - The AI suggestion
 * @returns Properly capitalized suggestion
 */
export const adjustSuggestionCapitalization = (
  userText: string,
  suggestion: string
): string => {
  if (!suggestion || !userText) return suggestion;

  // Create cache key from both inputs
  const cacheKey = `suggest:${userText}:${suggestion}`;
  const cached = capitalizationCache.get(cacheKey);
  if (cached !== undefined) {
    return cached;
  }

  // Make suggestion lowercase by default for natural flow
  let result = suggestion.toLowerCase();

  // Check if we're continuing after sentence-ending punctuation + space + new word
  if (AFTER_SENTENCE_PATTERN.test(userText)) {
    // We're in the middle of a new sentence, keep lowercase
    result = result;
  } else if (ENDS_WITH_SENTENCE.test(userText.trim())) {
    // Check if we're immediately after sentence ending
    // Capitalize first word of new sentence
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }
  // For all other cases (continuing same sentence), keep lowercase

  // Cache the result
  if (capitalizationCache.size < 1000) {
    capitalizationCache.set(cacheKey, result);
  }

  return result;
};

/**
 * Clear the capitalization cache
 * Useful for testing or memory management
 */
export const clearCapitalizationCache = (): void => {
  capitalizationCache.clear();
};

/**
 * Get capitalization cache stats
 */
export const getCapitalizationStats = () => ({
  cacheSize: capitalizationCache.size,
  maxCacheSize: 1000,
});
