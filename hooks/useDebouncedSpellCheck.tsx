import { useMemo } from "react";
import { useDebounce } from "use-debounce";
import useSpellCheck from "./useSpellCheck";

const useDebouncedSpellCheck = (text: string, delay: number = 800) => {
  const { getMisspelledWords, isLoading: spellCheckLoading, getSuggestions } = useSpellCheck();
  
  // Debounce the text input to prevent spell checking on every keystroke
  const [debouncedText] = useDebounce(text, delay);
  
  // Memoize spell check results to avoid unnecessary recalculations
  const misspelledWords = useMemo(() => {
    if (!debouncedText || spellCheckLoading) return [];
    return getMisspelledWords(debouncedText);
  }, [debouncedText, spellCheckLoading, getMisspelledWords]);
  
  // Cache for individual word suggestions to avoid repeated API calls
  const wordSuggestionsCache = useMemo(() => new Map<string, string[]>(), []);
  
  const getCachedSuggestions = (word: string): string[] => {
    if (wordSuggestionsCache.has(word)) {
      return wordSuggestionsCache.get(word) || [];
    }
    
    const suggestions = getSuggestions(word);
    wordSuggestionsCache.set(word, suggestions);
    return suggestions;
  };
  
  return {
    misspelledWords,
    isLoading: spellCheckLoading,
    getSuggestions: getCachedSuggestions,
    isProcessing: text !== debouncedText, // True when user is still typing
  };
};

export default useDebouncedSpellCheck;