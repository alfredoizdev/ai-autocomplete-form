import { useMemo, useState, useEffect } from "react";
import { useDebounce } from "use-debounce";
import useSpellCheck from "./useSpellCheck";

const useDebouncedSpellCheck = (text: string, delay?: number) => {
  // Progressive debouncing based on text length for better performance
  const calculateDelay = () => {
    if (!text) return 200;
    const wordCount = text.split(/\s+/).filter(w => w.length > 0).length;
    
    // Shorter delay for small texts, longer for large texts
    if (wordCount < 20) return 200;
    if (wordCount < 50) return 350;
    if (wordCount < 100) return 500;
    return 700; // Max delay for very long texts
  };
  
  const effectiveDelay = delay ?? calculateDelay();
  const { getMisspelledWords, isLoading: spellCheckLoading, getSuggestions } = useSpellCheck();
  const [isRapidTyping, setIsRapidTyping] = useState(false);
  const [lastTypeTime, setLastTypeTime] = useState(Date.now());
  
  // Detect rapid typing to temporarily disable spell check
  useEffect(() => {
    const now = Date.now();
    const timeSinceLastType = now - lastTypeTime;
    
    // If typing faster than 100ms between keystrokes, consider it rapid
    if (timeSinceLastType < 100) {
      setIsRapidTyping(true);
      // Reset rapid typing flag after a pause
      const timeout = setTimeout(() => setIsRapidTyping(false), 500);
      return () => clearTimeout(timeout);
    }
    
    setLastTypeTime(now);
  }, [text, lastTypeTime]);
  
  // Debounce the text input to prevent spell checking on every keystroke
  const [debouncedText] = useDebounce(text, effectiveDelay);
  
  // Memoize spell check results to avoid unnecessary recalculations
  const misspelledWords = useMemo(() => {
    if (!debouncedText || spellCheckLoading || isRapidTyping) return [];
    return getMisspelledWords(debouncedText);
  }, [debouncedText, spellCheckLoading, getMisspelledWords, isRapidTyping]);
  
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
    isProcessing: text !== debouncedText || isRapidTyping, // True when user is still typing
  };
};

export default useDebouncedSpellCheck;