import { useState, useEffect, useRef } from "react";

// Define types for Typo.js dictionary
interface TypoDictionary {
  check: (word: string) => boolean;
  suggest: (word: string) => string[];
}

declare global {
  interface Window {
    Typo: any;
  }
}

const useSpellCheck = () => {
  const [dictionary, setDictionary] = useState<TypoDictionary | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const typoRef = useRef<any>(null);

  // Initialize dictionary on mount
  useEffect(() => {
    const initDictionary = async () => {
      try {
        // Import Typo.js dynamically
        const Typo = await import("typo-js");
        typoRef.current = Typo.default || Typo;
        
        // Load dictionary files manually
        const [affResponse, dicResponse] = await Promise.all([
          fetch("/dictionaries/en_US/en_US.aff"),
          fetch("/dictionaries/en_US/en_US.dic")
        ]);
        
        const affData = await affResponse.text();
        const dicData = await dicResponse.text();
        
        // Initialize English dictionary with loaded data
        const dict = new typoRef.current("en_US", affData, dicData);
        setDictionary(dict);
        setIsLoading(false);
      } catch (error) {
        console.error("Failed to initialize spell checker:", error);
        setIsLoading(false);
      }
    };

    initDictionary();
  }, []);

  // Check if a word is spelled correctly
  const checkWord = (word: string): boolean => {
    if (!dictionary || !word) return true;
    
    // Clean the word of punctuation
    const cleanWord = word.replace(/[.,!?;:'"]/g, "");
    if (!cleanWord) return true;
    
    return dictionary.check(cleanWord);
  };

  // Get suggestions for a misspelled word
  const getSuggestions = (word: string): string[] => {
    if (!dictionary || !word) return [];
    
    // Clean the word of punctuation
    const cleanWord = word.replace(/[.,!?;:'"]/g, "");
    if (!cleanWord) return [];
    
    return dictionary.suggest(cleanWord);
  };

  // Check multiple words in a text
  const checkText = (text: string): Array<{ word: string; isCorrect: boolean; suggestions: string[] }> => {
    if (!text || !dictionary) return [];
    
    // Split text into words
    const words = text.match(/\b\w+\b/g) || [];
    
    return words.map(word => ({
      word,
      isCorrect: checkWord(word),
      suggestions: checkWord(word) ? [] : getSuggestions(word)
    }));
  };

  // Get misspelled words from text
  const getMisspelledWords = (text: string): Array<{ word: string; suggestions: string[] }> => {
    const results = checkText(text);
    return results.filter(result => !result.isCorrect).map(result => ({
      word: result.word,
      suggestions: result.suggestions
    }));
  };

  return {
    dictionary,
    isLoading,
    checkWord,
    getSuggestions,
    checkText,
    getMisspelledWords
  };
};

export default useSpellCheck;