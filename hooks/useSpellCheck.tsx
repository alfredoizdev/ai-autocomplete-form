import { useState, useEffect, useRef, useMemo } from "react";
import { useDebounce } from "use-debounce";
import { customDictionary } from "@/lib/customDictionary";

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

// Common contraction mappings for missing apostrophes
const contractionMappings: Record<string, string[]> = {
  // Don't variations
  dont: ["don't", "do not"],
  doesnt: ["doesn't", "does not"],
  didnt: ["didn't", "did not"],

  // Can't variations
  cant: ["can't", "cannot", "can not"],
  couldnt: ["couldn't", "could not"],
  wouldnt: ["wouldn't", "would not"],
  shouldnt: ["shouldn't", "should not"],

  // Won't and will variations
  wont: ["won't", "will not"],
  wasnt: ["wasn't", "was not"],
  werent: ["weren't", "were not"],
  weve: ["we've", "we have"],
  were: ["we're", "we are"],
  weld: ["we'd", "we would", "we had"],
  well: ["we'll", "we will"],

  // Have variations
  havent: ["haven't", "have not"],
  hasnt: ["hasn't", "has not"],
  hadnt: ["hadn't", "had not"],

  // Is/Are variations
  isnt: ["isn't", "is not"],
  arent: ["aren't", "are not"],
  aint: ["ain't", "am not", "are not", "is not"],

  // I variations
  im: ["I'm", "I am"],
  ive: ["I've", "I have"],
  ill: ["I'll", "I will"],
  id: ["I'd", "I would", "I had"],

  // You variations
  youre: ["you're", "you are"],
  youve: ["you've", "you have"],
  youll: ["you'll", "you will"],
  youd: ["you'd", "you would", "you had"],

  // They variations
  theyre: ["they're", "they are"],
  theyve: ["they've", "they have"],
  theyll: ["they'll", "they will"],
  theyd: ["they'd", "they would", "they had"],

  // It variations
  its: ["it's", "it is", "it has"],
  itd: ["it'd", "it would", "it had"],
  itll: ["it'll", "it will"],

  // There variations
  theres: ["there's", "there is", "there has"],
  thered: ["there'd", "there would", "there had"],
  therell: ["there'll", "there will"],

  // Here variations
  heres: ["here's", "here is", "here has"],
  hered: ["here'd", "here would", "here had"],
  herell: ["here'll", "here will"],

  // Where variations
  wheres: ["where's", "where is", "where has"],
  whered: ["where'd", "where would", "where had"],
  wherell: ["where'll", "where will"],

  // What variations
  whats: ["what's", "what is", "what has"],
  whatd: ["what'd", "what would", "what did"],
  whatll: ["what'll", "what will"],

  // Who variations
  whos: ["who's", "who is", "who has"],
  whod: ["who'd", "who would", "who had"],
  wholl: ["who'll", "who will"],

  // How variations
  hows: ["how's", "how is", "how has"],
  howd: ["how'd", "how would", "how did"],
  howll: ["how'll", "how will"],

  // When variations
  whens: ["when's", "when is", "when has"],
  whend: ["when'd", "when would", "when did"],
  whenll: ["when'll", "when will"],

  // Why variations
  whys: ["why's", "why is", "why has"],
  whyd: ["why'd", "why would", "why did"],
  whyll: ["why'll", "why will"],

  // Let's
  lets: ["let's", "let us"],

  // That's
  thats: ["that's", "that is", "that has"],
  thatd: ["that'd", "that would", "that had"],
  thatll: ["that'll", "that will"],
};

// Get contraction suggestions for a word
const getContractionSuggestions = (word: string): string[] => {
  const lowerWord = word.toLowerCase();
  return contractionMappings[lowerWord] || [];
};

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
          fetch("/dictionaries/en_US/en_US.dic"),
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
    if (!word) return true;

    // Clean the word of punctuation
    const cleanWord = word.replace(/[.,!?;:'"]/g, "");
    if (!cleanWord) return true;

    // First check custom dictionary
    if (customDictionary.hasWord(cleanWord)) {
      return true;
    }

    // Then check Typo.js dictionary if available
    if (!dictionary) return true;

    return dictionary.check(cleanWord);
  };

  // Get suggestions for a misspelled word
  const getSuggestions = (word: string): string[] => {
    if (!word) return [];

    // Clean the word of punctuation
    const cleanWord = word.replace(/[.,!?;:'"]/g, "");
    if (!cleanWord) return [];

    // First, check for custom word mappings (highest priority)
    const customMapping = customDictionary.getMapping(cleanWord);
    const customSuggestions = customMapping ? [customMapping] : [];

    // Then check for contraction suggestions
    const contractionSuggestions = getContractionSuggestions(cleanWord);

    // Then get dictionary suggestions if available
    const dictionarySuggestions = dictionary
      ? dictionary.suggest(cleanWord)
      : [];

    // Combine suggestions, prioritizing custom mappings, then contractions, then dictionary
    const allSuggestions = [
      ...customSuggestions,
      ...contractionSuggestions,
      ...dictionarySuggestions,
    ];

    // Remove duplicates and return first 8 suggestions
    return [...new Set(allSuggestions)].slice(0, 8);
  };

  // Check multiple words in a text
  const checkText = (
    text: string
  ): Array<{ word: string; isCorrect: boolean; suggestions: string[] }> => {
    if (!text || !dictionary) return [];

    // Split text into words
    const words = text.match(/\b\w+\b/g) || [];

    return words.map((word) => ({
      word,
      isCorrect: checkWord(word),
      suggestions: checkWord(word) ? [] : getSuggestions(word),
    }));
  };

  // Get misspelled words from text
  const getMisspelledWords = (
    text: string
  ): Array<{ word: string; suggestions: string[] }> => {
    const results = checkText(text);
    return results
      .filter((result) => !result.isCorrect)
      .map((result) => ({
        word: result.word,
        suggestions: result.suggestions,
      }));
  };

  return {
    dictionary,
    isLoading,
    checkWord,
    getSuggestions,
    checkText,
    getMisspelledWords,
    customDictionary,
  };
};

export default useSpellCheck;
