/**
 * Custom Dictionary Service for Spell Check
 * Manages user-defined words that should not be marked as misspelled
 * Uses localStorage for persistence across sessions
 */

const STORAGE_KEY = 'customSpellCheckDictionary';
const MAPPINGS_STORAGE_KEY = 'customSpellCheckMappings';

export interface CustomDictionaryService {
  addWord: (word: string) => boolean;
  removeWord: (word: string) => boolean;
  hasWord: (word: string) => boolean;
  getAllWords: () => string[];
  clearDictionary: () => void;
  getDictionarySize: () => number;
  addMapping: (misspelledWord: string, correctWord: string) => boolean;
  removeMapping: (misspelledWord: string) => boolean;
  getMapping: (misspelledWord: string) => string | null;
  getAllMappings: () => Record<string, string>;
  clearMappings: () => void;
  getMappingsSize: () => number;
}

/**
 * Get the current custom dictionary from localStorage
 */
const getStoredDictionary = (): string[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) return [];
    
    const parsed = JSON.parse(stored);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.warn('Failed to load custom dictionary from localStorage:', error);
    return [];
  }
};

/**
 * Save the custom dictionary to localStorage
 */
const saveStoredDictionary = (words: string[]): boolean => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(words));
    return true;
  } catch (error) {
    console.warn('Failed to save custom dictionary to localStorage:', error);
    return false;
  }
};

/**
 * Normalize word for consistent storage and lookup
 */
const normalizeWord = (word: string): string => {
  return word.toLowerCase().trim().replace(/[.,!?;:'"]/g, '');
};

/**
 * Add a word to the custom dictionary
 */
const addWord = (word: string): boolean => {
  const normalizedWord = normalizeWord(word);
  if (!normalizedWord) return false;
  
  const dictionary = getStoredDictionary();
  
  // Check if word already exists
  if (dictionary.includes(normalizedWord)) {
    return true; // Already exists, consider it successful
  }
  
  // Add word to dictionary
  dictionary.push(normalizedWord);
  
  // Save to localStorage
  return saveStoredDictionary(dictionary);
};

/**
 * Remove a word from the custom dictionary
 */
const removeWord = (word: string): boolean => {
  const normalizedWord = normalizeWord(word);
  if (!normalizedWord) return false;
  
  const dictionary = getStoredDictionary();
  const index = dictionary.indexOf(normalizedWord);
  
  if (index === -1) {
    return true; // Word doesn't exist, consider it successful
  }
  
  // Remove word from dictionary
  dictionary.splice(index, 1);
  
  // Save to localStorage
  return saveStoredDictionary(dictionary);
};

/**
 * Check if a word exists in the custom dictionary
 */
const hasWord = (word: string): boolean => {
  const normalizedWord = normalizeWord(word);
  if (!normalizedWord) return false;
  
  const dictionary = getStoredDictionary();
  return dictionary.includes(normalizedWord);
};

/**
 * Get all words in the custom dictionary
 */
const getAllWords = (): string[] => {
  return getStoredDictionary();
};

/**
 * Clear the entire custom dictionary
 */
const clearDictionary = (): void => {
  try {
    localStorage.removeItem(STORAGE_KEY);
  } catch (error) {
    console.warn('Failed to clear custom dictionary:', error);
  }
};

/**
 * Get the number of words in the custom dictionary
 */
const getDictionarySize = (): number => {
  return getStoredDictionary().length;
};

// ===== WORD MAPPINGS FUNCTIONALITY =====

/**
 * Get the current word mappings from localStorage
 */
const getStoredMappings = (): Record<string, string> => {
  try {
    const stored = localStorage.getItem(MAPPINGS_STORAGE_KEY);
    if (!stored) return {};
    
    const parsed = JSON.parse(stored);
    return typeof parsed === 'object' && parsed !== null ? parsed : {};
  } catch (error) {
    console.warn('Failed to load word mappings from localStorage:', error);
    return {};
  }
};

/**
 * Save word mappings to localStorage
 */
const saveStoredMappings = (mappings: Record<string, string>): boolean => {
  try {
    localStorage.setItem(MAPPINGS_STORAGE_KEY, JSON.stringify(mappings));
    return true;
  } catch (error) {
    console.warn('Failed to save word mappings to localStorage:', error);
    return false;
  }
};

/**
 * Add a word mapping (misspelled -> correct)
 */
const addMapping = (misspelledWord: string, correctWord: string): boolean => {
  const normalizedMisspelled = normalizeWord(misspelledWord);
  const normalizedCorrect = normalizeWord(correctWord);
  
  if (!normalizedMisspelled || !normalizedCorrect) return false;
  
  const mappings = getStoredMappings();
  mappings[normalizedMisspelled] = normalizedCorrect;
  
  return saveStoredMappings(mappings);
};

/**
 * Remove a word mapping
 */
const removeMapping = (misspelledWord: string): boolean => {
  const normalizedWord = normalizeWord(misspelledWord);
  if (!normalizedWord) return false;
  
  const mappings = getStoredMappings();
  delete mappings[normalizedWord];
  
  return saveStoredMappings(mappings);
};

/**
 * Get the correct spelling for a misspelled word
 */
const getMapping = (misspelledWord: string): string | null => {
  const normalizedWord = normalizeWord(misspelledWord);
  if (!normalizedWord) return null;
  
  const mappings = getStoredMappings();
  return mappings[normalizedWord] || null;
};

/**
 * Get all word mappings
 */
const getAllMappings = (): Record<string, string> => {
  return getStoredMappings();
};

/**
 * Clear all word mappings
 */
const clearMappings = (): void => {
  try {
    localStorage.removeItem(MAPPINGS_STORAGE_KEY);
  } catch (error) {
    console.warn('Failed to clear word mappings:', error);
  }
};

/**
 * Get the number of word mappings
 */
const getMappingsSize = (): number => {
  return Object.keys(getStoredMappings()).length;
};

/**
 * Custom Dictionary Service
 */
export const customDictionary: CustomDictionaryService = {
  addWord,
  removeWord,
  hasWord,
  getAllWords,
  clearDictionary,
  getDictionarySize,
  addMapping,
  removeMapping,
  getMapping,
  getAllMappings,
  clearMappings,
  getMappingsSize,
};

/**
 * Check if localStorage is available
 */
export const isLocalStorageAvailable = (): boolean => {
  try {
    const test = '__localStorage_test__';
    localStorage.setItem(test, 'test');
    localStorage.removeItem(test);
    return true;
  } catch {
    return false;
  }
};

export default customDictionary;