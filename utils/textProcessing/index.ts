/**
 * Text Processing Utilities
 * Optimized utilities for spell correction, capitalization, and word detection
 */

export {
  autoCorrectText,
  clearSpellCorrectionCache,
  getSpellCorrectionStats,
} from "./spellCorrection";

export {
  autoCapitalizeText,
  applyBasicCapitalization,
  adjustSuggestionCapitalization,
  clearCapitalizationCache,
  getCapitalizationStats,
} from "./capitalization";

export {
  isReadyForSuggestions,
  getLastWord,
  endsWithCompleteWord,
  countWords,
  clearWordDetectionCache,
  getWordDetectionStats,
} from "./wordDetection";
