import { askOllamaCompletationAction } from "@/actions/ai-text";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useTransition,
  useCallback,
} from "react";
import { useForm, SubmitHandler } from "react-hook-form";
import { useDebounce } from "use-debounce";

// Get the current sentence being typed based on cursor position
const getCurrentSentenceAtCursor = (
  text: string,
  cursorPos: number = text.length
): string => {
  if (!text || text.trim().length === 0) return "";

  // Find the start of current sentence by looking backwards for sentence boundaries
  let sentenceStart = 0;
  for (let i = cursorPos - 1; i >= 0; i--) {
    if (/[.!?]/.test(text[i])) {
      // Found sentence boundary, start after it (and any whitespace)
      sentenceStart = i + 1;
      while (sentenceStart < text.length && /\s/.test(text[sentenceStart])) {
        sentenceStart++;
      }
      break;
    }
  }

  // Extract current sentence from start to cursor position
  const currentSentence = text.substring(sentenceStart, cursorPos).trim();
  return currentSentence;
};

// Count words from sentence start to cursor position
const getWordsBeforeCursor = (
  text: string,
  cursorPos: number = text.length
): string[] => {
  const currentSentence = getCurrentSentenceAtCursor(text, cursorPos);

  if (!currentSentence) return [];

  return currentSentence
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0);
};

// Count total words from text start to cursor position
const getWordCountAtCursor = (
  text: string,
  cursorPos: number = text.length
): number => {
  if (!text || text.trim().length === 0) return 0;
  
  const textToCursor = text.substring(0, cursorPos).trim();
  if (!textToCursor) return 0;
  
  return textToCursor.split(/\s+/).filter(word => word.length > 0).length;
};

// Count words typed since a specific position in the text
const getWordCountSincePosition = (
  text: string,
  sincePosition: number,
  cursorPos: number = text.length
): number => {
  if (!text || sincePosition >= cursorPos) return 0;
  
  const textSincePosition = text.substring(sincePosition, cursorPos).trim();
  if (!textSincePosition) return 0;
  
  return textSincePosition.split(/\s+/).filter(word => word.length > 0).length;
};

// Check if we're ready for autocomplete suggestions
const isReadyForSuggestions = (
  text: string,
  cursorPos: number = text.length,
  lastAcceptedWordCount: number = 0,
  lastAcceptedPosition: number = 0,
  afterPunctuation: boolean = false
): boolean => {
  if (!text || text.trim().length === 0) return false;

  const words = getWordsBeforeCursor(text, cursorPos);
  
  // Relax word requirement after punctuation (natural pause)
  const minWordsRequired = afterPunctuation ? 5 : 5;
  
  // Must have minimum words in current sentence
  if (words.length < minWordsRequired) return false;
  
  // If we've never accepted a suggestion, we're ready
  if (lastAcceptedWordCount === 0) return true;
  
  // Must have typed at least 5 new words since last acceptance
  const newWordsTyped = getWordCountSincePosition(text, lastAcceptedPosition, cursorPos);
  return newWordsTyped >= 5;
};

// Determine if we need a space before the suggestion based on text context
const needsSpaceBeforeSuggestion = (text: string): boolean => {
  if (!text) return false;

  // If text ends with whitespace, no additional space needed
  if (/\s$/.test(text)) return false;

  // If text ends with punctuation, we need a space
  if (/[.,!?;:]$/.test(text)) return true;

  // If text ends with a word character, we need a space
  if (/\w$/.test(text)) return true;

  return false;
};

// Apply only basic capitalization while user is typing
const applyBasicCapitalization = (text: string): string => {
  if (!text) return text;

  let result = text;

  // Capitalize first character of the entire text
  if (result.length > 0) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  // Capitalize after sentence-ending punctuation followed by one or more spaces
  result = result.replace(
    /([.!?]\s+)([a-z])/g,
    (match, punctuation, letter) => {
      return punctuation + letter.toUpperCase();
    }
  );

  // Only capitalize standalone "I" if it's followed by a space (completed word)
  result = result.replace(/\b(i)\s/g, "I ");

  return result;
};

// Auto-capitalize user input text based on sentence context
const autoCapitalizeText = (text: string): string => {
  if (!text) return text;

  let result = text;

  // Capitalize first character of the entire text
  if (result.length > 0) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  // Capitalize after sentence-ending punctuation followed by one or more spaces
  result = result.replace(
    /([.!?]\s+)([a-z])/g,
    (match, punctuation, letter) => {
      return punctuation + letter.toUpperCase();
    }
  );

  // Capitalize "I" when it's a standalone word
  result = result.replace(/\b(i)\b/g, "I");

  // Capitalize proper nouns and common words that should be capitalized
  result = result.replace(/\b(we|us|our)\b/gi, (match) => {
    // Only capitalize if it's at start of sentence or after punctuation
    const beforeMatch = result.substring(0, result.indexOf(match));
    if (beforeMatch === "" || /[.!?]\s*$/.test(beforeMatch)) {
      return match.charAt(0).toUpperCase() + match.slice(1).toLowerCase();
    }
    return match.toLowerCase();
  });

  return result;
};

interface UseFormAutocompleteOptions {
  disableAutocomplete?: boolean;
}

const useFormAutocomplete = (options: UseFormAutocompleteOptions = {}) => {
  const { disableAutocomplete = false } = options;
  const [suggestion, setSuggestion] = useState("");
  const [isPending, startTransition] = useTransition();
  const [textareaHeight, setTextareaHeight] = useState("auto");
  const [overlayHeight, setOverlayHeight] = useState("auto");
  const [lastKnownHeight, setLastKnownHeight] = useState(0);
  const [lastAcceptedWordCount, setLastAcceptedWordCount] = useState(0);
  const [lastAcceptedPosition, setLastAcceptedPosition] = useState(0);
  const [justReplacedSpellCheckWord, setJustReplacedSpellCheckWord] = useState(false);
  const [isAutocompleteActive, setIsAutocompleteActive] = useState(false);
  const [lastAutocompleteRequest, setLastAutocompleteRequest] = useState<AbortController | null>(null);
  const [forceAutocompleteCheck, setForceAutocompleteCheck] = useState(0);
  const [lastSpellCheckCursorPos, setLastSpellCheckCursorPos] = useState<number | null>(null);
  const [savedSuggestion, setSavedSuggestion] = useState<string>("");
  const [immediateAutocomplete, setImmediateAutocomplete] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const measureRef = useRef<HTMLTextAreaElement>(null);

  // Simple state reset function for when textarea is emptied
  const resetAutocompleteState = () => {
    setSuggestion("");
    setLastAcceptedWordCount(0);
    setLastAcceptedPosition(0);
    setJustReplacedSpellCheckWord(false);
  };

  // Check if text is truly empty (handles whitespace-only content)
  const isTextEmpty = (text: string): boolean => {
    return !text || text.trim().length === 0;
  };

  // Callback to notify that a spell check word replacement occurred
  const notifySpellCheckReplacement = useCallback(() => {
    // Cancel any in-flight autocomplete request
    if (lastAutocompleteRequest) {
      lastAutocompleteRequest.abort();
      setLastAutocompleteRequest(null);
    }
    
    // Save the current suggestion before clearing
    if (suggestion) {
      setSavedSuggestion(suggestion);
    }
    
    setJustReplacedSpellCheckWord(true);
    setSuggestion(""); // Clear any existing suggestion immediately
    setIsAutocompleteActive(false);
    
    // Clear the flag and trigger immediate autocomplete check
    setTimeout(() => {
      setJustReplacedSpellCheckWord(false);
      
      // Restore the saved suggestion if it still exists
      if (savedSuggestion) {
        // Check if the text still makes sense for the saved suggestion
        if (textareaRef.current) {
          const currentText = textareaRef.current.value;
          const cursorPos = textareaRef.current.selectionStart || currentText.length;
          
          // Verify we still have at least 5 words
          const currentWordCount = getWordCountAtCursor(currentText, cursorPos);
          if (currentWordCount >= 5) {
            // Restore the suggestion
            setSuggestion(savedSuggestion);
            setSavedSuggestion(""); // Clear the saved suggestion
            
            // Don't update lastAcceptedPosition to preserve word count
            // This allows the autocomplete to remain visible
          }
        }
      } else {
        // Original behavior if no suggestion was saved
        if (textareaRef.current) {
          const currentText = textareaRef.current.value;
          const cursorPos = textareaRef.current.selectionStart || currentText.length;
          
          // Smart word count adjustment after spell check
          const currentWordCount = getWordCountAtCursor(currentText, cursorPos);
          if (currentWordCount >= 5) {
            // Adjust last accepted position to current cursor position
            // This allows autocomplete to resume naturally after spell check
            setLastAcceptedPosition(cursorPos);
            // Force autocomplete re-evaluation
            setForceAutocompleteCheck(prev => prev + 1);
          }
        }
      }
      // Clear the stored cursor position
      setLastSpellCheckCursorPos(null);
      
      // Force immediate autocomplete check if conditions are met
      if (textareaRef.current) {
        const currentText = textareaRef.current.value;
        const cursorPos = textareaRef.current.selectionStart || currentText.length;
        const wordCount = getWordCountAtCursor(currentText, cursorPos);
        
        if (wordCount >= 5) { // Consistent requirement after spell check
          // Trigger immediate autocomplete
          setImmediateAutocomplete(true);
          setForceAutocompleteCheck(prev => prev + 1);
        }
      }
    }, 150); // Balanced delay for spell check recovery
  }, [suggestion, savedSuggestion, lastAutocompleteRequest]);

  const {
    register,
    handleSubmit,
    formState: { errors },
    watch,
    setValue,
  } = useForm({
    defaultValues: {
      name: "",
      prompt: "",
    },
  });

  const promptValue = watch("prompt");
  
  // Smart debounce: 200ms after spell check, 1500ms normally (1.5 seconds for less aggressive autocomplete)
  const debounceDelay = justReplacedSpellCheckWord ? 200 : 1500;
  const [debouncedPrompt] = useDebounce(promptValue, debounceDelay);

  // Calculate textarea height based on content
  const calculateHeight = (text: string) => {
    if (!measureRef.current) return "auto";

    // Set the text and reset height to get accurate measurement
    measureRef.current.value = text;
    measureRef.current.style.height = "auto";

    // Force a reflow to ensure accurate scrollHeight
    void measureRef.current.offsetHeight;

    const scrollHeight = measureRef.current.scrollHeight;
    const calculatedHeight = Math.max(scrollHeight, 96); // Minimum 96px (4 rows)

    return `${calculatedHeight}px`;
  };

  // Clear suggestions when text is empty
  useEffect(() => {
    // If text is completely empty, reset autocomplete state
    if (isTextEmpty(promptValue)) {
      resetAutocompleteState();
    }
  }, [promptValue]);

  // ResizeObserver to monitor textarea size changes
  useLayoutEffect(() => {
    if (!textareaRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { height } = entry.contentRect;

        // Only update if height actually changed (prevent unnecessary updates)
        if (Math.abs(height - lastKnownHeight) > 0.5) {
          // 0.5px threshold for rounding
          const computedHeight = `${Math.max(height + 16, 96)}px`; // Add padding
          setLastKnownHeight(height);

          // Use double requestAnimationFrame for perfect synchronization
          requestAnimationFrame(() => {
            requestAnimationFrame(() => {
              setOverlayHeight(computedHeight);
            });
          });
        }
      }
    });

    resizeObserver.observe(textareaRef.current);

    return () => {
      resizeObserver.disconnect();
    };
  }, [lastKnownHeight]);

  // Update height when suggestion changes - use useLayoutEffect for immediate updates
  useLayoutEffect(() => {
    const fullText = promptValue + (suggestion ? " " + suggestion : "");
    const newHeight = calculateHeight(fullText);
    setTextareaHeight(newHeight);
  }, [promptValue, suggestion]);

  // Adjust suggestion capitalization based on sentence context
  const adjustSuggestionCapitalization = (
    userText: string,
    suggestion: string
  ): string => {
    if (!suggestion || !userText) return suggestion;

    // Make suggestion lowercase by default for natural flow
    const result = suggestion.toLowerCase();

    // Check if we're continuing after sentence-ending punctuation + space + new word
    const afterSentencePattern = /[.!?]\s+[a-zA-Z]+\s*$/;
    if (afterSentencePattern.test(userText)) {
      // We're in the middle of a new sentence, keep lowercase
      return result;
    }

    // Check if we're immediately after sentence ending (this shouldn't happen with our new logic)
    const endsWithSentence = /[.!?]\s*$/.test(userText.trim());
    if (endsWithSentence) {
      // Capitalize first word of new sentence
      return result.charAt(0).toUpperCase() + result.slice(1);
    }

    // For all other cases (continuing same sentence), keep lowercase
    return result;
  };

  // Word-based autocomplete logic - simple and effective
  useEffect(() => {
    // Skip if autocomplete is disabled (e.g., when kick detection is active)
    if (disableAutocomplete) {
      setSuggestion("");
      return;
    }
    
    // Skip if we just replaced a spell check word or if autocomplete is already active
    if (justReplacedSpellCheckWord || isAutocompleteActive) {
      setSuggestion("");
      return;
    }

    // Use prompt value directly for immediate autocomplete after spell check
    const textToCheck = immediateAutocomplete ? promptValue : debouncedPrompt;
    
    // Get current cursor position from the actual textarea element
    const cursorPos = textareaRef.current?.selectionStart ?? textareaRef.current?.selectionEnd ?? textToCheck.length;
    
    // Simple check for punctuation before cursor (natural pause point)
    const charBeforeCursor = cursorPos > 0 ? textToCheck[cursorPos - 1] : '';
    const afterPunctuation = ['.', '!', '?', ',', ';'].includes(charBeforeCursor);

    // Check if ready for suggestions
    const ready = isReadyForSuggestions(textToCheck, cursorPos, lastAcceptedWordCount, lastAcceptedPosition, afterPunctuation);
    
    
    // Clear suggestion if not ready
    if (!ready) {
      setSuggestion("");
      setIsAutocompleteActive(false);
      return;
    }

    // Cancel any previous request
    if (lastAutocompleteRequest) {
      lastAutocompleteRequest.abort();
    }

    // Create new abort controller for this request
    const abortController = new AbortController();
    setLastAutocompleteRequest(abortController);
    setIsAutocompleteActive(true);

    // Only start transition when we're definitely going to make an API call
    startTransition(async () => {
      try {
        const result = await askOllamaCompletationAction(textToCheck);
        
        // Check if request was aborted
        if (abortController.signal.aborted) {
          return;
        }
        
        if (result) {
          const processedSuggestion = adjustSuggestionCapitalization(
            textToCheck,
            result
          );
          setSuggestion(processedSuggestion);
        } else {
          setSuggestion("");
        }
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error("Autocomplete error:", error);
        }
        setSuggestion("");
      } finally {
        setIsAutocompleteActive(false);
        setLastAutocompleteRequest(null);
        // Clear immediate autocomplete flag
        if (immediateAutocomplete) {
          setImmediateAutocomplete(false);
        }
      }
    });
  }, [debouncedPrompt, promptValue, immediateAutocomplete, lastAcceptedWordCount, lastAcceptedPosition, justReplacedSpellCheckWord, forceAutocompleteCheck, disableAutocomplete]);

  const onSubmit: SubmitHandler<{ name: string; prompt: string }> = async (
    data
  ) => {
    // Handle form submission
    console.log("Form submitted:", data);
  };

  // Auto-capitalize the input text when it changes
  useEffect(() => {
    if (promptValue) {
      // Only apply corrections when user just finished typing a word (ends with space or punctuation)
      const shouldApplyFullCapitalization = /[\s.,!?;:]$/.test(promptValue);

      let processedValue = promptValue;

      // Apply full capitalization when word is complete
      if (shouldApplyFullCapitalization) {
        processedValue = autoCapitalizeText(promptValue);
      } else {
        // Apply only basic sentence capitalization while typing (first letter and after sentence endings)
        processedValue = applyBasicCapitalization(promptValue);
      }

      // Only update if capitalization changed to avoid infinite loops
      if (processedValue !== promptValue) {
        const cursorPosition = textareaRef.current?.selectionStart || 0;
        setValue("prompt", processedValue);

        // Restore cursor position after setValue
        setTimeout(() => {
          if (textareaRef.current) {
            textareaRef.current.setSelectionRange(
              cursorPosition,
              cursorPosition
            );
          }
        }, 0);
      }
    }
  }, [promptValue, setValue]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && suggestion && !disableAutocomplete) {
      e.preventDefault();
      const space = needsSpaceBeforeSuggestion(promptValue) ? " " : "";
      const newText = promptValue + space + suggestion;
      // Apply capitalization after accepting suggestion
      const processedText = autoCapitalizeText(newText);

      // Set processed text and track word count and position for future autocomplete blocking
      setValue("prompt", processedText);
      setSuggestion("");
      
      // Track word count and position at time of acceptance
      const cursorPos = textareaRef.current?.selectionStart || promptValue.length;
      setLastAcceptedWordCount(getWordCountAtCursor(processedText, processedText.length));
      setLastAcceptedPosition(processedText.length);
    }
  };

  return {
    register,
    handleSubmit,
    errors,
    onSubmit,
    textareaRef,
    measureRef,
    suggestion,
    isPending,
    handleKeyDown,
    setValue,
    setSuggestion,
    promptValue,
    textareaHeight,
    overlayHeight,
    needsSpaceBeforeSuggestion,
    notifySpellCheckReplacement,
    isAutocompleteActive,
  };
};

export default useFormAutocomplete;
