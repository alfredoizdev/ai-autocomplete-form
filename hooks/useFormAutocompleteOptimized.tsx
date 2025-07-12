import { streamOllamaCompletion } from "@/actions/ai-text-streaming";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useTransition,
  useCallback,
  useMemo,
  memo,
  useDeferredValue,
} from "react";
import { useForm, SubmitHandler } from "react-hook-form";
import { useDebounce } from "use-debounce";

// Memoized helper functions
const getCurrentSentenceAtCursor = memo((
  text: string,
  cursorPos: number = text.length
): string => {
  if (!text || text.trim().length === 0) return "";

  let sentenceStart = 0;
  for (let i = cursorPos - 1; i >= 0; i--) {
    if (/[.!?]/.test(text[i])) {
      sentenceStart = i + 1;
      while (sentenceStart < text.length && /\s/.test(text[sentenceStart])) {
        sentenceStart++;
      }
      break;
    }
  }

  const currentSentence = text.substring(sentenceStart, cursorPos).trim();
  return currentSentence;
});

const getWordsBeforeCursor = memo((
  text: string,
  cursorPos: number = text.length
): string[] => {
  const currentSentence = getCurrentSentenceAtCursor(text, cursorPos);

  if (!currentSentence) return [];

  return currentSentence
    .trim()
    .split(/\s+/)
    .filter((word) => word.length > 0);
});

const getWordCountAtCursor = memo((
  text: string,
  cursorPos: number = text.length
): number => {
  if (!text || text.trim().length === 0) return 0;
  
  const textToCursor = text.substring(0, cursorPos).trim();
  if (!textToCursor) return 0;
  
  return textToCursor.split(/\s+/).filter(word => word.length > 0).length;
});

const getWordCountSincePosition = memo((
  text: string,
  sincePosition: number,
  cursorPos: number = text.length
): number => {
  if (!text || sincePosition >= cursorPos) return 0;
  
  const textSincePosition = text.substring(sincePosition, cursorPos).trim();
  if (!textSincePosition) return 0;
  
  return textSincePosition.split(/\s+/).filter(word => word.length > 0).length;
});

const isReadyForSuggestions = memo((
  text: string,
  cursorPos: number = text.length,
  lastAcceptedWordCount: number = 0,
  lastAcceptedPosition: number = 0,
  afterPunctuation: boolean = false
): boolean => {
  if (!text || text.trim().length === 0) return false;

  const words = getWordsBeforeCursor(text, cursorPos);
  
  // Reduced word requirement for better responsiveness
  const minWordsRequired = afterPunctuation ? 3 : 4;
  
  if (words.length < minWordsRequired) return false;
  
  if (lastAcceptedWordCount === 0) return true;
  
  // Reduced to 3 new words for more frequent suggestions
  const newWordsTyped = getWordCountSincePosition(text, lastAcceptedPosition, cursorPos);
  return newWordsTyped >= 3;
});

const needsSpaceBeforeSuggestion = memo((text: string): boolean => {
  if (!text) return false;
  if (/\s$/.test(text)) return false;
  if (/[.,!?;:]$/.test(text)) return true;
  if (/\w$/.test(text)) return true;
  return false;
});

const applyBasicCapitalization = memo((text: string): string => {
  if (!text) return text;

  let result = text;

  if (result.length > 0) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  result = result.replace(
    /([.!?]\s+)([a-z])/g,
    (match, punctuation, letter) => {
      return punctuation + letter.toUpperCase();
    }
  );

  result = result.replace(/\b(i)\s/g, "I ");

  return result;
});

const autoCapitalizeText = memo((text: string): string => {
  if (!text) return text;

  let result = text;

  if (result.length > 0) {
    result = result.charAt(0).toUpperCase() + result.slice(1);
  }

  result = result.replace(
    /([.!?]\s+)([a-z])/g,
    (match, punctuation, letter) => {
      return punctuation + letter.toUpperCase();
    }
  );

  result = result.replace(/\b(i)\b/g, "I");

  result = result.replace(/\b(we|us|our)\b/gi, (match) => {
    const beforeMatch = result.substring(0, result.indexOf(match));
    if (beforeMatch === "" || /[.!?]\s*$/.test(beforeMatch)) {
      return match.charAt(0).toUpperCase() + match.slice(1).toLowerCase();
    }
    return match.toLowerCase();
  });

  return result;
});

interface UseFormAutocompleteOptions {
  disableAutocomplete?: boolean;
}

const useFormAutocompleteOptimized = (options: UseFormAutocompleteOptions = {}) => {
  const { disableAutocomplete = false } = options;
  const [suggestion, setSuggestion] = useState("");
  const [streamingSuggestion, setStreamingSuggestion] = useState("");
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
  const [savedSuggestion, setSavedSuggestion] = useState<string>("");
  const [immediateAutocomplete, setImmediateAutocomplete] = useState(false);
  const [typingSpeed, setTypingSpeed] = useState<number>(0);
  const [lastTypingTime, setLastTypingTime] = useState<number>(Date.now());
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const measureRef = useRef<HTMLTextAreaElement>(null);

  const resetAutocompleteState = useCallback(() => {
    setSuggestion("");
    setStreamingSuggestion("");
    setLastAcceptedWordCount(0);
    setLastAcceptedPosition(0);
    setJustReplacedSpellCheckWord(false);
  }, []);

  const isTextEmpty = useCallback((text: string): boolean => {
    return !text || text.trim().length === 0;
  }, []);

  const notifySpellCheckReplacement = useCallback(() => {
    if (lastAutocompleteRequest) {
      lastAutocompleteRequest.abort();
      setLastAutocompleteRequest(null);
    }
    
    if (suggestion || streamingSuggestion) {
      setSavedSuggestion(suggestion || streamingSuggestion);
    }
    
    setJustReplacedSpellCheckWord(true);
    setSuggestion("");
    setStreamingSuggestion("");
    setIsAutocompleteActive(false);
    
    setTimeout(() => {
      setJustReplacedSpellCheckWord(false);
      
      if (savedSuggestion && textareaRef.current) {
        const currentText = textareaRef.current.value;
        const cursorPos = textareaRef.current.selectionStart || currentText.length;
        
        const currentWordCount = getWordCountAtCursor(currentText, cursorPos);
        if (currentWordCount >= 4) {
          setSuggestion(savedSuggestion);
          setSavedSuggestion("");
        }
      } else if (textareaRef.current) {
        const currentText = textareaRef.current.value;
        const cursorPos = textareaRef.current.selectionStart || currentText.length;
        
        const currentWordCount = getWordCountAtCursor(currentText, cursorPos);
        if (currentWordCount >= 4) {
          setLastAcceptedPosition(cursorPos);
          setForceAutocompleteCheck(prev => prev + 1);
        }
      }
      
      if (textareaRef.current) {
        const currentText = textareaRef.current.value;
        const cursorPos = textareaRef.current.selectionStart || currentText.length;
        const wordCount = getWordCountAtCursor(currentText, cursorPos);
        
        if (wordCount >= 4) {
          setImmediateAutocomplete(true);
          setForceAutocompleteCheck(prev => prev + 1);
        }
      }
    }, 100); // Reduced delay for faster recovery
  }, [suggestion, streamingSuggestion, savedSuggestion, lastAutocompleteRequest]);

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
  
  // Adaptive debouncing based on typing speed and context
  const calculateDebounceDelay = useMemo(() => {
    if (justReplacedSpellCheckWord) return 100; // Very fast after spell check
    if (immediateAutocomplete) return 50; // Almost instant for immediate mode
    
    // Calculate typing speed (words per minute)
    const now = Date.now();
    const timeDiff = now - lastTypingTime;
    const wordsTyped = promptValue.split(/\s+/).filter(w => w.length > 0).length;
    const wpm = timeDiff > 0 ? (wordsTyped / timeDiff) * 60000 : 0;
    
    // Adaptive delay based on typing speed
    if (wpm > 80) return 200; // Fast typer - quick response
    if (wpm > 50) return 300; // Medium typer
    return 400; // Slower typer or paused
  }, [justReplacedSpellCheckWord, immediateAutocomplete, lastTypingTime, promptValue]);
  
  const [debouncedPrompt] = useDebounce(promptValue, calculateDebounceDelay);
  
  // Use deferred value for non-critical updates
  const deferredSuggestion = useDeferredValue(streamingSuggestion || suggestion);

  // Memoized height calculation
  const calculateHeight = useCallback((text: string) => {
    if (!measureRef.current) return "auto";

    measureRef.current.value = text;
    measureRef.current.style.height = "auto";

    void measureRef.current.offsetHeight;

    const scrollHeight = measureRef.current.scrollHeight;
    const calculatedHeight = Math.max(scrollHeight, 96);

    return `${calculatedHeight}px`;
  }, []);

  // Track typing speed
  useEffect(() => {
    setLastTypingTime(Date.now());
  }, [promptValue]);

  useEffect(() => {
    if (isTextEmpty(promptValue)) {
      resetAutocompleteState();
    }
  }, [promptValue, isTextEmpty, resetAutocompleteState]);

  useLayoutEffect(() => {
    if (!textareaRef.current) return;

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { height } = entry.contentRect;

        if (Math.abs(height - lastKnownHeight) > 0.5) {
          const computedHeight = `${Math.max(height + 16, 96)}px`;
          setLastKnownHeight(height);

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

  useLayoutEffect(() => {
    const fullText = promptValue + (deferredSuggestion ? " " + deferredSuggestion : "");
    const newHeight = calculateHeight(fullText);
    setTextareaHeight(newHeight);
  }, [promptValue, deferredSuggestion, calculateHeight]);

  const adjustSuggestionCapitalization = useCallback((
    userText: string,
    suggestion: string
  ): string => {
    if (!suggestion || !userText) return suggestion;

    const result = suggestion.toLowerCase();

    const afterSentencePattern = /[.!?]\s+[a-zA-Z]+\s*$/;
    if (afterSentencePattern.test(userText)) {
      return result;
    }

    const endsWithSentence = /[.!?]\s*$/.test(userText.trim());
    if (endsWithSentence) {
      return result.charAt(0).toUpperCase() + result.slice(1);
    }

    return result;
  }, []);

  // Streaming autocomplete with performance optimizations
  useEffect(() => {
    if (disableAutocomplete) {
      setSuggestion("");
      setStreamingSuggestion("");
      return;
    }
    
    if (justReplacedSpellCheckWord || isAutocompleteActive) {
      setSuggestion("");
      setStreamingSuggestion("");
      return;
    }

    const textToCheck = immediateAutocomplete ? promptValue : debouncedPrompt;
    
    const cursorPos = textareaRef.current?.selectionStart ?? textareaRef.current?.selectionEnd ?? textToCheck.length;
    
    const charBeforeCursor = cursorPos > 0 ? textToCheck[cursorPos - 1] : '';
    const afterPunctuation = ['.', '!', '?', ',', ';'].includes(charBeforeCursor);

    const ready = isReadyForSuggestions(textToCheck, cursorPos, lastAcceptedWordCount, lastAcceptedPosition, afterPunctuation);
    
    if (!ready) {
      setSuggestion("");
      setStreamingSuggestion("");
      setIsAutocompleteActive(false);
      return;
    }

    if (lastAutocompleteRequest) {
      lastAutocompleteRequest.abort();
    }

    const abortController = new AbortController();
    setLastAutocompleteRequest(abortController);
    setIsAutocompleteActive(true);

    startTransition(async () => {
      try {
        setStreamingSuggestion("");
        let accumulated = "";
        
        for await (const chunk of streamOllamaCompletion(textToCheck)) {
          if (abortController.signal.aborted) break;
          
          if (chunk.done) {
            if (accumulated) {
              const processedSuggestion = adjustSuggestionCapitalization(
                textToCheck,
                accumulated
              );
              setSuggestion(processedSuggestion);
              setStreamingSuggestion("");
            }
            break;
          } else {
            accumulated += chunk.text;
            const processedSuggestion = adjustSuggestionCapitalization(
              textToCheck,
              accumulated
            );
            setStreamingSuggestion(processedSuggestion);
          }
        }
      } catch (error: any) {
        if (error.name !== 'AbortError') {
          console.error("Autocomplete error:", error);
        }
        setSuggestion("");
        setStreamingSuggestion("");
      } finally {
        setIsAutocompleteActive(false);
        setLastAutocompleteRequest(null);
        if (immediateAutocomplete) {
          setImmediateAutocomplete(false);
        }
      }
    });
  }, [debouncedPrompt, promptValue, immediateAutocomplete, lastAcceptedWordCount, lastAcceptedPosition, justReplacedSpellCheckWord, forceAutocompleteCheck, disableAutocomplete, adjustSuggestionCapitalization]);

  const onSubmit: SubmitHandler<{ name: string; prompt: string }> = async (
    data
  ) => {
    console.log("Form submitted:", data);
  };

  useEffect(() => {
    if (promptValue) {
      const shouldApplyFullCapitalization = /[\s.,!?;:]$/.test(promptValue);

      let processedValue = promptValue;

      if (shouldApplyFullCapitalization) {
        processedValue = autoCapitalizeText(promptValue);
      } else {
        processedValue = applyBasicCapitalization(promptValue);
      }

      if (processedValue !== promptValue) {
        const cursorPosition = textareaRef.current?.selectionStart || 0;
        setValue("prompt", processedValue);

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

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && (suggestion || streamingSuggestion) && !disableAutocomplete) {
      e.preventDefault();
      const finalSuggestion = suggestion || streamingSuggestion;
      const space = needsSpaceBeforeSuggestion(promptValue) ? " " : "";
      const newText = promptValue + space + finalSuggestion;
      const processedText = autoCapitalizeText(newText);

      setValue("prompt", processedText);
      setSuggestion("");
      setStreamingSuggestion("");
      
      const cursorPos = textareaRef.current?.selectionStart || promptValue.length;
      setLastAcceptedWordCount(getWordCountAtCursor(processedText, processedText.length));
      setLastAcceptedPosition(processedText.length);
    }
  }, [suggestion, streamingSuggestion, disableAutocomplete, promptValue, setValue]);

  return {
    register,
    handleSubmit,
    errors,
    onSubmit,
    textareaRef,
    measureRef,
    suggestion: deferredSuggestion,
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

export default useFormAutocompleteOptimized;