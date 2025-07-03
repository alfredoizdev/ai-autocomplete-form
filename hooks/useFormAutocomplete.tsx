import { askOllamaCompletationAction } from "@/actions/ai-text";
import {
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  useTransition,
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

// Check if we're ready for autocomplete suggestions
const isReadyForSuggestions = (
  text: string,
  cursorPos: number = text.length
): boolean => {
  if (!text || text.trim().length === 0) return false;

  const words = getWordsBeforeCursor(text, cursorPos);

  // Must have at least 3 words in current sentence
  return words.length >= 3;
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

const useFormAutocomplete = () => {
  const [suggestion, setSuggestion] = useState("");
  const [isPending, startTransition] = useTransition();
  const [textareaHeight, setTextareaHeight] = useState("auto");
  const [overlayHeight, setOverlayHeight] = useState("auto");
  const [lastKnownHeight, setLastKnownHeight] = useState(0);
  const [justAcceptedSuggestion, setJustAcceptedSuggestion] = useState(false);
  const [lastAcceptedTextLength, setLastAcceptedTextLength] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const measureRef = useRef<HTMLTextAreaElement>(null);

  // Simple state reset function for when textarea is emptied
  const resetAutocompleteState = () => {
    setSuggestion("");
    setJustAcceptedSuggestion(false);
    setLastAcceptedTextLength(0);
  };

  // Check if text is truly empty (handles whitespace-only content)
  const isTextEmpty = (text: string): boolean => {
    return !text || text.trim().length === 0;
  };

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
  const [debouncedPrompt] = useDebounce(promptValue, 2000);

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

  // Simplified autocomplete logic - triggers for ANY sentence with 3+ words
  useEffect(() => {
    // Skip if we just accepted a suggestion
    if (justAcceptedSuggestion) {
      console.log("⏸️ Skipping autocomplete - just accepted suggestion");
      return;
    }

    // Skip if text hasn't grown beyond the last accepted suggestion (but allow empty text to reset)
    if (
      debouncedPrompt.length > 0 &&
      debouncedPrompt.length <= lastAcceptedTextLength
    ) {
      console.log(
        "⏸️ Skipping autocomplete - text hasn't grown beyond accepted suggestion"
      );
      return;
    }

    // Get current cursor position (defaults to end of text)
    const cursorPos =
      textareaRef.current?.selectionStart || debouncedPrompt.length;

    // Debug logging
    console.log("🔍 Autocomplete check:", {
      text: debouncedPrompt,
      cursorPos,
      currentSentence: getCurrentSentenceAtCursor(debouncedPrompt, cursorPos),
      words: getWordsBeforeCursor(debouncedPrompt, cursorPos),
      wordCount: getWordsBeforeCursor(debouncedPrompt, cursorPos).length,
      isReady: isReadyForSuggestions(debouncedPrompt, cursorPos),
    });

    // Clear suggestion if not ready
    if (!isReadyForSuggestions(debouncedPrompt, cursorPos)) {
      setSuggestion("");
      return;
    }

    // Only start transition when we're definitely going to make an API call
    console.log("✅ Starting autocomplete for:", debouncedPrompt);
    startTransition(async () => {
      const result = await askOllamaCompletationAction(debouncedPrompt);
      if (result) {
        const processedSuggestion = adjustSuggestionCapitalization(
          debouncedPrompt,
          result
        );
        setSuggestion(processedSuggestion);
        console.log("💡 Suggestion received:", processedSuggestion);
      } else {
        setSuggestion("");
        console.log("❌ No suggestion received");
      }
    });
  }, [debouncedPrompt, justAcceptedSuggestion, lastAcceptedTextLength]);

  const onSubmit: SubmitHandler<{ name: string; prompt: string }> = async (
    data
  ) => {
    console.log(data);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && suggestion) {
      e.preventDefault();
      const space = needsSpaceBeforeSuggestion(promptValue) ? " " : "";
      const newText = promptValue + space + suggestion;

      // Set new text and block future autocomplete
      setValue("prompt", newText);
      setSuggestion("");
      setJustAcceptedSuggestion(true);
      setLastAcceptedTextLength(newText.length);
      console.log("✅ Suggestion accepted:", suggestion);

      // Clear the block after delay that exceeds debounce time
      setTimeout(() => {
        setJustAcceptedSuggestion(false);
      }, 3000);
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
  };
};

export default useFormAutocomplete;
