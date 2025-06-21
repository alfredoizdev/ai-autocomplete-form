import { askOllamaCompletationAction } from "@/actions/ai-text";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  useTransition,
  useMemo,
} from "react";
import { useForm, SubmitHandler } from "react-hook-form";
import { useDebounce } from "use-debounce";
import {
  autoCorrectText,
  autoCapitalizeText,
  applyBasicCapitalization,
  adjustSuggestionCapitalization,
  isReadyForSuggestions,
} from "@/utils/textProcessing";
import {
  MIN_TEXT_LENGTH,
  MIN_NEW_CHARS,
  MIN_TEXTAREA_HEIGHT,
  DEBOUNCE_DELAY,
  CORRECTION_TRIGGER,
} from "@/utils/constants/regexPatterns";

const useFormAutocomplete = () => {
  const [suggestion, setSuggestion] = useState("");
  const [isPending, startTransition] = useTransition();
  const [lastAcceptedLength, setLastAcceptedLength] = useState(0);
  const [previousTextLength, setPreviousTextLength] = useState(0);
  const [justDeleted, setJustDeleted] = useState(false);
  const [textareaHeight, setTextareaHeight] = useState("auto");
  const [conversationHistory, setConversationHistory] = useState<string[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const measureRef = useRef<HTMLTextAreaElement>(null);

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
  const [debouncedPrompt] = useDebounce(promptValue, DEBOUNCE_DELAY);

  // Memoized conversation memory reset function
  const resetConversationMemory = useCallback(() => {
    setConversationHistory([]);
  }, []);

  // Memoized height calculation function
  const calculateHeight = useCallback((text: string) => {
    if (!measureRef.current) return "auto";

    measureRef.current.value = text;
    measureRef.current.style.height = "auto";
    const scrollHeight = measureRef.current.scrollHeight;
    return `${Math.max(scrollHeight, MIN_TEXTAREA_HEIGHT)}px`;
  }, []);

  // Memoized text processing
  const processedPromptValue = useMemo(() => {
    if (!promptValue) return promptValue;

    // Only apply corrections when user just finished typing a word (ends with space or punctuation)
    const shouldCorrectText = CORRECTION_TRIGGER.test(promptValue);

    if (shouldCorrectText) {
      // Apply spell correction and then full capitalization when word is complete
      const corrected = autoCorrectText(promptValue);
      return autoCapitalizeText(corrected);
    } else {
      // Apply only basic sentence capitalization while typing
      return applyBasicCapitalization(promptValue);
    }
  }, [promptValue]);

  // Clear suggestions immediately when text is deleted or becomes too short
  useEffect(() => {
    // Clear suggestion immediately if text becomes too short or empty
    if (!promptValue || promptValue.length < MIN_TEXT_LENGTH) {
      setSuggestion("");
    }

    // Clear suggestion immediately if text was deleted
    if (promptValue.length < previousTextLength) {
      setSuggestion("");
    }

    // Reset conversation memory if user clears all text (starts fresh)
    if (!promptValue || promptValue.trim().length === 0) {
      if (conversationHistory.length > 0) {
        resetConversationMemory();
      }
    }
  }, [
    promptValue,
    previousTextLength,
    conversationHistory.length,
    resetConversationMemory,
  ]);

  // Update height when suggestion changes - memoized for performance
  useEffect(() => {
    const fullText = promptValue + (suggestion ? " " + suggestion : "");
    const newHeight = calculateHeight(fullText);
    setTextareaHeight(newHeight);
  }, [promptValue, suggestion, calculateHeight]);

  // Optimized suggestion generation effect
  useEffect(() => {
    // Detect if user has deleted text
    if (debouncedPrompt.length < previousTextLength) {
      // User deleted text, reset tracking and mark as just deleted
      setLastAcceptedLength(
        Math.max(0, debouncedPrompt.length - MIN_NEW_CHARS)
      );
      setJustDeleted(true);
      setSuggestion(""); // Clear any existing suggestions
      setPreviousTextLength(debouncedPrompt.length);
      return;
    }

    // If user has typed new content after deletion, allow suggestions again
    if (justDeleted && debouncedPrompt.length > previousTextLength) {
      setJustDeleted(false);
    }

    setPreviousTextLength(debouncedPrompt.length);

    if (!debouncedPrompt || debouncedPrompt.length < MIN_TEXT_LENGTH) {
      setSuggestion("");
      return;
    }

    // Don't suggest immediately after deletion - wait for new typing
    if (justDeleted) {
      setSuggestion("");
      return;
    }

    // Only generate new suggestions if user has typed enough new content since last acceptance
    if (debouncedPrompt.length < lastAcceptedLength + MIN_NEW_CHARS) {
      return;
    }

    // Only suggest if text is ready for suggestions (memoized function)
    if (!isReadyForSuggestions(debouncedPrompt)) {
      setSuggestion("");
      return;
    }

    startTransition(async () => {
      const result = await askOllamaCompletationAction(
        debouncedPrompt,
        conversationHistory
      );
      if (result) {
        // Check if we're in the middle of a sentence and adjust capitalization (memoized)
        const processedSuggestion = adjustSuggestionCapitalization(
          debouncedPrompt,
          result
        );
        setSuggestion(processedSuggestion);
      } else {
        setSuggestion("");
      }
    });
  }, [
    debouncedPrompt,
    lastAcceptedLength,
    previousTextLength,
    justDeleted,
    conversationHistory,
  ]);

  // Memoized form submission handler
  const onSubmit: SubmitHandler<{ name: string; prompt: string }> =
    useCallback(async () => {
      // Reset conversation memory after successful form submission
      if (conversationHistory.length > 0) {
        resetConversationMemory();
      }
    }, [conversationHistory.length, resetConversationMemory]);

  // Auto-correct and auto-capitalize the input text when it changes
  useEffect(() => {
    // Only update if corrections or capitalization changed to avoid infinite loops
    if (processedPromptValue !== promptValue) {
      const cursorPosition = textareaRef.current?.selectionStart || 0;
      setValue("prompt", processedPromptValue);

      // Restore cursor position after setValue
      setTimeout(() => {
        if (textareaRef.current) {
          textareaRef.current.setSelectionRange(cursorPosition, cursorPosition);
        }
      }, 0);
    }
  }, [processedPromptValue, promptValue, setValue]);

  // Memoized key down handler
  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Tab" && suggestion) {
        e.preventDefault();
        const space = promptValue && !promptValue.endsWith(" ") ? " " : "";
        const newText = promptValue + space + suggestion;

        // Apply spell correction and then capitalization
        const correctedText = autoCorrectText(newText);
        const capitalizedText = autoCapitalizeText(correctedText);

        setValue("prompt", capitalizedText);
        setLastAcceptedLength(capitalizedText.length);

        // Add accepted suggestion to conversation history
        setConversationHistory((prev) => [...prev, suggestion]);

        setSuggestion("");
      }
    },
    [suggestion, promptValue, setValue]
  );

  // Memoized return object to prevent unnecessary re-renders
  return useMemo(
    () => ({
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
      resetConversationMemory,
      conversationHistory,
    }),
    [
      register,
      handleSubmit,
      errors,
      onSubmit,
      suggestion,
      isPending,
      handleKeyDown,
      setValue,
      promptValue,
      textareaHeight,
      resetConversationMemory,
      conversationHistory,
    ]
  );
};

export default useFormAutocomplete;
