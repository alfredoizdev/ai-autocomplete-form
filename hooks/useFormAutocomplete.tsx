import { askOllamaCompletationAction } from "@/actions/ai-text";
import { useEffect, useRef, useState, useTransition } from "react";
import { useForm, SubmitHandler } from "react-hook-form";
import { useDebounce } from "use-debounce";

// Spell correction dictionary for common misspellings (moved outside component for stability)
const spellCorrections: Record<string, string> = {
  // Common dating/relationship words
  beatiful: "beautiful",
  beautifull: "beautiful",
  beutiful: "beautiful",
  awsome: "awesome",
  realy: "really",
  definately: "definitely",
  defintely: "definitely",
  adventorous: "adventurous",
  expereinced: "experienced",
  expirenced: "experienced",
  experiance: "experience",
  expiriance: "experience",
  freindly: "friendly",
  freinds: "friends",
  coupl: "couple",
  sexyy: "sexy",
  sexxy: "sexy",
  exciteing: "exciting",
  excting: "exciting",
  laidback: "laid back",
  outgoin: "outgoing",
  profesional: "professional",
  proffesional: "professional",
  profesionnal: "professional",
  iam: "I am",
  were: "we are",
  wer: "we are",
  lookking: "looking",
  loking: "looking",
  meetig: "meeting",
  meting: "meeting",
  playfull: "playful",
  discret: "discreet",
  discrette: "discreet",
};

// Auto-correct misspelled words (moved outside component for stability)
const autoCorrectText = (text: string): string => {
  if (!text) return text;

  // Split into words while preserving spaces and punctuation
  const words = text.split(/(\s+|[.,!?;:])/);

  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    if (word && /^[a-zA-Z]+$/.test(word)) {
      // Only check actual words
      const lowerWord = word.toLowerCase();
      if (spellCorrections[lowerWord]) {
        // Preserve original capitalization pattern
        const correction = spellCorrections[lowerWord];
        if (word[0] === word[0].toUpperCase()) {
          words[i] = correction.charAt(0).toUpperCase() + correction.slice(1);
        } else {
          words[i] = correction;
        }
      }
    }
  }

  return words.join("");
};

const useFormAutocomplete = () => {
  const [suggestion, setSuggestion] = useState("");
  const [isPending, startTransition] = useTransition();
  const [previousTextLength, setPreviousTextLength] = useState(0);
  const [justDeleted, setJustDeleted] = useState(false);
  const [isBlockedAfterAcceptance, setIsBlockedAfterAcceptance] = useState(false);
  const [baselineTextForCounting, setBaselineTextForCounting] = useState("");
  const [textareaHeight, setTextareaHeight] = useState("auto");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const measureRef = useRef<HTMLTextAreaElement>(null);

  // Complete state reset function for when textarea is emptied
  const resetAllAutocompleteState = () => {
    setSuggestion("");
    setJustDeleted(false);
    setIsBlockedAfterAcceptance(false);
    setBaselineTextForCounting("");
    setPreviousTextLength(0);
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

  // Simple word counting function for new words after baseline
  const countNewWordsAfterBaseline = (currentText: string, baseline: string): number => {
    if (!baseline || currentText.length < baseline.length) return 0;
    const newText = currentText.substring(baseline.length);
    return newText.trim().split(/\s+/).filter(word => word.length > 0).length;
  };

  // Calculate textarea height based on content
  const calculateHeight = (text: string) => {
    if (!measureRef.current) return "auto";

    measureRef.current.value = text;
    measureRef.current.style.height = "auto";
    const scrollHeight = measureRef.current.scrollHeight;
    return `${Math.max(scrollHeight, 96)}px`; // Minimum 4 rows (24px * 4)
  };

  // Clear suggestions immediately when text is deleted or becomes too short
  useEffect(() => {
    // If text is completely empty, reset all autocomplete state
    if (isTextEmpty(promptValue)) {
      resetAllAutocompleteState();
      return;
    }

    // Count words and clear suggestion if less than 3 words
    const words = promptValue
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0);
    if (words.length < 3) {
      setSuggestion("");
    }

    // Clear suggestion immediately if text was deleted
    if (promptValue.length < previousTextLength) {
      setSuggestion("");
    }
    
    // Clear suggestion if user just accepted and hasn't typed 3 new words
    if (isBlockedAfterAcceptance && baselineTextForCounting) {
      const newWords = countNewWordsAfterBaseline(promptValue, baselineTextForCounting);
      if (newWords < 3) {
        setSuggestion("");
      }
    }
  }, [promptValue, previousTextLength, isBlockedAfterAcceptance, baselineTextForCounting]);

  // Update height when suggestion changes
  useEffect(() => {
    const fullText = promptValue + (suggestion ? " " + suggestion : "");
    const newHeight = calculateHeight(fullText);
    setTextareaHeight(newHeight);
  }, [promptValue, suggestion]);

  // Check if text is ready for suggestions (not in middle of typing a word)
  const isReadyForSuggestions = (text: string): boolean => {
    // Early exit for empty text
    if (isTextEmpty(text)) return false;

    // Count actual words (split by whitespace and filter out empty strings)
    const words = text
      .trim()
      .split(/\s+/)
      .filter((word) => word.length > 0);

    // Require at least 3 words before offering suggestions
    if (words.length < 3) return false;

    // Check if we're at the end of a sentence (period/question/exclamation + space)
    const endsWithSentenceSpace = /[.!?]\s+$/.test(text);
    if (endsWithSentenceSpace) {
      // Don't suggest immediately after sentence endings - wait for user to start next sentence
      return false;
    }

    // Check if we just finished a sentence and user has started typing a new word
    const sentencePattern = /[.!?]\s+([a-zA-Z]+)$/;
    const sentenceMatch = text.match(sentencePattern);
    if (sentenceMatch) {
      // User has started typing after a sentence ending, check if word is complete enough
      const newWord = sentenceMatch[1];
      return newWord.length >= 3 && /[aeiouAEIOU]/.test(newWord);
    }

    // For non-sentence-ending cases, check normal word completion
    const lastChar = text[text.length - 1];
    if (/[\s,;:]/.test(lastChar)) {
      // Ready after comma, semicolon, colon, or regular space (but not after sentence endings)
      return true;
    }

    // If text doesn't end with space/punctuation, check if last "word" is reasonable length
    const lastWord = words[words.length - 1];

    // Allow suggestions if the last word is at least 3 characters and seems complete
    return lastWord.length >= 3 && /[aeiouAEIOU]/.test(lastWord);
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

    // Capitalize "I" when it's a standalone word (but be more specific)
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

  // Apply only basic capitalization while user is typing (no word corrections like "I" -> "I am")
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

  // Main autocomplete logic after debounce
  useEffect(() => {
    // If text is completely empty, reset all state and exit early
    if (isTextEmpty(debouncedPrompt)) {
      resetAllAutocompleteState();
      return;
    }

    // Detect if user has deleted text
    if (debouncedPrompt.length < previousTextLength) {
      // User deleted text, reset all tracking and mark as just deleted
      setJustDeleted(true);
      setIsBlockedAfterAcceptance(false);
      setBaselineTextForCounting("");
      setSuggestion(""); // Clear any existing suggestions
      setPreviousTextLength(debouncedPrompt.length);
      return;
    }

    // If user has typed new content after deletion, check if they have enough words to reset
    if (justDeleted && debouncedPrompt.length > previousTextLength) {
      // Count words to see if user has typed enough meaningful content
      const words = debouncedPrompt
        .trim()
        .split(/\s+/)
        .filter((word) => word.length > 0);
      if (words.length >= 3) {
        setJustDeleted(false);
      }
    }

    // Clear justDeleted flag if user starts typing from empty state
    if (justDeleted && previousTextLength === 0 && debouncedPrompt.length > 0) {
      setJustDeleted(false);
    }

    setPreviousTextLength(debouncedPrompt.length);

    // Don't suggest immediately after deletion - wait for new typing
    if (justDeleted) {
      setSuggestion("");
      return;
    }

    // Check if blocked after tab acceptance
    if (isBlockedAfterAcceptance) {
      const newWords = countNewWordsAfterBaseline(debouncedPrompt, baselineTextForCounting);
      if (newWords < 3) {
        setSuggestion("");
        return;
      }
      // User has typed 3+ new words, unblock autocomplete
      setIsBlockedAfterAcceptance(false);
      setBaselineTextForCounting("");
    }

    // Only suggest if text is ready for suggestions
    if (!isReadyForSuggestions(debouncedPrompt)) {
      setSuggestion("");
      return;
    }

    // Only start transition (show thinking message) when we're actually going to fetch
    startTransition(async () => {
      const result = await askOllamaCompletationAction(debouncedPrompt);
      if (result) {
        // Check if we're in the middle of a sentence and adjust capitalization
        const processedSuggestion = adjustSuggestionCapitalization(
          debouncedPrompt,
          result
        );
        setSuggestion(processedSuggestion);
      } else {
        setSuggestion("");
      }
    });
  }, [debouncedPrompt, previousTextLength, justDeleted, isBlockedAfterAcceptance, baselineTextForCounting]);

  const onSubmit: SubmitHandler<{ name: string; prompt: string }> = async (
    data
  ) => {
    console.log(data);
  };

  // Auto-correct and auto-capitalize the input text when it changes
  useEffect(() => {
    if (promptValue) {
      // Only apply corrections when user just finished typing a word (ends with space or punctuation)
      const shouldCorrectText = /[\s.,!?;:]$/.test(promptValue);

      let processedValue = promptValue;

      // Apply spell correction only when word is complete
      if (shouldCorrectText) {
        processedValue = autoCorrectText(promptValue);
        // Apply full capitalization only when word is complete
        processedValue = autoCapitalizeText(processedValue);
      } else {
        // Apply only basic sentence capitalization while typing (first letter and after sentence endings)
        processedValue = applyBasicCapitalization(promptValue);
      }

      // Only update if corrections or capitalization changed to avoid infinite loops
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
    if (e.key === "Tab" && suggestion) {
      e.preventDefault();
      const space = promptValue && !promptValue.endsWith(" ") ? " " : "";
      const newText = promptValue + space + suggestion;
      // Apply spell correction and then capitalization
      const correctedText = autoCorrectText(newText);
      const capitalizedText = autoCapitalizeText(correctedText);
      setValue("prompt", capitalizedText);
      
      // Block autocomplete and set baseline for counting new words
      setIsBlockedAfterAcceptance(true);
      setBaselineTextForCounting(capitalizedText);
      setSuggestion("");
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
  };
};

export default useFormAutocomplete;
