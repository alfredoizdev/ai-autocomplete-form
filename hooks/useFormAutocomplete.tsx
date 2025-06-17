import { askOllamaCompletationAction } from "@/actions/ai";
import { useEffect, useRef, useState, useTransition } from "react";
import { useForm, SubmitHandler } from "react-hook-form";
import { useDebounce } from "use-debounce";

const useFormAutocomplete = () => {
  const [suggestion, setSuggestion] = useState("");
  const [isPending, startTransition] = useTransition();
  const [lastAcceptedLength, setLastAcceptedLength] = useState(0);
  const [previousTextLength, setPreviousTextLength] = useState(0);
  const [justDeleted, setJustDeleted] = useState(false);
  const [textareaHeight, setTextareaHeight] = useState("auto");
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
  const [debouncedPrompt] = useDebounce(promptValue, 1000);

  // Calculate textarea height based on content
  const calculateHeight = (text: string) => {
    if (!measureRef.current) return "auto";

    measureRef.current.value = text;
    measureRef.current.style.height = "auto";
    const scrollHeight = measureRef.current.scrollHeight;
    return `${Math.max(scrollHeight, 96)}px`; // Minimum 4 rows (24px * 4)
  };

  // Update height when suggestion changes
  useEffect(() => {
    const fullText = promptValue + (suggestion ? " " + suggestion : "");
    const newHeight = calculateHeight(fullText);
    setTextareaHeight(newHeight);
  }, [promptValue, suggestion]);

  // Check if text is ready for suggestions (not in middle of typing a word)
  const isReadyForSuggestions = (text: string): boolean => {
    if (!text) return false;

    // If text ends with space or punctuation, it's ready
    const lastChar = text[text.length - 1];
    if (/[\s.,!?;:]/.test(lastChar)) {
      return true;
    }

    // If text doesn't end with space/punctuation, check if last "word" is reasonable length
    // This allows suggestions after complete words even without trailing space
    const words = text.trim().split(/\s+/);
    const lastWord = words[words.length - 1];

    // Allow suggestions if the last word is at least 3 characters and seems complete
    // (contains vowels or common word patterns)
    return lastWord.length >= 3 && /[aeiouAEIOU]/.test(lastWord);
  };

  // Obtener sugerencia después del debounce
  useEffect(() => {
    // Detect if user has deleted text
    if (debouncedPrompt.length < previousTextLength) {
      // User deleted text, reset tracking and mark as just deleted
      setLastAcceptedLength(Math.max(0, debouncedPrompt.length - 2));
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

    if (!debouncedPrompt || debouncedPrompt.length < 10) {
      setSuggestion("");
      return;
    }

    // Don't suggest immediately after deletion - wait for new typing
    if (justDeleted) {
      setSuggestion("");
      return;
    }

    // Only generate new suggestions if user has typed enough new content since last acceptance
    // Require at least 2 new characters to prevent immediate re-suggestions
    if (debouncedPrompt.length < lastAcceptedLength + 2) {
      return;
    }

    // Only suggest if text is ready for suggestions
    if (!isReadyForSuggestions(debouncedPrompt)) {
      setSuggestion("");
      return;
    }

    startTransition(async () => {
      const result = await askOllamaCompletationAction(debouncedPrompt);
      setSuggestion(result || "");
    });
  }, [debouncedPrompt, lastAcceptedLength, previousTextLength, justDeleted]);

  const onSubmit: SubmitHandler<{ name: string; prompt: string }> = async (
    data
  ) => {
    console.log(data);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && suggestion) {
      e.preventDefault();
      const space =
        promptValue &&
        !promptValue.endsWith(" ") &&
        !/[.,!?;:]$/.test(promptValue)
          ? " "
          : "";
      const newText = promptValue + space + suggestion;
      setValue("prompt", newText);
      setLastAcceptedLength(newText.length);
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
