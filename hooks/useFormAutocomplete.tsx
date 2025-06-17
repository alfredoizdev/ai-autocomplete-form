import { askOllamaCompletationAction } from "@/actions/ai";
import { useEffect, useRef, useState, useTransition } from "react";
import { useForm, SubmitHandler } from "react-hook-form";
import { useDebounce } from "use-debounce";

const useFormAutocomplete = () => {
  const [suggestion, setSuggestion] = useState("");
  const [isPending, startTransition] = useTransition();
  const [lastAcceptedLength, setLastAcceptedLength] = useState(0);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

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

  // Obtener sugerencia después del debounce
  useEffect(() => {
    if (!debouncedPrompt || debouncedPrompt.length < 10) {
      setSuggestion("");
      return;
    }

    // Only generate new suggestions if user has typed beyond the last accepted suggestion
    if (debouncedPrompt.length <= lastAcceptedLength) {
      return;
    }

    startTransition(async () => {
      const result = await askOllamaCompletationAction(debouncedPrompt);
      setSuggestion(result || "");
    });
  }, [debouncedPrompt, lastAcceptedLength]);

  const onSubmit: SubmitHandler<{ name: string; prompt: string }> = async (
    data
  ) => {
    console.log(data);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab" && suggestion) {
      e.preventDefault();
      const newText = promptValue + " " + suggestion;
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
    suggestion,
    isPending,
    handleKeyDown,
    setValue,
    setSuggestion,
    promptValue,
  };
};

export default useFormAutocomplete;
