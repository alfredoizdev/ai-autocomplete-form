"use client";

import Image from "next/image";
import useFormAutocompleteOptimized from "@/hooks/useFormAutocompleteOptimized";
import { memo } from "react";

// Memoized suggestion overlay for performance
const SuggestionOverlay = memo(({ 
  promptValue, 
  suggestion, 
  overlayHeight,
  needsSpace 
}: {
  promptValue: string;
  suggestion: string;
  overlayHeight: string;
  needsSpace: boolean;
}) => {
  if (!suggestion) return null;
  
  return (
    <div
      className="absolute inset-0 pointer-events-none overflow-hidden whitespace-pre-wrap break-words"
      style={{ height: overlayHeight }}
    >
      <span className="invisible">{promptValue}</span>
      {needsSpace && <span className="invisible"> </span>}
      <span className="text-gray-400 animate-fade-in">{suggestion}</span>
    </div>
  );
});

SuggestionOverlay.displayName = 'SuggestionOverlay';

const FormOptimized = () => {
  const {
    register,
    handleSubmit,
    errors,
    onSubmit,
    textareaRef,
    measureRef,
    suggestion,
    isPending,
    handleKeyDown,
    textareaHeight,
    overlayHeight,
    promptValue,
    needsSpaceBeforeSuggestion,
  } = useFormAutocompleteOptimized();

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="flex flex-col gap-4 w-full max-w-[500px] py-6 px-10 bg-white border border-gray-200 rounded-[10px] mx-auto"
    >
      {/* Logo */}
      <div className="flex justify-center mb-1">
        <Image
          src="/images/logo-swing.svg"
          alt="Swing Logo"
          width={120}
          height={40}
          className="h-20 w-auto"
          priority
        />
      </div>

      {/* Header and description */}
      <div className="text-center mb-3">
        <h2 className="text-2xl font-semibold text-gray-900 mb-3">
          Create Bio
        </h2>
        <p className="text-gray-500 text-sm">
          Let AI help you craft the perfect bio with ultra-fast streaming suggestions
        </p>
      </div>

      {/* Name Field */}
      <div className="mb-3">
        <label
          htmlFor="name"
          className="block text-sm font-semibold text-gray-700"
        >
          Name:
        </label>
        <input
          {...register("name", {
            required: "Name is required",
            maxLength: {
              value: 50,
              message: "Name cannot exceed 50 characters",
            },
          })}
          type="text"
          id="name"
          className="mt-1 text-gray-900 placeholder:text-gray-400 block w-full h-[46px] p-2 border border-gray-200 rounded-[8px] focus:border-black active:border-black focus:outline-none transition-all duration-300 ease-out"
          style={{
            WebkitAppearance: "none",
            appearance: "none",
            fontSize: "16px",
          }}
          placeholder="Enter your name"
        />
        {errors.name && (
          <span className="text-red-500 text-sm">{errors.name.message}</span>
        )}
      </div>

      {/* Bio Textarea with Autocomplete */}
      <div className="mb-1">
        <label
          htmlFor="prompt"
          className="block text-sm font-semibold text-gray-700"
        >
          Bio:
        </label>
        <div className="relative mt-1">
          <textarea
            {...register("prompt", {
              required: "Bio text is required",
              minLength: {
                value: 10,
                message: "Bio must be at least 10 characters",
              },
              maxLength: {
                value: 500,
                message: "Bio cannot exceed 500 characters",
              },
            })}
            ref={textareaRef}
            id="prompt"
            className="block w-full p-3 min-h-[96px] border border-gray-200 rounded-[8px] focus:border-black active:border-black focus:outline-none transition-all duration-300 ease-out resize-none text-gray-900 placeholder:text-gray-400 relative z-10 bg-transparent"
            style={{
              WebkitAppearance: "none",
              appearance: "none",
              fontSize: "16px",
              lineHeight: "1.5",
              height: textareaHeight,
            }}
            placeholder="Start typing your bio..."
            rows={4}
            onKeyDown={handleKeyDown}
          />
          
          {/* Streaming Suggestion Overlay */}
          <SuggestionOverlay
            promptValue={promptValue}
            suggestion={suggestion}
            overlayHeight={overlayHeight}
            needsSpace={needsSpaceBeforeSuggestion(promptValue)}
          />
          
          {/* Hidden measure textarea */}
          <textarea
            ref={measureRef}
            className="absolute invisible pointer-events-none"
            style={{
              width: "100%",
              padding: "12px",
              fontSize: "16px",
              lineHeight: "1.5",
              border: "1px solid transparent",
            }}
            tabIndex={-1}
            aria-hidden="true"
          />
        </div>
        
        {errors.prompt && (
          <span className="text-red-500 text-sm">
            {errors.prompt.message}
          </span>
        )}
        
        {/* Performance indicator */}
        {isPending && (
          <div className="text-xs text-gray-500 mt-1 animate-pulse">
            AI is thinking...
          </div>
        )}
        
        {/* Tab hint for autocomplete */}
        {suggestion && (
          <div className="text-xs text-gray-500 mt-1">
            Press <kbd className="px-1.5 py-0.5 text-xs bg-gray-100 rounded">Tab</kbd> to accept suggestion
          </div>
        )}
      </div>

      {/* Submit Button */}
      <button
        type="submit"
        className="w-full py-3 mt-2 bg-black text-white font-medium rounded-[8px] hover:bg-gray-800 transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        disabled={isPending}
      >
        {isPending ? "Processing..." : "Submit Bio"}
      </button>
    </form>
  );
};

export default FormOptimized;