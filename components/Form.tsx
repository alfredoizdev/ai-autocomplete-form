"use client";

import Image from "next/image";
import useFormAutocomplete from "@/hooks/useFormAutocomplete";
import useSpellCheck from "@/hooks/useSpellCheck";
import SpellCheckPopup from "./SpellCheckPopup";
import { useState, useEffect } from "react";

const Form = () => {
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
    setValue,
    textareaHeight,
    overlayHeight,
    promptValue,
    needsSpaceBeforeSuggestion,
  } = useFormAutocomplete();

  const { getMisspelledWords, isLoading: spellCheckLoading, getSuggestions } = useSpellCheck();

  // State for spell check popup
  const [showPopup, setShowPopup] = useState(false);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [selectedWord, setSelectedWord] = useState("");
  const [wordSuggestions, setWordSuggestions] = useState<string[]>([]);

  // Get misspelled words for current text
  const misspelledWords = getMisspelledWords(promptValue);

  // Handle word replacement
  const replaceWord = (originalWord: string, newWord: string) => {
    if (!promptValue || !textareaRef.current) return;
    
    // Create a regex that matches the exact word with word boundaries
    const regex = new RegExp(`\\b${originalWord}\\b`, 'g');
    const newText = promptValue.replace(regex, newWord);
    
    // Update the form value
    setValue("prompt", newText);
    
    // Close popup
    setShowPopup(false);
    
    // Focus back on textarea
    setTimeout(() => {
      textareaRef.current?.focus();
    }, 0);
  };

  // Handle clicking on misspelled words
  const handleWordClick = (event: React.MouseEvent, word: string) => {
    event.preventDefault();
    event.stopPropagation();
    
    // Get click position
    const rect = event.currentTarget.getBoundingClientRect();
    let x = rect.left + rect.width / 2;
    let y = rect.top;
    
    // Prevent popup from going off-screen
    const popupWidth = 250;
    const popupHeight = 200;
    
    // Adjust horizontal position
    if (x + popupWidth / 2 > window.innerWidth) {
      x = window.innerWidth - popupWidth / 2 - 20;
    }
    if (x - popupWidth / 2 < 0) {
      x = popupWidth / 2 + 20;
    }
    
    // Adjust vertical position (show above the word)
    if (y - popupHeight < 0) {
      y = rect.bottom + 10; // Show below if not enough space above
    } else {
      y = y - 10; // Show above with some spacing
    }
    
    // Get suggestions for the word
    const suggestions = getSuggestions(word);
    
    // Set popup state
    setSelectedWord(word);
    setWordSuggestions(suggestions);
    setPopupPosition({ x, y });
    setShowPopup(true);
  };

  // Handle clicks on the spell check overlay
  const handleOverlayClick = (event: React.MouseEvent) => {
    const target = event.target as HTMLElement;
    const word = target.getAttribute('data-word');
    
    if (word && target.classList.contains('misspelled-word')) {
      handleWordClick(event, word);
    }
  };

  // Create highlighted text with misspelled words underlined and clickable
  const getHighlightedText = (text: string) => {
    if (!text || spellCheckLoading) return text;
    
    const words = text.match(/\b\w+\b/g) || [];
    const misspelled = misspelledWords.map(item => item.word.toLowerCase());
    
    let highlightedText = text;
    
    // Replace misspelled words with clickable underlined versions
    words.forEach(word => {
      if (misspelled.includes(word.toLowerCase())) {
        const regex = new RegExp(`\\b${word}\\b`, 'g');
        highlightedText = highlightedText.replace(regex, 
          `<span 
            class="misspelled-word"
            style="text-decoration: underline; text-decoration-color: red; text-decoration-style: wavy; text-underline-offset: 3px; cursor: pointer; user-select: none;" 
            data-word="${word}"
          >${word}</span>`
        );
      }
    });
    
    return highlightedText;
  };

  // Close popup when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (showPopup && !event.defaultPrevented) {
        setShowPopup(false);
      }
    };

    if (showPopup) {
      document.addEventListener("click", handleClickOutside);
    }

    return () => {
      document.removeEventListener("click", handleClickOutside);
    };
  }, [showPopup]);

  console.log("sugestion", suggestion);
  console.log("misspelled words", misspelledWords);

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className="flex flex-col gap-4 w-full max-w-[500px] py-6 px-10 bg-white border border-gray-200 rounded-[10px] mx-auto "
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
          Let AI help you craft the perfect bio
        </p>
      </div>

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
          <p className="text-red-500 text-sm mt-1">{errors.name.message}</p>
        )}
      </div>
      <div className="relative w-full">
        <label
          htmlFor="prompt"
          className="block text-sm font-semibold text-gray-700 mb-1"
        >
          Bio Description:
        </label>

        {/* Hidden textarea for height measurement */}
        <textarea
          ref={measureRef}
          className="absolute opacity-0 pointer-events-none -z-10 w-full p-2 border border-gray-200 rounded-[8px] resize-none whitespace-pre-wrap hide-scrollbar"
          style={{
            position: "absolute",
            left: "-9999px",
            top: "-9999px",
            fontSize: "16px",
            lineHeight: "24px",
            fontFamily: "var(--font-inter), Inter, sans-serif",
            boxSizing: "border-box",
            wordWrap: "break-word",
            overflowWrap: "break-word",
            textRendering: "geometricPrecision",
            padding: "8px",
            borderWidth: "1px",
            borderStyle: "solid",
            minHeight: "96px",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            verticalAlign: "bottom",
            overflow: "hidden",
            margin: 0,
            WebkitFontSmoothing: "antialiased",
            MozOsxFontSmoothing: "grayscale",
          }}
          tabIndex={-1}
        />

        {/* Container for textarea with absolute positioned overlay */}
        <div style={{ 
          position: "relative",
          width: "100%",
        }}>
          {/* Actual input textarea */}
          <textarea
            id="prompt"
            {...register("prompt", {
              required: "Description is required",
              maxLength: {
                value: 200,
                message: "Description cannot exceed 200 characters",
              },
            })}
            ref={(e) => {
              register("prompt").ref(e);
              textareaRef.current = e;
            }}
            onKeyDown={handleKeyDown}
            placeholder="Write a brief description about yourself..."
            style={{
              backgroundColor: "transparent",
              color: "#000000",
              height: textareaHeight,
              minHeight: "96px",
              fontSize: "16px",
              lineHeight: "24px", // Fixed line height in pixels
              fontFamily: "var(--font-inter), Inter, sans-serif",
              letterSpacing: "normal",
              WebkitAppearance: "none",
              appearance: "none",
              boxSizing: "border-box",
              wordWrap: "break-word",
              overflowWrap: "anywhere", // Better for long words
              textRendering: "geometricPrecision",
              padding: "8px",
              border: "1px solid #e5e7eb",
              borderRadius: "4px",
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              width: "100%",
              display: "block",
              resize: "none",
              outline: "none",
              overflow: "hidden",
              margin: 0,
              verticalAlign: "top", // Changed from bottom
              WebkitFontSmoothing: "subpixel-antialiased", // Better for monospace
              MozOsxFontSmoothing: "auto",
            }}
            className="hide-scrollbar focus:border-black active:border-black"
          />
          
          {/* Spell check overlay - positioned absolutely */}
          {promptValue && !spellCheckLoading && (
            <div
              onClick={handleOverlayClick}
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: overlayHeight || textareaHeight,
                pointerEvents: "auto",
                padding: "8px",
                border: "1px solid transparent",
                fontSize: "16px",
                lineHeight: "24px",
                fontFamily: "var(--font-inter), Inter, sans-serif",
                letterSpacing: "normal",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                wordWrap: "break-word",
                overflowWrap: "anywhere",
                textRendering: "geometricPrecision",
                WebkitFontSmoothing: "subpixel-antialiased",
                MozOsxFontSmoothing: "auto",
                boxSizing: "border-box",
                overflow: "hidden",
                margin: 0,
                verticalAlign: "top",
                borderRadius: "4px",
                color: "transparent",
                zIndex: 1,
              }}
              dangerouslySetInnerHTML={{ __html: getHighlightedText(promptValue) }}
            />
          )}

          {/* Suggestion overlay - positioned absolutely with ResizeObserver height */}
          {suggestion && (
            <div
              style={{
                position: "absolute",
                top: 0,
                left: 0,
                right: 0,
                height: overlayHeight || textareaHeight,
                pointerEvents: "none",
                padding: "8px",
                border: "1px solid transparent",
                fontSize: "16px",
                lineHeight: "24px", // Same fixed line height
                fontFamily: "var(--font-inter), Inter, sans-serif",
                letterSpacing: "normal",
                whiteSpace: "pre-wrap",
                wordBreak: "break-word",
                wordWrap: "break-word",
                overflowWrap: "anywhere",
                textRendering: "geometricPrecision",
                WebkitFontSmoothing: "subpixel-antialiased",
                MozOsxFontSmoothing: "auto",
                boxSizing: "border-box",
                overflow: "hidden",
                margin: 0,
                verticalAlign: "top",
                borderRadius: "4px",
                zIndex: 2,
              }}
            >
              <span style={{ visibility: "hidden" }}>{promptValue}</span>
              <span style={{ color: "#9CA3AF" }}>
                {needsSpaceBeforeSuggestion(promptValue) ? " " : ""}
                {suggestion}
              </span>
            </div>
          )}
        </div>

        {/* Instructional message */}
        <div className="mt-3 text-xs text-gray-500 min-h-[16px]">
          {suggestion ? (
            <span>
              💡 Press{" "}
              <kbd className="px-1 py-0.5 bg-gray-100 border border-gray-300 rounded text-xs">
                Tab
              </kbd>{" "}
              to accept suggestion
            </span>
          ) : isPending ? (
            <span className="animate-pulse">🤔 Thinking of suggestions...</span>
          ) : null}
        </div>

        {errors.prompt && (
          <p className="text-red-500 text-sm mt-1">{errors.prompt.message}</p>
        )}
      </div>

      {/* Spell Check Popup */}
      {showPopup && (
        <SpellCheckPopup
          word={selectedWord}
          suggestions={wordSuggestions}
          position={popupPosition}
          onSelect={(suggestion) => replaceWord(selectedWord, suggestion)}
          onIgnore={() => setShowPopup(false)}
          onClose={() => setShowPopup(false)}
        />
      )}

      <button
        type="submit"
        disabled={isPending}
        className="cursor-pointer w-full bg-gray-900 text-white font-semibold py-2 px-4 rounded hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 "
      >
        Submit
      </button>
      <div className="text-center text-sm text-gray-500 mt-4">
        Brought to you by{" "}
        <a
          href="https://swing.com"
          target="_blank"
          rel="noopener noreferrer"
          className="text-gray-700 font-semibold hover:text-yellow-500"
        >
          Swing.com
        </a>
      </div>
    </form>
  );
};

export default Form;
