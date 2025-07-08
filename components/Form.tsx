"use client";

import Image from "next/image";
import useFormAutocomplete from "@/hooks/useFormAutocomplete";
import useDebouncedSpellCheck from "@/hooks/useDebouncedSpellCheck";
import SpellCheckPopup from "./SpellCheckPopup";
import SpellCheckOverlay from "./SpellCheckOverlay";
import { useState, useEffect, useCallback } from "react";
import useTextFeatureCoordinator, { TextFeature } from "@/hooks/useTextFeatureCoordinator";

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
    notifySpellCheckReplacement,
  } = useFormAutocomplete();

  const { misspelledWords, isLoading: spellCheckLoading, getSuggestions, isProcessing, customDictionary, refreshSpellCheck } = useDebouncedSpellCheck(promptValue);
  
  // Use the text feature coordinator to prevent conflicts
  const coordinator = useTextFeatureCoordinator();

  // State for spell check popup
  const [showPopup, setShowPopup] = useState(false);
  const [popupPosition, setPopupPosition] = useState({ x: 0, y: 0 });
  const [selectedWord, setSelectedWord] = useState("");
  const [wordSuggestions, setWordSuggestions] = useState<string[]>([]);


  // Handle word replacement - memoized for performance
  const replaceWord = useCallback((originalWord: string, newWord: string) => {
    if (!promptValue || !textareaRef.current) return;
    
    // Lock autocomplete feature during spell check replacement
    coordinator.lockFeature(TextFeature.AUTOCOMPLETE, 150);
    coordinator.setActiveFeature(TextFeature.SPELLCHECK);
    
    // Store current cursor position
    const currentCursorPos = textareaRef.current.selectionStart;
    
    // Create a regex that matches the exact word with word boundaries
    const regex = new RegExp(`\\b${originalWord}\\b`, 'g');
    const newText = promptValue.replace(regex, newWord);
    
    // Calculate the difference in length to adjust cursor position
    const lengthDiff = newWord.length - originalWord.length;
    
    // Close popup first
    setShowPopup(false);
    
    // Notify autocomplete that a spell check replacement occurred
    notifySpellCheckReplacement();
    
    // Update the form value
    setValue("prompt", newText);
    
    // Restore focus and cursor position immediately
    requestAnimationFrame(() => {
      if (textareaRef.current) {
        textareaRef.current.focus();
        
        // Adjust cursor position if the replacement happened before current position
        const adjustedCursorPos = currentCursorPos + lengthDiff;
        textareaRef.current.setSelectionRange(adjustedCursorPos, adjustedCursorPos);
        
        // Dispatch input event to trigger React Hook Form updates and autocomplete
        const inputEvent = new Event('input', {
          bubbles: true,
          cancelable: true,
        });
        textareaRef.current.dispatchEvent(inputEvent);
        
        // Also dispatch change event for completeness
        const changeEvent = new Event('change', {
          bubbles: true,
          cancelable: true,
        });
        textareaRef.current.dispatchEvent(changeEvent);
      }
      
      // Clear active feature after UI update
      coordinator.setActiveFeature(null);
    });
  }, [promptValue, setValue, textareaRef, notifySpellCheckReplacement, coordinator]);

  // Handle adding word to custom dictionary - memoized for performance
  const handleAddToDictionary = useCallback((word: string) => {
    // Add word to custom dictionary
    const success = customDictionary.addWord(word);
    
    if (success) {
      // Close popup after successful addition
      setShowPopup(false);
      
      // Optional: Show brief success feedback (could be implemented with toast)
      console.log(`Added "${word}" to custom dictionary`);
    } else {
      // Handle error case
      console.error(`Failed to add "${word}" to custom dictionary`);
    }
  }, [customDictionary]);

  // Handle teaching word correction - memoized for performance
  const handleTeachCorrection = useCallback((misspelledWord: string, correctWord: string) => {
    // Add word mapping to custom dictionary
    const success = customDictionary.addMapping(misspelledWord, correctWord);
    
    if (success) {
      // Close popup first
      setShowPopup(false);
      
      // Automatically replace the misspelled word in the textarea with the correct word
      if (promptValue && textareaRef.current) {
        // Lock autocomplete feature during replacement
        coordinator.lockFeature(TextFeature.AUTOCOMPLETE, 150);
        coordinator.setActiveFeature(TextFeature.SPELLCHECK);
        
        // Store current cursor position
        const currentCursorPos = textareaRef.current.selectionStart;
        
        // Create a regex that matches the exact word with word boundaries
        const regex = new RegExp(`\\b${misspelledWord}\\b`, 'g');
        const newText = promptValue.replace(regex, correctWord);
        
        // Calculate the difference in length to adjust cursor position
        const lengthDiff = correctWord.length - misspelledWord.length;
        
        // Notify autocomplete that a replacement occurred
        notifySpellCheckReplacement();
        
        // Update the form value
        setValue("prompt", newText);
        
        // Restore focus and cursor position
        requestAnimationFrame(() => {
          if (textareaRef.current) {
            textareaRef.current.focus();
            
            // Adjust cursor position if the replacement happened before current position
            const adjustedCursorPos = currentCursorPos + lengthDiff;
            textareaRef.current.setSelectionRange(adjustedCursorPos, adjustedCursorPos);
            
            // Dispatch input event to trigger React Hook Form updates
            const inputEvent = new Event('input', {
              bubbles: true,
              cancelable: true,
            });
            textareaRef.current.dispatchEvent(inputEvent);
            
            // Also dispatch change event for completeness
            const changeEvent = new Event('change', {
              bubbles: true,
              cancelable: true,
            });
            textareaRef.current.dispatchEvent(changeEvent);
          }
          
          // Clear active feature after UI update
          coordinator.setActiveFeature(null);
          
          // Force spell check refresh to ensure new mapping is available
          setTimeout(() => {
            refreshSpellCheck();
          }, 200);
        });
      }
      
      // Show success feedback
      console.log(`Taught correction: "${misspelledWord}" → "${correctWord}" and replaced in text`);
    } else {
      // Handle error case
      console.error(`Failed to teach correction: "${misspelledWord}" → "${correctWord}"`);
    }
  }, [customDictionary, promptValue, textareaRef, coordinator, notifySpellCheckReplacement, setValue, refreshSpellCheck]);

  // Handle clicking on misspelled words - memoized for performance
  const handleWordClick = useCallback((event: React.MouseEvent, word: string) => {
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
      y = y; // Show directly above with no gap
    }
    
    // Get suggestions for the word
    const suggestions = getSuggestions(word);
    
    // Set popup state
    setSelectedWord(word);
    setWordSuggestions(suggestions);
    setPopupPosition({ x, y });
    setShowPopup(true);
  }, [getSuggestions]);



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

  // Debug logging removed for production

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
        <div 
          style={{ 
            position: "relative",
            width: "100%",
          }}
          onClick={() => {
            // Ensure textarea gets focus when clicking container
            if (textareaRef.current) {
              textareaRef.current.focus();
            }
          }}
        >
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
            className="hide-scrollbar border border-gray-200 rounded-[8px] focus:border-black active:border-black focus:outline-none transition-all duration-300 ease-out"
          />
          
          {/* Spell check overlay - now a separate optimized component */}
          <SpellCheckOverlay
            promptValue={promptValue}
            misspelledWords={misspelledWords}
            isLoading={spellCheckLoading}
            isProcessing={isProcessing}
            canActivateFeature={coordinator.canActivateFeature}
            textareaHeight={textareaHeight}
            overlayHeight={overlayHeight}
            onWordClick={handleWordClick}
          />

          {/* Suggestion overlay - positioned absolutely with ResizeObserver height */}
          {suggestion && coordinator.canActivateFeature(TextFeature.AUTOCOMPLETE) && (
            console.log("🎯 Showing suggestion overlay:", { 
              suggestion: suggestion.substring(0, 30) + "...", 
              canActivate: coordinator.canActivateFeature(TextFeature.AUTOCOMPLETE),
              activeFeature: coordinator.activeFeature 
            }),
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
          onIgnore={() => handleAddToDictionary(selectedWord)}
          onAddToDictionary={handleAddToDictionary}
          onTeachCorrection={handleTeachCorrection}
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
