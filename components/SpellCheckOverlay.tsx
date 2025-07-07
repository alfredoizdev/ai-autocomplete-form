import React, { useMemo, useDeferredValue, memo } from "react";
import { TextFeature } from "@/hooks/useTextFeatureCoordinator";

interface SpellCheckOverlayProps {
  promptValue: string;
  misspelledWords: Array<{ word: string; suggestions: string[] }>;
  isLoading: boolean;
  isProcessing: boolean;
  canActivateFeature: (feature: TextFeature) => boolean;
  textareaHeight: string;
  overlayHeight: string;
  onWordClick: (event: React.MouseEvent, word: string) => void;
}

const SpellCheckOverlay = memo(({
  promptValue,
  misspelledWords,
  isLoading,
  isProcessing,
  canActivateFeature,
  textareaHeight,
  overlayHeight,
  onWordClick,
}: SpellCheckOverlayProps) => {
  // Use deferred value for spell check to prevent blocking typing
  const deferredPromptValue = useDeferredValue(promptValue);
  const deferredMisspelledWords = useDeferredValue(misspelledWords);
  
  // Create highlighted text with misspelled words underlined and clickable
  const highlightedText = useMemo(() => {
    if (!deferredPromptValue || isLoading || isProcessing) return deferredPromptValue;
    
    const words = deferredPromptValue.match(/\b\w+\b/g) || [];
    const misspelled = deferredMisspelledWords.map(item => item.word.toLowerCase());
    
    let result = deferredPromptValue;
    
    // Replace misspelled words with clickable underlined versions
    words.forEach(word => {
      if (misspelled.includes(word.toLowerCase())) {
        const regex = new RegExp(`\\b${word}\\b`, 'g');
        result = result.replace(regex, 
          `<span 
            class="misspelled-word"
            data-word="${word}"
          >${word}</span>`
        );
      }
    });
    
    return result;
  }, [deferredPromptValue, deferredMisspelledWords, isLoading, isProcessing]);
  
  // Handle overlay click events
  const handleOverlayClick = (event: React.MouseEvent) => {
    const target = event.target as HTMLElement;
    const word = target.getAttribute('data-word');
    
    if (word && target.classList.contains('misspelled-word')) {
      onWordClick(event, word);
    }
  };
  
  // Don't render if conditions aren't met
  if (!promptValue || isLoading || isProcessing || !canActivateFeature(TextFeature.SPELLCHECK)) {
    return null;
  }
  
  return (
    <div
      onClick={handleOverlayClick}
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
        zIndex: 3,
      }}
      dangerouslySetInnerHTML={{ __html: highlightedText }}
    />
  );
});

SpellCheckOverlay.displayName = "SpellCheckOverlay";

export default SpellCheckOverlay;