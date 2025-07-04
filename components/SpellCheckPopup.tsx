import React, { useState } from "react";

interface SpellCheckPopupProps {
  word: string;
  suggestions: string[];
  position: { x: number; y: number };
  onSelect: (suggestion: string) => void;
  onIgnore: () => void;
  onAddToDictionary: (word: string) => void;
  onTeachCorrection: (misspelledWord: string, correctWord: string) => void;
}

const SpellCheckPopup: React.FC<SpellCheckPopupProps> = ({
  word,
  suggestions,
  position,
  onSelect,
  onIgnore,
  onAddToDictionary,
  onTeachCorrection,
}) => {
  const [isTeachMode, setIsTeachMode] = useState(false);
  const [correctionInput, setCorrectionInput] = useState("");

  const handlePopupClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    event.preventDefault();
  };

  const handleTeachSubmit = () => {
    if (correctionInput.trim()) {
      onTeachCorrection(word, correctionInput.trim());
      setIsTeachMode(false);
      setCorrectionInput("");
    }
  };

  const handleTeachCancel = () => {
    setIsTeachMode(false);
    setCorrectionInput("");
  };

  return (
    <div
      className="fixed bg-white border border-gray-300 rounded-lg shadow-lg z-50 min-w-[200px] max-w-[300px]"
      style={{
        left: `${position.x}px`,
        top: `${position.y}px`,
        transform: "translateX(-50%) translateY(-100%)",
      }}
      onClick={handlePopupClick}
    >
      <div className="p-3">
        <div className="text-sm font-semibold text-gray-900 mb-2">
          Suggestions for &quot;{word}&quot;
        </div>

        {!isTeachMode ? (
          <>
            {suggestions.length > 0 ? (
              <div className="space-y-1 mb-3">
                {suggestions.slice(0, 3).map((suggestion, index) => (
                  <button
                    key={index}
                    onClick={() => onSelect(suggestion)}
                    className="w-full text-left px-2 py-1 text-sm text-gray-700 hover:bg-gray-100 rounded transition-colors"
                  >
                    {suggestion}
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-sm text-gray-500 mb-3">
                No suggestions available
              </div>
            )}
          </>
        ) : (
          <div className="mb-3">
            <div className="text-sm text-gray-700 mb-2">
              What is the correct spelling?
            </div>
            <input
              type="text"
              value={correctionInput}
              onChange={(e) => setCorrectionInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleTeachSubmit();
                } else if (e.key === "Escape") {
                  handleTeachCancel();
                }
              }}
              placeholder="Type correct spelling..."
              className="w-full px-2 py-1 text-sm text-gray-900 border border-gray-200 rounded focus:outline-none focus:border-black"
              autoFocus
            />
            <div className="flex gap-2 mt-2">
              <button
                onClick={handleTeachSubmit}
                disabled={!correctionInput.trim()}
                className="px-3 py-1 text-xs bg-gray-900 text-white rounded hover:bg-gray-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors"
              >
                Save
              </button>
              <button
                onClick={handleTeachCancel}
                className="px-3 py-1 text-xs bg-gray-200 text-gray-700 rounded hover:bg-gray-300 transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        )}

        {!isTeachMode && (
          <div className="border-t pt-2">
            <div className="grid grid-cols-3 gap-2">
              <button
                onClick={() => setIsTeachMode(true)}
                className="text-xs bg-gray-900 text-white hover:bg-gray-700 px-2 py-1 rounded transition-colors font-medium"
              >
                Teach Correction
              </button>
              <button
                onClick={() => onAddToDictionary(word)}
                className="text-xs bg-gray-900 text-white hover:bg-gray-700 px-2 py-1 rounded transition-colors font-medium"
              >
                Add to Dictionary
              </button>
              <button
                onClick={onIgnore}
                className="text-xs bg-gray-200 text-gray-700 hover:bg-gray-300 px-2 py-1 rounded transition-colors"
              >
                Ignore
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SpellCheckPopup;
