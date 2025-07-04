import React from "react";

interface SpellCheckPopupProps {
  word: string;
  suggestions: string[];
  position: { x: number; y: number };
  onSelect: (suggestion: string) => void;
  onIgnore: () => void;
  onClose: () => void;
}

const SpellCheckPopup: React.FC<SpellCheckPopupProps> = ({
  word,
  suggestions,
  position,
  onSelect,
  onIgnore,
  onClose,
}) => {
  const handlePopupClick = (event: React.MouseEvent) => {
    event.stopPropagation();
    event.preventDefault();
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
        
        <div className="border-t pt-2 flex justify-between">
          <button
            onClick={onIgnore}
            className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-100 transition-colors"
          >
            Ignore
          </button>
          <button
            onClick={onClose}
            className="text-xs text-gray-500 hover:text-gray-700 px-2 py-1 rounded hover:bg-gray-100 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};

export default SpellCheckPopup;