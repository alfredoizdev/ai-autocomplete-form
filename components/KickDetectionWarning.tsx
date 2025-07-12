import React from 'react';
import { DetectionResult } from '@/lib/kickDetection';

interface KickDetectionWarningProps {
  detection: DetectionResult;
  onDismiss: () => void;
  onAcknowledge?: () => void;
  className?: string;
}

export const KickDetectionWarning: React.FC<KickDetectionWarningProps> = ({
  detection,
  onDismiss,
  onAcknowledge,
  className = ''
}) => {
  // Don't show if not detected or confidence too low
  if (!detection.detected || detection.confidence < 30) {
    return null;
  }
  
  const getSeverityStyles = () => {
    if (detection.confidence >= 90) {
      return 'bg-red-50 border-red-300 text-red-800';
    } else if (detection.confidence >= 70) {
      return 'bg-orange-50 border-orange-300 text-orange-800';
    } else {
      return 'bg-yellow-50 border-yellow-300 text-yellow-800';
    }
  };
  
  const getSeverityIcon = () => {
    if (detection.confidence >= 90) {
      return (
        <svg className="h-5 w-5 text-red-400" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
        </svg>
      );
    } else {
      return (
        <svg className="h-5 w-5 text-yellow-400" viewBox="0 0 20 20" fill="currentColor">
          <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
        </svg>
      );
    }
  };
  
  const getTechniqueDescription = (technique: string): string => {
    const descriptions: Record<string, string> = {
      'character_substitution': 'character substitution',
      'separators': 'separator characters',
      'character_repetition': 'repeated characters',
      'alternative_spelling': 'alternative spelling',
      'parentheses': 'parentheses',
      'underscores': 'underscores',
      'domain_pattern': 'domain pattern',
      'homoglyph': 'look-alike characters',
      'fuzzy_match': 'similar spelling',
      'streaming_context': 'streaming context',
      'url_context': 'URL context',
      'platform_context': 'platform references'
    };
    
    return descriptions[technique] || technique;
  };
  
  return (
    <div className={`rounded-md border p-4 ${getSeverityStyles()} ${className}`}>
      <div className="flex">
        <div className="flex-shrink-0">
          {getSeverityIcon()}
        </div>
        <div className="ml-3 flex-1">
          <h3 className="text-sm font-medium">
            Prohibited Content Detected
          </h3>
          <div className="mt-2 text-sm">
            <p>
              Links to external streaming platforms are not allowed in user bios.
            </p>
            
            {/* Show detection details for high confidence */}
            {detection.confidence >= 70 && detection.matches.length > 0 && (
              <div className="mt-2">
                <p className="font-medium">
                  Detected: <span className="font-mono">{detection.matches.join(', ')}</span>
                </p>
                {detection.techniques.length > 0 && (
                  <p className="text-xs mt-1 opacity-75">
                    Detection method: {detection.techniques.map(getTechniqueDescription).join(', ')}
                  </p>
                )}
              </div>
            )}
            
            {/* Confidence indicator */}
            <div className="mt-3">
              <div className="flex items-center justify-between text-xs">
                <span>Detection confidence</span>
                <span className="font-medium">{detection.confidence}%</span>
              </div>
              <div className="mt-1 w-full bg-gray-200 rounded-full h-2">
                <div
                  className="h-2 rounded-full transition-all duration-300"
                  style={{
                    width: `${detection.confidence}%`,
                    backgroundColor: detection.confidence >= 90 ? '#ef4444' : 
                                   detection.confidence >= 70 ? '#f97316' : '#eab308'
                  }}
                />
              </div>
            </div>
          </div>
          
          {/* Action buttons */}
          <div className="mt-4 flex gap-2">
            {onAcknowledge && (
              <button
                type="button"
                onClick={onAcknowledge}
                className="text-sm font-medium hover:underline focus:outline-none"
              >
                I understand
              </button>
            )}
            <button
              type="button"
              onClick={onDismiss}
              className="text-sm font-medium hover:underline focus:outline-none ml-auto"
            >
              Dismiss
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

// Inline warning component (shows next to input field)
interface InlineKickWarningProps {
  detection: DetectionResult;
  className?: string;
}

export const InlineKickWarning: React.FC<InlineKickWarningProps> = ({
  detection,
  className = ''
}) => {
  if (!detection.detected || detection.confidence < 30) {
    return null;
  }
  
  const getWarningColor = () => {
    if (detection.confidence >= 90) return 'text-red-600';
    if (detection.confidence >= 70) return 'text-orange-600';
    return 'text-yellow-600';
  };
  
  return (
    <div className={`flex items-center gap-1 text-sm ${getWarningColor()} ${className}`}>
      <svg className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
      </svg>
      <span>External platform links not allowed</span>
    </div>
  );
};