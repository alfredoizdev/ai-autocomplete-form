import { useState, useEffect, useCallback } from 'react';
import { detectKickLinks, type DetectionResult } from '@/lib/kickDetectionImproved';

interface UseKickDetectionOptions {
  enabled?: boolean;
  debounceMs?: number;
  contextAnalysis?: boolean;
  confidenceThreshold?: number;
}

interface UseKickDetectionResult {
  detection: DetectionResult | null;
  isChecking: boolean;
  clearDetection: () => void;
}

export function useKickDetection(
  text: string,
  options: UseKickDetectionOptions = {}
): UseKickDetectionResult {
  const {
    enabled = true,
    debounceMs = 300,
    contextAnalysis: enableContext = true,
    confidenceThreshold = 0.6
  } = options;
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  const [debounceTimer, setDebounceTimer] = useState<NodeJS.Timeout | null>(null);

  const performDetection = useCallback((text: string) => {
    setIsChecking(true);
    
    try {
      const result = detectKickLinks(text);
      
      // Only flag if confidence meets threshold
      const detected = result.detected && result.confidence >= confidenceThreshold;
      
      // Update detection result
      setDetection(detected ? result : null);
      
      // Log detection for debugging (only in development)
      if (process.env.NODE_ENV === 'development' && detected) {
        console.log('Kick link detected:', {
          confidence: `${(result.confidence * 100).toFixed(0)}%`,
          matches: result.matches,
          techniques: result.techniques,
          hasLegitimateUsage: result.hasLegitimateUsage
        });
      }
    } catch (error) {
      console.error('Error in kick detection:', error);
      setDetection(null);
    } finally {
      setIsChecking(false);
    }
  }, [confidenceThreshold]);

  const checkText = useCallback((text: string) => {
    // Clear existing timer
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }

    // If text is empty, clear detection
    if (!text || text.trim().length === 0) {
      setDetection(null);
      setIsChecking(false);
      return;
    }

    // Set checking state
    setIsChecking(true);

    // Set new timer
    const timer = setTimeout(() => {
      performDetection(text);
    }, debounceMs);

    setDebounceTimer(timer);
  }, [debounceMs, debounceTimer, performDetection]);

  const clearDetection = useCallback(() => {
    if (debounceTimer) {
      clearTimeout(debounceTimer);
    }
    setDetection(null);
    setIsChecking(false);
  }, [debounceTimer]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimer) {
        clearTimeout(debounceTimer);
      }
    };
  }, [debounceTimer]);

  // Automatically check text when it changes
  useEffect(() => {
    if (!enabled) {
      clearDetection();
      return;
    }
    
    checkText(text);
  }, [text, enabled, checkText, clearDetection]);

  return {
    detection,
    isChecking,
    clearDetection
  };
}