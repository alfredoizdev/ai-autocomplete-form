import { useState, useEffect, useCallback } from 'react';
import { useDebounce } from 'use-debounce';
import { progressiveDetection, contextualAnalysis, type DetectionResult } from '@/lib/kickDetection';

interface UseKickDetectionOptions {
  enabled?: boolean;
  debounceMs?: number;
  contextAnalysis?: boolean;
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
    contextAnalysis: enableContext = true
  } = options;
  
  const [detection, setDetection] = useState<DetectionResult | null>(null);
  const [isChecking, setIsChecking] = useState(false);
  
  // Debounce the text input to avoid excessive checks
  const [debouncedText] = useDebounce(text, debounceMs);
  
  // Clear detection result
  const clearDetection = useCallback(() => {
    setDetection(null);
  }, []);
  
  useEffect(() => {
    // Skip if disabled or text is too short
    if (!enabled || !debouncedText || debouncedText.length < 3) {
      setDetection(null);
      setIsChecking(false);
      return;
    }
    
    // Perform detection
    const checkText = async () => {
      setIsChecking(true);
      
      try {
        // Use progressive detection for performance
        let result = progressiveDetection(debouncedText);
        
        // Apply context analysis if enabled and something was detected
        if (enableContext && result.detected) {
          result = contextualAnalysis(debouncedText, result);
        }
        
        setDetection(result);
      } catch (error) {
        console.error('Error in kick detection:', error);
        setDetection(null);
      } finally {
        setIsChecking(false);
      }
    };
    
    checkText();
  }, [debouncedText, enabled, enableContext]);
  
  return {
    detection,
    isChecking,
    clearDetection
  };
}

// Hook for logging detected patterns (for analytics/improvement)
interface DetectionLog {
  timestamp: Date;
  text: string;
  result: DetectionResult;
  userAction: 'submitted' | 'edited' | 'dismissed';
  sessionId: string;
}

export function useKickDetectionLogger() {
  const [logs, setLogs] = useState<DetectionLog[]>([]);
  const [sessionId] = useState(() => 
    `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  );
  
  const logDetection = useCallback((
    text: string,
    result: DetectionResult,
    userAction: DetectionLog['userAction']
  ) => {
    const log: DetectionLog = {
      timestamp: new Date(),
      text,
      result,
      userAction,
      sessionId
    };
    
    setLogs(prev => [...prev, log]);
    
    // In production, you would send this to your analytics endpoint
    if (process.env.NODE_ENV === 'production') {
      // Example: sendToAnalytics(log);
    }
  }, [sessionId]);
  
  const flushLogs = useCallback(async () => {
    if (logs.length === 0) return;
    
    try {
      // Send logs to server
      await fetch('/api/kick-detection-logs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(logs)
      });
      
      // Clear logs after successful send
      setLogs([]);
    } catch (error) {
      console.error('Failed to flush detection logs:', error);
    }
  }, [logs]);
  
  // Auto-flush logs when they reach a certain size
  useEffect(() => {
    if (logs.length >= 10) {
      flushLogs();
    }
  }, [logs.length, flushLogs]);
  
  // Flush logs on unmount
  useEffect(() => {
    return () => {
      if (logs.length > 0) {
        flushLogs();
      }
    };
  }, [logs.length, flushLogs]);
  
  return {
    logDetection,
    flushLogs,
    logsCount: logs.length
  };
}