import { useRef, useEffect, useMemo, useCallback } from 'react';

/**
 * Custom debounce hook that maintains stable function reference while accessing latest state
 * Solves the race condition issue when form is cleared and debounce fires with stale values
 */
export const useStableDebounce = <T extends (...args: any[]) => any>(
  callback: T,
  delay: number
) => {
  // Store the latest callback in a ref
  const callbackRef = useRef(callback);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isAbortedRef = useRef(false);
  
  // Update the callback ref whenever it changes
  useEffect(() => {
    callbackRef.current = callback;
  }, [callback]);
  
  // Create a stable debounced function
  const debouncedCallback = useMemo(() => {
    const func = (...args: Parameters<T>) => {
      // Clear any existing timeout
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
      
      // Reset abort flag
      isAbortedRef.current = false;
      
      // Set new timeout
      timeoutRef.current = setTimeout(() => {
        if (!isAbortedRef.current) {
          // Call the latest callback from ref
          callbackRef.current?.(...args);
        }
      }, delay);
    };
    
    return func as T;
  }, [delay]); // Only recreate if delay changes
  
  // Cancel function to abort pending debounced calls
  const cancel = useCallback(() => {
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current);
      timeoutRef.current = null;
    }
    isAbortedRef.current = true;
  }, []);
  
  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, []);
  
  return { debouncedCallback, cancel };
};