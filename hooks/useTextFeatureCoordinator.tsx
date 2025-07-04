import { useState, useCallback, useRef } from "react";

export enum TextFeature {
  AUTOCOMPLETE = "autocomplete",
  SPELLCHECK = "spellcheck",
  CAPITALIZATION = "capitalization",
}

interface FeatureState {
  activeFeature: TextFeature | null;
  featureLocks: Set<TextFeature>;
}

const useTextFeatureCoordinator = () => {
  const [state, setState] = useState<FeatureState>({
    activeFeature: null,
    featureLocks: new Set(),
  });
  
  const timeoutRefs = useRef<Map<TextFeature, NodeJS.Timeout>>(new Map());

  // Lock a feature temporarily
  const lockFeature = useCallback((feature: TextFeature, duration: number = 200) => {
    setState(prev => ({
      ...prev,
      featureLocks: new Set([...prev.featureLocks, feature]),
    }));

    // Clear existing timeout for this feature
    const existingTimeout = timeoutRefs.current.get(feature);
    if (existingTimeout) {
      clearTimeout(existingTimeout);
    }

    // Set new timeout to unlock
    const timeout = setTimeout(() => {
      setState(prev => {
        const newLocks = new Set(prev.featureLocks);
        newLocks.delete(feature);
        return {
          ...prev,
          featureLocks: newLocks,
        };
      });
      timeoutRefs.current.delete(feature);
    }, duration);

    timeoutRefs.current.set(feature, timeout);
  }, []);

  // Check if a feature can be activated
  const canActivateFeature = useCallback((feature: TextFeature): boolean => {
    // Can't activate if this feature is locked
    if (state.featureLocks.has(feature)) return false;
    
    // Allow autocomplete and spellcheck to coexist
    if (feature === TextFeature.AUTOCOMPLETE && state.activeFeature === TextFeature.SPELLCHECK) {
      return true;
    }
    if (feature === TextFeature.SPELLCHECK && state.activeFeature === TextFeature.AUTOCOMPLETE) {
      return true;
    }
    
    // For other cases, only allow if no other feature is active
    return state.activeFeature === null;
  }, [state]);

  // Set active feature
  const setActiveFeature = useCallback((feature: TextFeature | null) => {
    setState(prev => ({
      ...prev,
      activeFeature: feature,
    }));
  }, []);

  // Check if a specific feature is active
  const isFeatureActive = useCallback((feature: TextFeature): boolean => {
    return state.activeFeature === feature;
  }, [state]);

  // Reset all states
  const reset = useCallback(() => {
    // Clear all timeouts
    timeoutRefs.current.forEach(timeout => clearTimeout(timeout));
    timeoutRefs.current.clear();
    
    setState({
      activeFeature: null,
      featureLocks: new Set(),
    });
  }, []);

  return {
    canActivateFeature,
    setActiveFeature,
    isFeatureActive,
    lockFeature,
    reset,
    activeFeature: state.activeFeature,
  };
};

export default useTextFeatureCoordinator;