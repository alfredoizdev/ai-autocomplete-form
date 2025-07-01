# Fix Autocomplete: Simple 3-Word + 2-Second Rule

## Problem Analysis
Looking at the screenshot, autocomplete is showing "years, still playfully entwined." immediately after the user reaches the bottom of the textarea, without waiting for 3 new words after tab acceptance. The current implementation is too complex and has interference between multiple effects.

## User Requirements (Keep It Simple)
1. User types ≥3 words → pause 2 seconds → autocomplete shows
2. User presses Tab → autocomplete STOPS completely  
3. User must type ≥3 NEW words → pause 2 seconds → autocomplete resumes

## Root Cause Analysis
After examining the current code, the issues are:

1. **Debounce is 1 second, user wants 2 seconds**
2. **Complex word tracking logic is unreliable** - multiple effects interfering
3. **Auto-correction effects disrupting the tracking**
4. **Height calculation effects may trigger unwanted behavior**
5. **Too many state variables creating race conditions**

## Simplified Solution Strategy

Instead of trying to fix the complex tracking, implement a clean, simple approach:

### Core Logic
- Use a single "blocked" state after tab acceptance
- Count words from the EXACT point where tab was pressed
- Block ALL autocomplete until exactly 3 new words typed
- Use 2-second debounce as requested

## Detailed Implementation Plan

### Task 1: Clean up current implementation
- [x] Remove complex `justAcceptedSuggestion` and `wordsAfterAcceptance` tracking
- [x] Remove the character-based checks (10+ chars) - too complex
- [x] Simplify to just one tracking mechanism

### Task 2: Implement simple word-position tracking
- [x] Store the exact text content at the moment tab is pressed
- [x] Track new words by comparing current text to stored baseline
- [x] Use simple, reliable word counting (split on whitespace, filter empty)

### Task 3: Update debounce and timing
- [x] Change debounce from 1000ms to 2000ms (user requirement)
- [x] Ensure blocked state prevents debounce from triggering

### Task 4: Separate autocomplete logic from other effects
- [x] Make sure auto-correction doesn't interfere with word counting
- [x] Ensure height calculation doesn't trigger unwanted autocomplete
- [x] Keep autocomplete logic isolated and simple

### Task 5: Add robust reset mechanism
- [x] Clear blocked state only when exactly 3 new words detected
- [x] Handle edge cases (deletion, correction) by resetting if needed
- [x] Ensure state stays consistent

## Implementation Details

### New State Structure (Simplified)
```typescript
const [isBlockedAfterAcceptance, setIsBlockedAfterAcceptance] = useState(false);
const [baselineTextForCounting, setBaselineTextForCounting] = useState("");
```

### Word Counting Logic
```typescript
const countNewWordsAfterBaseline = (currentText: string, baseline: string): number => {
  const newText = currentText.substring(baseline.length);
  return newText.trim().split(/\s+/).filter(word => word.length > 0).length;
};
```

### Tab Handler
```typescript
// On tab press:
setIsBlockedAfterAcceptance(true);
setBaselineTextForCounting(finalTextAfterAcceptance);
```

### Debounced Effect Check
```typescript
// In main debounced effect:
if (isBlockedAfterAcceptance) {
  const newWords = countNewWordsAfterBaseline(debouncedPrompt, baselineTextForCounting);
  if (newWords < 3) {
    setSuggestion("");
    return;
  }
  // User has typed 3+ new words, unblock
  setIsBlockedAfterAcceptance(false);
}
```

## Success Criteria
- [x] Autocomplete waits 2 seconds after user stops typing
- [x] After tab acceptance, autocomplete completely stops
- [x] User must type exactly 3 new words before autocomplete resumes
- [x] No interference from auto-correction or height changes
- [x] Simple, reliable behavior every time

## Review

### Changes Made
1. **Simplified State Management** - Replaced complex tracking variables with just two simple ones:
   - `isBlockedAfterAcceptance` - Boolean flag set when Tab is pressed
   - `baselineTextForCounting` - Stores exact text content at moment of acceptance

2. **Updated Timing** - Changed debounce from 1000ms to 2000ms as requested

3. **Clean Word Counting** - Implemented simple `countNewWordsAfterBaseline()` function that:
   - Compares current text to stored baseline
   - Counts only new words added after the baseline
   - Uses reliable whitespace splitting and empty filtering

4. **Isolated Logic** - Completely separated autocomplete blocking from auto-correction and height effects

5. **Robust Reset** - Autocomplete automatically unblocks when exactly 3 new words are detected

### Key Improvements
- **Predictable behavior**: Simple state machine with clear conditions
- **No interference**: Auto-correction and formatting don't affect word counting
- **Proper timing**: 2-second debounce as requested
- **Clean blocking**: Tab press completely stops autocomplete until requirements met
- **Reliable unblocking**: Automatic resume when user types 3+ new words

The implementation now matches your exact requirements: Type 3+ words → pause 2 seconds → autocomplete shows → press Tab → completely stops → type 3+ new words → pause 2 seconds → resumes.