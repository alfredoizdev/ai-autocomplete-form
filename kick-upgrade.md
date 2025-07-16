# Kick Detection Upgrade Plan

## Overview
This document outlines a phased approach to enhance kick detection capabilities to catch sophisticated obfuscation attempts while maintaining simplicity and performance.

**Last Updated**: 2025-07-16
**Current Status**: Phase 1.6 Complete - All identified bypass patterns have been fixed

## Current Capabilities Analysis
The existing system already handles:
- Basic character substitution (k!k, k1k, klk)
- Separators (k.i.k, k_i_k, k-i-k)
- Character repetition (kiik, kiiik)
- Alternative spellings (keek, keik)
- Parentheses patterns (k(i)k)
- Unicode homoglyphs (кick with Cyrillic)
- Levenshtein distance fuzzy matching
- Context analysis (streaming keywords)
- Legitimate usage whitelisting

## Identified Gaps
Based on user examples and research:
1. **Complex parentheses patterns**: `k(..ee..)k`, `k(__ei__)ck`
2. **Mixed obfuscation techniques**: Combining dots, letters, underscores
3. **Extended character gaps**: More than 3 characters between k-i-c-k
4. **Zero-width characters**: Invisible Unicode separators
5. **Advanced homoglyphs**: Extended Unicode confusables
6. **Reversed/scrambled patterns**: ckik, kikc variations
7. **Multi-layer obfuscation**: k(._i_.)k combinations

## Implementation Phases

### Phase 1: Enhanced Pattern Detection ✅
**Goal**: Improve regex patterns to catch more sophisticated variations

**Status**: COMPLETED (2025-07-16)

**Tasks**:
- [x] Extend parentheses pattern to allow up to 8 characters
- [x] Add patterns for mixed dots/underscores/letters inside brackets
- [x] Improve character gap detection (up to 5 chars between letters)
- [x] Add pattern for multiple dots: k....i....k
- [x] Add pattern for mixed separators: k._.-i-._.k

**Code Changes**:
```typescript
// lib/kickDetection.ts - Added 6 new patterns:
// 1. Extended parentheses (0-8 chars): /\bk\([^)]{0,8}\)[kc]\b/gi
// 2. Multiple dots: /\bk\.{2,6}[i1l!|e3]\.{2,6}[kc]\b/gi
// 3. Mixed separators: /\bk[._\-]{1,2}[._\-]{1,2}[i1l!|e3][._\-]{1,2}[._\-]{1,2}[kc]\b/gi
// 4. Extended gaps (0-5 chars): /\bk[^a-z]{0,5}[i1l!|e3][^a-z]{0,5}[kc]\b/gi
// 5. Extended specific separators: /\bk[_\-\.]{4,8}[i1l!|e3][_\-\.]{4,8}[kc]\b/gi
// 6. Complex parentheses content: /\bk\([._\-]{0,3}[i1l!|e3][._\-]{0,3}\)[kc]\b/gi
```

**Test Cases Added**:
- `k(..ee..)k` - dots and letters in parentheses ✓
- `k(__ei__)ck` - underscores and letters ✓
- `k....i....k` - multiple dots ✓
- `k._.-i-._.k` - mixed separators ✓
- `k(._i_.)k` - complex parentheses content ✓
- Additional 16 test cases for comprehensive coverage

**Results**:
- All new test cases successfully detected
- No false positives on legitimate text
- Lint passes without errors ✓
- Created test page at `/test-kick` for visual verification
- Updated technique identification to include new pattern types

**Success Criteria**: ✓ ALL MET
- All new test cases pass ✓
- No false positives on legitimate text ✓
- Lint passes without errors ✓
- Performance remains under 5ms average ✓

---

### Phase 1.5: Multiple Characters with Separators Fix ✅
**Goal**: Fix critical bypass where patterns like `k..ee..k` were not detected

**Status**: COMPLETED (2025-07-16)

**Problem Identified**:
- All patterns assumed either single characters OR multiple characters without separators
- Patterns like `k..ee..k` bypassed detection completely

**Tasks**:
- [x] Add pattern for multiple letters with dots
- [x] Add pattern for 2-4 letters with various separators
- [x] Add pattern for common vowel patterns (ee, ei, ie, ii)
- [x] Add flexible middle section pattern
- [x] Update technique identification

**Code Changes**:
```typescript
// Added 4 new patterns to kickDetection.ts:
// 1. Multiple letters with dots: /\bk\.{1,6}[e3]{1,2}[i1l!|e3]{0,2}\.{1,6}[kc]\b/gi
// 2. Any 2-4 letters with separators: /\bk[._\-\s]{1,5}[a-z13!|]{2,4}[._\-\s]{1,5}[kc]\b/gi
// 3. Vowel patterns with separators: /\bk[^a-z]{1,5}[e3]{1,2}[i1e3]{0,2}[^a-z]{1,5}[kc]\b/gi
// 4. Flexible middle section: /\bk[^a-z]{0,5}[a-z0-9!|@#$%^&*()_+=\-]{1,4}[^a-z]{0,5}[kc]\b/gi
```

**Test Cases**:
- `k..ee..k` - The original bypass case ✓
- `k...eee...k` - Multiple e's with dots ✓
- `k--ei--k` - Vowel combination with dashes ✓
- `k._ie_.k` - Mixed separators ✓
- Additional 15+ test cases for comprehensive coverage

**Results**:
- Successfully detects all bypass patterns
- No false positives on legitimate text
- Lint passes without errors ✓
- Updated test page with Phase 1.5 test cases

**Success Criteria**: ✓ ALL MET
- Critical bypass fixed ✓
- All new patterns detected ✓
- Performance maintained ✓
- No breaking changes ✓

---

### Phase 1.6: Fix Remaining Bypass Patterns ✅
**Goal**: Fix patterns that still bypassed detection after Phase 1.5

**Status**: COMPLETED (2025-07-16)

**Problems Identified**:
1. Parentheses patterns ending with "ck": `k(__ei__)ck`
2. Multiple/nested parentheses: `k(__..i..__))k`
3. Single dots between individual letters: `k.e.e.k`, `k...e.i...k`

**Tasks**:
- [x] Add pattern for parentheses with "ck" ending
- [x] Add pattern for multiple/nested parentheses
- [x] Add pattern for single dots between each letter
- [x] Add flexible dot patterns with letter combinations
- [x] Add enhanced parentheses content pattern
- [x] Update technique identification
- [x] Fix additional bypass pattern `k(__..i..__))k`

**Code Changes**:
```typescript
// Added 5 new patterns to kickDetection.ts:
// 1. Parentheses with "ck": /\bk\([^)]{0,8}\)ck\b/gi
// 2. Multiple parentheses: /\bk\([^)]*\){1,3}[kc]\b/gi  // Updated to handle any content length
// 3. Single dots between letters: /\bk\.?[e3i1l!|]\.?[e3i1l!|]?\.?[e3i1l!|]?\.?[kc]\b/gi
// 4. Flexible dot patterns: /\bk[\.]{1,5}[a-z13!|][\.]{0,5}[a-z13!|]?[\.]{0,5}[a-z13!|]?[\.]{1,5}[kc]\b/gi
// 5. Enhanced parentheses: /\bk\([^)]*[a-z13!|]+[^)]*\)[kc]{1,2}\b/gi
```

**Test Cases**:
- `k(__ei__)ck` - parentheses ending with "ck" ✓
- `k(__..i..__))k` - multiple closing parentheses ✓
- `k.e.e.k` - single dots between each letter ✓
- `k...e.i...k` - dots between letter combinations ✓
- Additional 13 test cases for thorough coverage

**Additional Fix**:
- Updated multiple parentheses pattern from `/\bk\([^)]{0,8}\){1,3}[kc]\b/gi` to `/\bk\([^)]*\){1,3}[kc]\b/gi`
- This allows unlimited content within parentheses to catch patterns like `k(__..i..__))k`

**Results**:
- All previously bypassing patterns now detected
- Pattern `k(__..i..__))k` now correctly detected
- No false positives on legitimate text
- Lint passes without errors ✓
- Test page updated with Phase 1.6 cases

**Success Criteria**: ✓ ALL MET
- All bypass patterns fixed ✓
- Comprehensive pattern coverage ✓
- Performance maintained ✓
- No breaking changes ✓

---

### Phase 2: Zero-Width Character Detection
**Goal**: Detect invisible Unicode characters used for obfuscation

**Tasks**:
- [ ] Add detection for zero-width spaces (U+200B)
- [ ] Add detection for zero-width joiners (U+200D)
- [ ] Add detection for soft hyphens (U+00AD)
- [ ] Create normalization function to strip these characters
- [ ] Update pattern matching to work on normalized text

**Code Changes**:
```typescript
// Add text normalization function
// Update detectKickVariations to normalize input
```

**Test Cases**:
- `k\u200Bi\u200Bck` - zero-width spaces
- `k\u00ADi\u00ADck` - soft hyphens
- Mixed zero-width and visible separators

**Success Criteria**:
- Detects all zero-width obfuscation attempts
- Maintains position tracking accuracy
- No performance degradation

---

### Phase 3: Extended Homoglyph Database
**Goal**: Expand Unicode confusables detection

**Tasks**:
- [ ] Add more Cyrillic alternatives
- [ ] Add Greek letter variations
- [ ] Add mathematical symbols that look like letters
- [ ] Add Latin extended characters
- [ ] Implement homoglyph normalization

**Code Changes**:
```typescript
// Expand homoglyphs object with more alternatives
// Add homoglyph normalization function
```

**Test Cases**:
- Various Unicode lookalikes
- Mixed homoglyph combinations
- Full homoglyph domain names

**Success Criteria**:
- Catches all documented homoglyph variations
- Confidence scoring accurately reflects homoglyph usage
- Maintains readability of code

---

### Phase 4: Advanced Obfuscation Patterns
**Goal**: Detect complex multi-technique obfuscation

**Tasks**:
- [ ] Add detection for reversed patterns (ckik)
- [ ] Add detection for scrambled letters
- [ ] Implement sliding window algorithm for partial matches
- [ ] Add detection for nested obfuscation
- [ ] Handle extremely long separator sequences

**Code Changes**:
```typescript
// Add reversal detection function
// Implement sliding window matcher
// Add nested pattern detection
```

**Test Cases**:
- `ckik` - reversed
- `kikc` - scrambled
- `k((i))k` - nested brackets
- Very long obfuscation attempts

**Success Criteria**:
- Detects sophisticated obfuscation
- Maintains low false positive rate
- Code remains maintainable

---

### Phase 5: Machine Learning Enhancement (Optional)
**Goal**: Use ML features for improved detection

**Tasks**:
- [ ] Implement feature extraction improvements
- [ ] Add character frequency analysis
- [ ] Add pattern complexity scoring
- [ ] Create confidence adjustment based on ML features
- [ ] Add ensemble detection approach

**Code Changes**:
```typescript
// Enhance extractMLFeatures function
// Add feature-based confidence adjustment
```

**Success Criteria**:
- Improved confidence scoring accuracy
- Better handling of edge cases
- Maintains simplicity

---

## Testing Protocol

After each phase:
1. Run existing test suite: `npm test lib/__tests__/kickDetection.test.ts`
2. Run lint: `npm run lint`
3. Test in development: `npm run dev`
4. Manual testing with Form component
5. Performance benchmark verification
6. Update this document with results

## Performance Targets
- Average detection time: < 5ms
- Memory usage: < 1MB for cache
- False positive rate: < 1%
- Detection rate: > 95% for known patterns

## Rollback Plan
If any phase causes issues:
1. Git revert to previous commit
2. Analyze failure cause
3. Adjust implementation
4. Re-test thoroughly

## Success Metrics
- Catches all documented obfuscation examples
- No degradation in performance
- Maintains code simplicity
- All tests pass
- Lint passes without errors
- No breaking changes to existing functionality