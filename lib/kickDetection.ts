// Kick.com link detection module
// Detects various obfuscation attempts to hide kick.com links in user bios

export interface DetectionResult {
  detected: boolean;
  confidence: number;
  matches: string[];
  techniques: string[];
  positions: Array<{ start: number; end: number }>;
  hasLegitimateUsage?: boolean;
}

// Core pattern definitions based on real user examples
const kickVariationPatterns = [
  // Basic character substitution and spacing (with word boundaries where possible)
  /\bk\s*[i1l!|]\s*[kc]\b/gi,
  
  // Patterns with separators (dots, underscores, dashes)
  /\bk[._\-]{1,3}[i1l!|][._\-]{0,3}[kc]\b/gi,
  
  // Pattern specifically for k.i.ck format (separator before AND after i)
  /\bk[._\-]{1,3}[i1l!|][._\-]{1,3}ck\b/gi,
  
  // Repeated characters
  /\bk[i1l!|]{2,4}[kc]\b/gi,
  
  // Alternative spellings (keek, keik, kiek)
  /\bk[e3][e3i1][kc]\b/gi,
  
  // Enhanced parentheses patterns - catches k(__)k, k(_)k, k( )k, k(i)k, etc.
  /\bk\([^)]{0,3}\)[kc]\b/gi,
  
  // Multiple underscores
  /\bk_{1,3}[i1l!|]_{0,3}[kc]\b/gi,
  
  // Advanced pattern with any non-letter chars between
  /\bk[^a-z]{0,3}[i1l!|e3][^a-z]{0,3}[kc]\b/gi,
  
  // Any bracket type with any content - k[_]k, k{i}k, k<_>k, etc.
  /\bk[(\[{<][^)\]}>]{0,3}[)\]}>][kc]\b/gi,
  
  // Missing letter patterns - k__k, k--k, k..k, k~~k
  /\bk[_\-\.~]{1,3}[kc]\b/gi,
  
  // Multiple spaces pattern - k  k, k   k
  /\bk\s{1,3}[kc]\b/gi,
  
  // General k***k pattern where *** is any non-letter chars (catches missing i)
  /\bk[^a-z]{1,3}[kc]\b/gi,
  
  // PHASE 1 ADDITIONS:
  
  // Extended parentheses patterns (up to 8 chars) - k(..ee..)k, k(__ei__)ck
  /\bk\([^)]{0,8}\)[kc]\b/gi,
  
  // Multiple dots pattern - k....i....k
  /\bk\.{2,6}[i1l!|e3]\.{2,6}[kc]\b/gi,
  
  // Mixed separators pattern - k._.-i-._.k
  /\bk[._\-]{1,2}[._\-]{1,2}[i1l!|e3][._\-]{1,2}[._\-]{1,2}[kc]\b/gi,
  
  // Extended character gaps (up to 5 chars between letters)
  /\bk[^a-z]{0,5}[i1l!|e3][^a-z]{0,5}[kc]\b/gi,
  
  // Extended gaps with specific separators
  /\bk[_\-\.]{4,8}[i1l!|e3][_\-\.]{4,8}[kc]\b/gi,
  
  // Complex parentheses with mixed content - k(._i_.)k, k(.._i_..)k
  /\bk\([._\-]{0,3}[i1l!|e3][._\-]{0,3}\)[kc]\b/gi,
  
  // PHASE 1.5 ADDITIONS - Fix for k..ee..k bypass:
  
  // Multiple letters with dots (catches k..ee..k, k...ei...k)
  /\bk\.{1,6}[e3]{1,2}[i1l!|e3]{0,2}\.{1,6}[kc]\b/gi,
  
  // Any 2-4 letters with various separators
  /\bk[._\-\s]{1,5}[a-z13!|]{2,4}[._\-\s]{1,5}[kc]\b/gi,
  
  // Common vowel patterns with separators (ee, ei, ie, ii)
  /\bk[^a-z]{1,5}[e3]{1,2}[i1e3]{0,2}[^a-z]{1,5}[kc]\b/gi,
  
  // Flexible middle section (1-4 chars, must include special chars or numbers)
  /\bk[^a-z]{0,5}[a-z0-9!|@#$%^&*()_+=\-]*[0-9!|@#$%^&*()_+=\-]+[a-z0-9!|@#$%^&*()_+=\-]*[^a-z]{0,5}[kc]\b/gi,
  
  // PHASE 1.6 ADDITIONS - Fix remaining bypasses:
  
  // Parentheses with "ck" ending - k(__ei__)ck
  /\bk\([^)]{0,8}\)ck\b/gi,
  
  // Multiple/nested parentheses - k(__..i..__))k
  /\bk\([^)]*\){1,3}[kc]\b/gi,
  
  // Single dots between each letter - k.e.e.k, k.i.c.k
  /\bk\.?[e3i1l!|]\.?[e3i1l!|]?\.?[e3i1l!|]?\.?[kc]\b/gi,
  
  // Flexible dot patterns with letter combinations - k...e.i...k
  /\bk[\.]{1,5}[a-z13!|][\.]{0,5}[a-z13!|]?[\.]{0,5}[a-z13!|]?[\.]{1,5}[kc]\b/gi,
  
  // Enhanced parentheses content (letters AND separators)
  /\bk\([^)]*[a-z13!|]+[^)]*\)[kc]{1,2}\b/gi,
  
  // PHASE 1.7 ADDITIONS - Phonetic variations:
  
  // Double vowel patterns (keek, kook, kuuk)
  /\bk[e3]{2}[kc]\b/gi,
  /\bk[aeiou]{2}[kc]\b/gi,
  
  // Vowel variations with y (kyck, kyyk)
  /\bky{1,2}[kc]\b/gi,
  
  // Single vowel sound-alike patterns (keak, kouk, kaik)
  /\bk[aeiouey]{1,2}[kc]\b/gi,
  
  // Short variations (kic, kiq, kik)
  /\bki[cqk]\b/gi,
  
  // Mixed number-letter vowel patterns (k3ek, ke3k)
  /\bk[e3][aeiou3][kc]\b/gi,
  
  // Domain patterns (more flexible boundaries for URLs)
  /k[i1l!|._\-\s]{1,4}[kc]\s*[\.\,\·\•]\s*c[o0]m/gi,
  /k[i1l!|._\-\s]{1,4}[kc]\s+dot\s+c[o0]m/gi,
  /k[i1l!|._\-\s]{1,4}[kc]\[?\.\]?\s*c[o0]m/gi,
];

// Unicode confusables that look like 'kick' characters
const homoglyphs: Record<string, string[]> = {
  'k': ['к', 'κ', 'ķ', 'ҡ', 'ҝ', 'ќ'],  // Cyrillic and Greek
  'i': ['і', 'í', 'ì', 'ï', 'ı', '1', 'l', '|', '!'],
  'c': ['с', 'ς', 'ċ', 'ĉ', 'ć', 'č'],
};

// Zero-width and invisible Unicode characters used for obfuscation
const ZERO_WIDTH_CHARS = [
  '\u200B', // Zero-width space
  '\u200C', // Zero-width non-joiner
  '\u200D', // Zero-width joiner
  '\u00AD', // Soft hyphen
  '\uFEFF', // Zero-width no-break space
  '\u2060', // Word joiner
];

// Create regex pattern for zero-width characters
const ZERO_WIDTH_PATTERN = new RegExp(`[${ZERO_WIDTH_CHARS.join('')}]`, 'g');

// Normalize text by removing zero-width characters while preserving position mapping
export function normalizeText(text: string): {
  normalized: string;
  hasZeroWidth: boolean;
  positionMap: number[]; // Maps normalized position to original position
} {
  const normalized = text.replace(ZERO_WIDTH_PATTERN, '');
  const hasZeroWidth = normalized.length !== text.length;
  
  // Build position map for accurate tracking
  const positionMap: number[] = [];
  let originalPos = 0;
  
  for (let i = 0; i < normalized.length; i++) {
    // Skip zero-width characters in original text
    while (originalPos < text.length && ZERO_WIDTH_CHARS.includes(text[originalPos])) {
      originalPos++;
    }
    positionMap.push(originalPos);
    originalPos++;
  }
  
  return { normalized, hasZeroWidth, positionMap };
}

// Common legitimate phrases containing "kick"
const KICK_WHITELIST_PHRASES = [
  'kick back',
  'kick the ball',
  'kick off',
  'kickstart',
  'kick start', 
  'kick in',
  'kick out',
  'kick ass',
  'kick butt',
  'kick around',
  'kick up',
  'get a kick',
  'for kicks',
  'kick the bucket',
  'kick the habit',
  'kick into gear',
  'alive and kicking',
  'kick yourself',
  'kick some',
  'kick my',
  'kick your',
  'kick his',
  'kick her',
  'kick their',
  'side kick',
  'free kick',
  'penalty kick',
  'karate kick',
  'soccer kick',
  'football kick'
];

// Check if "kick" appears in a legitimate context
function isLegitimateKickUsage(text: string, matchPosition: number): boolean {
  const lowerText = text.toLowerCase();
  const contextWindow = 50; // Characters to check before and after
  
  // Get context around the match
  const start = Math.max(0, matchPosition - contextWindow);
  const end = Math.min(text.length, matchPosition + 4 + contextWindow); // 4 for "kick"
  const context = lowerText.substring(start, end);
  
  // Check if it's part of a whitelisted phrase
  for (const phrase of KICK_WHITELIST_PHRASES) {
    if (context.includes(phrase)) {
      return true;
    }
  }
  
  // Check for verb usage patterns (kick + preposition/article/pronoun)
  const verbPatterns = [
    /kick\s+(the|a|an|my|your|his|her|their|some|any)\s+\w+/,
    /to\s+kick\s+/,
    /will\s+kick\s+/,
    /would\s+kick\s+/,
    /could\s+kick\s+/,
    /should\s+kick\s+/,
    /might\s+kick\s+/,
    /can\s+kick\s+/,
    /gonna\s+kick\s+/,
    /wanna\s+kick\s+/,
    /let\'s\s+kick\s+/,
    /like\s+to\s+kick\s+/,
    /love\s+to\s+kick\s+/,
    /want\s+to\s+kick\s+/
  ];
  
  for (const pattern of verbPatterns) {
    if (pattern.test(context)) {
      return true;
    }
  }
  
  // Check if surrounded by normal words (not special characters)
  const wordBoundaryCheck = /\w+\s+kick\s+\w+/;
  if (wordBoundaryCheck.test(context)) {
    // Make sure it's not followed by domain-like patterns
    const domainCheck = /kick\s*[\.\[]\s*c[o0]m/;
    if (!domainCheck.test(context)) {
      return true;
    }
  }
  
  return false;
}

// Calculate Levenshtein distance between two strings
function levenshteinDistance(str1: string, str2: string): number {
  const matrix: number[][] = [];
  
  // Initialize the first column
  for (let i = 0; i <= str2.length; i++) {
    matrix[i] = [i];
  }
  
  // Initialize the first row
  for (let j = 0; j <= str1.length; j++) {
    matrix[0][j] = j;
  }
  
  // Fill in the rest of the matrix
  for (let i = 1; i <= str2.length; i++) {
    for (let j = 1; j <= str1.length; j++) {
      if (str2.charAt(i - 1) === str1.charAt(j - 1)) {
        matrix[i][j] = matrix[i - 1][j - 1];
      } else {
        matrix[i][j] = Math.min(
          matrix[i - 1][j - 1] + 1, // substitution
          matrix[i][j - 1] + 1,     // insertion
          matrix[i - 1][j] + 1      // deletion
        );
      }
    }
  }
  
  return matrix[str2.length][str1.length];
}

// Calculate confidence score based on detection results
function calculateConfidence(results: Partial<DetectionResult>): number {
  if (!results.matches || results.matches.length === 0) return 0;
  
  let confidence = 0;
  
  // Direct matches need context analysis
  const hasDirectMatch = results.matches.some(m => 
    m.toLowerCase().replace(/[^a-z]/g, '') === 'kick'
  );
  
  if (hasDirectMatch) {
    // Check if it's legitimate usage
    if (results.hasLegitimateUsage) {
      // Legitimate usage gets very low confidence
      confidence = 15;
    } else {
      // Non-legitimate direct match still gets moderate confidence
      // (not 100% because it could still be a borderline case)
      confidence = 60;
    }
  } else {
    // Base confidence on number of matches and techniques
    const baseConfidence = (results.matches.length * 20) + 
      ((results.techniques?.length || 0) * 15);
    
    // Add technique-specific boosts for highly suspicious patterns
    let techniqueBoost = 0;
    if (results.techniques?.includes('parentheses')) {
      techniqueBoost += 20; // Parentheses are very suspicious
    }
    if (results.techniques?.includes('domain_pattern')) {
      techniqueBoost += 25; // Domain patterns are highly suspicious
    }
    if (results.techniques?.includes('character_substitution') && 
        results.matches.some(m => /[1!|]/.test(m))) {
      techniqueBoost += 15; // Number/symbol substitutions are suspicious
    }
    if (results.techniques?.includes('separators') && 
        results.matches.some(m => m.length > 5)) {
      techniqueBoost += 10; // Long separated patterns are suspicious
    }
    
    confidence = Math.min(95, baseConfidence + techniqueBoost);
  }
  
  // Apply legitimate usage penalty if detected
  if (results.hasLegitimateUsage && confidence > 30) {
    confidence = Math.floor(confidence * 0.3); // Reduce by 70%
  }
  
  return confidence;
}

// Check for homoglyph attacks using Unicode confusables
function detectHomoglyphs(text: string): { detected: boolean; matches: string[] } {
  const matches: string[] = [];
  
  // Build regex pattern for homoglyphs
  const kChars = ['k', ...homoglyphs.k].join('');
  const iChars = ['i', ...homoglyphs.i].join('');
  const cChars = ['c', ...homoglyphs.c].join('');
  
  // Create pattern that matches any combination of these characters
  const homoglyphPattern = new RegExp(
    `[${kChars}]\\s*[${iChars}]\\s*[${cChars}]\\s*[${kChars}]`,
    'gi'
  );
  
  let match;
  while ((match = homoglyphPattern.exec(text)) !== null) {
    matches.push(match[0]);
  }
  
  return {
    detected: matches.length > 0,
    matches
  };
}

// Main detection function
export function detectKickVariations(text: string): DetectionResult {
  const results: DetectionResult = {
    detected: false,
    confidence: 0,
    matches: [],
    techniques: [],
    positions: [],
    hasLegitimateUsage: false
  };
  
  // First, check for zero-width characters
  const { normalized: zeroWidthNormalized, hasZeroWidth, positionMap } = normalizeText(text);
  
  // If zero-width characters were found, add to techniques
  if (hasZeroWidth) {
    results.techniques.push('zero_width');
  }
  
  // Normalize for analysis (but keep original for position tracking)
  const normalizedText = zeroWidthNormalized.toLowerCase();
  
  // Pattern matching with position tracking
  kickVariationPatterns.forEach((pattern, index) => {
    const regex = new RegExp(pattern.source, pattern.flags);
    let match;
    
    while ((match = regex.exec(normalizedText)) !== null) {
      results.detected = true;
      
      // Avoid duplicate matches
      if (!results.matches.includes(match[0])) {
        results.matches.push(match[0]);
        
        // Map positions back to original text if zero-width characters were present
        const start = hasZeroWidth && match.index < positionMap.length ? 
          positionMap[match.index] : match.index;
        const endNormalizedPos = match.index + match[0].length - 1;
        const end = hasZeroWidth && endNormalizedPos < positionMap.length ? 
          positionMap[endNormalizedPos] + 1 : match.index + match[0].length;
        
        results.positions.push({
          start,
          end
        });
        
        // Identify technique used
        if (index === 0) {
          results.techniques.push('character_substitution');
        } else if (index === 1) {
          results.techniques.push('separators');
        } else if (index === 2) {
          results.techniques.push('double_separators');
        } else if (index === 3) {
          results.techniques.push('character_repetition');
        } else if (index === 4) {
          results.techniques.push('alternative_spelling');
        } else if (index === 5) {
          results.techniques.push('parentheses');
        } else if (index === 6) {
          results.techniques.push('underscores');
        } else if (index === 7) {
          results.techniques.push('advanced_pattern');
        } else if (index === 8) {
          results.techniques.push('brackets');
        } else if (index === 9) {
          results.techniques.push('missing_letter');
        } else if (index === 10) {
          results.techniques.push('spaces');
        } else if (index === 11) {
          results.techniques.push('general_obfuscation');
        } else if (index === 12) {
          results.techniques.push('extended_parentheses');
        } else if (index === 13) {
          results.techniques.push('multiple_dots');
        } else if (index === 14) {
          results.techniques.push('mixed_separators');
        } else if (index === 15) {
          results.techniques.push('extended_gaps');
        } else if (index === 16) {
          results.techniques.push('extended_gaps');
        } else if (index === 17) {
          results.techniques.push('parentheses');
        } else if (index === 18) {
          results.techniques.push('multi_char_dots');
        } else if (index === 19) {
          results.techniques.push('multi_char_separators');
        } else if (index === 20) {
          results.techniques.push('vowel_patterns');
        } else if (index === 21) {
          results.techniques.push('flexible_middle');
        } else if (index === 22) {
          results.techniques.push('parentheses_ck');
        } else if (index === 23) {
          results.techniques.push('multiple_parentheses');
        } else if (index === 24) {
          results.techniques.push('single_dots');
        } else if (index === 25) {
          results.techniques.push('flexible_dots');
        } else if (index === 26) {
          results.techniques.push('enhanced_parentheses');
        } else if (index === 27) {
          results.techniques.push('double_vowel');
        } else if (index === 28) {
          results.techniques.push('double_vowel');
        } else if (index === 29) {
          results.techniques.push('y_vowel');
        } else if (index === 30) {
          results.techniques.push('vowel_variation');
        } else if (index === 31) {
          results.techniques.push('short_variation');
        } else if (index === 32) {
          results.techniques.push('mixed_vowel');
        } else if (index >= 33) {
          results.techniques.push('domain_pattern');
        }
      }
    }
  });
  
  // Check for homoglyphs (on zero-width normalized text for consistency)
  const homoglyphResult = detectHomoglyphs(normalizedText);
  if (homoglyphResult.detected) {
    results.detected = true;
    results.matches.push(...homoglyphResult.matches);
    results.techniques.push('homoglyph');
  }
  
  // Levenshtein distance check for fuzzy matching
  const words = normalizedText.split(/[\s._\-]+/);
  words.forEach((word) => {
    const cleaned = word.replace(/[^a-z0-9]/g, '');
    if (cleaned.length >= 3 && cleaned.length <= 6) {
      const distance = levenshteinDistance(cleaned, 'kick');
      
      // Known phonetic variations that sound like "kick" (distance 2)
      const phoneticVariations = ['keek', 'keak', 'kyck', 'kyek', 'kouk', 'kaik'];
      const isPhoneticVariation = phoneticVariations.includes(cleaned);
      
      // Allow distance 2 for known phonetic variations, distance 1 for others
      const maxDistance = isPhoneticVariation ? 2 : 1;
      
      if (distance <= maxDistance && distance > 0 && !results.matches.includes(word)) {
        // Additional check: ensure it's not a common English word
        const commonWords = ['tick', 'pick', 'lick', 'sick', 'wick', 'dick', 'nick', 'rick', 
                           'back', 'pack', 'lack', 'sack', 'rack', 'tack', 'hack',
                           'peek', 'meek', 'seek', 'week', 'keep', 'keen', 'kayak'];
        if (!commonWords.includes(cleaned)) {
          results.detected = true;
          results.matches.push(word);
          results.techniques.push('fuzzy_match');
          
          // Find position of this word in original text
          const wordIndex = normalizedText.indexOf(word);
          if (wordIndex !== -1) {
            results.positions.push({
              start: wordIndex,
              end: wordIndex + word.length
            });
          }
        }
      }
    }
  });
  
  // Remove duplicate techniques
  results.techniques = [...new Set(results.techniques)];
  
  // Check for legitimate usage if we found "kick"
  if (results.matches.some(m => m.toLowerCase().replace(/[^a-z]/g, '') === 'kick')) {
    // Find the position of "kick" in the text
    const kickIndex = normalizedText.indexOf('kick');
    if (kickIndex !== -1) {
      results.hasLegitimateUsage = isLegitimateKickUsage(text, kickIndex);
    }
  }
  
  // Calculate confidence based on match quality
  results.confidence = calculateConfidence(results);
  
  return results;
}

// Context-aware analysis to improve detection accuracy
export function contextualAnalysis(text: string, detectionResult: DetectionResult): DetectionResult {
  const result = { ...detectionResult };
  
  // Check surrounding context for streaming/platform references
  const streamingKeywords = /(stream|live|channel|watch|follow|subscribe|viewer)/i;
  const urlContext = /(https?|www|\.com|\.tv|link|url|visit|check out)/i;
  const platformContext = /(twitch|youtube|platform|broadcast|content)/i;
  
  if (result.detected) {
    let contextBoost = 0;
    
    // Increase confidence if streaming context found
    if (streamingKeywords.test(text)) {
      contextBoost += 15;
      result.techniques.push('streaming_context');
    }
    
    if (urlContext.test(text)) {
      contextBoost += 10;
      result.techniques.push('url_context');
    }
    
    if (platformContext.test(text)) {
      contextBoost += 10;
      result.techniques.push('platform_context');
    }
    
    result.confidence = Math.min(100, result.confidence + contextBoost);
  }
  
  return result;
}

// Cache implementation for performance
const detectionCache = new Map<string, { result: DetectionResult; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

export function cachedDetection(text: string): DetectionResult {
  const cacheKey = text.toLowerCase().trim();
  const cached = detectionCache.get(cacheKey);
  
  if (cached && Date.now() - cached.timestamp < CACHE_TTL) {
    return cached.result;
  }
  
  const result = detectKickVariations(text);
  detectionCache.set(cacheKey, { result, timestamp: Date.now() });
  
  // Limit cache size
  if (detectionCache.size > 1000) {
    const firstKey = detectionCache.keys().next().value;
    if (firstKey) detectionCache.delete(firstKey);
  }
  
  return result;
}

// Progressive detection for performance optimization
export function progressiveDetection(text: string): DetectionResult {
  // Level 1: Quick pattern check
  // Updated pattern to catch phonetic variations like "keek", "kyck", etc.
  // Matches: k + (various middle patterns) + optional [kcq]
  const quickCheck = /k(?:[^a-z]{0,3}[i1l!|e3aeiouey][^a-z]{0,3}|[aeiouey0-9]{1,2}|\W{0,5}|i[cqk])[kcq]?/i;
  if (!quickCheck.test(text.toLowerCase())) {
    return { 
      detected: false, 
      confidence: 0, 
      matches: [], 
      techniques: [], 
      positions: [] 
    };
  }
  
  // Level 2: Full pattern matching
  const fullResult = detectKickVariations(text);
  
  // Level 3: Context analysis for borderline cases
  if (fullResult.confidence > 40 && fullResult.confidence < 80) {
    return contextualAnalysis(text, fullResult);
  }
  
  return fullResult;
}

// Extract features for machine learning
export interface MLFeatures {
  hasKSound: boolean;
  hasISound: boolean;
  hasCSound: boolean;
  specialCharDensity: number;
  averageWordLength: number;
  maxCharacterGap: number;
  suspiciousPatternCount: number;
  wordCount: number;
}

export function extractMLFeatures(text: string): MLFeatures {
  const words = text.split(/\s+/);
  const specialChars = text.match(/[^a-zA-Z0-9\s]/g) || [];
  const normalized = text.toLowerCase();
  
  // Find maximum gap between characters in potential kick variations
  let maxGap = 0;
  const gapPattern = /k[^a-z]*[i1l!|][^a-z]*[kc]/gi;
  let gapMatch;
  while ((gapMatch = gapPattern.exec(normalized)) !== null) {
    const nonLetters = gapMatch[0].match(/[^a-z]/g) || [];
    maxGap = Math.max(maxGap, nonLetters.length);
  }
  
  return {
    hasKSound: /[kкκ]/i.test(text),
    hasISound: /[i1l!|іí]/i.test(text),
    hasCSound: /[kcсς]/i.test(text),
    specialCharDensity: specialChars.length / Math.max(1, text.length),
    averageWordLength: text.replace(/\s+/g, '').length / Math.max(1, words.length),
    maxCharacterGap: maxGap,
    suspiciousPatternCount: (text.match(/k.{0,5}[i1l!|].{0,5}[kc]/gi) || []).length,
    wordCount: words.length
  };
}