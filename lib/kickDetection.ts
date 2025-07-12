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
  
  // Repeated characters
  /\bk[i1l!|]{2,4}[kc]\b/gi,
  
  // Alternative spellings (keek, keik, kiek)
  /\bk[e3][e3i1][kc]\b/gi,
  
  // Parentheses patterns
  /\bk\([i1l!|]\)[kc]\b/gi,
  
  // Multiple underscores
  /\bk_{1,3}[i1l!|]_{0,3}[kc]\b/gi,
  
  // Advanced pattern with any non-letter chars between
  /\bk[^a-z]{0,3}[i1l!|e3][^a-z]{0,3}[kc]\b/gi,
  
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
function calculateConfidence(results: Partial<DetectionResult>, text?: string): number {
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
  
  // Normalize for analysis (but keep original for position tracking)
  const normalizedText = text.toLowerCase();
  
  // Pattern matching with position tracking
  kickVariationPatterns.forEach((pattern, index) => {
    const regex = new RegExp(pattern.source, pattern.flags);
    let match;
    
    while ((match = regex.exec(normalizedText)) !== null) {
      results.detected = true;
      
      // Avoid duplicate matches
      if (!results.matches.includes(match[0])) {
        results.matches.push(match[0]);
        results.positions.push({
          start: match.index,
          end: match.index + match[0].length
        });
        
        // Identify technique used
        if (index === 0) {
          results.techniques.push('character_substitution');
        } else if (index === 1) {
          results.techniques.push('separators');
        } else if (index === 2) {
          results.techniques.push('character_repetition');
        } else if (index === 3) {
          results.techniques.push('alternative_spelling');
        } else if (index === 4) {
          results.techniques.push('parentheses');
        } else if (index === 5) {
          results.techniques.push('underscores');
        } else if (index >= 7) {
          results.techniques.push('domain_pattern');
        }
      }
    }
  });
  
  // Check for homoglyphs
  const homoglyphResult = detectHomoglyphs(normalizedText);
  if (homoglyphResult.detected) {
    results.detected = true;
    results.matches.push(...homoglyphResult.matches);
    results.techniques.push('homoglyph');
  }
  
  // Levenshtein distance check for fuzzy matching
  const words = normalizedText.split(/[\s._\-]+/);
  words.forEach((word, wordIdx) => {
    const cleaned = word.replace(/[^a-z0-9]/g, '');
    if (cleaned.length >= 3 && cleaned.length <= 6) {
      const distance = levenshteinDistance(cleaned, 'kick');
      // More strict threshold: only 1 edit for "kick" variations
      // This prevents matching words like "back", "tick", "pick", etc.
      if (distance === 1 && !results.matches.includes(word)) {
        // Additional check: ensure it's not a common English word
        const commonWords = ['tick', 'pick', 'lick', 'sick', 'wick', 'dick', 'nick', 'rick'];
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
  results.confidence = calculateConfidence(results, text);
  
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
  const quickCheck = /k[^a-z]{0,3}[i1l!|][^a-z]{0,3}[kc]/i;
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