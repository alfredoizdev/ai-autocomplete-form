// Improved Kick.com link detection with false positive reduction
// Uses whitelisting, negative lookahead, and context-aware detection

export interface DetectionResult {
  detected: boolean;
  confidence: number;
  matches: string[];
  techniques: string[];
  positions: Array<{ start: number; end: number }>;
  hasLegitimateUsage?: boolean;
}

// Common phrases that should NOT trigger detection (whitelist)
const LEGITIMATE_PHRASES = [
  'take charge',
  'take care',
  'make changes',
  'cake choice',
  'fake charm',
  'make chemistry',
  'rake chores',
  'wake children',
  'stake claim',
  'brake check',
  'shake champagne',
  'bake chicken',
  'lake charles',
  'make choices',
  'awake challenge',
  'mistake chance',
  'blake chapman',
  'jake chang',
  'drake chase',
  'flake chocolate'
];

// Context words that indicate legitimate usage
const SAFE_CONTEXT_WORDS = [
  'take', 'make', 'wake', 'bake', 'cake', 'fake', 'rake', 'sake', 'brake', 'shake', 'stake',
  'charge', 'change', 'chance', 'choice', 'charm', 'chase', 'check', 'chemistry', 'champagne',
  'chicken', 'children', 'chocolate', 'charles', 'chapman', 'challenge'
];

// Function to check if text contains whitelisted phrases
function containsWhitelistedPhrase(text: string): boolean {
  const lowerText = text.toLowerCase();
  return LEGITIMATE_PHRASES.some(phrase => lowerText.includes(phrase));
}

// Function to check surrounding context for legitimacy
function hasLegitimateContext(text: string, matchPosition: number): boolean {
  const contextRadius = 20; // Check 20 chars before and after
  const start = Math.max(0, matchPosition - contextRadius);
  const end = Math.min(text.length, matchPosition + contextRadius);
  const context = text.substring(start, end).toLowerCase();
  
  // Check if any safe context words are present
  return SAFE_CONTEXT_WORDS.some(word => context.includes(word));
}

// Improved patterns with negative lookahead
const improvedKickPatterns = [
  // Basic "kick" with negative lookahead for common false positives
  /(?<!ta|ma|wa|ba|ca|fa|ra|sa|bra|sha|sta)k[i1l!|]c(?!k)/gi,
  
  // Separated k.i.c patterns (but exclude legitimate phrases)
  /(?<!ta|ma|wa|ba|ca|fa|ra|sa|bra|sha|sta)k[._\-]{1,3}[i1l!|][._\-]{0,3}[c](?!h|k)/gi,
  
  // More specific patterns with word boundaries
  /\b(?<!ta|ma|wa|ba|ca|fa|ra|sa|bra|sha|sta)k[._\-]{1,3}[i1l!|][._\-]{1,3}ck\b/gi,
  
  // Patterns that require non-letter characters around them (more likely to be obfuscation)
  /[^a-z]k[^a-z]{0,3}[i1l!|][^a-z]{0,3}[kc][^a-z]/gi,
  
  // Parentheses patterns (these are highly suspicious)
  /\bk\([^)]{0,3}\)[kc]\b/gi,
  /\([kc]\)[^a-z]{0,3}[i1l!|][^a-z]{0,3}[kc]\b/gi,
  
  // Multiple underscores or dots (clear obfuscation)
  /\bk[_\.]{2,}[i1l!|][_\.]{0,}[kc]\b/gi,
  
  // Zero-width and special Unicode patterns (highly suspicious)
  /k[\u200B-\u200D\uFEFF\u00AD]*[i1l!|][\u200B-\u200D\uFEFF\u00AD]*[kc]/gi,
];

// Function to calculate confidence based on obfuscation technique
function calculateConfidence(match: string, technique: string): number {
  let confidence = 0.5; // Base confidence
  
  // High confidence for clear obfuscation patterns
  if (technique.includes('parentheses') || technique.includes('zero-width')) {
    confidence = 0.9;
  }
  // Medium-high confidence for multiple separators
  else if (technique.includes('multiple-separators') || match.includes('__') || match.includes('..')) {
    confidence = 0.8;
  }
  // Lower confidence for simple variations
  else if (technique.includes('basic')) {
    confidence = 0.6;
  }
  
  // Reduce confidence if match is very short
  if (match.length <= 4) {
    confidence *= 0.7;
  }
  
  return confidence;
}

export function detectKickLinks(text: string): DetectionResult {
  const matches: string[] = [];
  const techniques: string[] = [];
  const positions: Array<{ start: number; end: number }> = [];
  let highestConfidence = 0;
  
  // First check: if text contains whitelisted phrases, skip certain patterns
  const hasWhitelisted = containsWhitelistedPhrase(text);
  
  // Note: We could remove zero-width characters for analysis if needed
  // const cleanText = text.replace(/[\u200B-\u200D\uFEFF\u00AD]/g, '');
  
  // Check each pattern
  improvedKickPatterns.forEach((pattern, index) => {
    // Skip basic patterns if we have whitelisted phrases
    if (hasWhitelisted && index < 3) {
      return;
    }
    
    const regex = new RegExp(pattern.source, pattern.flags);
    let match;
    
    while ((match = regex.exec(text)) !== null) {
      const matchText = match[0];
      const position = match.index;
      
      // Check if this match has legitimate context
      if (hasLegitimateContext(text, position)) {
        continue; // Skip this match
      }
      
      // Determine technique used
      let technique = 'unknown';
      if (matchText.includes('(') || matchText.includes(')')) {
        technique = 'parentheses-obfuscation';
      } else if (matchText.includes('__') || matchText.includes('..') || matchText.includes('--')) {
        technique = 'multiple-separators';
      } else if (/[\u200B-\u200D\uFEFF\u00AD]/.test(matchText)) {
        technique = 'zero-width-characters';
      } else if (matchText.includes('_') || matchText.includes('.') || matchText.includes('-')) {
        technique = 'separator-obfuscation';
      } else if (/[1l!|]/.test(matchText)) {
        technique = 'character-substitution';
      } else {
        technique = 'basic-variation';
      }
      
      // Calculate confidence for this match
      const confidence = calculateConfidence(matchText, technique);
      if (confidence > highestConfidence) {
        highestConfidence = confidence;
      }
      
      // Only include if confidence is above threshold
      if (confidence >= 0.6) {
        matches.push(matchText);
        techniques.push(technique);
        positions.push({ start: position, end: position + matchText.length });
      }
    }
  });
  
  // Additional check: look for ".com" near potential kick references
  const hasDotCom = /\.(com|net|org|tv|live)/i.test(text);
  if (hasDotCom && matches.length > 0) {
    highestConfidence = Math.min(highestConfidence * 1.2, 1.0);
  }
  
  return {
    detected: matches.length > 0 && highestConfidence >= 0.6,
    confidence: highestConfidence,
    matches: [...new Set(matches)], // Remove duplicates
    techniques: [...new Set(techniques)],
    positions,
    hasLegitimateUsage: hasWhitelisted
  };
}

// Function to test if a specific phrase would trigger detection
export function testPhrase(phrase: string): { wouldTrigger: boolean; reason?: string } {
  const result = detectKickLinks(phrase);
  
  if (!result.detected) {
    return { wouldTrigger: false };
  }
  
  // Check if it's a known false positive
  if (containsWhitelistedPhrase(phrase)) {
    return { 
      wouldTrigger: false, 
      reason: 'Phrase is whitelisted as legitimate usage' 
    };
  }
  
  return {
    wouldTrigger: true,
    reason: `Detected with ${(result.confidence * 100).toFixed(0)}% confidence using ${result.techniques.join(', ')}`
  };
}

// Export a function to add custom whitelist entries
export function addToWhitelist(phrase: string): void {
  if (!LEGITIMATE_PHRASES.includes(phrase.toLowerCase())) {
    LEGITIMATE_PHRASES.push(phrase.toLowerCase());
  }
}

// Export a function to check current whitelist
export function getWhitelist(): string[] {
  return [...LEGITIMATE_PHRASES];
}