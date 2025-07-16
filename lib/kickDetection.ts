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
  // Direct "kick" detection (for platform references)
  /\bkick\b/gi,
  
  // Basic character substitution and spacing (with word boundaries where possible)
  /\bk\s*[i1l!|]\s*[kc]\b/gi,
  
  // Specific pattern for 'l' substitution (kilk, kllk, klck)
  /\bk[il1]+[kc]\b/gi,
  
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
  // Removed - this was causing too many false positives like "pack", "back", "deck"
  // /\bk[^a-z]{1,3}[kc]\b/gi,
  
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
  
  // PHASE 2.3 ADDITIONS - "hk" ending patterns:
  
  // Basic "hk" endings (kihk, k1hk, klhk)
  /\bk[i1l!|]hk\b/gi,
  
  // "hk" with separators (k..i..hk, k-i-hk, k_i_hk)
  /\bk[._\-]{1,3}[i1l!|][._\-]{0,3}hk\b/gi,
  
  // General pattern with "hk" ending
  /\bk[^a-z]{0,5}[i1l!|e3][^a-z]{0,5}hk\b/gi,
  
  // Phonetic variations with "hk" (keehk, kaihk, kyahk)
  /\bk[aeiouey]{1,2}hk\b/gi,
  
  // Complex patterns with "hk" (k(i)hk, k(..i..)hk)
  /\bk\([^)]{0,8}\)hk\b/gi,
  
  // Extended "hk" endings (khk, chk) - k..i..khk, k..i..chk
  /\bk[^a-z]{0,5}[i1l!|e3][^a-z]{0,5}[kc]hk\b/gi,
  
  // Domain patterns (more flexible boundaries for URLs)
  /k[i1l!|._\-\s]{1,4}[kc]\s*[\.\,\·\•]\s*c[o0]m/gi,
  /k[i1l!|._\-\s]{1,4}[kc]\s+dot\s+c[o0]m/gi,
  /k[i1l!|._\-\s]{1,4}[kc]\[?\.\]?\s*c[o0]m/gi,
];

// Unicode confusables that look like 'kick' characters
// Comprehensive database of homoglyphs from various Unicode blocks
const homoglyphs: Record<string, string[]> = {
  'k': [
    // Cyrillic letters
    'к',     // U+043A - Cyrillic small letter ka
    'К',     // U+041A - Cyrillic capital letter ka
    'ҡ',     // U+04A1 - Cyrillic small letter bashkir ka
    'ҝ',     // U+049D - Cyrillic small letter ka with vertical stroke
    'ќ',     // U+045C - Cyrillic small letter kje
    'қ',     // U+049B - Cyrillic small letter ka with descender
    'ҟ',     // U+049F - Cyrillic small letter ka with stroke
    'ҝ',     // U+049D - Cyrillic small letter ka with vertical stroke
    
    // Greek letters
    'κ',     // U+03BA - Greek small letter kappa
    'Κ',     // U+039A - Greek capital letter kappa
    'ϰ',     // U+03F0 - Greek kappa symbol
    
    // Latin extended
    'ķ',     // U+0137 - Latin small letter k with cedilla
    'ĸ',     // U+0138 - Latin small letter kra
    'ḱ',     // U+1E31 - Latin small letter k with acute
    'ḳ',     // U+1E33 - Latin small letter k with dot below
    'ḵ',     // U+1E35 - Latin small letter k with line below
    'ⱪ',     // U+2C6A - Latin small letter k with descender
    '𝐤',    // U+1D424 - Mathematical bold small k
    '𝑘',    // U+1D458 - Mathematical italic small k
    '𝒌',    // U+1D48C - Mathematical bold italic small k
    '𝓀',    // U+1D4C0 - Mathematical script small k
    '𝔨',    // U+1D528 - Mathematical fraktur small k
    '𝕜',    // U+1D55C - Mathematical double-struck small k
    
    // Fullwidth
    'ｋ',    // U+FF4B - Fullwidth Latin small letter k
  ],
  
  'i': [
    // Cyrillic letters
    'і',     // U+0456 - Cyrillic small letter byelorussian-ukrainian i
    'І',     // U+0406 - Cyrillic capital letter byelorussian-ukrainian i
    'ї',     // U+0457 - Cyrillic small letter yi
    'Ї',     // U+0407 - Cyrillic capital letter yi
    
    // Greek letters
    'ι',     // U+03B9 - Greek small letter iota
    'Ι',     // U+0399 - Greek capital letter iota
    'ί',     // U+03AF - Greek small letter iota with tonos
    'ἰ',     // U+1F30 - Greek small letter iota with psili
    'ἱ',     // U+1F31 - Greek small letter iota with dasia
    'ϊ',     // U+03CA - Greek small letter iota with dialytika
    
    // Latin extended and diacritics
    'í',     // U+00ED - Latin small letter i with acute
    'ì',     // U+00EC - Latin small letter i with grave
    'ï',     // U+00EF - Latin small letter i with diaeresis
    'î',     // U+00EE - Latin small letter i with circumflex
    'ī',     // U+012B - Latin small letter i with macron
    'ĭ',     // U+012D - Latin small letter i with breve
    'į',     // U+012F - Latin small letter i with ogonek
    'ı',     // U+0131 - Latin small letter dotless i
    'ḭ',     // U+1E2D - Latin small letter i with tilde below
    'ḯ',     // U+1E2F - Latin small letter i with diaeresis and acute
    'ỉ',     // U+1EC9 - Latin small letter i with hook above
    'ị',     // U+1ECB - Latin small letter i with dot below
    
    // Mathematical symbols
    '𝐢',    // U+1D422 - Mathematical bold small i
    '𝑖',    // U+1D456 - Mathematical italic small i  
    '𝒊',    // U+1D48A - Mathematical bold italic small i
    '𝓲',    // U+1D4F2 - Mathematical script small i
    '𝔦',    // U+1D526 - Mathematical fraktur small i
    '𝕚',    // U+1D55A - Mathematical double-struck small i
    
    // Look-alike numbers and symbols
    '1',     // Digit one
    'l',     // Latin small letter l
    '|',     // Vertical bar
    '!',     // Exclamation mark
    'ǀ',     // U+01C0 - Latin letter dental click
    'ⅰ',     // U+2170 - Small Roman numeral one
    'Ⅰ',     // U+2160 - Roman numeral one
    '⏽',     // U+23FD - Power on symbol
    '│',     // U+2502 - Box drawings light vertical
    '┃',     // U+2503 - Box drawings heavy vertical
    '∣',     // U+2223 - Divides
    
    // Fullwidth
    'ｉ',    // U+FF49 - Fullwidth Latin small letter i
  ],
  
  'c': [
    // Cyrillic letters
    'с',     // U+0441 - Cyrillic small letter es
    'С',     // U+0421 - Cyrillic capital letter es
    'ҫ',     // U+04AB - Cyrillic small letter es with descender
    
    // Greek letters
    'ς',     // U+03C2 - Greek small letter final sigma
    'σ',     // U+03C3 - Greek small letter sigma (in some fonts)
    'ϲ',     // U+03F2 - Greek lunate sigma symbol
    'Ϲ',     // U+03F9 - Greek capital lunate sigma symbol
    
    // Latin extended
    'ċ',     // U+010B - Latin small letter c with dot above
    'ĉ',     // U+0109 - Latin small letter c with circumflex
    'ć',     // U+0107 - Latin small letter c with acute
    'č',     // U+010D - Latin small letter c with caron
    'ç',     // U+00E7 - Latin small letter c with cedilla
    'ḉ',     // U+1E09 - Latin small letter c with cedilla and acute
    'ȼ',     // U+023C - Latin small letter c with stroke
    'ƈ',     // U+0188 - Latin small letter c with hook
    
    // Mathematical symbols
    '𝐜',    // U+1D41C - Mathematical bold small c
    '𝑐',    // U+1D450 - Mathematical italic small c
    '𝒄',    // U+1D484 - Mathematical bold italic small c
    '𝓬',    // U+1D4EC - Mathematical script small c
    '𝔠',    // U+1D520 - Mathematical fraktur small c
    '𝕔',    // U+1D554 - Mathematical double-struck small c
    
    // Look-alike symbols
    '⊂',     // U+2282 - Subset of
    '⟨',     // U+27E8 - Mathematical left angle bracket
    '〈',     // U+3008 - Left angle bracket
    '﹤',     // U+FE64 - Small less-than sign
    '＜',     // U+FF1C - Fullwidth less-than sign
    'ϲ',     // U+03F2 - Greek lunate sigma symbol
    
    // Removed regular parenthesis '(' as it breaks pattern matching
    // Only keeping decorative parentheses that actually look like 'c'
    '❨',     // U+2768 - Medium left parenthesis ornament
    '⁽',     // U+207D - Superscript left parenthesis
    
    // Fullwidth
    'ｃ',    // U+FF43 - Fullwidth Latin small letter c
  ],
  
  // Additional mapping for 'h' to detect "hk" endings with homoglyphs
  'h': [
    // Cyrillic letters
    'һ',     // U+04BB - Cyrillic small letter shha
    'Һ',     // U+04BA - Cyrillic capital letter shha
    'н',     // U+043D - Cyrillic small letter en (looks like h in some fonts)
    'Н',     // U+041D - Cyrillic capital letter en
    
    // Greek letters
    'η',     // U+03B7 - Greek small letter eta (in some fonts)
    'ή',     // U+03AE - Greek small letter eta with tonos
    
    // Latin extended
    'ħ',     // U+0127 - Latin small letter h with stroke
    'ĥ',     // U+0125 - Latin small letter h with circumflex
    'ḣ',     // U+1E23 - Latin small letter h with dot above
    'ḥ',     // U+1E25 - Latin small letter h with dot below
    'ḧ',     // U+1E27 - Latin small letter h with diaeresis
    'ḩ',     // U+1E29 - Latin small letter h with cedilla
    'ḫ',     // U+1E2B - Latin small letter h with breve below
    'ẖ',     // U+1E96 - Latin small letter h with line below
    
    // Mathematical symbols
    '𝐡',    // U+1D421 - Mathematical bold small h
    '𝒉',    // U+1D489 - Mathematical bold italic small h
    'ℎ',     // U+210E - Planck constant
    
    // Fullwidth
    'ｈ',    // U+FF48 - Fullwidth Latin small letter h
  ]
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

// Create reverse mapping for homoglyph normalization
const homoglyphToBase: Map<string, string> = new Map();

// Initialize the reverse mapping
function initializeHomoglyphMapping() {
  for (const [base, variants] of Object.entries(homoglyphs)) {
    for (const variant of variants) {
      homoglyphToBase.set(variant, base);
    }
  }
  // Also map the base characters to themselves
  homoglyphToBase.set('k', 'k');
  homoglyphToBase.set('i', 'i');
  homoglyphToBase.set('c', 'c');
  homoglyphToBase.set('h', 'h');
}

// Initialize on module load
initializeHomoglyphMapping();

// Normalize homoglyphs to their base Latin characters
export function normalizeHomoglyphs(text: string): {
  normalized: string;
  hasHomoglyphs: boolean;
  homoglyphCount: number;
  detectedHomoglyphs: string[];
} {
  let normalized = '';
  let hasHomoglyphs = false;
  let homoglyphCount = 0;
  const detectedHomoglyphs: string[] = [];
  
  for (const char of text) {
    const baseChar = homoglyphToBase.get(char);
    if (baseChar && baseChar !== char) {
      // Found a homoglyph
      normalized += baseChar;
      hasHomoglyphs = true;
      homoglyphCount++;
      if (!detectedHomoglyphs.includes(char)) {
        detectedHomoglyphs.push(char);
      }
    } else {
      // Regular character or unmapped character
      normalized += char;
    }
  }
  
  return {
    normalized,
    hasHomoglyphs,
    homoglyphCount,
    detectedHomoglyphs
  };
}

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

// Common words to exclude from detection
const EXCLUDED_WORDS = [
  // Common -ck ending words
  'back', 'pack', 'lack', 'sack', 'rack', 'tack', 'hack', 'stack', 'track', 'crack',
  'black', 'attack', 'slack', 'snack', 'whack', 'jack', 'quack', 'shack',
  'deck', 'neck', 'check', 'wreck', 'peck', 'speck', 'fleck',
  'pick', 'tick', 'sick', 'quick', 'stick', 'trick', 'thick', 'click', 'brick', 'flick', 
  'slick', 'chick', 'wick', 'dick', 'nick', 'rick', 'mick',
  'rock', 'lock', 'dock', 'cock', 'shock', 'stock', 'block', 'clock', 'knock', 'flock',
  'mock', 'sock',
  'duck', 'luck', 'suck', 'truck', 'stuck', 'chuck', 'buck', 'muck', 'tuck', 'fuck',
  'pluck', 'struck',
  // -ing words that might trigger patterns
  'tracking', 'picking', 'bucking', 'fucking', 'lacking', 'packing', 'backing',
  'stacking', 'attacking', 'hacking', 'cracking', 'sticking', 'clicking',
  'rocking', 'locking', 'docking', 'shocking', 'stocking', 'blocking', 'knocking',
  'mocking', 'sucking', 'trucking', 'stucking', 'chucking', 'plucking',
  // Other common words
  'neck', 'wreck', 'check'
];

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
  'football kick',
  // Sports/game related
  'kick ball',
  'kickball',
  'play kick',
  'playing kick',
  'played kick',
  'plays kick',
  'kick boxing',
  'kickboxing',
  'field kick',
  'goal kick',
  'corner kick',
  'drop kick',
  'place kick',
  'kick return',
  'kick serve',
  'high kick',
  'low kick',
  'roundhouse kick',
  'flying kick',
  'bicycle kick',
  'scissor kick',
  'kick scooter',
  'kick flip',
  'kick turn'
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
    /want\s+to\s+kick\s+/,
    // Sports/game patterns
    /play\s+kick\s+/,
    /playing\s+kick\s+/,
    /played\s+kick\s+/,
    /plays\s+kick\s+/,
    /game\s+of\s+kick\s+/,
    /kick\s+(game|sport|match|tournament)/,
    /practice\s+kick/,
    /practicing\s+kick/,
    /learn\s+to\s+kick/,
    /learning\s+to\s+kick/,
    /teach\s+.*\s+kick/,
    /coach\s+.*\s+kick/
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
  
  // Check for sports/recreational context
  const sportsKeywords = /(play|game|sport|ball|team|field|court|match|practice|coach|player|athlete|exercise|workout|training|gym|fitness)/i;
  if (sportsKeywords.test(context)) {
    // Double-check it's not a disguised URL
    const suspiciousPatterns = /(kick\s*[\.\/]\s*com|kick\s+dot\s+com|visit\s+kick|go\s+to\s+kick|check\s+out\s+kick)/i;
    if (!suspiciousPatterns.test(context)) {
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
    if (results.techniques?.includes('hk_ending')) {
      techniqueBoost += 15; // "hk" endings are suspicious obfuscation attempts
    }
    if (results.techniques?.includes('advanced_homoglyph')) {
      techniqueBoost += 25; // Advanced homoglyphs are highly suspicious
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
  
  // Build regex pattern for homoglyphs - need to escape special regex characters
  const escapeRegex = (str: string) => str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  
  // Build character sets for each position
  const kChars = ['k', 'K', ...homoglyphs.k].map(escapeRegex).join('');
  const iChars = ['i', 'I', ...homoglyphs.i].map(escapeRegex).join('');
  const cChars = ['c', 'C', ...homoglyphs.c].map(escapeRegex).join('');
  const hChars = homoglyphs.h ? ['h', 'H', ...homoglyphs.h].map(escapeRegex).join('') : '';
  
  // Create patterns for various combinations
  const patterns = [
    // Basic kick pattern with homoglyphs
    new RegExp(`[${kChars}]\\s*[${iChars}]\\s*[${cChars}]\\s*[${kChars}]`, 'gi'),
    // kick with spaces/separators
    new RegExp(`[${kChars}][._\\-\\s]{0,3}[${iChars}][._\\-\\s]{0,3}[${cChars}][._\\-\\s]{0,3}[${kChars}]`, 'gi'),
    // kihk pattern with homoglyphs (hk ending)
    new RegExp(`[${kChars}]\\s*[${iChars}]\\s*[${hChars}]\\s*[${kChars}]`, 'gi'),
    // Just k-i-c (without trailing k)
    new RegExp(`[${kChars}][._\\-\\s]{0,3}[${iChars}][._\\-\\s]{0,3}[${cChars}](?!\\w)`, 'gi'),
  ];
  
  // Apply each pattern
  for (const pattern of patterns) {
    let match;
    while ((match = pattern.exec(text)) !== null) {
      // Check if this contains actual homoglyphs (not just regular letters)
      const hasActualHomoglyph = [...match[0]].some(char => {
        const baseChar = homoglyphToBase.get(char);
        return baseChar && baseChar !== char.toLowerCase();
      });
      
      if (hasActualHomoglyph && !matches.includes(match[0])) {
        matches.push(match[0]);
      }
    }
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
  
  // Then check for homoglyphs (for detection only, not for pattern matching)
  const { 
    hasHomoglyphs, 
    detectedHomoglyphs 
  } = normalizeHomoglyphs(zeroWidthNormalized);
  
  // If homoglyphs were found, add to techniques
  if (hasHomoglyphs) {
    results.techniques.push('advanced_homoglyph');
    // Add specific homoglyph info to matches for debugging
    if (detectedHomoglyphs.length > 0) {
      results.matches.push(`[Homoglyphs: ${detectedHomoglyphs.join(', ')}]`);
    }
  }
  
  // Normalize for analysis (but keep original for position tracking)
  // Use zero-width normalized text for pattern matching to preserve pattern characters
  const normalizedText = zeroWidthNormalized.toLowerCase();
  
  // Pattern matching with position tracking
  kickVariationPatterns.forEach((pattern, index) => {
    const regex = new RegExp(pattern.source, pattern.flags);
    let match;
    
    while ((match = regex.exec(normalizedText)) !== null) {
      // Check if the match is an excluded word
      const matchLower = match[0].toLowerCase().trim();
      if (EXCLUDED_WORDS.includes(matchLower)) {
        continue; // Skip excluded words
      }
      
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
        
        // For direct "kick" pattern, check if it's legitimate usage
        if (index === 0) {
          // Check if this is a legitimate usage
          const originalPosition = hasZeroWidth && match.index < positionMap.length ? 
            positionMap[match.index] : match.index;
          
          if (isLegitimateKickUsage(text, originalPosition)) {
            // Remove this match as it's legitimate
            results.matches.pop();
            results.positions.pop();
            // Don't mark as detected if this was the only match
            if (results.matches.length === 0) {
              results.detected = false;
            }
            continue; // Skip to next match
          }
          results.techniques.push('direct_kick');
        } else if (index === 1) {
          results.techniques.push('character_substitution');
        } else if (index === 2) {
          results.techniques.push('l_substitution');
        } else if (index === 3) {
          results.techniques.push('separators');
        } else if (index === 4) {
          results.techniques.push('double_separators');
        } else if (index === 5) {
          results.techniques.push('character_repetition');
        } else if (index === 6) {
          results.techniques.push('alternative_spelling');
        } else if (index === 7) {
          results.techniques.push('parentheses');
        } else if (index === 8) {
          results.techniques.push('underscores');
        } else if (index === 9) {
          results.techniques.push('advanced_pattern');
        } else if (index === 10) {
          results.techniques.push('brackets');
        } else if (index === 11) {
          results.techniques.push('missing_letter');
        } else if (index === 12) {
          results.techniques.push('spaces');
        } else if (index === 13) {
          results.techniques.push('general_obfuscation');
        } else if (index === 14) {
          results.techniques.push('extended_parentheses');
        } else if (index === 15) {
          results.techniques.push('multiple_dots');
        } else if (index === 16) {
          results.techniques.push('mixed_separators');
        } else if (index === 17) {
          results.techniques.push('extended_gaps');
        } else if (index === 18) {
          results.techniques.push('extended_gaps');
        } else if (index === 19) {
          results.techniques.push('parentheses');
        } else if (index === 20) {
          results.techniques.push('multi_char_dots');
        } else if (index === 21) {
          results.techniques.push('multi_char_separators');
        } else if (index === 22) {
          results.techniques.push('vowel_patterns');
        } else if (index === 23) {
          results.techniques.push('flexible_middle');
        } else if (index === 24) {
          results.techniques.push('parentheses_ck');
        } else if (index === 25) {
          results.techniques.push('multiple_parentheses');
        } else if (index === 26) {
          results.techniques.push('single_dots');
        } else if (index === 27) {
          results.techniques.push('flexible_dots');
        } else if (index === 28) {
          results.techniques.push('enhanced_parentheses');
        } else if (index === 29) {
          results.techniques.push('double_vowel');
        } else if (index === 30) {
          results.techniques.push('double_vowel');
        } else if (index === 31) {
          results.techniques.push('y_vowel');
        } else if (index === 32) {
          results.techniques.push('vowel_variation');
        } else if (index === 33) {
          results.techniques.push('short_variation');
        } else if (index === 34) {
          results.techniques.push('mixed_vowel');
        } else if (index === 35) {
          results.techniques.push('hk_ending');
        } else if (index === 36) {
          results.techniques.push('hk_ending');
        } else if (index === 37) {
          results.techniques.push('hk_ending');
        } else if (index === 38) {
          results.techniques.push('hk_ending');
        } else if (index === 39) {
          results.techniques.push('hk_ending');
        } else if (index === 40) {
          results.techniques.push('hk_ending');
        } else if (index >= 41) {
          results.techniques.push('domain_pattern');
        }
      }
    }
  });
  
  // Check for homoglyphs on the original text (after zero-width normalization)
  // This ensures we detect the actual homoglyphs before normalization
  const homoglyphResult = detectHomoglyphs(zeroWidthNormalized);
  if (homoglyphResult.detected) {
    results.detected = true;
    results.matches.push(...homoglyphResult.matches);
    if (!results.techniques.includes('homoglyph')) {
      results.techniques.push('homoglyph');
    }
  }
  
  // Levenshtein distance check for fuzzy matching
  const words = normalizedText.split(/[\s._\-]+/);
  words.forEach((word) => {
    const cleaned = word.replace(/[^a-z0-9]/g, '');
    if (cleaned.length >= 3 && cleaned.length <= 6) {
      // Skip if the word is in the excluded list
      if (EXCLUDED_WORDS.includes(cleaned) || EXCLUDED_WORDS.includes(word.toLowerCase())) {
        return;
      }
      
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
  // Updated pattern to catch phonetic variations like "keek", "kyck", etc. and "hk" endings
  // Matches: k + (various middle patterns) + optional [kchq] or hk
  const quickCheck = /k(?:[^a-z]{0,3}[i1l!|e3aeiouey][^a-z]{0,3}|[aeiouey0-9]{1,2}|\W{0,5}|i[cqk])(?:[kchq]{0,2}|hk)?/i;
  if (!quickCheck.test(text.toLowerCase())) {
    return { 
      detected: false, 
      confidence: 0, 
      matches: [], 
      techniques: [], 
      positions: [] 
    };
  }
  
  // Check if the text contains only excluded words
  const words = text.toLowerCase().split(/[\s._\-]+/);
  const nonExcludedWords = words.filter(word => !EXCLUDED_WORDS.includes(word));
  if (nonExcludedWords.length === 0) {
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