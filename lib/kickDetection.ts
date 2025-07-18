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
  
  // PHASE 2.3 ADDITIONS - "hk" ending patterns:
  
  // Basic "hk" endings (kihk, k1hk, klhk)
  /\bk[i1l!|]hk\b/gi,
  
  // "hk" with separators (k..i..hk, k-i-hk, k_i_hk)
  /\bk[._\-]{1,3}[i1l!|][._\-]{0,3}hk\b/gi,
  
  // General pattern with "hk" ending (increased character limit from 5 to 10)
  /\bk[^a-z]{0,10}[i1l!|e3][^a-z]{0,10}hk\b/gi,
  
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
  
  // PHASE 4 ADDITIONS - Advanced Obfuscation Patterns
  
  // Reversed patterns (with strict boundaries)
  /\b[ck][ck][i1!|][kc]\b/gi,        // ckik, ccik, kkic, kkik
  /\b[ck][i1!|][ck][kc]\b/gi,        // cikk, cick, kick
  
  // Nested brackets (2-3 levels deep)
  /\bk\({2,3}[i1!|l]\){2,3}[kc]\b/gi,     // k((i))k, k(((i)))k
  /\bk\[{2,3}[i1!|l]\]{2,3}[kc]\b/gi,     // k[[i]]k, k[[[i]]]k
  /\bk\{{2,3}[i1!|l]\}{2,3}[kc]\b/gi,     // k{{i}}k, k{{{i}}}k
  /\bk<{2,3}[i1!|l]>{2,3}[kc]\b/gi,       // k<<i>>k, k<<<i>>>k
  
  // Extreme gaps with reasonable limits (6-12 chars)
  /\bk[^a-z]{6,12}[i1!|l][^a-z]{6,12}[kc]\b/gi,  // k------i------k
  
  // PHASE 5 ADDITIONS - Parentheses-wrapped K patterns
  
  // K wrapped in parentheses at the beginning
  /\([kc]\)[^a-z]{0,20}[i1l!|e3][^a-z]{0,20}[kc]\b/gi,      // (k)...i...k
  
  // K wrapped in parentheses at the end
  /\b[kc][^a-z]{0,20}[i1l!|e3][^a-z]{0,20}\([kc]\)/gi,      // k...i...(k)
  
  // Both K's wrapped in parentheses
  /\([kc]\)[^a-z]{0,20}[i1l!|e3][^a-z]{0,20}\([kc]\)/gi,    // (k)...i...(k)
  
  // Mixed dots and underscores with parentheses-wrapped K
  /\([kc]\)[._]{2,}[^a-z]*[i1l!|e3][^a-z]*[._]{2,}[kc]\b/gi, // (k)..___i___..k
  
  // Extreme separator patterns (up to 20 chars)
  /\bk[^a-z]{13,20}[i1l!|e3][^a-z]{13,20}[kc]\b/gi,         // k-----(many)-----i-----(many)-----k
  
  // Parentheses + any separators pattern
  /\([kc]\)[^a-z]*[i1l!|e3][^a-z]*[kc]\b/gi,                // (k)[anything]i[anything]k
  /\b[kc][^a-z]*[i1l!|e3][^a-z]*\([kc]\)/gi,                // k[anything]i[anything](k)
  
  // PHASE 6 ADDITIONS - Multiple distributed parentheses patterns
  
  // Middle vowel/character wrapped in parentheses
  /\([kc]\)[^a-z]*\([i1l!|e3aeiouey]\)[^a-z]*[kc]\b/gi,     // (k)...(i)...k
  /\b[kc][^a-z]*\([i1l!|e3aeiouey]\)[^a-z]*\([kc]\)/gi,     // k...(i)...(k)
  /\([kc]\)[^a-z]*\([i1l!|e3aeiouey]\)[^a-z]*\([kc]\)/gi,   // (k)...(i)...(k)
  
  // H-wrapped endings (like (h)k or (h)c)
  /\b[kc][^a-z]*\([i1l!|e3aeiouey]\)[^a-z]*\(h\)[kc]/gi,    // k...(i)...(h)k
  /\([kc]\)[^a-z]*\([i1l!|e3aeiouey]\)[^a-z]*\(h\)[kc]/gi,  // (k)...(i)...(h)k
  
  // Multiple separate parentheses groups (2-3 groups)
  /\([kc]\)[^a-z]*\([^)]+\)[^a-z]*\([^)]+\)[^a-z]*[kc]/gi,  // (k)...(any)...(any)...k
  /\b[kc][^a-z]*\([^)]+\)[^a-z]*\([^)]+\)[^a-z]*[kc]\b/gi,  // k...(any)...(any)...k
  
  // Specific pattern for (k)__(I)..__(h)k style
  /\([kc]\)[_\.]{2,}\([i1l!|e3aeiouey]\)[_\.]{2,}\(h\)[kc]/gi,  // (k)__..(I).._.(h)k
  
  // PHASE 7 ADDITIONS - Unclosed/unmatched parentheses patterns
  
  // Unclosed opening parenthesis at start
  /\([kc][^)]*[i1l!|e3aeiouey][^a-z]*[kc]\b/gi,              // (k__i__k (no closing paren after k)
  /\([kc][^)]*\([i1l!|e3aeiouey][^)]*\([hc]\)[kc]/gi,        // (k__(I..__(h)k (multiple unclosed)
  /\([kc][^)]*[i1l!|e3aeiouey][^a-z]*\([hc]\)[kc]/gi,        // (k__i__(h)k
  
  // Mixed parentheses - some wrapped, some not
  /\([kc]\)[^a-z]*[i1l!|e3aeiouey][^a-z]*\([hc]\)[kc]/gi,    // (k)__I__(h)k (middle char unwrapped)
  /\([kc][^)]*\)[^a-z]*[i1l!|e3aeiouey][^a-z]*[hc]\b/gi,     // (k__)__i__hk
  
  // Flexible parentheses patterns - handle various combinations
  /\(?[kc]\)?[^a-z]*\(?[i1l!|e3aeiouey]\)?[^a-z]*\([hc]\)[kc]/gi,  // Optional parens, but (h)k required
  /\([kc][^)]{0,10}[i1l!|e3aeiouey][^a-z]*[!@#$%^&*]?[hc][^a-z]*[kc]/gi,  // (k__)i__..!h..k pattern
  
  // Catch-all for complex unmatched parentheses with k-vowel-k structure
  /\([kc][^kc]{1,30}[i1l!|e3aeiouey][^kc]{0,30}[kc]\b/gi,    // Very flexible unclosed parenthesis pattern

  // PHASE 8 ADDITIONS - Ultra-complex mixed separator obfuscation
  
  // Angle bracket wrapped characters - k<char>k, k...<char>...k
  /\bk[^a-z]*<[i1l!|e3aeioueyh]>[^a-z]*[kc]\b/gi,           // k<i>k, k...<h>...k
  /\bk<[^>]{0,3}>[^a-z]*[kc]\b/gi,                          // k<i>k, k<..>k
  /\bk[^a-z]*<[^>]{0,3}>[kc]\b/gi,                          // k...<i>k, k<.>k
  
  // Angle bracket wrapped endings - k...<h>k, k<c>k, k<ch>k
  /\bk[^a-z]{0,10}[i1l!|e3aeiouey][^a-z]{0,10}<[hckc]>[kc]?/gi,  // k..i..<h>k, k<c>k
  /\bk[^a-z]{0,10}[i1l!|e3aeiouey][^a-z]{0,10}<[hc][kc]>/gi,     // k..i..<hk>, k<ck>
  
  // Complex mixed separator patterns (dots + exclamation + parentheses + angle brackets)
  /\bk[.\-_!@#$%^&*()]*[i1l!|e3aeiouey][.\-_!@#$%^&*()]*<[^>]*>[kc]*/gi,  // k..!()..<h>k
  /\bk[.\-_!@#$%^&*()<>]*[i1l!|e3aeiouey][.\-_!@#$%^&*()<>]*[kc]\b/gi,    // k..!()<>i<>!..k
  
  // Ultra-complex obfuscation - any combination of special chars with angle brackets
  /\bk[^a-z]{1,15}[i1l!|e3aeiouey][^a-z]{1,15}[kc]\b/gi,       // k(ultra-complex)i(ultra-complex)k
  
  // Specific pattern for the reported case: k..!()..<h>k
  /\bk[.\-_!@#$%^&*()]{2,10}<[hckci1l!|e3aeiouey]>[kc]*/gi,     // k..!()..<h>k style
  
  // Nested angle brackets - k<<i>>k, k<<<h>>>k
  /\bk<{2,4}[i1l!|e3aeioueyh]>{2,4}[kc]\b/gi,                  // k<<i>>k, k<<<h>>>k
  
  // Mixed bracket types with angle brackets - k(<i>)k, k[<h>]k, k{<c>}k
  /\bk[(\[{][^)\]}]*<[^>]*>[^)\]}]*[)\]}][kc]\b/gi,             // k(<i>)k, k[<h>]k
  
  // Extreme mixed separators (10+ character combinations)
  /\bk[^a-z]{10,20}[i1l!|e3aeiouey][^a-z]{10,20}[kc]\b/gi,     // k(many chars)i(many chars)k

  // PHASE 9 ADDITIONS - Truncated/Incomplete obfuscation patterns
  
  // Unclosed angle brackets with phonetic endings - k<..eek, k<..ick, k<..ook (excluding HTML tags)
  /\bk<(?!(?:div|span|body|head|meta|link|input|script|style|html|form|table|button|img|a|p|h[1-6]|br|hr|ul|ol|li|td|tr|th|nav|main|section|article|aside|header|footer|address|blockquote|pre|code|em|strong|small|mark|del|ins|sub|sup|i|b|u|s|q|cite|abbr|dfn|time|data|var|samp|kbd|output|progress|meter|details|summary|fieldset|legend|label|select|optgroup|option|textarea|keygen|datalist|ruby|rt|rp|bdi|bdo|wbr)\b)[^>]{0,10}[eioauy]{2,4}[kc]*\b/gi,  // k<..eek, k<..ick, k<..ook
  /\bk<(?!(?:div|span|body|head|meta|link|input|script|style|html|form|table|button|img|a|p|h[1-6]|br|hr|ul|ol|li|td|tr|th|nav|main|section|article|aside|header|footer|address|blockquote|pre|code|em|strong|small|mark|del|ins|sub|sup|i|b|u|s|q|cite|abbr|dfn|time|data|var|samp|kbd|output|progress|meter|details|summary|fieldset|legend|label|select|optgroup|option|textarea|keygen|datalist|ruby|rt|rp|bdi|bdo|wbr)\b)[^>]{1,10}[i1l!|][kc]{1,2}\b/gi,  // k<..ick, k<..ic
  
  // Unclosed angle brackets with any suspicious content (excluding common HTML tags)
  /\bk<(?!(?:div|span|body|head|meta|link|input|script|style|html|form|table|button|img|a|p|h[1-6]|br|hr|ul|ol|li|td|tr|th|nav|main|section|article|aside|header|footer|address|blockquote|pre|code|em|strong|small|mark|del|ins|sub|sup|i|b|u|s|q|cite|abbr|dfn|time|data|var|samp|kbd|output|progress|meter|details|summary|fieldset|legend|label|select|optgroup|option|textarea|keygen|datalist|ruby|rt|rp|bdi|bdo|wbr)\b)[^>]{2,8}[eioauy]+\b/gi,  // k<..ee, k<..oo, k<..ea
  /\bk<(?!(?:div|span|body|head|meta|link|input|script|style|html|form|table|button|img|a|p|h[1-6]|br|hr|ul|ol|li|td|tr|th|nav|main|section|article|aside|header|footer|address|blockquote|pre|code|em|strong|small|mark|del|ins|sub|sup|i|b|u|s|q|cite|abbr|dfn|time|data|var|samp|kbd|output|progress|meter|details|summary|fieldset|legend|label|select|optgroup|option|textarea|keygen|datalist|ruby|rt|rp|bdi|bdo|wbr)\b)[^>]*[i1l!|e3][^>]*[eioauy]*\b/gi,  // k<..i..e, k<1..o (excluding HTML tags)
  
  // Reversed/malformed angle brackets - k>..something, >k..something  
  /\bk>[^<]{1,8}[eioauy]{2,4}[kc]*\b/gi,                       // k>..eek, k>..ick
  /\b>[kc][^<]{1,8}[eioauy]{2,4}\b/gi,                         // >k..eek, >c..ick
  
  // Single angle bracket with minimal content - k<.., k>.., k<., k>.
  /\bk[<>][.\-_!@#$%^&*()]{1,5}[eioauy]{1,3}[kc]*\b/gi,       // k<..e, k>..o, k<.ea
  /\bk[<>][.\-_!@#$%^&*()]{2,8}\b/gi,                          // k<.., k>.., k<...
  
  // Truncated mixed patterns - combinations that got cut off
  /\bk[<>][^<>a-z]{1,6}[i1l!|e3][^<>a-z]{0,6}[eioauy]*[kc]*\b/gi,  // k<..i..e, k>..1..o
  
  // Ultra-specific for the reported case and variations
  /\bk<[.\-_!@#$%^&*()]{2,6}[eioauy]{2,4}\b/gi,                // k<..eek, k<...ook, k<..ick
  
  // Phonetic variations after unclosed brackets
  /\bk[<>][^<>a-z]*[kq][eioauy]{1,3}[kc]*\b/gi,                // k<..keek, k>..qeek
  /\bk[<>][^<>a-z]*[y][kc]{1,2}\b/gi,                          // k<..yck, k>..yk
  
  // Missing closing bracket with obvious intent - k<i.., k<e.., k<o.. (excluding HTML tags)
  /\bk<(?!(?:div|span|body|head|meta|link|input|script|style|html|form|table|button|img|a|p|h[1-6]|br|hr|ul|ol|li|td|tr|th|nav|main|section|article|aside|header|footer|address|blockquote|pre|code|em|strong|small|mark|del|ins|sub|sup|i|b|u|s|q|cite|abbr|dfn|time|data|var|samp|kbd|output|progress|meter|details|summary|fieldset|legend|label|select|optgroup|option|textarea|keygen|datalist|ruby|rt|rp|bdi|bdo|wbr)\b)[i1l!|e3aeiouey][^>]{1,8}\b/gi,  // k<i.., k<e.., k<1..
  
  // Additional comprehensive "hk" pattern with larger character limits (no word boundary at end for flexibility)
  /\bk[^a-z]{0,15}[i1l!|e3aeiouey][^a-z]{0,15}hk/gi,
  
  // PHASE 10 ADDITIONS - Asterisk-based obfuscation patterns
  
  // Asterisk-wrapped characters - k*i*k, k*e*k, k*I*k
  /\bk[^a-z]*\*[i1l!|e3aeiouey]\*[^a-z]*[kchq]/gi,
  
  // Asterisk-wrapped with "hk" ending - k*i*hk, k*I*...hk
  /\bk[^a-z]*\*[i1l!|e3aeiouey]\*[^a-z]*hk/gi,
  
  // Multiple asterisks patterns - k**i**k, k***e***k
  /\bk\*{1,5}[i1l!|e3aeiouey]\*{1,5}[kchq]/gi,
  
  // Mixed asterisks with other separators - k*i*..__<hk, k*..e..*k
  /\bk\*[^*]*\*[^a-z]*[kchq]/gi,
  /\bk\*[^*]*\*[^a-z]*hk/gi,
  
  // Flexible asterisk patterns with any content between
  /\bk[^a-z]*\*[^*]{1,10}\*[^a-z]*[kchq]/gi,
  
  // Asterisk at various positions - *k*i*k*, k*i*k, etc.
  /\*?k[^a-z]*\*?[i1l!|e3aeiouey]\*?[^a-z]*[kchq]\*?/gi,
  
  // Complex asterisk patterns with mixed separators and angle brackets
  /\bk\*[^*]*\*[._\-<>!@#$%^&()]*[kchq]/gi,
  /\bk\*[^*]*\*[._\-<>!@#$%^&()]*hk/gi,
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

// PHASE 4 ADDITIONS - False Positive Prevention Infrastructure

// Common words that contain k,i,c letters that should NOT be detected
const PHASE4_FALSE_POSITIVE_WORDS = [
  // Words containing k,i,c letters
  'quick', 'quickly', 'quickest', 'quicken', 'quicksand', 'quickie',
  'stick', 'sticker', 'sticky', 'sticks', 'drumstick', 'lipstick', 'chopstick',
  'thick', 'thicker', 'thickest', 'thickness', 'thicken',
  'trick', 'tricky', 'trickster', 'trickle', 'trickery',
  'chicken', 'chick', 'chickpea', 'chicks',
  'cricket', 'click', 'clicked', 'clicking', 'clicker', 'clickbait',
  'tickle', 'pickle', 'nickle', 'fickle', 'trickle', 'prickle',
  'picnic', 'hispanic', 'aspic',
  'brick', 'prick', 'slick', 'flick', 'hickory', 'rickety',
  'ticket', 'wicket', 'thicket', 'picket', 'rickshaw',
  'sidekick', 'homesick', 'seasick', 'carsick', 'airsick',
  // Additional safety words
  'sticking', 'picking', 'kicking', 'licking', 'ticking',
  'quicksilver', 'quicksort', 'quickfire', 'quickdraw'
];

// Helper interface for Phase 4 match results
interface Phase4MatchResult {
  text: string;
  position: number;
  technique: string;
}

// Enhanced context checking for Phase 4 patterns
function isLegitimatePhase4Usage(text: string, match: string, position: number): boolean {
  const contextRadius = 100; // Larger context window
  const context = text.slice(
    Math.max(0, position - contextRadius),
    Math.min(text.length, position + match.length + contextRadius)
  ).toLowerCase();
  
  // Check if part of a false positive word
  for (const word of PHASE4_FALSE_POSITIVE_WORDS) {
    const wordIndex = context.indexOf(word);
    if (wordIndex !== -1) {
      // Calculate if our match is within this word
      const wordStartInText = position - contextRadius + wordIndex;
      const wordEndInText = wordStartInText + word.length;
      if (position >= wordStartInText && position < wordEndInText) {
        return true; // It's part of a legitimate word
      }
    }
  }
  
  // Check if it's part of any whitelisted phrase
  for (const phrase of KICK_WHITELIST_PHRASES) {
    if (context.includes(phrase)) {
      return true;
    }
  }
  
  // Check for sentence structure (capital letter, punctuation)
  const beforeMatch = text.slice(Math.max(0, position - 50), position);
  const afterMatch = text.slice(position + match.length, Math.min(text.length, position + match.length + 50));
  
  // If it's at the start of a sentence or after punctuation, likely legitimate
  if (/^[A-Z]/.test(text.slice(position)) || /[.!?]\s*$/.test(beforeMatch)) {
    // But only if followed by normal words
    if (/^[a-z\s]+[.!?,]?$/i.test(afterMatch.slice(0, 20))) {
      return true;
    }
  }
  
  // Check if surrounded by alphabetic characters (likely part of a word)
  if (position > 0 && position + match.length < text.length) {
    const charBefore = text[position - 1];
    const charAfter = text[position + match.length];
    if (/[a-zA-Z]/.test(charBefore) || /[a-zA-Z]/.test(charAfter)) {
      return true; // Part of a larger word
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

// PHASE 4 DETECTION FUNCTIONS

// Ultra-conservative scrambled pattern detection
function detectScrambledPatterns(text: string): Phase4MatchResult[] {
  const results: Phase4MatchResult[] = [];
  
  // Only check 4-character sequences that are completely isolated
  const words = text.split(/[^a-zA-Z0-9!@#$%^&*()_+=\-|\\]+/);
  
  for (const word of words) {
    // Skip if too short or too long
    if (word.length !== 4) continue;
    
    const wordLower = word.toLowerCase();
    
    // Must contain at least one special character or number (evidence of obfuscation)
    if (!/[0-9!@#$%^&*()_+=\-|\\]/.test(word)) {
      continue;
    }
    
    // Check if it's a known false positive
    if (PHASE4_FALSE_POSITIVE_WORDS.some(fp => fp.includes(wordLower))) {
      continue;
    }
    
    // Count character frequencies
    const chars = wordLower.split('');
    const hasK = chars.some(c => c === 'k' || c === 'c');
    const hasI = chars.some(c => c === 'i' || c === '1' || c === '!' || c === '|' || c === 'l');
    const hasC = chars.filter(c => c === 'k' || c === 'c').length >= 2; // Need 2 k/c chars
    
    // Must have the right characters but in wrong order
    if (hasK && hasI && hasC) {
      // Check if it's NOT already in correct order (kick)
      if (!/k[i1!|l][ck]{2}/.test(wordLower) && !/[ck]{2}[i1!|l]k/.test(wordLower)) {
        // Find position in original text
        const position = text.indexOf(word);
        if (position !== -1) {
          // Final safety check
          if (!isLegitimatePhase4Usage(text, word, position)) {
            results.push({
              text: word,
              position: position,
              technique: 'scrambled_pattern'
            });
          }
        }
      }
    }
  }
  
  return results;
}

// Extremely conservative sliding window detection
function slidingWindowDetection(text: string): Phase4MatchResult[] {
  // Only run on short texts to prevent performance issues
  if (text.length > 500) return [];
  
  const results: Phase4MatchResult[] = [];
  
  for (let windowSize = 4; windowSize <= 6; windowSize++) {
    for (let i = 0; i <= text.length - windowSize; i++) {
      const window = text.slice(i, i + windowSize);
      const windowLower = window.toLowerCase();
      
      // Skip if it's a normal word (all alphabetic)
      if (/^[a-z]+$/.test(windowLower)) {
        continue;
      }
      
      // Must have special characters OTHER THAN SPACES (evidence of obfuscation)
      // Don't count spaces as they're normal in text
      const specialChars = (window.match(/[^a-zA-Z\s]/g) || []).length;
      if (specialChars === 0) {
        continue;
      }
      
      // The window itself must contain obfuscation characters, not just be near spaces
      if (!/[0-9!@#$%^&*()_+=\-|\\]/.test(window)) {
        continue;
      }
      
      // Special char density check (between 20% and 60%)
      const specialDensity = specialChars / window.length;
      if (specialDensity < 0.2 || specialDensity > 0.6) {
        continue;
      }
      
      // Check for required characters
      const hasK = /[kc]/i.test(window);
      const hasI = /[i1!|l]/i.test(window);
      const hasSecondK = (window.match(/[kc]/gi) || []).length >= 2;
      
      if (hasK && hasI && hasSecondK) {
        // Check it's not a false positive word
        let isFalsePositive = false;
        
        // First check if the window itself is a common word
        if (PHASE4_FALSE_POSITIVE_WORDS.includes(windowLower.replace(/[^a-z]/g, ''))) {
          isFalsePositive = true;
        }
        
        // Then check if we're part of a larger word
        if (!isFalsePositive) {
          const extendedWindow = text.slice(Math.max(0, i - 10), Math.min(text.length, i + windowSize + 10));
          for (const fpWord of PHASE4_FALSE_POSITIVE_WORDS) {
            if (extendedWindow.toLowerCase().includes(fpWord)) {
              // Check if our window is actually part of this word
              const fpIndex = extendedWindow.toLowerCase().indexOf(fpWord);
              const fpStart = Math.max(0, i - 10) + fpIndex;
              const fpEnd = fpStart + fpWord.length;
              if (i >= fpStart && i < fpEnd) {
                isFalsePositive = true;
                break;
              }
            }
          }
        }
        
        if (!isFalsePositive && !isLegitimatePhase4Usage(text, window, i)) {
          results.push({
            text: window,
            position: i,
            technique: 'sliding_window'
          });
          
          // Skip ahead to avoid overlapping detections
          i += windowSize - 1;
        }
      }
    }
  }
  
  return results;
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
        } else if (index === 33) {
          results.techniques.push('hk_ending');
        } else if (index === 34) {
          results.techniques.push('hk_ending');
        } else if (index === 35) {
          results.techniques.push('hk_ending');
        } else if (index === 36) {
          results.techniques.push('hk_ending');
        } else if (index === 37) {
          results.techniques.push('hk_ending');
        } else if (index === 38) {
          results.techniques.push('hk_ending');
        } else if (index >= 39 && index <= 41) {
          results.techniques.push('domain_pattern');
        } else if (index === 42 || index === 43) {
          results.techniques.push('reversed_pattern');
        } else if (index >= 44 && index <= 47) {
          results.techniques.push('nested_brackets');
        } else if (index === 48) {
          results.techniques.push('extreme_gaps');
        } else if (index >= 49 && index <= 51) {
          results.techniques.push('parentheses_wrapped_k');
        } else if (index === 52) {
          results.techniques.push('mixed_dots_underscores');
        } else if (index === 53) {
          results.techniques.push('extreme_separators');
        } else if (index >= 54 && index <= 55) {
          results.techniques.push('parentheses_any_separators');
        } else if (index >= 56 && index <= 58) {
          results.techniques.push('middle_vowel_wrapped');
        } else if (index >= 59 && index <= 60) {
          results.techniques.push('h_wrapped_ending');
        } else if (index >= 61 && index <= 62) {
          results.techniques.push('multiple_parentheses_groups');
        } else if (index === 63) {
          results.techniques.push('distributed_parentheses');
        } else if (index >= 64 && index <= 66) {
          results.techniques.push('unclosed_parentheses');
        } else if (index >= 67 && index <= 68) {
          results.techniques.push('mixed_parentheses');
        } else if (index >= 69 && index <= 70) {
          results.techniques.push('flexible_parentheses');
        } else if (index === 71) {
          results.techniques.push('complex_unmatched_parentheses');
        } else if (index >= 72 && index <= 74) {
          results.techniques.push('angle_bracket_wrapped');
        } else if (index >= 75 && index <= 76) {
          results.techniques.push('angle_bracket_ending');
        } else if (index >= 77 && index <= 78) {
          results.techniques.push('complex_mixed_separators');
        } else if (index === 79) {
          results.techniques.push('ultra_complex_obfuscation');
        } else if (index === 80) {
          results.techniques.push('specific_mixed_pattern');
        } else if (index === 81) {
          results.techniques.push('nested_angle_brackets');
        } else if (index === 82) {
          results.techniques.push('mixed_bracket_types');
        } else if (index === 83) {
          results.techniques.push('extreme_mixed_separators');
        } else if (index >= 84 && index <= 85) {
          results.techniques.push('unclosed_angle_bracket_phonetic');
        } else if (index >= 86 && index <= 87) {
          results.techniques.push('unclosed_angle_bracket_suspicious');
        } else if (index >= 88 && index <= 89) {
          results.techniques.push('reversed_malformed_angle_bracket');
        } else if (index >= 90 && index <= 91) {
          results.techniques.push('single_angle_bracket_minimal');
        } else if (index === 92) {
          results.techniques.push('truncated_mixed_patterns');
        } else if (index === 93) {
          results.techniques.push('ultra_specific_truncated');
        } else if (index >= 94 && index <= 95) {
          results.techniques.push('phonetic_after_unclosed_bracket');
        } else if (index === 96) {
          results.techniques.push('missing_closing_bracket_obvious');
        } else if (index === 97) {
          results.techniques.push('comprehensive_hk_pattern');
        } else if (index === 98) {
          results.techniques.push('asterisk_wrapped');
        } else if (index === 99) {
          results.techniques.push('asterisk_wrapped_hk');
        } else if (index === 100) {
          results.techniques.push('multiple_asterisks');
        } else if (index === 101 || index === 102) {
          results.techniques.push('mixed_asterisk_separators');
        } else if (index === 103) {
          results.techniques.push('flexible_asterisk_pattern');
        } else if (index === 104) {
          results.techniques.push('asterisk_various_positions');
        } else if (index === 105 || index === 106) {
          results.techniques.push('complex_asterisk_patterns');
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
  
  // PHASE 4: Advanced obfuscation detection
  // Only run if we haven't detected anything yet and text is reasonable length
  if (!results.detected && text.length < 1000) {
    // Try scrambled pattern detection
    const scrambledMatches = detectScrambledPatterns(normalizedText);
    for (const match of scrambledMatches) {
      // Double-check it's not legitimate usage
      if (!isLegitimatePhase4Usage(text, match.text, match.position)) {
        results.detected = true;
        results.matches.push(match.text);
        results.techniques.push(match.technique);
        results.positions.push({
          start: match.position,
          end: match.position + match.text.length
        });
      }
    }
    
    // If still not detected, try sliding window (most expensive)
    if (!results.detected) {
      const slidingMatches = slidingWindowDetection(normalizedText);
      for (const match of slidingMatches) {
        // Final safety check
        if (!isLegitimatePhase4Usage(text, match.text, match.position)) {
          results.detected = true;
          results.matches.push(match.text);
          results.techniques.push(match.technique);
          results.positions.push({
            start: match.position,
            end: match.position + match.text.length
          });
        }
      }
    }
  }
  
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
  // Updated pattern to catch phonetic variations like "keek", "kyck", etc., "hk" endings, Phase 4 reversed patterns, Phase 5 parentheses-wrapped K, and Phase 7 unclosed parentheses
  // Matches: k + (various middle patterns) + optional [kchq] or hk, OR reversed patterns like ckik, OR (k) patterns, OR unclosed parentheses
  const quickCheck = /k(?:[^a-z]{0,3}[i1l!|e3aeiouey][^a-z]{0,3}|[aeiouey0-9]{1,2}|\W{0,5}|i[cqk])(?:[kchq]{0,2}|hk)?|[ck]{2}[i1l!|][kc]|k\({2,}|k\[{2,}|k\{{2,}|k<{2,}|k[^a-z]{6,}|\([kc]\)[^a-z]*[i1l!|e3]|\([kc]\)[^a-z]*\([i1l!|e3aeiouey]\)|\([i1l!|e3aeiouey]\)[^a-z]*[kc]|\(h\)[kc]|\([kc][^)]*[i1l!|e3aeiouey]/i;
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