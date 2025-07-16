"use client";

import { detectKickVariations } from "@/lib/kickDetection";
import { useEffect, useState } from "react";

interface TestResult {
  input: string;
  detected: boolean;
  confidence: number;
  techniques: string[];
  matches: string[];
  hasZeroWidth?: boolean;
  normalizedInput?: string;
}

export default function TestKickPage() {
  const [results, setResults] = useState<TestResult[]>([]);

  useEffect(() => {
    const phase1TestCases = [
      // Extended parentheses patterns
      'k(..ee..)k',
      'k(__ei__)ck',
      'k(._i_.)k',
      'k(..i..)k',
      'k(...i...)k',
      'k(....i....)k',
      
      // Multiple dots patterns
      'k....i....k',
      'k.....i.....k',
      'k......i......k',
      
      // Mixed separators
      'k._.-i-._.k',
      'k-._i_.-k',
      'k_.-._i_.-._k',
      
      // Extended character gaps
      'k_____i_____k',
      'k-----i-----k',
      'k.....i.....c.....k',
      
      // Complex mixed patterns
      'k(.._i_..)k',
      'k(__..i..__))k',
      'k(.-_i_-.)k',
      
      // Legitimate text that should NOT be detected
      'I like to kick the ball',
      'kickstart your day',
      'Let\'s kick off the meeting',
      'lets play kick ball',
      'want to play kickball?',
      'join our kickball team',
      'practicing my soccer kick',
      'learned a new karate kick',
      'playing kick the can',
      'kick boxing class tonight',
      'coach taught me to kick',
      'kick ball tournament tomorrow',
      'kids love to play kick ball',
      'kickball game after school',
      
      // PHASE 1.5 TEST CASES
      '--- PHASE 1.5 TESTS ---',
      
      // The case that bypassed detection
      'k..ee..k',
      
      // Similar patterns with multiple chars
      'k...eee...k',
      'k--ei--k',
      'k._ie_.k',
      'k~~ii~~k',
      'k..e..k',
      'k.e.e.k',
      'k-e-i-k',
      
      // More complex variations
      'k...ee...k',
      'k----ei----k',
      'k._.ie._.k',
      'k..eei..k',
      'k--iee--k',
      'k...e.i...k',
      'k-.-e-e-.-k',
      
      // Mixed separators with multiple chars
      'k.-_ee_-.k',
      'k__..ei..__)k',
      'k~~~ie~~~k',
      
      // PHASE 1.6 TEST CASES - Remaining bypasses
      '--- PHASE 1.6 TESTS ---',
      
      // Patterns that were still bypassing
      'k(__ei__)ck',
      'k(__..i..__))k',
      'k.e.e.k',
      'k...e.i...k',
      
      // Additional test cases
      'k.i.c.k',
      'k(i)ck',
      'k((i))k',
      'k.e.i.c.k',
      'k(ei)ck',
      'k(..i..)ck',
      'k)))k',
      'k.k.i.c.k',
      
      // Complex combinations
      'k(._._i_._.)ck',
      'k....e....i....k',
      'k.e.i.k',
      'k(eick)k',
      'k(e.i)ck',
      
      // PHASE 1.7 TEST CASES - Phonetic variations
      '--- PHASE 1.7 TESTS ---',
      
      // The original issue
      'keek',
      'find me on keek',
      'Follow me on keek',
      'check out my keek channel',
      
      // Other phonetic variations
      'keak',
      'kyck',
      'kyyk',
      'kouk',
      'kaik',
      'kic',
      'kiq',
      
      // Double vowel patterns
      'kook',
      'kuuk',
      'kiik',
      
      // More complex phonetic variations
      'k33k',
      'ke3k',
      'k3ek',
      
      // Should NOT detect these
      'peek',
      'meek',
      'seek',
      'week',
      'kayak',
      'keep',
      'keen',
      
      // PHASE 2 TEST CASES - Zero-Width Characters
      '--- PHASE 2 TESTS ---',
      
      // Zero-width space (U+200B)
      'k\u200Bi\u200Bck',
      'k\u200B\u200Bi\u200B\u200Bck',
      
      // Zero-width non-joiner (U+200C)
      'k\u200Ci\u200Cck',
      
      // Zero-width joiner (U+200D)
      'k\u200Di\u200Dck',
      
      // Soft hyphen (U+00AD)
      'k\u00ADi\u00ADck',
      
      // Zero-width no-break space (U+FEFF)
      'k\uFEFFi\uFEFFck',
      
      // Word joiner (U+2060)
      'k\u2060i\u2060ck',
      
      // Mixed zero-width characters
      'k\u200B\u00ADi\u200D\u200Cck',
      
      // Zero-width + character substitution
      'k\u200B1\u200Bck',
      'k\u00AD!\u00ADck',
      
      // Zero-width + visible separators
      'k\u200B.\u200Bi\u200B.\u200Bck',
      'k_\u200Bi\u200B_k',
      
      // Zero-width + parentheses
      'k(\u200Bi\u200B)k',
      'k\u200B(i)\u200Bk',
      
      // Zero-width + phonetic variations
      'k\u200Be\u200Be\u200Bk',
      'k\u00ADy\u00ADck',
      
      // Complex zero-width patterns
      'Find me on k\u200Bi\u200Bck for updates',
      'k\u200B\u200C\u200D\u00AD\uFEFF\u2060i\u200B\u200C\u200D\u00AD\uFEFF\u2060ck',
      
      // PHASE 2.3 TEST CASES - "hk" Ending Patterns
      '--- PHASE 2.3 TESTS ---',
      
      // The reported bypass case
      'k..i..hk',
      'find me at k..i..hk',
      
      // Basic "hk" endings
      'kihk',
      'k1hk',
      'klhk',
      'k!hk',
      
      // "hk" with separators
      'k-i-hk',
      'k_i_hk',
      'k.i.hk',
      'k...i...hk',
      'k--i--hk',
      'k__i__hk',
      
      // General patterns with "hk"
      'k i hk',
      'k  i  hk',
      'k(i)hk',
      'k[i]hk',
      'k{i}hk',
      
      // Phonetic variations with "hk"
      'keehk',
      'kaihk',
      'kyahk',
      'kouhk',
      'keahk',
      
      // Complex patterns with "hk"
      'k(..i..)hk',
      'k(__i__)hk',
      'k(._i_.)hk',
      'k(...ei...)hk',
      
      // Extended "hk" endings
      'k..i..khk',
      'k..i..chk',
      'k-i-khk',
      'k_i_chk',
      
      // Zero-width + "hk" endings
      'k\u200Bi\u200Bhk',
      'k\u200B.\u200Bi\u200B.\u200Bhk',
      
      // PHASE 3 TEST CASES - Extended Homoglyphs
      '--- PHASE 3 TESTS ---',
      
      // Cyrillic homoglyphs
      'кick',  // Cyrillic к
      'kiск',  // Cyrillic с
      'кiск',  // Both Cyrillic
      'КІск',  // All caps Cyrillic
      'ҡіҫк',  // Various Cyrillic variants
      
      // Greek homoglyphs
      'κick',  // Greek kappa
      'kiςk',  // Greek final sigma
      'κιϲκ',  // All Greek
      'ΚΙϹΚ',  // All caps Greek
      
      // Latin extended homoglyphs
      'ķíčķ',  // Latin with diacritics
      'ḳīċḵ',  // Latin extended
      'ĸìćķ',  // More Latin variants
      
      // Mathematical symbols
      '𝐤𝐢𝐜𝐤',  // Mathematical bold
      '𝑘𝑖𝑐𝑘',  // Mathematical italic
      '𝒌𝒊𝒄𝒌',  // Mathematical bold italic
      '𝓀𝓲𝓬𝓀',  // Mathematical script
      '𝔨𝔦𝔠𝔨',  // Mathematical fraktur
      '𝕜𝕚𝕔𝕜',  // Mathematical double-struck
      
      // Mixed homoglyphs
      'к1ϲk',  // Cyrillic k, digit 1, Greek c
      'ķi|k',  // Latin k with cedilla, pipe for i
      '𝐤ιck',  // Math bold k, Greek iota
      'кїςк',  // Cyrillic k, Cyrillic yi, Greek sigma
      
      // Fullwidth characters
      'ｋｉｃｋ',  // All fullwidth
      'ｋick',    // Mixed fullwidth
      
      // Look-alike symbols for i
      'k|ck',   // Pipe
      'k!ck',   // Exclamation
      'k1ck',   // Digit one
      'kⅰck',   // Roman numeral
      'k│ck',   // Box drawing
      
      // Look-alike symbols for c
      'ki(k',   // Parenthesis as c
      'ki⊂k',   // Subset symbol
      'ki⟨k',   // Angle bracket
      
      // Complex mixed homoglyphs
      'ķ│ςķ',   // Latin k, box drawing, Greek sigma
      '𝔨!ϲ𝔨',   // Fraktur k, exclamation, Greek lunate sigma
      'К1СК',   // All caps Cyrillic with digit 1
      
      // Homoglyphs with separators
      'к.i.c.k',  // Cyrillic k with dots
      'κ_ι_ς_κ',  // Greek with underscores
      '𝐤-𝐢-𝐜-𝐤',  // Math bold with dashes
      
      // Homoglyphs with zero-width
      'к\u200Bi\u200Bck',  // Cyrillic k + zero-width
      'κ\u00ADι\u00ADck',  // Greek kappa + soft hyphen
      
      // "hk" endings with homoglyphs
      'кihk',   // Cyrillic k
      'κιhκ',   // Greek
      'ķīħķ',   // Latin extended
      'k𝐢𝐡𝐤',   // Math bold
      'кїнк',   // All Cyrillic (н looks like h)
      
      // Should NOT detect regular text with Unicode
      'I like café',
      'Résumé attached',
      'Naïve approach',
    ];

    const testResults = phase1TestCases.map(testCase => {
      const result = detectKickVariations(testCase);
      
      // Check if input contains zero-width characters
      const hasZeroWidth = result.techniques.includes('zero_width');
      const normalizedInput = hasZeroWidth ? 
        testCase.replace(/[\u200B\u200C\u200D\u00AD\uFEFF\u2060]/g, '') : 
        undefined;
      
      return {
        input: testCase,
        detected: result.detected,
        confidence: result.confidence,
        techniques: result.techniques,
        matches: result.matches,
        hasZeroWidth,
        normalizedInput
      };
    });

    setResults(testResults);
  }, []);

  const detectedCount = results.filter((r) => {
    // Skip the separator lines
    if (r.input.startsWith('--- PHASE')) return true;
    
    // Legitimate text should NOT be detected
    const legitimatePatterns = ['kick the', 'kickstart', 'kick off'];
    const isLegitimate = legitimatePatterns.some(pattern => r.input.includes(pattern));
    
    const shouldDetect = !isLegitimate;
    return r.detected === shouldDetect;
  }).length;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="p-8 max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-white">Kick Detection Test Results (Phase 1, 2 & 3)</h1>
        
        <div className="mb-6 p-4 bg-gray-900 border border-gray-800 rounded-lg">
          <h2 className="text-lg font-semibold text-gray-100">Summary: {detectedCount}/{results.length} tests passed</h2>
        </div>

        <div className="space-y-4">
          {results.map((result, idx) => (
            <div 
              key={idx} 
              className={`p-4 border rounded-lg ${
                result.detected ? 'border-red-500/50 bg-red-950/20' : 'border-green-500/50 bg-green-950/20'
              }`}
            >
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <div className="font-mono text-sm mb-2 text-gray-200">
                  &quot;{result.input}&quot;
                  {result.hasZeroWidth && (
                    <span className="ml-2 text-xs text-purple-400 font-semibold">
                      [Contains Zero-Width Characters]
                    </span>
                  )}
                </div>
                {result.normalizedInput && (
                  <div className="font-mono text-xs text-gray-400 mb-1">
                    Normalized: &quot;{result.normalizedInput}&quot;
                  </div>
                )}
                <div className="text-sm">
                  Status: <span className={`font-semibold ${result.detected ? 'text-red-400' : 'text-green-400'}`}>
                    {result.detected ? 'DETECTED' : 'NOT DETECTED'}
                  </span>
                  {result.detected && (
                    <span className="ml-2 text-gray-300">
                      (Confidence: {result.confidence}%)
                    </span>
                  )}
                </div>
                {result.detected && result.techniques.length > 0 && (
                  <div className="text-xs text-gray-400 mt-1">
                    Techniques: {result.techniques.join(', ')}
                  </div>
                )}
                {result.detected && result.matches.length > 0 && (
                  <div className="text-xs text-gray-400">
                    Matches: {result.matches.join(', ')}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        </div>
      </div>
    </div>
  );
}