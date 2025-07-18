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
      
      // PHASE 4 TEST CASES - Advanced Obfuscation Patterns
      '--- PHASE 4 TESTS ---',
      
      // Reversed patterns (should detect)
      'ckik',
      'ckic',
      'ccik',
      'kkic',
      'cikk',
      'c1kk',
      'ck!k',
      
      // Scrambled patterns with special chars (should detect)
      'k!kc',    // Scrambled with special char
      'k1ck',    // Scrambled with number
      'c!kk',    // Scrambled with special
      'ki$k',    // With $ symbol
      
      // Nested brackets
      'k((i))k',
      'k(((i)))k',
      'k[[i]]k',
      'k[[[i]]]k',
      'k{{i}}k',
      'k{{{i}}}k',
      'k<<i>>k',
      'k<<<i>>>k',
      
      // Extreme gaps
      'k------i------k',
      'k__________i__________k',
      'k............i............k',
      'k~~~~~~i~~~~~~k',
      
      // FALSE POSITIVE TESTS - These should NOT be detected
      '--- PHASE 4 FALSE POSITIVE TESTS ---',
      
      // Legitimate sentences
      'Let\'s kick back and relax',
      'Think quickly about this',
      'The chicken tastes great',
      'Use a stick for that',
      'Click here to continue',
      'That\'s a clever trick',
      'Going on a picnic tomorrow',
      'Cricket is a fun sport',
      'The sauce is too thick',
      'Quicken your pace',
      'This is a sticky situation',
      'Stop trying to trick me',
      'I love fried chicken',
      'The quick brown fox',
      'Stick to the plan',
      'Pick up the stick',
      'Lick the ice cream',
      'This music is sick',
      'Light the wick',
      'Choose your pick',
      'Check the ticket',
      'Build with bricks',
      'Avoid the prick',
      'The road is slick',
      'Quick flick of the wrist',
      
      // Complex false positive scenarios
      'I kick the ball quickly',
      'She has a sidekick named Rick',
      'The homesick kid wants chicken',
      'Quick! Kick the soccer ball!',
      'They trick or treat for candy',
      
      // Edge cases that should be detected
      '--- PHASE 4 EDGE CASES ---',
      
      // Mixed techniques
      'k((1))k',     // Nested with number
      'k<<e>>k',     // Nested with vowel
      'ck(i)k',      // Reversed with parentheses
      'k\u200B!kc',  // Scrambled with zero-width
      
      // PHASE 5 TEST CASES - Parentheses-wrapped K patterns
      '--- PHASE 5 TESTS ---',
      
      // The reported edge case
      '(k)I..___________k',
      
      // Basic parentheses-wrapped K at beginning
      '(k)ick',
      '(k)i k',
      '(k)..i..k',
      '(k)___i___k',
      '(k)...i...k',
      '(c)ick',
      
      // Basic parentheses-wrapped K at end
      'kick(k)',
      'ki (k)',
      'k..i..(k)',
      'k___i___(k)',
      'k...i...(k)',
      'kic(c)',
      
      // Both wrapped
      '(k)i(k)',
      '(k)..i..(k)',
      '(k)___i___(k)',
      '(c)..i..(c)',
      
      // Mixed dots and underscores
      '(k)..__i__.._k',
      '(k)_._i_._k',
      '(k)..___..i..___..k',
      
      // Extreme separators
      'k_____________i_____________k',
      'k.............i.............k',
      'k-------------i-------------k',
      'k~~~~~~~~~~~~~i~~~~~~~~~~~~~k',
      
      // Complex combinations
      '(k)._._._.i._._._.k',
      '(k)____....i....____k',
      '(k)~^~^~i~^~^~k',
      '(k)!@#$%i%$#@!k',
      
      // With numbers and special chars
      '(k)1..___________k',
      '(k)|..___________k',
      '(k)!..___________k',
      
      // Capital letters
      '(K)I..___________K',
      '(K)...I...K',
      
      // Zero-width with parentheses
      '(k)\u200Bi\u200Bk',
      '(k)\u200B..i..\u200Bk',
    ];

    // Phase 6 test cases - Multiple distributed parentheses
    const phase6TestCases = [
      // The reported edge case
      '(k__(I..__(h)k',
      'find me on (k__(I..__(h)k',
      
      // Variations
      '(k)__(I)..__(h)k',
      '(k__(i..__(h)k',
      '(k)__(i)..__(h)k',
      'k__(I)..__(h)k',
      '(k)__I..__(h)k',
      '(k)__(I)..__(h)(k)',
      '(k)_(I)_(h)k',
      '(k)(I)(h)k',
      '(k)...(I)...(h)k',
      'k(I)hk',
      'k__(h)k',
      '(h)k',
      
      // More complex variations
      '(k)...(I)...(h)...k',
      '(k)___(i)___(h)___k',
      '(k)..(1)..(h)..k',
      '(k)__(e)__(h)k',
      'k(i)(c)(h)k',
      '(k)(i)(c)(k)',
      
      // Mixed with other techniques
      '(k)​__(I)​..​__(h)​k', // With zero-width spaces
      '(K)__(I)..__(H)K',    // Capital letters
      '(k)__(!)..__(h)k',    // Special chars in middle
    ];

    // Phase 7 test cases - Unclosed/unmatched parentheses
    const phase7TestCases = [
      '--- PHASE 7 TESTS ---',
      
      // The new reported edge cases
      '(k__)i__..!h..k',
      '(k__(I..__(h)k',
      'find me on (k__(I..__(h)k',
      '(k__(i..__(h)k',
      '(k)__I..__(h)k',
      
      // More unclosed parentheses variations
      '(k__i__k',
      '(k___i___k',
      '(k..i..k',
      '(k i k',
      '(ki k',
      '(k ick',
      
      // Multiple unclosed parentheses
      '(k__(i__(h)k',
      '(k__(I__(h)k',
      '(k(i(h)k',
      '(k_(i_(h)k',
      
      // Mixed closed and unclosed
      '(k)__i__(h)k',
      '(k__i__(h)k',
      '(k)__(i__(h)k',
      
      // With special characters
      '(k__!i__k',
      '(k__1__(h)k',
      '(k__|__(h)k',
      '(k__i__!h__k',
      
      // Complex patterns
      '(k__)__i__..!h..k',
      '(k__))i((__(h)k',
      '(k__]i[__(h)k',
      '(k__)i)__(h)k',
      
      // Edge cases with no vowel parentheses
      '(k)i(h)k',
      '(k)__i__(h)k',
      '(k)..i..(h)k',
      '(k)___i___(h)k',
      
      // Very complex unmatched
      '((k__i__(h))k',
      '(((k__i__k',
      '(k__(((i__(h)k',
    ];

    // Phase 8 test cases - Ultra-complex mixed separator obfuscation
    const phase8TestCases = [
      '--- PHASE 8 TESTS ---',
      
      // The original reported case
      'k..!()..<h>k',
      'find me on k..!()..<h>k',
      
      // Angle bracket wrapped characters
      'k<i>k',
      'k<h>k',
      'k<c>k',
      'k...<i>...k',
      'k___<h>___k',
      'k--<e>--k',
      'k<.>k',
      'k<..>k',
      'k<...>k',
      
      // Angle bracket wrapped endings
      'ki<h>k',
      'ke<c>k',
      'k..i..<h>k',
      'k___i___<c>k',
      'k--e--<h>',
      'ki<hk>',
      'ke<ck>',
      
      // Complex mixed separator patterns
      'k..!()..<h>k',
      'k._@()..<i>k',
      'k#$%<e>^&*k',
      'k..!()<>i<>!..k',
      'k()!..<>e<>.!()k',
      
      // Ultra-complex obfuscation
      'k!@#$%^&*()i!@#$%^&*()k',
      'k._._._._.i._._._._.k',
      'k()()()()i()()()()k',
      'k<><><><>i<><><><>k',
      
      // Nested angle brackets
      'k<<i>>k',
      'k<<<h>>>k',
      'k<<<<e>>>>k',
      'k<<>>k',
      'k<<<>>>k',
      
      // Mixed bracket types with angle brackets
      'k(<i>)k',
      'k[<h>]k',
      'k{<c>}k',
      'k(<.>)k',
      'k[<..>]k',
      'k{<...>}k',
      'k((<i>))k',
      'k[[<h>]]k',
      
      // Extreme mixed separators
      'k!@#$%^&*()!@#$%^&*()i!@#$%^&*()!@#$%^&*()k',
      'k..........i..........k',
      'k__________i__________k',
      'k----------i----------k',
      'k~~~~~~~~~~i~~~~~~~~~~k',
      
      // Variations of the original case
      'k..!()..<c>k',
      'k..!()..<i>k',
      'k..@()..<h>k',
      'k..#()..<h>k',
      'k..![].<h>k',
      'k..!{}..<h>k',
      'k()!..<h>k',
      'k..!(}<h>k',
      
      // Mixed with zero-width characters
      'k\u200B..!()..<h>\u200Bk',
      'k\u200B<i>\u200Bk',
      'k..!\u200B()..<h>k',
      
      // Capital letter variations
      'K..!()..<H>K',
      'K<I>K',
      'K<<<H>>>K',
      
      // Should NOT detect (false positive tests)
      'I like <HTML> tags',
      'Use k<something>k format',
      'The key<value> pair',
      'Pick <option> from list',
      'Click <button> here',
      'Check <input> field',
    ];

    // Phase 9 test cases - Truncated/Incomplete obfuscation patterns
    const phase9TestCases = [
      '--- PHASE 9 TESTS ---',
      
      // The original reported case and variations
      'k<..eek',
      'find me on k<..eek',
      'k<..ick',
      'k<..ook',
      'k<...eek',
      'k<....ick',
      
      // Unclosed angle brackets with phonetic endings
      'k<eek',
      'k<ick',
      'k<ook',
      'k<..eak',
      'k<...oak',
      'k<--eek',
      'k<__ick',
      'k<##ook',
      'k<@@eek',
      
      // Variations with different separators
      'k<.-eek',
      'k<_.ick',
      'k<!.ook',
      'k<@#eek',
      'k<$%ick',
      'k<^&ook',
      
      // Unclosed brackets with single chars
      'k<i',
      'k<e',
      'k<o',
      'k<1',
      'k<!',
      'k<.',
      'k<_',
      'k<-',
      
      // Reversed/malformed angle brackets
      'k>..eek',
      'k>..ick',
      'k>..ook',
      '>k..eek',
      '>k..ick',
      '>c..ook',
      'k>eek',
      'k>ick',
      '>keek',
      '>kick',
      
      // Single angle bracket with minimal content
      'k<..',
      'k>..',
      'k<.',
      'k>.',
      'k<-',
      'k>-',
      'k<_',
      'k>_',
      'k<!',
      'k>!',
      'k<@',
      'k>#',
      
      // Truncated mixed patterns
      'k<..i..',
      'k>..1..',
      'k<..!..',
      'k>..e..',
      'k<.i.e',
      'k>.1.o',
      'k<#i#',
      'k>$1$',
      
      // Phonetic variations after unclosed brackets
      'k<..keek',
      'k>..qeek',
      'k<..yck',
      'k>..yk',
      'k<kyck',
      'k>qeek',
      
      // Missing closing bracket with obvious intent
      'k<i..',
      'k<e..',
      'k<o..',
      'k<1..',
      'k<!..',
      'k<...',
      'k<____',
      'k<----',
      
      // Complex truncated variations
      'k<..!i',
      'k>..@e',
      'k<#$%eek',
      'k>^&*ick',
      'k<()ook',
      'k>[]eek',
      'k<{}ick',
      
      // Capital letter variations
      'K<..EEK',
      'K<..ICK',
      'K>..OOK',
      'K<EEK',
      'K>ICK',
      
      // Zero-width with truncated patterns
      'k\u200B<..eek',
      'k<..\u200Beek',
      'k<\u200B..eek',
      
      // Should NOT detect (false positive tests for Phase 9)
      'I use k<div> tags in HTML',
      'The k>value comparison',
      'Check k<script>alert()',
      'Use k<input> field',
      'The k>0 condition',
      'In k<style> sheets',
      'For k<body> content',
      'With k<head> section',
      'The k<meta> tag',
      'A k<link> element',
    ];

    // Phase 10 test cases - Asterisk-based obfuscation patterns
    const phase10TestCases = [
      '--- PHASE 10 TESTS ---',
      
      // The reported pattern that wasn't detected
      'k*I*..__<hk',
      'find me on k*I*..__<hk',
      
      // Basic asterisk-wrapped characters
      'k*i*k',
      'k*e*k',
      'k*I*k',
      'k*1*k',
      'k*!*k',
      'k*l*k',
      
      // Asterisk-wrapped with hk ending
      'k*i*hk',
      'k*I*hk',
      'k*e*hk',
      'k*i*..hk',
      'k*I*__hk',
      'k*i*..__hk',
      
      // Multiple asterisks
      'k**i**k',
      'k***e***k',
      'k****I****k',
      'k*****i*****k',
      
      // Mixed asterisks with other separators
      'k*i*..__<k',
      'k*e*..--k',
      'k*I*____k',
      'k*i*....k',
      'k*..e..*k',
      'k*.i.*k',
      
      // Complex asterisk patterns
      'k*i*..__<>k',
      'k*I*..!()k',
      'k*e*#$%^k',
      'k*i*@#$%hk',
      
      // Asterisk at various positions
      '*k*i*k*',
      'k*i*k*',
      '*k*i*k',
      'k*i*ck',
      
      // Capital letter variations
      'K*I*K',
      'K*E*HK',
      'K*I*..__<HK',
      
      // Zero-width with asterisks
      'k\u200B*i*\u200Bk',
      'k*\u200Bi\u200B*k',
      
      // Edge cases
      'k*eek',
      'k*ick',
      'k*ook',
      'k*yck',
      
      // Should NOT detect (legitimate asterisk usage)
      'I give this 5k* rating',
      'The k* value is important',
      'Check k*args in Python',
    ];

    // Phase 11 test cases - Reversed angle bracket patterns (>k patterns)
    const phase11TestCases = [
      '--- PHASE 11 TESTS ---',
      
      // The reported pattern that wasn't detected
      '>k..eek',
      'find me on >k..eek',
      
      // Basic reversed angle bracket patterns
      '>k.eek',
      '>keek',
      '>k...ick',
      '>k....ook',
      '>c..eek',
      '>c...ick',
      
      // With different separators
      '>k--eek',
      '>k__ick',
      '>k~~ook',
      '>k##eek',
      '>k@@ick',
      '>k$$ook',
      
      // With vowel variations
      '>k..i..k',
      '>k__e__k',
      '>k--I--k',
      '>k..1..k',
      '>k__!__k',
      
      // Complex patterns
      '>k....i....k',
      '>k_._._i_._._k',
      '>k--__--i--__--k',
      
      // Capital letter variations
      '>K..EEK',
      '>K__ICK',
      '>K--OOK',
      '>K..I..K',
      
      // Zero-width with reversed brackets
      '>k\u200B..eek',
      '>k..\u200Beek',
      
      // Edge cases
      '>kik',
      '>kick',
      '>kic',
      '>khk',
      
      // Should NOT detect (legitimate usage)
      'The value >k is greater',
      'Check if x >k in the equation',
      'Use >key for sorting',
    ];

    // Combine all test cases
    const allTestCases = [...phase1TestCases, ...phase6TestCases, ...phase7TestCases, ...phase8TestCases, ...phase9TestCases, ...phase10TestCases, ...phase11TestCases];
    
    const testResults = allTestCases.map(testCase => {
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
    
    // Check if it's in the false positive test section
    const inputIndex = results.findIndex(res => res.input === r.input);
    const falsePositiveStartIndex = results.findIndex(res => res.input === '--- PHASE 4 FALSE POSITIVE TESTS ---');
    const edgeCaseStartIndex = results.findIndex(res => res.input === '--- PHASE 4 EDGE CASES ---');
    
    let shouldDetect = true;
    
    // If it's in the false positive section, it should NOT be detected
    if (falsePositiveStartIndex !== -1 && inputIndex > falsePositiveStartIndex && 
        (edgeCaseStartIndex === -1 || inputIndex < edgeCaseStartIndex)) {
      shouldDetect = false;
    }
    
    // Additional check for known legitimate patterns
    const legitimatePatterns = ['kick the', 'kickstart', 'kick off', 'kick back'];
    if (legitimatePatterns.some(pattern => r.input.toLowerCase().includes(pattern))) {
      shouldDetect = false;
    }
    
    return r.detected === shouldDetect;
  }).length;

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="p-8 max-w-4xl mx-auto">
        <h1 className="text-3xl font-bold mb-6 text-white">Kick Detection Test Results (Phase 1, 2, 4, 5, 6, 7, 8 & 9)</h1>
        
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