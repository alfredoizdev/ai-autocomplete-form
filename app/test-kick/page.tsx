"use client";

import { detectKickVariations } from "@/lib/kickDetection";
import { useEffect, useState } from "react";

interface TestResult {
  input: string;
  detected: boolean;
  confidence: number;
  techniques: string[];
  matches: string[];
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
    ];

    const testResults = phase1TestCases.map(testCase => {
      const result = detectKickVariations(testCase);
      return {
        input: testCase,
        detected: result.detected,
        confidence: result.confidence,
        techniques: result.techniques,
        matches: result.matches
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
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Phase 1 Kick Detection Test Results</h1>
      
      <div className="mb-6 p-4 bg-gray-100 rounded">
        <h2 className="text-lg font-semibold">Summary: {detectedCount}/{results.length} tests passed</h2>
      </div>

      <div className="space-y-4">
        {results.map((result, idx) => (
          <div 
            key={idx} 
            className={`p-4 border rounded ${
              result.detected ? 'border-red-500 bg-red-50' : 'border-green-500 bg-green-50'
            }`}
          >
            <div className="flex justify-between items-start">
              <div className="flex-1">
                <div className="font-mono text-sm mb-2">&quot;{result.input}&quot;</div>
                <div className="text-sm">
                  Status: <span className={`font-semibold ${result.detected ? 'text-red-600' : 'text-green-600'}`}>
                    {result.detected ? 'DETECTED' : 'NOT DETECTED'}
                  </span>
                  {result.detected && (
                    <span className="ml-2">
                      (Confidence: {result.confidence}%)
                    </span>
                  )}
                </div>
                {result.detected && result.techniques.length > 0 && (
                  <div className="text-xs text-gray-600 mt-1">
                    Techniques: {result.techniques.join(', ')}
                  </div>
                )}
                {result.detected && result.matches.length > 0 && (
                  <div className="text-xs text-gray-600">
                    Matches: {result.matches.join(', ')}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}