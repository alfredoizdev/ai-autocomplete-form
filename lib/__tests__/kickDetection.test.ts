import { 
  detectKickVariations, 
  contextualAnalysis, 
  progressiveDetection,
  extractMLFeatures
} from '../kickDetection';

describe('Kick.com Detection Module', () => {
  describe('Real User Examples Detection', () => {
    // All 28 real examples from users
    const userExamples = [
      'k i k', 'K ik', 'k!k', 'k_!_k', 'k-i-k', 'k..i..k',
      'k.i.k', 'ki.k', 'kiik', 'kik', 'keek', 'kiek',
      'keik', 'kick', 'k(i)k', 'k___ik', 'k_i_k', 'k_l_k',
      'klk', 'kiik', 'kiilk', 'kiiik', 'kilk', 'killk',
      'k1k', 'k_1_k', 'k1_k', 'k_1k', 'klik', 'k11k', 'kii_k'
    ];
    
    test.each(userExamples)('should detect "%s"', (example) => {
      const result = detectKickVariations(example);
      expect(result.detected).toBe(true);
      expect(result.confidence).toBeGreaterThan(50);
      expect(result.matches.length).toBeGreaterThan(0);
    });
    
    test('should identify correct techniques for each pattern', () => {
      const techniqueTests = [
        { input: 'k i k', expectedTechnique: 'character_substitution' },
        { input: 'k.i.k', expectedTechnique: 'separators' },
        { input: 'kiiik', expectedTechnique: 'character_repetition' },
        { input: 'keek', expectedTechnique: 'alternative_spelling' },
        { input: 'k(i)k', expectedTechnique: 'parentheses' },
        { input: 'k___ik', expectedTechnique: 'underscores' },
        { input: 'kik', expectedTechnique: 'fuzzy_match' },
      ];
      
      techniqueTests.forEach(({ input, expectedTechnique }) => {
        const result = detectKickVariations(input);
        expect(result.techniques).toContain(expectedTechnique);
      });
    });
  });
  
  describe('Domain Variation Detection', () => {
    const domainExamples = [
      'k!ck.com',
      'k i c k . c o m',
      'kick[.]com',
      'kick dot com',
      'k1ck[dot]c0m',
      'kick.c0m',
      'k.i.c.k.com',
      'k_i_c_k.com',
      'k-i-c-k.com',
      'kiick.com'
    ];
    
    test.each(domainExamples)('should detect domain variation "%s"', (domain) => {
      const result = detectKickVariations(domain);
      expect(result.detected).toBe(true);
      expect(result.techniques).toContain('domain_pattern');
    });
  });
  
  describe('Homoglyph Detection', () => {
    const homoglyphExamples = [
      'кick',      // Cyrillic k
      'kiсk',      // Cyrillic c
      'kіck',      // Cyrillic i
      'κick',      // Greek kappa
      'ķick',      // Latin k with cedilla
    ];
    
    test.each(homoglyphExamples)('should detect homoglyph "%s"', (example) => {
      const result = detectKickVariations(example);
      expect(result.detected).toBe(true);
      expect(result.techniques).toContain('homoglyph');
    });
  });
  
  describe('False Positive Prevention', () => {
    const legitimateText = [
      'I like to kick the ball',
      'kickstart your day',
      'a quick brown fox',
      'kitchen kicks',
      'kicker position',
      'click here',
      'stick figure',
      'thick forest',
      'trick or treat',
      'brick wall'
    ];
    
    test.each(legitimateText)('should not flag legitimate text "%s"', (text) => {
      const result = detectKickVariations(text);
      expect(result.detected).toBe(false);
      expect(result.confidence).toBe(0);
    });
  });
  
  describe('Context Analysis', () => {
    test('should increase confidence with streaming context', () => {
      const text = 'check out my k!ck channel for live streams';
      const baseResult = detectKickVariations(text);
      const contextResult = contextualAnalysis(text, baseResult);
      
      expect(contextResult.confidence).toBeGreaterThan(baseResult.confidence);
      expect(contextResult.techniques).toContain('streaming_context');
    });
    
    test('should detect URL context', () => {
      const text = 'visit my page at k.i.c.k dot com';
      const baseResult = detectKickVariations(text);
      const contextResult = contextualAnalysis(text, baseResult);
      
      expect(contextResult.techniques).toContain('url_context');
    });
    
    test('should not increase confidence without relevant context', () => {
      const text = 'k!k';
      const baseResult = detectKickVariations(text);
      const contextResult = contextualAnalysis(text, baseResult);
      
      expect(contextResult.confidence).toBe(baseResult.confidence);
    });
  });
  
  describe('Position Tracking', () => {
    test('should correctly track match positions', () => {
      const text = 'my username is k!ck and I stream';
      const result = detectKickVariations(text);
      
      expect(result.positions.length).toBeGreaterThan(0);
      const firstPosition = result.positions[0];
      expect(text.substring(firstPosition.start, firstPosition.end)).toMatch(/k!ck/i);
    });
    
    test('should track multiple match positions', () => {
      const text = 'k!ck is my platform, visit k.i.c.k';
      const result = detectKickVariations(text);
      
      expect(result.positions.length).toBe(2);
      expect(result.matches.length).toBe(2);
    });
  });
  
  describe('Progressive Detection Performance', () => {
    test('should skip full detection for obvious non-matches', () => {
      const text = 'This is a normal bio without any suspicious content';
      const result = progressiveDetection(text);
      
      expect(result.detected).toBe(false);
      expect(result.matches).toHaveLength(0);
    });
    
    test('should perform full detection for potential matches', () => {
      const text = 'follow me on k!ck';
      const result = progressiveDetection(text);
      
      expect(result.detected).toBe(true);
      expect(result.matches.length).toBeGreaterThan(0);
    });
  });
  
  describe('ML Feature Extraction', () => {
    test('should extract correct features', () => {
      const text = 'k!ck.com';
      const features = extractMLFeatures(text);
      
      expect(features.hasKSound).toBe(true);
      expect(features.hasCSound).toBe(true);
      expect(features.specialCharDensity).toBeGreaterThan(0);
      expect(features.suspiciousPatternCount).toBeGreaterThan(0);
    });
    
    test('should calculate character gaps correctly', () => {
      const text = 'k___i___k';
      const features = extractMLFeatures(text);
      
      expect(features.maxCharacterGap).toBe(6); // 3 underscores + 3 underscores
    });
  });
  
  describe('Edge Cases', () => {
    test('should handle empty string', () => {
      const result = detectKickVariations('');
      expect(result.detected).toBe(false);
      expect(result.matches).toHaveLength(0);
    });
    
    test('should handle very long text', () => {
      const longText = 'Lorem ipsum '.repeat(100) + 'k!ck' + ' dolor sit amet'.repeat(100);
      const result = detectKickVariations(longText);
      
      expect(result.detected).toBe(true);
      expect(result.matches).toContain('k!ck');
    });
    
    test('should handle mixed case variations', () => {
      const mixedCase = 'K!Ck KiCk kIcK';
      const result = detectKickVariations(mixedCase);
      
      expect(result.detected).toBe(true);
      expect(result.matches.length).toBeGreaterThan(0);
    });
    
    test('should handle repeated patterns', () => {
      const repeated = 'k!ck k!ck k!ck';
      const result = detectKickVariations(repeated);
      
      expect(result.detected).toBe(true);
      // Should not have duplicate matches
      expect(result.matches).toEqual(['k!ck']);
    });
  });
  
  describe('Confidence Scoring', () => {
    test('should give 100% confidence for exact match', () => {
      const result = detectKickVariations('kick');
      expect(result.confidence).toBe(100);
    });
    
    test('should give high confidence for obvious variations', () => {
      const result = detectKickVariations('k!ck.com');
      expect(result.confidence).toBeGreaterThan(80);
    });
    
    test('should give lower confidence for fuzzy matches', () => {
      const result = detectKickVariations('kik');
      expect(result.confidence).toBeLessThan(80);
      expect(result.confidence).toBeGreaterThan(50);
    });
  });
  
  describe('Complex Real-World Scenarios', () => {
    const complexExamples = [
      {
        text: 'Hey everyone! Follow me on k i c k dot com for amazing content!',
        shouldDetect: true,
        minConfidence: 80
      },
      {
        text: 'My streaming schedule: Mon-Fri 8pm. Platform: k!ck',
        shouldDetect: true,
        minConfidence: 70
      },
      {
        text: 'I love playing soccer and practicing my kick technique',
        shouldDetect: false,
        minConfidence: 0
      },
      {
        text: 'username: coolstreamer | platform: k___i___c___k',
        shouldDetect: true,
        minConfidence: 60
      },
      {
        text: 'Check out кick.com/mystream (using Cyrillic to bypass filters)',
        shouldDetect: true,
        minConfidence: 90
      }
    ];
    
    test.each(complexExamples)(
      'should correctly analyze: "$text"',
      ({ text, shouldDetect, minConfidence }) => {
        const result = detectKickVariations(text);
        expect(result.detected).toBe(shouldDetect);
        if (shouldDetect) {
          expect(result.confidence).toBeGreaterThanOrEqual(minConfidence);
        }
      }
    );
  });
  
  describe('Performance Benchmarks', () => {
    test('should process text quickly', () => {
      const text = 'This is a test bio with k!ck in it';
      const iterations = 1000;
      
      const start = Date.now();
      for (let i = 0; i < iterations; i++) {
        detectKickVariations(text);
      }
      const end = Date.now();
      
      const avgTime = (end - start) / iterations;
      expect(avgTime).toBeLessThan(5); // Should process in less than 5ms on average
    });
  });
  
  describe('Phase 1: Enhanced Pattern Detection', () => {
    const phase1Examples = [
      // Extended parentheses patterns (up to 8 chars)
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
      
      // Extended character gaps (up to 5 chars)
      'k_____i_____k',
      'k-----i-----k',
      'k.....i.....c.....k',
      
      // Complex mixed patterns
      'k(.._i_..)k',
      'k(__..i..__))k',
      'k(.-_i_-.)k'
    ];
    
    test.each(phase1Examples)('should detect complex pattern "%s"', (example) => {
      const result = detectKickVariations(example);
      expect(result.detected).toBe(true);
      expect(result.confidence).toBeGreaterThan(50);
      expect(result.matches.length).toBeGreaterThan(0);
    });
    
    test('should still not flag legitimate text after Phase 1 updates', () => {
      const legitimateText = [
        'I like to kick the ball',
        'kickstart your day',
        'Let\'s kick off the meeting',
        'He kicked the habit',
        'A quick kick in the pants'
      ];
      
      legitimateText.forEach(text => {
        const result = detectKickVariations(text);
        expect(result.detected).toBe(false);
      });
    });
    
    test('should identify correct techniques for Phase 1 patterns', () => {
      const techniqueTests = [
        { input: 'k(..ee..)k', expectedTechnique: 'extended_parentheses' },
        { input: 'k....i....k', expectedTechnique: 'multiple_dots' },
        { input: 'k._.-i-._.k', expectedTechnique: 'mixed_separators' },
        { input: 'k_____i_____k', expectedTechnique: 'extended_gaps' }
      ];
      
      techniqueTests.forEach(({ input, expectedTechnique }) => {
        const result = detectKickVariations(input);
        expect(result.detected).toBe(true);
        expect(result.techniques.some(t => 
          t === expectedTechnique || 
          t === 'parentheses' || 
          t === 'separators' ||
          t === 'underscores' ||
          t === 'missing_letter'
        )).toBe(true);
      });
    });
  });
  
  describe('Phase 1.5: Multiple Characters with Separators Fix', () => {
    const criticalTestCases = [
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
      'k~~~ie~~~k'
    ];
    
    test.each(criticalTestCases)('should detect pattern "%s"', (example) => {
      const result = detectKickVariations(example);
      expect(result.detected).toBe(true);
      expect(result.confidence).toBeGreaterThan(50);
      expect(result.matches.length).toBeGreaterThan(0);
    });
    
    test('should still not flag legitimate text after Phase 1.5', () => {
      const legitimateText = [
        'I like to kick the ball',
        'kickstart your day',
        'Let\'s kick off the meeting',
        'Keep calm and kick on',
        'A quick kick in the pants',
        'He needs a kick up the backside'
      ];
      
      legitimateText.forEach(text => {
        const result = detectKickVariations(text);
        expect(result.detected).toBe(false);
      });
    });
  });
  
  describe('Phase 1.6: Fix Remaining Bypass Patterns', () => {
    const phase16TestCases = [
      // Patterns that were still bypassing detection
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
      'k(e.i)ck'
    ];
    
    test.each(phase16TestCases)('should detect pattern "%s"', (example) => {
      const result = detectKickVariations(example);
      expect(result.detected).toBe(true);
      expect(result.confidence).toBeGreaterThan(50);
      expect(result.matches.length).toBeGreaterThan(0);
    });
    
    test('should identify correct techniques for Phase 1.6 patterns', () => {
      const techniqueTests = [
        { input: 'k(__ei__)ck', expectedTechniques: ['parentheses_ck', 'enhanced_parentheses'] },
        { input: 'k(__..i..__))k', expectedTechniques: ['multiple_parentheses'] },
        { input: 'k.e.e.k', expectedTechniques: ['single_dots'] },
        { input: 'k...e.i...k', expectedTechniques: ['flexible_dots'] }
      ];
      
      techniqueTests.forEach(({ input, expectedTechniques }) => {
        const result = detectKickVariations(input);
        expect(result.detected).toBe(true);
        expect(expectedTechniques.some(tech => result.techniques.includes(tech))).toBe(true);
      });
    });
    
    test('should still maintain legitimate text detection after Phase 1.6', () => {
      const legitimateText = [
        'I like to kick the ball',
        'kickstart your day',
        'Let\'s kick off the meeting',
        'He kicked the habit',
        'A quick kick in the pants',
        'Keep calm and kick on'
      ];
      
      legitimateText.forEach(text => {
        const result = detectKickVariations(text);
        expect(result.detected).toBe(false);
      });
    });
  });
  
  describe('Phase 1.7: Phonetic Variations', () => {
    describe('Phonetic Sound-Alike Detection', () => {
      const phoneticVariations = [
        { input: 'keek', expectedTechnique: 'alternative_spelling' },
        { input: 'find me on keek', expectedTechnique: 'alternative_spelling' },
        { input: 'keak', expectedTechnique: 'vowel_variation' },
        { input: 'kyck', expectedTechnique: 'y_vowel' },
        { input: 'kyyk', expectedTechnique: 'y_vowel' },
        { input: 'kouk', expectedTechnique: 'vowel_variation' },
        { input: 'kaik', expectedTechnique: 'vowel_variation' },
        { input: 'kic', expectedTechnique: 'short_variation' },
        { input: 'kiq', expectedTechnique: 'short_variation' },
        { input: 'kook', expectedTechnique: 'double_vowel' },
        { input: 'kuuk', expectedTechnique: 'double_vowel' },
        { input: 'k33k', expectedTechnique: 'alternative_spelling' },
        { input: 'ke3k', expectedTechnique: 'mixed_vowel' },
        { input: 'k3ek', expectedTechnique: 'mixed_vowel' }
      ];
      
      test.each(phoneticVariations)('should detect phonetic variation "%s"', ({ input, expectedTechnique }) => {
        const result = detectKickVariations(input);
        expect(result.detected).toBe(true);
        expect(result.techniques).toContain(expectedTechnique);
        expect(result.confidence).toBeGreaterThan(30);
      });
    });
    
    describe('Phonetic False Positive Prevention', () => {
      const legitimateWords = [
        'peek', 'meek', 'seek', 'week', 'geek',
        'keep', 'keen', 'peep', 'beep', 'deep',
        'cook', 'book', 'look', 'took', 'hook',
        'kayak', 'peak', 'leak', 'weak', 'beak'
      ];
      
      test.each(legitimateWords)('should NOT detect legitimate word "%s"', (word) => {
        const result = detectKickVariations(word);
        expect(result.detected).toBe(false);
        expect(result.confidence).toBe(0);
      });
    });
    
    test('should detect "keek" with fuzzy match as well', () => {
      const result = detectKickVariations('keek');
      expect(result.detected).toBe(true);
      // Should be detected by both alternative_spelling pattern and fuzzy match
      expect(result.techniques.length).toBeGreaterThan(0);
    });
    
    test('should handle mixed context with phonetic variations', () => {
      const testCases = [
        'Follow me on keek for gaming streams',
        'check out my kyck channel',
        'streaming on keak platform'
      ];
      
      testCases.forEach(text => {
        const result = detectKickVariations(text);
        const contextResult = contextualAnalysis(text, result);
        expect(contextResult.detected).toBe(true);
        expect(contextResult.confidence).toBeGreaterThan(60);
      });
    });
  });
});