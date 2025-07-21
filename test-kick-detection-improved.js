// Test the improved kick detection
const { detectKickLinks, testPhrase } = require('./lib/kickDetectionImproved.ts');

console.log('🧪 Testing Improved Kick Detection\n');
console.log('=' * 50);

// Test cases
const testCases = [
  // Should NOT trigger (legitimate phrases)
  { text: "I want to find a strong confident woman that will be dominant and take charge.", expected: false },
  { text: "Let's make changes to our relationship", expected: false },
  { text: "I'll take care of everything", expected: false },
  { text: "We need to make chemistry happen", expected: false },
  { text: "Don't make choices you'll regret", expected: false },
  { text: "Time to bake chicken for dinner", expected: false },
  { text: "Let's shake champagne and celebrate", expected: false },
  
  // Should trigger (actual obfuscation attempts)
  { text: "Check out my profile on k.i.c.k", expected: true },
  { text: "Find me on k__ck", expected: true },
  { text: "My username is @user on k(i)ck", expected: true },
  { text: "Visit k...i...c...k.com", expected: true },
  { text: "k i c k dot com", expected: true },
  { text: "My k!ck is @username", expected: true },
  { text: "(k)ick me at @user", expected: true },
];

console.log('\nTest Results:\n');

testCases.forEach(({ text, expected }, index) => {
  const result = detectKickLinks(text);
  const passed = result.detected === expected;
  
  console.log(`Test ${index + 1}: ${passed ? '✅ PASS' : '❌ FAIL'}`);
  console.log(`Text: "${text}"`);
  console.log(`Expected: ${expected}, Got: ${result.detected}`);
  
  if (result.detected) {
    console.log(`Confidence: ${(result.confidence * 100).toFixed(0)}%`);
    console.log(`Techniques: ${result.techniques.join(', ')}`);
    console.log(`Matches: ${result.matches.join(', ')}`);
  }
  
  console.log('-'.repeat(50));
});

// Summary
const passedTests = testCases.filter(({ text, expected }) => {
  const result = detectKickLinks(text);
  return result.detected === expected;
}).length;

console.log(`\n📊 Summary: ${passedTests}/${testCases.length} tests passed`);

// Additional test for the specific phrase function
console.log('\n🔍 Testing specific phrases:\n');
const phrasesToTest = [
  "take charge",
  "k.i.c.k",
  "make changes",
  "k__i__c__k"
];

phrasesToTest.forEach(phrase => {
  const result = testPhrase(phrase);
  console.log(`"${phrase}": ${result.wouldTrigger ? '🚫 Would trigger' : '✅ Safe'}`);
  if (result.reason) {
    console.log(`  Reason: ${result.reason}`);
  }
});