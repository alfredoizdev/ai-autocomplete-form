// Simple test script for kick detection module
const { detectKickVariations } = require('./lib/kickDetection');

// Test all 28 user examples
const userExamples = [
  'k i k', 'K ik', 'k!k', 'k_!_k', 'k-i-k', 'k..i..k',
  'k.i.k', 'ki.k', 'kiik', 'kik', 'keek', 'kiek',
  'keik', 'kick', 'k(i)k', 'k___ik', 'k_i_k', 'k_l_k',
  'klk', 'kiik', 'kiilk', 'kiiik', 'kilk', 'killk',
  'k1k', 'k_1_k', 'k1_k', 'k_1k', 'klik', 'k11k', 'kii_k'
];

console.log('Testing Kick Detection Module\n');
console.log('=' .repeat(50));

// Test user examples
console.log('\nTesting User Examples:');
userExamples.forEach(example => {
  const result = detectKickVariations(example);
  console.log(`"${example}" - Detected: ${result.detected ? '✓' : '✗'} | Confidence: ${result.confidence}%`);
});

// Test domain variations
console.log('\n\nTesting Domain Variations:');
const domainExamples = [
  'k!ck.com',
  'k i c k . c o m',
  'kick[.]com',
  'kick dot com',
  'check out my k!ck channel'
];

domainExamples.forEach(example => {
  const result = detectKickVariations(example);
  console.log(`"${example}" - Detected: ${result.detected ? '✓' : '✗'} | Confidence: ${result.confidence}%`);
});

// Test false positives
console.log('\n\nTesting False Positives (should NOT be detected):');
const legitimateText = [
  'I like to kick the ball',
  'kickstart your day',
  'a quick brown fox',
  'kitchen kicks'
];

legitimateText.forEach(example => {
  const result = detectKickVariations(example);
  console.log(`"${example}" - Detected: ${result.detected ? '✓' : '✗'} | Confidence: ${result.confidence}%`);
});

console.log('\n' + '=' .repeat(50));
console.log('Test complete!');