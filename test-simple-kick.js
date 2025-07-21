// Simple test for kick detection
const improvedDetection = require('./lib/kickDetectionImproved.ts');

// Test the specific case that was failing
const testText = "I want to find a strong confident woman that will be dominant and take charge.";

console.log("Testing: ", testText);
console.log("\nOld pattern would match 'ake ch' as potential 'k...i...ck'");

// The improved detection should not flag this
const result = improvedDetection.detectKickLinks(testText);
console.log("\nImproved detection result:");
console.log("Detected:", result.detected);
console.log("Confidence:", result.confidence);
console.log("Has legitimate usage:", result.hasLegitimateUsage);

// Test actual kick references
const actualKick = "Check out my k.i.c.k profile";
const result2 = improvedDetection.detectKickLinks(actualKick);
console.log("\n\nTesting actual kick reference:", actualKick);
console.log("Detected:", result2.detected);
console.log("Confidence:", result2.confidence);
console.log("Techniques:", result2.techniques);