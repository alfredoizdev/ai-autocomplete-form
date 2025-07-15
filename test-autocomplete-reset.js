// Test script to verify autocomplete reset bug is fixed

async function testAutocomplete(prompt, testName) {
  console.log(`\n📝 ${testName}`);
  console.log(`   Input: "${prompt}"`);
  
  try {
    const response = await fetch('http://localhost:8001/api/autocomplete/hybrid', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ prompt })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    
    console.log(`   ✅ Suggestions:`);
    data.combined_suggestions.forEach((suggestion, i) => {
      console.log(`      ${i + 1}. ${suggestion}`);
    });
    
    return data.combined_suggestions;
    
  } catch (error) {
    console.error(`   ❌ Error: ${error.message}`);
    return [];
  }
}

async function simulateClearAndRestart() {
  console.log('\n🧪 Testing Autocomplete Reset Bug Fix\n');
  console.log('This test simulates the user behavior that causes the bug:');
  console.log('1. Type a bio');
  console.log('2. Clear it');
  console.log('3. Type a new bio');
  console.log('4. Check if suggestions are appropriate\n');
  
  // Step 1: First bio entry
  console.log('==== STEP 1: First Bio Entry ====');
  await testAutocomplete(
    "We are an older couple looking for a playful duo ready to intertwine our bodies and unleash a symphony of moans tonight.",
    "First complete bio"
  );
  
  // Step 2: Start typing second sentence
  await testAutocomplete(
    "We are an older couple looking for a playful duo ready to intertwine our bodies and unleash a symphony of moans tonight. If you email us and we dont respond",
    "Adding second sentence"
  );
  
  // Step 3: Simulate clearing (empty string)
  console.log('\n==== STEP 2: Clear Textarea ====');
  console.log('   [User clears the textarea completely]');
  
  // Step 4: Start typing new bio
  console.log('\n==== STEP 3: New Bio After Clear ====');
  await testAutocomplete(
    "I am a young male swinger looking for",
    "New bio after clear"
  );
  
  // Step 5: Continue new bio
  await testAutocomplete(
    "I am a young male swinger looking for couples who want to explore. If you message me and I dont respond",
    "Second sentence of new bio"
  );
  
  console.log('\n📊 Test Analysis:');
  console.log('If the bug is FIXED:');
  console.log('  - Suggestions should be contextually appropriate');
  console.log('  - No weird fragments like ", we so look forward"');
  console.log('  - Each bio should get fresh, relevant suggestions');
  console.log('\nIf the bug PERSISTS:');
  console.log('  - You might see fragments from the old bio');
  console.log('  - Suggestions may not make sense in context');
}

// Run the test
simulateClearAndRestart().catch(console.error);