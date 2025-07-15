// Test script for the integrated autocomplete with trained model

async function testAutocomplete(prompt) {
  console.log(`\n🔍 Testing: "${prompt}"`);
  
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
    
    console.log(`⏱️  Response time: ${data.elapsed_ms}ms`);
    console.log(`📊 Context used: ${data.context_used}`);
    console.log(`\n📝 Results:`);
    console.log(`  - Exact matches: ${data.exact_matches.length}`);
    console.log(`  - AI completions: ${data.llm_completions.length}`);
    console.log(`\n✨ Top suggestions:`);
    data.combined_suggestions.forEach((suggestion, i) => {
      console.log(`  ${i + 1}. ${suggestion}`);
    });
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

async function checkServers() {
  console.log('🔍 Checking server status...\n');
  
  // Check API server
  try {
    const apiResponse = await fetch('http://localhost:8001/');
    const apiData = await apiResponse.json();
    console.log('✅ API Server (8001):', apiData.status);
  } catch (error) {
    console.log('❌ API Server (8001): Not running');
  }
  
  // Check trained model server
  try {
    const modelResponse = await fetch('http://localhost:8002/health');
    const modelData = await modelResponse.json();
    console.log('✅ Trained Model Server (8002):', modelData.status);
  } catch (error) {
    console.log('❌ Trained Model Server (8002): Not running');
  }
  
  console.log('');
}

async function main() {
  console.log('🚀 Testing Integrated Autocomplete System\n');
  
  // Check servers first
  await checkServers();
  
  // Test prompts
  const testPrompts = [
    "I am a young male swinger looking for",
    "I would love to find a woman that",
    "We are a couple who",
    "Looking for friends to",
    "My biggest turn on is"
  ];
  
  for (const prompt of testPrompts) {
    await testAutocomplete(prompt);
    // Small delay between tests
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  console.log('\n✅ Test complete!');
}

main().catch(console.error);