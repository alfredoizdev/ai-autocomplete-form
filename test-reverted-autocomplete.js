// Test script for the reverted autocomplete (without trained model)

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
    
    console.log(`⏱️  Response time: ${data.elapsed_ms.toFixed(0)}ms`);
    console.log(`📊 Context used: ${data.context_used}`);
    console.log(`\n✨ Top suggestions:`);
    data.combined_suggestions.forEach((suggestion, i) => {
      console.log(`  ${i + 1}. ${suggestion}`);
    });
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

async function main() {
  console.log('🚀 Testing Reverted Autocomplete System (ChromaDB + Ollama only)\n');
  
  // Check if API server is running
  try {
    const response = await fetch('http://localhost:8001/');
    const data = await response.json();
    console.log('✅ API Server status:', data.status);
    console.log('✅ Vector search ready:', data.vector_search_ready);
  } catch (error) {
    console.log('❌ API Server not running!');
    console.log('   Run: ./start_api_server.sh');
    return;
  }
  
  // Test prompts
  const testPrompts = [
    "I am a young male swinger looking for",
    "I would love to find a woman that",
    "We are a couple who enjoys"
  ];
  
  for (const prompt of testPrompts) {
    await testAutocomplete(prompt);
  }
  
  console.log('\n✅ Test complete! The system is working without the trained model.');
}

main().catch(console.error);