// Test script to check if the Python API server is running

async function testApiHealth() {
  console.log('Testing Python API server health...\n');
  
  try {
    // Test health endpoint
    console.log('1. Testing health endpoint (http://localhost:8001/)...');
    const healthResponse = await fetch('http://localhost:8001/');
    
    if (healthResponse.ok) {
      const healthData = await healthResponse.json();
      console.log('✅ Server is running!');
      console.log('Response:', JSON.stringify(healthData, null, 2));
    } else {
      console.log('❌ Server returned error:', healthResponse.status);
    }
    
    // Test stats endpoint
    console.log('\n2. Testing stats endpoint (http://localhost:8001/api/stats)...');
    const statsResponse = await fetch('http://localhost:8001/api/stats');
    
    if (statsResponse.ok) {
      const statsData = await statsResponse.json();
      console.log('✅ Stats endpoint working!');
      console.log('Response:', JSON.stringify(statsData, null, 2));
    }
    
    // Test autocomplete with a sample prompt
    console.log('\n3. Testing hybrid autocomplete endpoint...');
    const testPrompt = 'I am a young male swinger looking for';
    
    const autocompleteResponse = await fetch('http://localhost:8001/api/autocomplete/hybrid', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: testPrompt
      }),
    });
    
    if (autocompleteResponse.ok) {
      const autocompleteData = await autocompleteResponse.json();
      console.log('✅ Autocomplete endpoint working!');
      console.log('Test prompt:', testPrompt);
      console.log('Response:', JSON.stringify(autocompleteData, null, 2));
    } else {
      console.log('❌ Autocomplete error:', autocompleteResponse.status);
    }
    
  } catch (error) {
    if (error.cause?.code === 'ECONNREFUSED') {
      console.log('\n❌ ERROR: Python API server is not running!');
      console.log('\nTo start the server:');
      console.log('1. Open a new terminal');
      console.log('2. Run: ./start_api_server.sh');
      console.log('   OR');
      console.log('   cd python && python3 api/api_server.py');
      console.log('\nMake sure Ollama is also running (ollama serve)');
    } else {
      console.log('\n❌ ERROR:', error.message);
    }
  }
}

// Run the test
testApiHealth();