// Test script for the autocomplete API
const testQueries = [
  "Looking for friends who",
  "I am a software",
  "My hobbies include",
  "Looking for couples",
  "I enjoy",
  "We are a",
  "Seeking fun"
];

async function testAutocomplete() {
  console.log("Testing Autocomplete API...\n");
  
  for (const query of testQueries) {
    try {
      const response = await fetch('http://localhost:8001/api/autocomplete', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ prompt: query })
      });
      
      const data = await response.json();
      
      console.log(`Query: "${query}"`);
      console.log(`Completion: "${data.completion}"`);
      console.log(`Time: ${data.elapsed_ms.toFixed(1)}ms`);
      console.log(`Suggestions: ${data.suggestions.length}`);
      console.log('---');
    } catch (error) {
      console.error(`Error for query "${query}":`, error.message);
    }
  }
}

testAutocomplete();