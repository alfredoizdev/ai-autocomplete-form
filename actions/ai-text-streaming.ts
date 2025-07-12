"use server";

interface StreamChunk {
  text: string;
  done: boolean;
}

// Cache for autocomplete suggestions
const suggestionCache = new Map<string, { suggestion: string; timestamp: number }>();
const CACHE_TTL = 5 * 60 * 1000; // 5 minutes

// Clean old cache entries
const cleanCache = () => {
  const now = Date.now();
  for (const [key, value] of suggestionCache.entries()) {
    if (now - value.timestamp > CACHE_TTL) {
      suggestionCache.delete(key);
    }
  }
};

// Get cache key from prompt (normalize for better hits)
const getCacheKey = (prompt: string): string => {
  return prompt.trim().toLowerCase().slice(-50); // Last 50 chars for context
};

export async function* streamOllamaCompletion(input: string) {
  // Check cache first
  cleanCache();
  const cacheKey = getCacheKey(input);
  const cached = suggestionCache.get(cacheKey);
  
  if (cached) {
    yield { text: cached.suggestion, done: true } as StreamChunk;
    return;
  }

  try {
    // Try hybrid API first with streaming support
    const hybridResponse = await fetch('http://localhost:8001/api/autocomplete/hybrid', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        prompt: input,
        stream: true // Request streaming if supported
      }),
    });

    if (hybridResponse.ok && hybridResponse.body) {
      const reader = hybridResponse.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = '';

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split('\n');

          for (const line of lines) {
            if (line.trim()) {
              try {
                const data = JSON.parse(line);
                if (data.chunk) {
                  accumulated += data.chunk;
                  yield { text: data.chunk, done: false } as StreamChunk;
                }
              } catch (e) {
                // Handle non-JSON lines
              }
            }
          }
        }

        // Cache the complete suggestion
        if (accumulated) {
          suggestionCache.set(cacheKey, { suggestion: accumulated, timestamp: Date.now() });
        }

        yield { text: '', done: true } as StreamChunk;
        return;
      } catch (error) {
        console.error('Streaming error:', error);
      }
    }
  } catch (error) {
    console.error('Hybrid API streaming error:', error);
  }

  // Fallback to direct Ollama with streaming
  try {
    const messages = [
      {
        role: "system",
        content: `You write seductive bio completions that make swingers desperately want to meet and play.

Write like you're seducing someone into your bed. Make them feel desired, excited, and eager to explore.

CRITICAL: Always complete the thought in a way that:
- Creates anticipation and desire
- Makes them imagine the pleasure ahead
- Sounds confident, playful, and irresistibly naughty
- Forms a complete, enticing sentence (8-20 words)
- Uses explicit language about pleasure, orgasms, and desires

Examples of seductive completions:
"I am a young male swinger looking for" → "sexy couples who want to explore every inch of pleasure together tonight"
"We love meeting" → "adventurous souls ready to turn our wildest fantasies into unforgettable orgasmic reality"
"My biggest turn on is" → "watching my partner moan with pleasure while we share you completely"
"Looking for couples who" → "crave intense oral pleasure and aren't afraid to get deliciously dirty with us"

Be explicitly sexual. Make them want to message immediately. Output ONLY the seductive completion.`,
      },
      {
        role: "user",
        content: `Complete this bio text: ${input}`,
      },
    ];

    const response = await fetch(`${process.env.OLLAMA_PATH_API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma3:12b",
        messages,
        stream: true, // Enable streaming
        temperature: 0.85,
        top_p: 0.95,
        max_tokens: 100,
        stop: ["\n", "\n\n"],
      }),
    });

    if (!response.body) throw new Error('No response body');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulated = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);
            if (data.message?.content) {
              const content = data.message.content;
              accumulated += content;
              
              // Process and yield the chunk
              let processedChunk = content
                ?.replace(/\.{3,}/g, '')
                ?.replace(/…/g, '')
                ?.trim();

              // Handle capitalization for the first chunk
              if (accumulated === content && input) {
                const lastChar = input.trim().slice(-1);
                if (lastChar === ',' || lastChar === ':' || lastChar === ';' || 
                    (lastChar && !['.' , '!', '?'].includes(lastChar))) {
                  processedChunk = processedChunk.charAt(0).toLowerCase() + processedChunk.slice(1);
                }
              }

              if (processedChunk) {
                yield { text: processedChunk, done: false } as StreamChunk;
              }
            }
          } catch (e) {
            // Handle non-JSON lines
          }
        }
      }
    }

    // Cache the complete suggestion
    if (accumulated) {
      suggestionCache.set(cacheKey, { suggestion: accumulated, timestamp: Date.now() });
    }

    yield { text: '', done: true } as StreamChunk;
  } catch (error) {
    console.error('Ollama streaming error:', error);
    yield { text: 'No answer found', done: true } as StreamChunk;
  }
}