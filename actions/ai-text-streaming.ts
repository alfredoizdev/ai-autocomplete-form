"use server";

interface StreamChunk {
  text: string;
  done: boolean;
}

// Cache for autocomplete suggestions
const suggestionCache = new Map<
  string,
  { suggestion: string; timestamp: number }
>();
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

// Clear all cache - useful when text is cleared or significantly changed
export const clearSuggestionCache = async () => {
  suggestionCache.clear();
  console.log("Suggestion cache cleared");
};

// Helper function to remove the prompt from the beginning of the AI response
const stripPromptFromResponse = (prompt: string, response: string): string => {
  if (!prompt || !response) return response;

  // Normalize both strings for comparison (trim and lowercase)
  const normalizedPrompt = prompt.trim().toLowerCase();
  const normalizedResponse = response.trim().toLowerCase();

  // Check if response starts with the prompt
  if (normalizedResponse.startsWith(normalizedPrompt)) {
    // Remove the prompt portion, preserving original casing
    const cleanedResponse = response
      .trim()
      .substring(prompt.trim().length)
      .trim();
    return cleanedResponse;
  }

  // Also check if response contains prompt with slight variations (extra spaces, punctuation)
  const promptWords = normalizedPrompt.split(/\s+/);
  const responseWords = normalizedResponse.split(/\s+/);

  // If first N words match (where N is number of words in prompt), strip them
  if (promptWords.length > 0 && responseWords.length >= promptWords.length) {
    let matches = true;
    for (let i = 0; i < promptWords.length; i++) {
      if (promptWords[i] !== responseWords[i]) {
        matches = false;
        break;
      }
    }

    if (matches) {
      // Find where to cut in the original response
      let cutIndex = 0;
      let wordCount = 0;
      for (let i = 0; i < response.length; i++) {
        if (/\s/.test(response[i])) {
          if (wordCount === promptWords.length - 1) {
            cutIndex = i;
            break;
          }
          // Skip consecutive spaces
          while (i < response.length - 1 && /\s/.test(response[i + 1])) {
            i++;
          }
          wordCount++;
        }
      }

      if (cutIndex > 0) {
        return response.substring(cutIndex).trim();
      }
    }
  }

  return response;
};

// Simple hash function for better cache key generation
const simpleHash = (str: string): string => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i);
    hash = (hash << 5) - hash + char;
    hash = hash & hash; // Convert to 32-bit integer
  }
  return Math.abs(hash).toString(36);
};

// Get cache key from prompt (use full prompt hash to avoid false matches)
const getCacheKey = (prompt: string): string => {
  const normalized = prompt.trim().toLowerCase();
  // Combine hash with last 20 chars for context
  const contextPart = normalized.slice(-20);
  return `${simpleHash(normalized)}_${contextPart}`;
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
    const hybridResponse = await fetch(
      "http://localhost:8001/api/autocomplete/hybrid",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          prompt: input,
          stream: true, // Request streaming if supported
        }),
      }
    );

    if (hybridResponse.ok && hybridResponse.body) {
      const reader = hybridResponse.body.getReader();
      const decoder = new TextDecoder();
      let accumulated = "";

      try {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          const lines = chunk.split("\n");

          for (const line of lines) {
            if (line.trim()) {
              try {
                const data = JSON.parse(line);
                if (data.chunk) {
                  const previousLength = accumulated.length;
                  accumulated += data.chunk;

                  // For streaming, we need to be careful about when to strip the prompt
                  // Only strip from the first chunk that completes the prompt
                  if (
                    previousLength < input.length &&
                    accumulated.length >= input.length
                  ) {
                    // We just crossed the prompt boundary, check if we need to strip
                    const strippedAccumulated = stripPromptFromResponse(
                      input,
                      accumulated
                    );
                    if (strippedAccumulated !== accumulated) {
                      // The prompt was included, yield only the new part
                      const newContent = strippedAccumulated.substring(
                        previousLength > 0 ? previousLength - input.length : 0
                      );
                      if (newContent) {
                        yield { text: newContent, done: false } as StreamChunk;
                      }
                      accumulated = strippedAccumulated;
                    } else {
                      // Normal chunk, yield as is
                      yield { text: data.chunk, done: false } as StreamChunk;
                    }
                  } else if (previousLength >= input.length) {
                    // We're past the prompt, yield normally
                    yield { text: data.chunk, done: false } as StreamChunk;
                  }
                  // If we're still within prompt length, accumulate but don't yield yet
                }
              } catch (e) {
                // Handle non-JSON lines
              }
            }
          }
        }

        // Cache the complete suggestion after stripping the prompt
        if (accumulated) {
          const cleaned = stripPromptFromResponse(input, accumulated);
          suggestionCache.set(cacheKey, {
            suggestion: cleaned,
            timestamp: Date.now(),
          });
        }

        yield { text: "", done: true } as StreamChunk;
        return;
      } catch (error) {
        console.error("Streaming error:", error);
      }
    }
  } catch (error) {
    console.error("Hybrid API streaming error:", error);
  }

  // Fallback to direct Ollama with streaming
  try {
    const messages = [
      {
        role: "system",
        content: `You complete dating/lifestyle bios using ONLY this specific vocabulary and style:

MUST USE THESE EXACT PHRASES AND WORDS:
- "fun in and out of the bedroom"
- "no drama" / "drama free"
- "same room" / "full swap" / "soft swap"
- "d&d free" / "ddf" / "clean"
- "discrete" or "discreet"
- "down to earth"
- "open minded"
- "laid back"
- "friends first" / "friendship and fun"
- "see where it goes" / "if chemistry is right"
- "NO SINGLE MEN" (in caps when rejecting)
- "select singles" / "couples and single ladies"
- "NSA" / "FWB" / "no strings"
- "good times" / "adult fun"
- "meet for drinks/dinner"
- "hot tub" / "house parties"
- "must be real" / "no games"

WRITE LIKE THIS:
- Short, casual sentences (5-20 words)
- Missing punctuation is OK
- Use "..." for trailing thoughts
- Fragment sentences are fine
- Run-on sentences with commas
- Very direct about what you want
- Mix activities: social (dinner, drinks) + sexual
- List format: "looking for couples, singles, groups"

CRITICAL: Output ONLY the completion. NEVER repeat the user's input.

REAL EXAMPLES from actual bios:
"We are looking for" → " couples and single ladies to join us for dinner, fine wine, rich conversation and all the other benefits that come along with the lifestyle"
"Looking for" → " fun loving couples to share dinners,dancing,more..."
"Couple seeking" → " other couples for friendship and fun in and out of the bedroom"
"We enjoy" → " meeting new people for drinks, dancing and if the chemistry is right some adult fun"
"Looking to meet" → " down to earth couples who are drama free and know how to have a good time"

Use multiple periods... casual spelling... incomplete sentences`,
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

    if (!response.body) throw new Error("No response body");

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let accumulated = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split("\n");

      for (const line of lines) {
        if (line.trim()) {
          try {
            const data = JSON.parse(line);
            if (data.message?.content) {
              const content = data.message.content;
              accumulated += content;

              // For Ollama streaming, we need to handle accumulated text differently
              const previousLength = accumulated.length;

              // Process the chunk
              let processedChunk = content
                ?.replace(/\.{3,}/g, "")
                ?.replace(/…/g, "")
                ?.trim();

              // Check if we need to strip the prompt from accumulated text
              if (accumulated.toLowerCase().startsWith(input.toLowerCase())) {
                // The AI is including the prompt, we need to strip it
                const strippedAccumulated = stripPromptFromResponse(
                  input,
                  accumulated
                );

                // Calculate what part of the stripped text is new
                if (previousLength <= input.length) {
                  // We were still in the prompt part, yield only truly new content
                  const newContent = strippedAccumulated.substring(
                    0,
                    content.length
                  );
                  if (newContent) {
                    processedChunk = newContent;
                  } else {
                    // This chunk was all prompt, skip it
                    continue;
                  }
                } else {
                  // We were already past the prompt
                  processedChunk = content;
                }

                // Update accumulated to the stripped version
                accumulated = strippedAccumulated;
              }

              // Handle capitalization for the first real chunk
              if (processedChunk && input) {
                const lastChar = input.trim().slice(-1);
                if (
                  lastChar === "," ||
                  lastChar === ":" ||
                  lastChar === ";" ||
                  (lastChar && ![".", "!", "?"].includes(lastChar))
                ) {
                  processedChunk =
                    processedChunk.charAt(0).toLowerCase() +
                    processedChunk.slice(1);
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

    // Cache the complete suggestion after stripping the prompt
    if (accumulated) {
      const cleaned = stripPromptFromResponse(input, accumulated);
      suggestionCache.set(cacheKey, {
        suggestion: cleaned,
        timestamp: Date.now(),
      });
    }

    yield { text: "", done: true } as StreamChunk;
  } catch (error) {
    console.error("Ollama streaming error:", error);
    yield { text: "No answer found", done: true } as StreamChunk;
  }
}
