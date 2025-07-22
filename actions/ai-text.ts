"use server";
// import { Bios } from "@/data/Bios";
// import { Bio } from "@/type/Collection";

// Removed global chat history - autocomplete should be stateless
// Each request should be independent without carrying previous context

// Cache for API server status
let apiServerAvailable: boolean | null = null;
let lastHealthCheck = 0;
const HEALTH_CHECK_INTERVAL = 60000; // Check every 60 seconds

// Helper function to check if the Python API server is running
async function checkApiServerHealth(): Promise<boolean> {
  const now = Date.now();

  // Use cached result if recent
  if (
    apiServerAvailable !== null &&
    now - lastHealthCheck < HEALTH_CHECK_INTERVAL
  ) {
    return apiServerAvailable;
  }

  try {
    const response = await fetch("http://localhost:8001/", {
      method: "GET",
      signal: AbortSignal.timeout(2000), // 2 second timeout
    });

    apiServerAvailable = response.ok;
    lastHealthCheck = now;

    if (apiServerAvailable) {
      console.log("✅ Python API server is running on port 8001");
    }

    return apiServerAvailable;
  } catch (error) {
    apiServerAvailable = false;
    lastHealthCheck = now;
    return false;
  }
}

// This function is no longer used - vector search is handled by the Python API
// export const setCollectionForVectorDB = async () => {
//   const client = await weaviate.connectToLocal();
//
//   const collection = client.collections.use<Bio>("Bio2");
//
//   const entries = Bios.map((bio, index) => ({
//     title: `Bio2 ${index + 1}`,
//     body: bio,
//   }));
//
//   await collection.data.insertMany(entries);
//   console.log(`✅ Inserted ${entries.length} bios into Weaviate.`);
// };

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

export const askOllamaCompletationAction = async (input: string) => {
  // Autocomplete is now stateless - no history tracking
  
  // Check which mode to use
  const mode = process.env.AUTOCOMPLETE_MODE || 'hybrid';
  
  if (mode === 'trained') {
    // Use trained MLX model
    try {
      const mlxResponse = await fetch(
        "http://localhost:8003/api/autocomplete/mlx",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            prompt: input,
            max_tokens: 50,  // Increased to ensure complete sentences
            temperature: 0.7,
            stop: [".", "!", "?", "\n"]  // Stop at sentence endings
          }),
        }
      );

      if (mlxResponse.ok) {
        const data = await mlxResponse.json();
        console.log(`Trained model autocomplete: ${data.elapsed_ms}ms`);
        console.log(`Using: ${data.model_name}`);
        
        // Return the completion after stripping the prompt
        if (data.completion) {
          return stripPromptFromResponse(input, data.completion);
        }
      }
    } catch (error: any) {
      console.error("Trained model error:", error);
      // Don't fall back to hybrid in trained mode - user explicitly chose this mode
      console.error("Make sure MLX server is running: ./start_trained.sh");
      return "Model not available - check server";
    }
  } else if (mode === 'hybrid') {
    // Use hybrid mode (vector search + AI)
    const serverAvailable = await checkApiServerHealth();

    if (serverAvailable) {
      try {
        // Use the new hybrid endpoint that combines vector search with LLM generation
        const hybridResponse = await fetch(
          "http://localhost:8001/api/autocomplete/hybrid",
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              prompt: input,
            }),
          }
        );

        if (hybridResponse.ok) {
          const data = await hybridResponse.json();

          // Log performance metrics
          console.log(`Hybrid autocomplete: ${data.elapsed_ms}ms`);
          console.log(`Context used: ${data.context_used}`);
          console.log(
            `Suggestions: ${data.combined_suggestions.length} (${data.exact_matches.length} exact, ${data.llm_completions.length} generated)`
          );

          // Return the first combined suggestion after stripping the prompt
          if (data.combined_suggestions && data.combined_suggestions.length > 0) {
            const suggestion = data.combined_suggestions[0];
            return stripPromptFromResponse(input, suggestion);
          }
        }
      } catch (error: any) {
        // Check if it's a connection error
        if (error.cause?.code === "ECONNREFUSED") {
          console.error("❌ API server is not running on port 8001");
          console.error("To start hybrid mode, run: ./start_hybrid.sh");
        } else {
          console.error("Hybrid autocomplete API error:", error);
        }
      }
    } else {
      console.log(
        "⚠️ API server not available. Make sure to run: ./start_hybrid.sh"
      );
      return "Hybrid mode not available - check server";
    }
  }

  // Fallback to direct Ollama method if vector search fails or returns no results
  try {
    // 📜 Mensajes para Ollama (without Weaviate context)
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
      // No chat history - each autocomplete request is independent
      {
        role: "user",
        content: `Complete this bio text: ${input}`,
      },
    ];

    // 🧠 Llamada a Ollama
    const response = await fetch(`${process.env.OLLAMA_PATH_API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: "gemma3:12b",
        messages,
        stream: false,
        temperature: 0.9, // Increased for more variety
        top_p: 0.92, // Slightly lower to focus on likely completions
        top_k: 50, // Add top_k to limit vocabulary choices
        max_tokens: 50, // Shorter to match bio style
        repeat_penalty: 1.1, // Prevent repetitive phrases
        stop: ["\n", "\n\n", ".", "!", "?"], // Stop at sentence end
      }),
    });

    const data = await response.json();

    let output = data?.message?.content
      ?.trim()
      ?.replace(/\.{3,}/g, "") // Remove any ellipsis (3 or more dots)
      ?.replace(/…/g, "") // Remove single ellipsis character
      ?.trim(); // Trim again after cleaning

    // Strip the prompt from the response if it was repeated
    if (output) {
      output = stripPromptFromResponse(input, output);
    }

    // Post-process: ensure lowercase if input ends with comma or no sentence-ending punctuation
    if (output && input) {
      const lastChar = input.trim().slice(-1);
      if (
        lastChar === "," ||
        lastChar === ":" ||
        lastChar === ";" ||
        (lastChar && ![".", "!", "?"].includes(lastChar))
      ) {
        // Force lowercase on first character
        output = output.charAt(0).toLowerCase() + output.slice(1);
      }
    }

    // No longer storing chat history - each request is independent

    return output || "No answer found";
  } catch (fallbackError) {
    console.error("Ollama fallback error:", fallbackError);
    return "No answer found";
  }
};
