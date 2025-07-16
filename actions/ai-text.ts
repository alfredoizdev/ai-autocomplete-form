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
  // Check if API server is available
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
        console.error("❌ Python API server is not running on port 8001");
        console.error("To start the server, run: ./start_api_server.sh");
        console.error("Or manually: cd python && python3 api/api_server.py");
      } else {
        console.error("Hybrid autocomplete API error:", error);
      }
    }
  } else {
    console.log(
      "⚠️ Python API server not available, using fallback Ollama method"
    );
  }

  // Fallback to direct Ollama method if vector search fails or returns no results
  try {
    // 📜 Mensajes para Ollama (without Weaviate context)
    const messages = [
      {
        role: "system",
        content: `You write seductive bio completions that make swingers desperately want to meet and play.

Write like you're seducing someone into your bed. Make them feel desired, excited, and eager to explore.

CRITICAL INSTRUCTION: You must output ONLY the completion text, NOT the original prompt. Never repeat what the user has already written.

Rules for completion:
- Do not sound too fancy it doesnt work with swingers
- Creates anticipation and desire
- Makes them imagine the pleasure ahead
- Be direct yet creative
- Sounds confident, kinky and irresistibly naughty
- Forms a complete, enticing sentence (8-20 words)
- Uses explicit language about pleasure, orgasms

Examples of CORRECT completions (notice we only return the new part):
"I am looking for couples or single ladies" → "am ddf, free, and looking for friends and playmates"
"Looking for fun and discret pleasures" → " with adventurous souls ready to turn our wildest fantasies into reality"
"I am a young male swinger looking for" → "partners who appreciate a dominant lover and crave deep, throbbing, orgasmic release."
"We like quiet evenings at our house" → "and to soft swing to full swap with playful couples"

Examples of INCORRECT completions (DO NOT do this):
"I am a young male swinger looking for" → "I am a young male swinger looking for sexy couples..."
"We love meeting" → "We love meeting adventurous souls..."

Be explicitly sexual. Make them want to message immediately. Output ONLY the continuation, NEVER repeat the input.`,
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
        temperature: 0.85,
        top_p: 0.95,
        max_tokens: 100,
        stop: ["\n", "\n\n"], // para evitar que inicie nuevo párrafo
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
