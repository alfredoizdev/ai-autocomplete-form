"use server";
// import { Bios } from "@/data/Bios";
// import { Bio } from "@/type/Collection";

const chatHistory: { role: "user" | "assistant"; content: string }[] = [];

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

export const askOllamaCompletationAction = async (input: string) => {
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

        // Return the first combined suggestion
        if (data.combined_suggestions && data.combined_suggestions.length > 0) {
          return data.combined_suggestions[0];
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
      ...chatHistory,
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

    if (output) {
      chatHistory.push({ role: "user", content: input });
      chatHistory.push({ role: "assistant", content: output });
    }

    return output || "No answer found";
  } catch (fallbackError) {
    console.error("Ollama fallback error:", fallbackError);
    return "No answer found";
  }
};
