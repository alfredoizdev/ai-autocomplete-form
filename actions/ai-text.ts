"use server";
import { Bios } from "@/data/Bios";
import { Bio } from "@/type/Collection";
import weaviate from "weaviate-client";

const chatHistory: { role: "user" | "assistant"; content: string }[] = [];

export const setCollectionForVectorDB = async () => {
  const client = await weaviate.connectToLocal();

  const collection = client.collections.use<Bio>("Bio2");

  const entries = Bios.map((bio, index) => ({
    title: `Bio2 ${index + 1}`,
    body: bio,
  }));

  await collection.data.insertMany(entries);
  console.log(`✅ Inserted ${entries.length} bios into Weaviate.`);
};

export const askOllamaCompletationAction = async (input: string) => {
  const client = await weaviate.connectToLocal();
  const collection = client.collections.get<Bio>("Bio2");

  // 🔍 Buscar contexto relevante
  const result = await collection.query.nearText([input], { limit: 3 });
  const context = result.objects.map((obj) => obj.properties.body).join("\n\n");

  // 📜 Mensajes para Ollama
  const messages = [
    {
      role: "system",
      content: `You write seductive bio completions that make swingers desperately want to meet and play.

Write like you're seducing someone into your bed. Make them feel desired, excited, and eager to explore.

CRITICAL: Always complete the thought in a way that:
- Creates anticipation and desire
- Makes them imagine the pleasure ahead
- Sounds confident, playful, and irresistibly naughty
- Forms a complete, enticing sentence (5-10 words)
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
      content: `Relevant swinger bios for context:\n${context}`,
    },
    ...chatHistory,
    {
      role: "user",
      content: `Complete the following bio:\n"${input}"`,
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
      temperature: 0.75,
      top_p: 0.9,
      max_tokens: 40,
      stop: [".", "\n", "!", "?", "...", "…"], // para evitar que inicie nuevo párrafo
    }),
  });

  const data = await response.json();

  const output = data?.message?.content
    ?.trim()
    ?.replace(/\.{3,}/g, "") // Remove any ellipsis (3 or more dots)
    ?.replace(/…/g, "") // Remove single ellipsis character
    ?.trim(); // Trim again after cleaning

  if (output) {
    chatHistory.push({ role: "user", content: input });
    chatHistory.push({ role: "assistant", content: output });
  }

  return output || "No answer found";
};
