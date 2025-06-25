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
      content: `You are an assistant that completes bios for people in the swinger and open lifestyle community.
Your job is to take the user's partial bio and COMPLETE IT without repeating it.
Make the tone sexy, confident, and playful. Focus on lifestyle themes: soft swap, bisexual women, drama-free fun, open-minded couples, and mutual pleasure.
Do not start a new sentence. Continue the user sentence naturally in the same grammatical person (I, we, she, he).
Respond with ONLY the completion (no quotes, no intro, no repetition). Respond with 3 to 5 words MAX.`,
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
      stop: [".", "\n", "!", "?"], // para evitar que inicie nuevo párrafo
    }),
  });

  const data = await response.json();

  const output = data?.message?.content?.trim();

  if (output) {
    chatHistory.push({ role: "user", content: input });
    chatHistory.push({ role: "assistant", content: output });
  }

  return output || "No answer found";
};
