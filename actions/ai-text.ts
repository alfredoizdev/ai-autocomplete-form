"use server";

function cleanCompletion(text: string): string {
  // Remove any quotes, punctuation, and unwanted characters
  const cleaned = text
    .replace(/^["'`\.\s]*/, "") // Remove starting quotes, dots, spaces
    .replace(/["'`\.\s]*$/, "") // Remove ending quotes, dots, spaces
    .replace(/\n.*$/g, "") // Remove everything after first line break
    .replace(/[.!?;:,'""`]/g, "") // Remove all punctuation
    .trim();

  // Split into words and take only first 3-5 words
  const words = cleaned.split(/\s+/).filter((word) => word.length > 0);
  const limitedWords = words.slice(0, 5); // Maximum 5 words

  // Only return if we have at least 3 words
  if (limitedWords.length >= 3) {
    return limitedWords.join(" ");
  }

  return "";
}

export async function askOllamaCompletationAction(
  userInputs: string
): Promise<string | null> {
  try {
    // Ensure userInputs is a string and trim it

    // Create the prompt with few-shot examples
    const prompt = `Complete this dating profile sentence with 3-5 natural words:

"${userInputs.trim()}"

Continue the sentence naturally. Write how real people talk, not fancy or literary. Keep it conversational and authentic.

Examples of good continuations:
- "We love to explore" → "new places together"
- "Looking for couples who" → "know how to party"
- "We are adventurous and" → "love meeting new people"

Rules:
- 3-5 words maximum
- Natural everyday language
- No punctuation or quotes
- Complete the thought naturally

Continue with:`;

    const res = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: "user",
        temperature: 0.3,
        top_p: 0.4,
        model: "mistral:7b",
        prompt,
        stream: false,
        max_tokens: 10, // Limit to very short response (3-5 words)
      }),
    });

    if (!res.ok) {
      console.error("Ollama API error:", res.statusText);
      return null;
    }

    const data = await res.json();

    const raw = data.response?.trim() ?? "";
    const cleaned = cleanCompletion(raw);
    return cleaned || null;
  } catch (err) {
    console.error("Ollama error:", err);
    return null;
  }
}
