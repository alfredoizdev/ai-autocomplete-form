"use server";

function cleanCompletion(text: string, originalText: string): string {
  // Remove any quotes, punctuation, and unwanted characters
  const cleaned = text
    .replace(/^["'`\.\s]*/, "") // Remove starting quotes, dots, spaces
    .replace(/["'`\.\s]*$/, "") // Remove ending quotes, dots, spaces
    .replace(/\n.*$/g, "") // Remove everything after first line break
    .replace(/[.!?;:,'""`]/g, "") // Remove all punctuation
    .trim();

  // Split into words and take only first 3-5 words
  const words = cleaned.split(/\s+/).filter((word) => word.length > 0);
  const limitedWords = words.slice(0, 8); // Maximum 5 words

  // Check for word repetition with original text
  const originalWords = originalText
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0);

  // Filter out any words that already exist in the original text
  const filteredWords = limitedWords.filter((word) => {
    const lowerWord = word.toLowerCase();
    return !originalWords.includes(lowerWord);
  });

  // Return if we have at least 2 unique words
  if (filteredWords.length >= 2) {
    return filteredWords.join(" ");
  }

  return "";
}

export async function askOllamaCompletationAction(
  userInputs: string
): Promise<string | null> {
  try {
    // Ensure userInputs is a string and trim it

    // Create the prompt with few-shot examples for swinger dating profiles
    const prompt = `Complete this dating profile sentence with 2-4 words. Don't repeat words already used:

"${userInputs.trim()}"

Examples:
"I am looking" → "for other couples"
"Young man looking" → "to have fun"
"Older couple looking" → "to meet others"
"Older couple looking for" → "new experiences"
"We love meeting" → "cool new people"
"Hot couple ready" → "to play tonight"
"Looking for someone" → "who likes fun"
"We want to" → "meet new friends"
"Seeking couples and" → "single women"

Complete naturally:`;

    const res = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        role: "user",
        temperature: 0.2,
        top_p: 0.8,
        model: "mistral:7b",
        prompt,
        stream: false,
        max_tokens: 20, // Limit to very short response (3-5 words)
        frequency_penalty: 0.3,
      }),
    });

    if (!res.ok) {
      console.error("Ollama API error:", res.statusText);
      return null;
    }

    const data = await res.json();

    const raw = data.response?.trim() ?? "";
    const cleaned = cleanCompletion(raw, userInputs.trim());
    return cleaned || null;
  } catch (err) {
    console.error("Ollama error:", err);
    return null;
  }
}
