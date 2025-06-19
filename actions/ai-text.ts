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
  const limitedWords = words.slice(0, 8); // Maximum 8 words

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

  // Debug logging
  // console.log("AI Response:", text);
  // console.log("Cleaned:", cleaned);
  // console.log("Limited words:", limitedWords);
  // console.log("Original words:", originalWords);
  // console.log("Filtered words:", filteredWords);

  // Return if we have at least 1 unique word (relaxed from 2)
  if (filteredWords.length >= 1) {
    return filteredWords.join(" ");
  }

  // If no unique words, return the original cleaned response (fallback)
  if (limitedWords.length >= 2) {
    return limitedWords.join(" ");
  }

  return "";
}

export async function askOllamaCompletationAction(
  userInputs: string
): Promise<string | null> {
  try {
    // Ensure userInputs is a string and trim it

    // Create the prompt with few-shot examples for swinger dating profiles
    const prompt = `You are a respectful, open-minded assistant who helps users write short, engaging bios and messages for swinger and lifestyle dating platforms. Your tone is confident, playful, and tasteful. Avoid explicit language. Emphasize honesty, mutual respect, and fun. Write in short, natural-sounding sentences. Do not judge or shame. Never sound robotic.

Complete this dating profile sentence with 2-4 words. Don't repeat words already used:

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
        temperature: 0.6,
        top_p: 0.9,
        model: "gemma3:12b",
        prompt,
        stream: false,
        max_tokens: 30, // Limit to very short response (3-5 words)
        frequency_penalty: 0.3,
        stop_tokens: ["\n", ".", "!", "?"],
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
