"use server";

function cleanCompletion(text: string, originalText: string): string {
  // Remove any quotes, punctuation, and unwanted characters more aggressively
  const cleaned = text
    .replace(/^["'`\.\s*→←↑↓▲▼►◄]*/, "") // Remove starting quotes, dots, spaces, arrows, stars
    .replace(/["'`\.\s*→←↑↓▲▼►◄]*$/, "") // Remove ending quotes, dots, spaces, arrows, stars
    .replace(/\n.*$/g, "") // Remove everything after first line break
    .replace(/[.!?;:,'""`*→←↑↓▲▼►◄]/g, "") // Remove all punctuation and special characters
    .replace(/\s+/g, " ") // Normalize multiple spaces to single space
    .trim();

  // Split into words and filter out any non-alphabetic words
  const words = cleaned
    .split(/\s+/)
    .filter((word) => word.length > 0)
    .filter((word) => /^[a-zA-Z]+$/.test(word)) // Only allow pure alphabetic words
    .slice(0, 6); // Maximum 6 words

  // Check for word repetition with original text
  const originalWords = originalText
    .toLowerCase()
    .split(/\s+/)
    .filter((w) => w.length > 0)
    .map((w) => w.replace(/[^a-zA-Z]/g, "")); // Clean original words too

  // Filter out any words that already exist in the original text
  const filteredWords = words.filter((word) => {
    const lowerWord = word.toLowerCase();
    return !originalWords.includes(lowerWord);
  });

  // Debug logging
  // console.log("AI Response:", text);
  // console.log("Cleaned:", cleaned);
  // console.log("Words:", words);
  // console.log("Original words:", originalWords);
  // console.log("Filtered words:", filteredWords);

  // Return if we have at least 1 unique word
  if (filteredWords.length >= 1) {
    return filteredWords.join(" ");
  }

  // If no unique words, return some cleaned words as fallback
  if (words.length >= 1) {
    return words.slice(0, 3).join(" ");
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

IMPORTANT: Only respond with plain words. Do not use any special characters, arrows (→), stars (*), punctuation, or symbols. Just provide 2-4 simple words to complete the sentence.

Complete this dating profile sentence with 2-4 words. Don't repeat words already used:

"${userInputs.trim()}"

Examples:
"I am looking" → for other couples
"Young man looking" → to have fun
"Older couple looking" → to meet others
"Older couple looking for" → new experiences
"We love meeting" → cool new people
"Hot couple ready" → to play tonight
"Looking for someone" → who likes fun
"We want to" → meet new friends
"Seeking couples and" → single women

Complete naturally with plain words only:`;

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
