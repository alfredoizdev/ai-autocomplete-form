'use server'

import bios from '@/data/bio.json'

function cleanCompletion(text: string): string {
  return text.replace(/^\.*\s*/, '') // elimina puntos suspensivos iniciales y espacios
}

export async function askOllamaCompletationAction(
  userInputs: string
): Promise<string | null> {
  try {
    // Ensure userInputs is a string and trim it
    const biosExamples = bios
      .slice(0, 5) // puedes ajustar cuántos usar
      .map((bio, i) => `Bio ${i + 1}:\n${bio}`)
      .join('\n\n')

    // Create the prompt with few-shot examples
    const prompt = `
You are writing short, sexy, and confident bios for a swinger dating app.

The user is writing their profile and wants help making it sound more flirty and bold.

Instructions:
- Your sentence should sound sexy, open-minded, and adventurous
- NO quotes, punctuation, or poetic style
- Only write 6 to 10 **new words**
- Do NOT repeat words from the user's input
- Use direct language suitable for adults looking to meet others
- NO explanations, only the phrase
- Keep it concise and impactful
- Don't use dounle quotes or any other punctuation

Example user bios:
${biosExamples}

User input:
${userInputs.trim()}

Your continuation:
`

    const res = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        role: 'user',
        temperature: 0.1,
        top_p: 0.2,
        model: 'mistral:7b',
        prompt,
        stream: false,
        max_tokens: 20, // Limit to a short response
      }),
    })

    if (!res.ok) {
      console.error('Ollama API error:', res.statusText)
      return null
    }

    const data = await res.json()

    const raw = data.response?.trim() ?? ''
    const cleaned = cleanCompletion(raw)
    return cleaned || null
  } catch (err) {
    console.error('Ollama error:', err)
    return null
  }
}
