'use server'
// import { openaiClient } from '@/lib/openai'

// function extractFirstQuotedSentence(text: string): string {
//   // Intenta extraer lo que esté entre comillas dobles
//   const quoted = text.match(/"([^"]+)"/)
//   if (quoted?.[1]) return quoted[1]

//   // Si no hay comillas, devuelve la primera oración
//   const sentence = text.split(/[.?!]/)[0]
//   return sentence.trim()
//}

// export const aiAutocomplateAction = async (
//   prompt: string
// ): Promise<string | null> => {
//   try {
//     const completion = await openaiClient.chat.completions.create({
//       model: 'gpt-4.1-nano',
//       messages: [
//         {
//           role: 'system',
//           content:
//             'You are an assistant that helps improve user biographies with friendly and creative suggestions.',
//         },
//         {
//           role: 'user',
//           content: `Help me complete this description: ${prompt}`,
//         },
//       ],
//       max_tokens: 30,
//     })

//     if (!completion.choices || completion.choices.length === 0) {
//       console.log('No choices returned from OpenAI API')
//       return null
//     }

//     const raw = completion.choices?.[0]?.message?.content ?? ''
//     const cleaned = extractFirstQuotedSentence(raw)

//     return cleaned
//   } catch (error) {
//     if (error instanceof Error) {
//       console.log('Error generating completion:', error.message)
//       return null
//     }
//     console.log('An unexpected error occurred:', error)
//     return null
//   }
// }

function cleanCompletion(text: string): string {
  return text.replace(/^\.*\s*/, '') // elimina puntos suspensivos iniciales y espacios
}

export async function askOllamaCompletationAction(
  prompt: string
): Promise<string | null> {
  try {
    const res = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'llama3.2',
        prompt: `Continue this personal bio with a single short sentence, no quotes, no alternatives, no explanations:\n${prompt.trim()} `,
        stream: false,
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
