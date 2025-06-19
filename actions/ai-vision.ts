'use server'

import path from 'path'
import { writeFile } from 'fs/promises'

export async function analyzeImage(formData: FormData) {
  const file = formData.get('image') as File
  if (!file) return { error: 'No image provided' }

  const bytes = await file.arrayBuffer()
  const buffer = Buffer.from(bytes)

  // save the file temporarily
  const tempPath = path.join('/tmp', file.name)
  await writeFile(tempPath, buffer)

  // call Ollama API to analyze the image
  const res = await fetch(`${process.env.OLLAMA_PATH_API}/generate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: 'gemma3:4b',
      prompt: `Analyze this image and respond only in JSON format. Does it contain naked people? Are there children visible in the image: { "naked": true|false, "kids": true|false }`,
      images: [buffer.toString('base64')],
    }),
  })

  const data = await res.json()

  return {
    response: data.response,
  }
}
