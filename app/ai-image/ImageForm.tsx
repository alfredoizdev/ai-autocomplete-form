'use client'

import { analyzeImage } from '@/actions/ai-vision'
import { useState } from 'react'

export default function ImageForm() {
  const [result, setResult] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  async function handleSubmit(formData: FormData) {
    setLoading(true)
    const response = await analyzeImage(formData)
    setResult(response?.response || 'Error procesando imagen.')
    setLoading(false)
  }

  return (
    <form action={handleSubmit} className='flex flex-col gap-4'>
      <input type='file' name='image' accept='image/*' required />
      <button
        type='submit'
        className='bg-blue-600 text-white px-4 py-2 rounded'
      >
        Analizar Imagen
      </button>

      {loading && <p className='text-gray-600'>Analizando...</p>}

      {result && (
        <pre className='bg-gray-100 p-4 rounded text-sm whitespace-pre-wrap'>
          {result}
        </pre>
      )}
    </form>
  )
}
