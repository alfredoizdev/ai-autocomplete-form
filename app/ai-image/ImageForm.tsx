'use client'

import { analyzeImage } from '@/actions/ai-vision'
import { useState } from 'react'
import ImageResultCard from './ImageResultCard'
import Image from 'next/image'

interface ImageAnalysis {
  file: File
  result: { naked: boolean; kids: boolean } | string
}

export default function ImageForm() {
  const [imageFiles, setImageFiles] = useState<File[]>([])
  const [results, setResults] = useState<ImageAnalysis[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleSubmit() {
    setLoading(true)

    const analysisResults: ImageAnalysis[] = []

    for (const file of imageFiles) {
      const formData = new FormData()
      formData.append('image', file)

      const response = await analyzeImage(formData)
      const result = response?.result || 'Error procesando imagen.'

      analysisResults.push({ file, result })
    }

    setResults(analysisResults)
    setLoading(false)
  }

  function handleChange(e: React.ChangeEvent<HTMLInputElement>) {
    if (e.target.files) {
      const files = Array.from(e.target.files)
      const maxSize = 1 * 1024 * 1024 // 1MB

      const validFiles = files.filter((file) => {
        if (file.size > maxSize) {
          setError(
            `The File ${file.name} is more them 1mb please select other image.`
          )
          return false
        }
        return true
      })

      setImageFiles(validFiles)
      setResults([])
    }
  }

  return (
    <>
      <form
        onSubmit={(e) => {
          e.preventDefault()
          handleSubmit()
        }}
        className='flex flex-col gap-4 w-full max-w-[500px] py-6 px-10 bg-white border border-gray-200 rounded-[10px] mx-auto'
      >
        {/* Logo */}
        <div className='flex justify-center mb-1'>
          <Image
            src='/images/logo-swing.svg'
            alt='Swing Logo'
            width={120}
            height={40}
            className='h-20 w-auto'
          />
        </div>

        {/* Header and description */}
        <div className='text-center mb-3'>
          <h2 className='text-2xl font-semibold text-gray-900 mb-3'>
            Let AI Analyze Your Images
          </h2>
          <p className='text-gray-500 text-sm'>
            Let AI help you determine if images contain nudity or children.
          </p>
        </div>

        <input
          type='file'
          name='image'
          accept='image/*'
          multiple
          required
          className='text-gray-900'
          onChange={handleChange}
          title='Upload images to analyze less than 1MB each'
        />
        <button
          type='submit'
          disabled={loading || imageFiles.length === 0 || error !== null}
          className='cursor-pointer w-full bg-gray-900 text-white font-semibold py-2 px-4 rounded hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50 '
        >
          Submit
        </button>

        {loading && (
          <span className='animate-pulse text-gray-700'>
            🤔 Analyzing images...
          </span>
        )}

        {error && <span className='text-red-500'> {error}</span>}

        {results.length === 0 && !loading && (
          <span className='text-gray-500'>
            No results yet. Please upload images to analyze.
          </span>
        )}

        <div className='text-center text-sm text-gray-500 mt-4'>
          Brought to you by{' '}
          <a
            href='https://swing.com'
            target='_blank'
            rel='noopener noreferrer'
            className='text-gray-700 font-semibold hover:text-yellow-500'
          >
            Swing.com
          </a>
        </div>
      </form>

      {results.map(({ file, result }, idx) => (
        <ImageResultCard
          key={idx}
          imageUrl={URL.createObjectURL(file)}
          result={result}
        />
      ))}
    </>
  )
}
