'use client'

import useFormAutocomplete from '@/hooks/useFormAutocomplete'

const Form = () => {
  const {
    register,
    handleSubmit,
    errors,
    onSubmit,
    textareaRef,
    suggestion,
    isPending,
    handleKeyDown,
    setValue,
    setSuggestion,
    promptValue,
  } = useFormAutocomplete()

  console.log('sugestion', suggestion)

  return (
    <form
      onSubmit={handleSubmit(onSubmit)}
      className='flex flex-col gap-4 w-full max-w-md p-6 bg-white shadow-md rounded-none mx-auto'
    >
      <div className='mb-3'>
        <label
          htmlFor='name'
          className='block text-sm font-medium text-gray-700'
        >
          Name:
        </label>
        <input
          {...register('name', {
            required: 'Name is required',
            maxLength: {
              value: 50,
              message: 'Name cannot exceed 50 characters',
            },
          })}
          type='text'
          id='name'
          className='mt-1 text-gray-900 placeholder:text-gray-400 block w-full p-2 border border-gray-300 rounded-none shadow-sm focus:ring-gray-500 focus:border-gray-500'
          placeholder='Enter your name'
        />
        {errors.name && (
          <p className='text-red-500 text-sm mt-1'>{errors.name.message}</p>
        )}
      </div>
      <div className='relative w-full'>
        {/* Ghost text */}
        <textarea
          id='prompt'
          {...register('prompt', {
            required: 'Description is required',
            maxLength: {
              value: 200,
              message: 'Description cannot exceed 200 characters',
            },
          })}
          ref={(e) => {
            register('prompt').ref(e)
            textareaRef.current = e
          }}
          onKeyDown={handleKeyDown}
          rows={4}
          className='relative z-10 bg-transparent text-black placeholder:text-gray-400 block w-full p-2 border border-gray-300 rounded-none shadow-sm focus:ring-gray-500 focus:border-gray-500 resize-none'
          placeholder='Write a brief description about yourself...'
        />
        {isPending && (
          <div className='mt-2 bg-gray-100 border border-gray-300 text-sm text-gray-700 p-2 rounded cursor-pointer hover:bg-gray-200 transition'>
            <span className='text-gray-500 animate-pulse duration-100'>
              Loading suggestion...
            </span>
          </div>
        )}
        {suggestion && (
          <div
            className='mt-2 bg-gray-100 border border-gray-300 text-sm text-gray-700 p-2 rounded cursor-pointer hover:bg-gray-200 transition'
            onClick={() => {
              setValue('prompt', promptValue + ' ' + suggestion)
              setSuggestion('')
            }}
          >
            💡 <span className='font-medium'>Suggestion:</span> {suggestion}
          </div>
        )}
      </div>
      <button
        type='submit'
        disabled={isPending}
        className='cursor-pointer w-full bg-gray-900 text-white font-semibold py-2 px-4 rounded-none hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-opacity-50'
      >
        Submit
      </button>
    </form>
  )
}

export default Form
