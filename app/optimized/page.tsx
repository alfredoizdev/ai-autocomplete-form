import FormOptimized from '@/components/FormOptimized'

export default function OptimizedPage() {
  return (
    <div className='flex flex-col items-center justify-center min-h-screen bg-white p-6'>
      <div className="mb-4 text-center">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Optimized Autocomplete Demo</h1>
        <p className="text-gray-600">Experience ultra-fast streaming AI suggestions with adaptive debouncing</p>
      </div>
      <FormOptimized />
    </div>
  )
}