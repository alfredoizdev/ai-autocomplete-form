import Image from 'next/image'

interface Props {
  imageUrl: string
  result: { naked: boolean; kids: boolean } | string | null
}

export default function ImageResultCard({ imageUrl, result }: Props) {
  const Icon = ({ ok }: { ok: boolean }) => (
    <span
      className={`text-2xl font-bold ${ok ? 'text-green-600' : 'text-red-600'}`}
    >
      {ok ? '✔' : '✖'}
    </span>
  )

  return (
    <div className='mt-2 flex items-center justify-center gap-4 w-full max-w-[500px] py-6 px-10 bg-white border border-gray-200 rounded-[10px] mx-auto'>
      <div className='w-24 h-24 relative overflow-hidden rounded'>
        <Image src={imageUrl} alt='Thumbnail' fill className='object-cover' />
      </div>

      <div className='flex-1'>
        <div className='flex justify-between items-center mb-2'>
          <span className='font-medium text-gray-800'>Contains Children:</span>
          <Icon
            ok={
              result !== null &&
              typeof result === 'object' &&
              result.kids === true
            }
          />
        </div>
        <div className='flex justify-between items-center'>
          <span className='font-medium text-gray-800'>Contains Nudity:</span>
          <Icon
            ok={
              result !== null &&
              typeof result === 'object' &&
              result.naked === true
            }
          />
        </div>
      </div>
    </div>
  )
}
