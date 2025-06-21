import { setCollectionForVectorDB } from '@/actions/ai-text'
import Form from '@/components/Form'

export default async function Home() {
  await setCollectionForVectorDB()

  return (
    <div className='flex flex-col items-center justify-center min-h-screen bg-white p-6'>
      <Form />
    </div>
  )
}
