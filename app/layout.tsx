import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'
import Navbar from '@/components/Navbar'

const geistSans = Geist({
  variable: '--font-geist-sans',
  subsets: ['latin'],
})

const geistMono = Geist_Mono({
  variable: '--font-geist-mono',
  subsets: ['latin'],
})

export const metadata: Metadata = {
  title: 'AI Bio Generator | Brought to you by Swing.com',
  description: 'Nextjs AI Bio Generator',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang='en'>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className='flex flex-col min-h-screen'>
          {/* Navbar can be added here if needed */}
          <Navbar />
          <main className='flex-1'>{children}</main>
          {/* Footer can be added here if needed */}
        </div>
      </body>
    </html>
  )
}
