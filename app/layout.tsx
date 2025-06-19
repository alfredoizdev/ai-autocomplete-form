import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import Navbar from "@/components/Navbar";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "AI Bio Generator and Image Analyzer | Brought to you by Swing.com",
  description: "Nextjs AI Bio Generator and Image Analyzer",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body className={`${inter.variable} antialiased font-inter`}>
        <div className="flex flex-col min-h-screen">
          {/* Navbar can be added here if needed */}
          <Navbar />
          <main className="flex-1">{children}</main>
          {/* Footer can be added here if needed */}
        </div>
      </body>
    </html>
  );
}
