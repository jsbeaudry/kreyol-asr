import type { Metadata } from "next";
import { Geist } from "next/font/google";

import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: {
    default: "Vwa.ai — Haitian Creole speech AI",
    template: "%s · Vwa.ai",
  },
  description:
    "Text to speech and speech to text for Haitian Creole. Eight voices and a " +
    "streaming transcription model trained on Kreyòl rather than bolted onto a " +
    "multilingual checkpoint.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${geistSans.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col">{children}</body>
    </html>
  );
}
