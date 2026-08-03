"use client";

import { useState } from "react";
import Link from "next/link";

import { PlayerBar, type Track } from "@/components/PlayerBar";
import { TtsPanel } from "@/components/TtsPanel";

export default function TextToSpeechPage() {
  const [track, setTrack] = useState<Track | null>(null);

  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4">
        <h1 className="text-lg font-semibold">Text to Speech</h1>
        <Link
          href="/docs"
          className="rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700"
        >
          Try with API
        </Link>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto px-5 py-8">
        <div className="mx-auto max-w-2xl">
          <div className="text-center">
            <h2 className="text-3xl font-bold tracking-tight">Bonjou 👋</h2>
            <p className="mx-auto mt-2 max-w-md text-muted">
              Eight Haitian Creole voices. Type a sentence or paste a paragraph —
              long text is split and joined back into one clip.
            </p>
          </div>

          <div className="mt-8">
            <TtsPanel onTrack={setTrack} />
          </div>

          <p className="mt-4 text-center text-xs text-muted">
            The first request after an idle period takes ~1–2 minutes while a GPU
            worker starts. After that it is a couple of seconds per segment.
          </p>
        </div>
      </div>

      <PlayerBar track={track} />
    </>
  );
}
