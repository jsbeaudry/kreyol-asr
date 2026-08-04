import Link from "next/link";

import { AsrPanel } from "@/components/AsrPanel";

export const metadata = { title: "Speech to Text" };

export default function SpeechToTextPage() {
  return (
    <>
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4">
        <h1 className="text-lg font-semibold">Speech to Text</h1>
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
            <h2 className="text-3xl font-bold tracking-tight">Transcribe Kreyòl</h2>
            <p className="mx-auto mt-2 max-w-md text-muted">
              A streaming FastConformer RNN-T fine-tuned on 58.8 h of Haitian
              Creole. <strong className="font-semibold text-ink">10.1% WER</strong>{" "}
              at 320 ms on a speaker-disjoint test set.
            </p>
          </div>

          <div className="mt-8">
            <AsrPanel />
          </div>

          <p className="mt-4 text-center text-xs text-muted">
            Haitian Creole only. Fine-tuning on Creole-only data moved the shared
            encoder and decoder, so this model does not retain the base model&apos;s
            other languages.
          </p>
        </div>
      </div>
    </>
  );
}
