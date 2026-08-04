import Link from "next/link";

import { Logo, MicIcon, SpeakerIcon } from "@/components/Icons";

const FEATURES = [
  {
    title: "Eight Kreyòl voices",
    body:
      "nana, deniz, mako, mariz, klodin, jan, job and leo — 22 050 Hz mono, with " +
      "temperature, top-p and repetition penalty exposed on every request.",
    icon: <SpeakerIcon className="h-5 w-5" />,
  },
  {
    title: "Streaming transcription",
    body:
      "A 0.6B FastConformer RNN-T fine-tuned on 58.8 h of Haitian Creole. " +
      "10.1% WER at 320 ms, and you pick the latency that suits your product.",
    icon: <MicIcon className="h-5 w-5" />,
  },
];

const NUMBERS = [
  { value: "10.1%", label: "WER at 320 ms", note: "332-clip speaker-disjoint test set" },
  { value: "98.6%", label: "WER of the base model", note: "same clips, no Creole slot" },
  { value: "58.8 h", label: "of Kreyòl in training", note: "41 h human-transcribed, 18 h machine-transcribed archival radio" },
];

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-5">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo />
          <span className="text-lg font-bold tracking-tight">
            Vwa<span className="text-brand-600">.ai</span>
          </span>
        </Link>
        <nav className="flex items-center gap-1 text-sm">
          <Link href="/pricing" className="rounded-lg px-3 py-2 text-muted hover:text-ink">
            Pricing
          </Link>
          <Link href="/docs" className="rounded-lg px-3 py-2 text-muted hover:text-ink">
            API
          </Link>
          <Link
            href="/playground"
            className="ml-1 rounded-xl bg-brand-600 px-4 py-2 font-semibold text-white shadow-sm transition hover:bg-brand-700"
          >
            Open playground
          </Link>
        </nav>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-5">
        <section className="py-14 text-center sm:py-20">
          <p className="inline-flex items-center gap-2 rounded-full bg-brand-50 px-3.5 py-1.5 text-xs font-medium text-brand-800 ring-1 ring-brand-100">
            Built for Haitian Creole, not translated into it
          </p>
          <h1 className="mx-auto mt-6 max-w-3xl text-4xl font-bold leading-[1.1] tracking-tight sm:text-6xl">
            Speech AI that actually
            <span className="text-brand-600"> speaks Kreyòl</span>
          </h1>
          <p className="mx-auto mt-5 max-w-2xl text-lg leading-relaxed text-muted">
            Text to speech and speech to text for Haitian Creole. Both models were
            fine-tuned on Creole audio rather than bolted onto a multilingual
            checkpoint — which is why the base model scores 98.6% WER on our test
            clips and ours scores 10.1%.
          </p>
          <div className="mt-9 flex flex-wrap justify-center gap-3">
            <Link
              href="/playground"
              className="rounded-xl bg-brand-600 px-6 py-3 font-semibold text-white shadow-sm transition hover:bg-brand-700"
            >
              Try it free
            </Link>
            <Link
              href="/pricing"
              className="rounded-xl border border-line bg-surface px-6 py-3 font-semibold transition hover:bg-canvas"
            >
              See pricing
            </Link>
          </div>
        </section>

        <section className="grid gap-4 sm:grid-cols-3">
          {NUMBERS.map((n) => (
            <div key={n.label} className="rounded-2xl border border-line bg-surface p-6 shadow-sm">
              <p className="text-3xl font-bold tracking-tight text-brand-600">{n.value}</p>
              <p className="mt-1 text-sm font-medium">{n.label}</p>
              <p className="mt-0.5 text-xs text-muted">{n.note}</p>
            </div>
          ))}
        </section>

        <section className="mt-4 grid gap-4 sm:grid-cols-2">
          {FEATURES.map((f) => (
            <div key={f.title} className="rounded-2xl border border-line bg-surface p-6 shadow-sm">
              <span className="grid h-10 w-10 place-items-center rounded-xl bg-brand-50 text-brand-600">
                {f.icon}
              </span>
              <h2 className="mt-4 text-lg font-semibold">{f.title}</h2>
              <p className="mt-1.5 text-sm leading-relaxed text-muted">{f.body}</p>
            </div>
          ))}
        </section>

        <section className="my-16 rounded-3xl bg-ink px-6 py-14 text-center text-white sm:px-12">
          <h2 className="text-3xl font-bold tracking-tight">Hear it before you decide</h2>
          <p className="mx-auto mt-3 max-w-xl text-white/70">
            The playground is free and needs no account. Paste a paragraph, pick a
            voice, and judge the output yourself.
          </p>
          <Link
            href="/playground"
            className="mt-7 inline-flex rounded-xl bg-white px-6 py-3 font-semibold text-ink transition hover:bg-white/90"
          >
            Open the playground
          </Link>
        </section>
      </main>

      <footer className="border-t border-line py-8">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-5 text-sm text-muted">
          <p>© {new Date().getFullYear()} Vwa.ai</p>
          <nav className="flex gap-5">
            <Link href="/playground" className="hover:text-ink">
              Playground
            </Link>
            <Link href="/pricing" className="hover:text-ink">
              Pricing
            </Link>
            <Link href="/docs" className="hover:text-ink">
              API
            </Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
