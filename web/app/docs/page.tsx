import Link from "next/link";

import { Logo } from "@/components/Icons";

export const metadata = { title: "API" };

const TTS_EXAMPLE = `curl -X POST https://your-deployment/api/tts \\
  -H 'Content-Type: application/json' \\
  -d '{
    "text": "Bonjou, koman ou ye?",
    "voice": "nana",
    "temperature": 1.0,
    "top_p": 0.95,
    "repetition_penalty": 1.1
  }'
# -> { "jobId": "..." }

curl https://your-deployment/api/job/tts/<jobId>
# -> { "status": "COMPLETED", "output": { "audio_base64": "...", "sample_rate": 22050 } }`;

const ASR_EXAMPLE = `curl -X POST https://your-deployment/api/asr \\
  -H 'Content-Type: application/json' \\
  -d '{ "audio_base64": "<16 kHz mono WAV>", "latency_ms": 320 }'
# -> { "jobId": "..." }

curl https://your-deployment/api/job/asr/<jobId>
# -> { "status": "COMPLETED", "output": { "text": "bonjou koman ou ye" } }`;

function Code({ children }: { children: string }) {
  return (
    <pre className="overflow-x-auto rounded-2xl bg-ink p-5 text-[13px] leading-relaxed text-white/90">
      <code>{children}</code>
    </pre>
  );
}

export default function DocsPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto flex w-full max-w-4xl items-center justify-between px-5 py-5">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo />
          <span className="text-lg font-bold tracking-tight">
            Vwa<span className="text-brand-600">.ai</span>
          </span>
        </Link>
        <Link
          href="/playground"
          className="rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700"
        >
          Open playground
        </Link>
      </header>

      <main className="mx-auto w-full max-w-4xl flex-1 px-5 pb-20">
        <h1 className="mt-8 text-4xl font-bold tracking-tight">API</h1>
        <p className="mt-4 max-w-2xl text-lg text-muted">
          Two routes, both asynchronous: submit a job, then poll it. Nothing here
          takes an API key yet — these are your own deployment&apos;s routes, and
          they are rate limited per IP until accounts ship.
        </p>

        <div className="mt-6 rounded-2xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900">
          <p className="font-semibold">Not a public API yet</p>
          <p className="mt-1 leading-relaxed">
            These endpoints are unauthenticated and meant for your own front end.
            Do not hand them to customers until there are keys and quotas behind
            them — anyone with the URL can spend your GPU budget.
          </p>
        </div>

        <section className="mt-10">
          <h2 className="text-2xl font-semibold tracking-tight">Why submit-and-poll</h2>
          <p className="mt-3 leading-relaxed text-muted">
            A cold GPU worker pulls more than 10 GB before it can answer, which
            takes 1–2 minutes. A single blocking request would die at the CDN&apos;s
            proxy limit and again at the serverless function timeout, so the
            browser holds the loop instead and each HTTP request stays short.
          </p>
        </section>

        <section className="mt-10">
          <h2 className="text-2xl font-semibold tracking-tight">Text to speech</h2>
          <p className="mt-3 text-muted">
            One request is one segment, capped at 120 characters — the worker&apos;s
            own limit. Longer text is split at sentence boundaries client-side and
            the audio is joined back together.
          </p>
          <div className="mt-4">
            <Code>{TTS_EXAMPLE}</Code>
          </div>
          <table className="mt-5 w-full overflow-hidden rounded-xl border border-line text-sm">
            <thead className="bg-canvas text-left">
              <tr>
                <th className="px-4 py-2.5 font-semibold">Field</th>
                <th className="px-4 py-2.5 font-semibold">Default</th>
                <th className="px-4 py-2.5 font-semibold">Range</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-line">
              <tr>
                <td className="px-4 py-2.5 font-mono text-xs">voice</td>
                <td className="px-4 py-2.5">nana</td>
                <td className="px-4 py-2.5 text-muted">
                  nana, deniz, mako, mariz, klodin, jan, job, leo
                </td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-mono text-xs">temperature</td>
                <td className="px-4 py-2.5">1.0</td>
                <td className="px-4 py-2.5 text-muted">0.1 – 2.0</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-mono text-xs">top_p</td>
                <td className="px-4 py-2.5">0.95</td>
                <td className="px-4 py-2.5 text-muted">0.05 – 1.0</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5 font-mono text-xs">repetition_penalty</td>
                <td className="px-4 py-2.5">1.1</td>
                <td className="px-4 py-2.5 text-muted">1.0 – 2.0</td>
              </tr>
            </tbody>
          </table>
        </section>

        <section className="mt-10">
          <h2 className="text-2xl font-semibold tracking-tight">Speech to text</h2>
          <p className="mt-3 text-muted">
            Send 16 kHz mono PCM16 WAV as base64. <code className="font-mono text-xs">latency_ms</code>{" "}
            is one of 80, 320, 560 or 1120 — less lookahead is faster and less
            accurate.
          </p>
          <div className="mt-4">
            <Code>{ASR_EXAMPLE}</Code>
          </div>
        </section>

        <section className="mt-10">
          <h2 className="text-2xl font-semibold tracking-tight">Haitian Creole only</h2>
          <p className="mt-3 leading-relaxed text-muted">
            Fine-tuning on Creole-only data moved the shared encoder and decoder, so
            the transcription model does not retain the base model&apos;s other
            languages — prompting it with <code className="font-mono text-xs">fr-FR</code> or{" "}
            <code className="font-mono text-xs">en-US</code> still returns Creole.
          </p>
        </section>
      </main>
    </div>
  );
}
