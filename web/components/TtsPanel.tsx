"use client";

import { useMemo, useRef, useState } from "react";

import {
  base64ToBytes,
  decodeToMono,
  joinSegments,
  peakOf,
  wavBlobUrl,
} from "@/lib/audio";
import { JobError, runJob } from "@/lib/client";
import { MAX_CHARS, MAX_TOTAL_CHARS, splitForTts } from "@/lib/segment";
import type { Track } from "./PlayerBar";
import { ChevronDown, SpinnerIcon } from "./Icons";

const VOICES = ["nana", "deniz", "mako", "mariz", "klodin", "jan", "job", "leo"] as const;

const KNOBS = [
  {
    name: "temperature",
    label: "Temperature",
    min: 0.1,
    max: 2.0,
    step: 0.05,
    def: 1.0,
    hint: "Higher is more varied, less stable.",
  },
  {
    name: "top_p",
    label: "Top-p",
    min: 0.05,
    max: 1.0,
    step: 0.01,
    def: 0.95,
    hint: "Nucleus sampling cutoff.",
  },
  {
    name: "repetition_penalty",
    label: "Repetition penalty",
    min: 1.0,
    max: 2.0,
    step: 0.05,
    def: 1.1,
    hint: "Higher discourages loops and stuck syllables.",
  },
] as const;

type Knobs = Record<(typeof KNOBS)[number]["name"], number>;
const DEFAULTS: Knobs = { temperature: 1.0, top_p: 0.95, repetition_penalty: 1.1 };

type TtsOutput = {
  audio_base64?: string;
  sample_rate?: number;
  duration_s?: number;
  device?: string;
};

export function TtsPanel({ onTrack }: { onTrack: (t: Track) => void }) {
  const [text, setText] = useState(
    "Bonjou, koman ou ye? Mwen kontan tande ou jodi a.",
  );
  const [voice, setVoice] = useState<string>("nana");
  const [knobs, setKnobs] = useState<Knobs>(DEFAULTS);
  const [openSettings, setOpenSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [result, setResult] = useState("");
  const abortRef = useRef<AbortController | null>(null);

  const segments = useMemo(() => splitForTts(text), [text]);
  const overLimit = text.length > MAX_TOTAL_CHARS;

  async function generate() {
    setError("");
    setResult("");
    if (!text.trim()) {
      setError("Type some Creole text first.");
      return;
    }
    if (overLimit) {
      setError(`That is ${text.length} characters; the limit is ${MAX_TOTAL_CHARS}.`);
      return;
    }

    const controller = new AbortController();
    abortRef.current = controller;
    setBusy(true);
    const started = performance.now();

    try {
      const chunks: Float32Array[] = [];
      let sampleRate = 0;
      let execTotal = 0;
      let device = "";

      for (const [i, segment] of segments.entries()) {
        const label = segments.length > 1 ? `Segment ${i + 1}/${segments.length} · ` : "";
        setStatus(`${label}sending…`);

        const { output, executionTime } = await runJob<TtsOutput>(
          "tts",
          { text: segment, voice, ...knobs },
          (p) => setStatus(`${label}${p.note}`),
          controller.signal,
        );

        if (!output.audio_base64) {
          throw new JobError(`Segment ${i + 1} came back with no audio.`);
        }
        const { samples, sampleRate: sr } = await decodeToMono(
          base64ToBytes(output.audio_base64),
        );
        // Joining mismatched rates would silently pitch-shift part of the clip.
        if (sampleRate && sr !== sampleRate) {
          throw new JobError(
            `Segment ${i + 1} came back at ${sr} Hz but segment 1 was ${sampleRate} Hz; refusing to join them.`,
          );
        }
        sampleRate = sr;
        execTotal += executionTime / 1000;
        device = output.device ?? device;
        chunks.push(samples);
      }

      const audio = joinSegments(chunks, sampleRate);
      const url = wavBlobUrl(audio, sampleRate);
      const seconds = audio.length / sampleRate;
      const wall = (performance.now() - started) / 1000;

      onTrack({
        url,
        title: text.trim().slice(0, 70) + (text.trim().length > 70 ? "…" : ""),
        subtitle: `${voice} · generated just now`,
        filename: `vwa-${voice}-${Date.now()}.wav`,
      });
      setResult(
        `${segments.length > 1 ? `${segments.length} segments · ` : ""}` +
          `${seconds.toFixed(2)}s @ ${sampleRate} Hz · inference ${execTotal.toFixed(2)}s · ` +
          `total ${wall.toFixed(1)}s · peak ${peakOf(audio).toFixed(3)}` +
          (device ? ` · ${device}` : ""),
      );
      setStatus("");
    } catch (e) {
      setStatus("");
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="rounded-2xl border border-line bg-surface shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3.5">
        <h2 className="text-sm font-semibold">Text to Speech</h2>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-muted">Voice</span>
          <span className="relative">
            <select
              value={voice}
              onChange={(e) => setVoice(e.target.value)}
              disabled={busy}
              className="appearance-none rounded-lg border border-line bg-surface py-1.5 pl-3 pr-8 text-sm font-medium capitalize outline-none focus:ring-2 focus:ring-brand-300"
            >
              {VOICES.map((v) => (
                <option key={v} value={v} className="capitalize">
                  {v}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          </span>
        </label>
      </div>

      <div className="px-5 pt-4">
        <label htmlFor="tts-text" className="sr-only">
          Creole text
        </label>
        <textarea
          id="tts-text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          disabled={busy}
          rows={6}
          placeholder="Bonjou, koman ou ye?"
          className="w-full resize-y rounded-xl border border-transparent bg-transparent text-[15px] leading-relaxed outline-none placeholder:text-muted/60 focus:border-brand-200"
        />
        <div className="flex flex-wrap items-center justify-between gap-2 pb-3 text-xs text-muted">
          <span>
            {segments.length > 1
              ? `Split into ${segments.length} segments of ≤${MAX_CHARS} characters, joined into one clip.`
              : "Longer text is split at sentence boundaries and joined back together."}
          </span>
          <span className={overLimit ? "font-semibold text-red-600" : undefined}>
            {text.length.toLocaleString()} / {MAX_TOTAL_CHARS.toLocaleString()} characters
          </span>
        </div>
      </div>

      <div className="border-t border-line px-5 py-3">
        <button
          type="button"
          onClick={() => setOpenSettings((v) => !v)}
          aria-expanded={openSettings}
          className="flex w-full items-center justify-between text-sm font-medium text-muted transition hover:text-ink"
        >
          Generation settings
          <ChevronDown className={`h-4 w-4 transition ${openSettings ? "rotate-180" : ""}`} />
        </button>

        {openSettings && (
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            {KNOBS.map((k) => (
              <div key={k.name}>
                <div className="flex items-baseline justify-between">
                  <label htmlFor={k.name} className="text-xs font-medium">
                    {k.label}
                  </label>
                  <span className="text-xs tabular-nums text-muted">{knobs[k.name]}</span>
                </div>
                <input
                  id={k.name}
                  type="range"
                  min={k.min}
                  max={k.max}
                  step={k.step}
                  value={knobs[k.name]}
                  disabled={busy}
                  onChange={(e) =>
                    setKnobs((s) => ({ ...s, [k.name]: Number(e.target.value) }))
                  }
                  className="mt-2 w-full"
                />
                <p className="mt-1 text-[11px] leading-snug text-muted">{k.hint}</p>
              </div>
            ))}
            <div className="sm:col-span-3">
              <button
                type="button"
                onClick={() => setKnobs(DEFAULTS)}
                disabled={busy}
                className="text-xs font-medium text-brand-700 hover:underline disabled:opacity-50"
              >
                Reset to defaults
              </button>
            </div>
          </div>
        )}
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line px-5 py-4">
        <div className="min-h-[20px] text-xs">
          {status && (
            <span className="flex items-center gap-2 text-muted">
              <SpinnerIcon /> {status}
            </span>
          )}
          {!status && error && <span className="text-red-600">{error}</span>}
          {!status && !error && result && <span className="text-muted">{result}</span>}
        </div>
        <button
          type="button"
          onClick={generate}
          disabled={busy || overLimit}
          className="rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? "Generating…" : "Generate speech"}
        </button>
      </div>
    </div>
  );
}
