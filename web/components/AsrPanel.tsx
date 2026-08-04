"use client";

import { useEffect, useRef, useState } from "react";

import { toAsrBase64 } from "@/lib/audio";
import { runJob } from "@/lib/client";
import { ChevronDown, MicIcon, SpinnerIcon, UploadIcon } from "./Icons";

// Right context -> latency, with WER from a 332-clip speaker-disjoint test set.
const LATENCIES = [
  { ms: 80, label: "80 ms — 11.9% WER" },
  { ms: 320, label: "320 ms — 10.1% WER" },
  { ms: 560, label: "560 ms — 10.3% WER" },
  { ms: 1120, label: "1120 ms — 10.1% WER" },
] as const;

type AsrOutput = {
  text?: string;
  note?: string;
  duration_s?: number;
  latency_ms?: number;
  audio_rms?: number;
  audio_peak?: number;
  gain_applied?: number;
  device?: string;
};

type Clip = { blob: Blob; url: string; name: string };

export function AsrPanel() {
  const [latency, setLatency] = useState(320);
  const [clip, setClip] = useState<Clip | null>(null);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");
  const [transcript, setTranscript] = useState("");
  const [meta, setMeta] = useState("");
  const [recording, setRecording] = useState(false);

  const recorderRef = useRef<MediaRecorder | null>(null);
  const clipUrlRef = useRef<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  // Blob URLs are not garbage collected on their own.
  useEffect(
    () => () => {
      if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
      recorderRef.current?.stream.getTracks().forEach((t) => t.stop());
    },
    [],
  );

  /**
   * Staging a clip never submits it. Transcription is a billed GPU call, so it
   * waits for an explicit Transcribe.
   */
  function stageClip(blob: Blob, name: string) {
    if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
    const url = URL.createObjectURL(blob);
    clipUrlRef.current = url;
    setClip({ blob, url, name });
    // The old transcript belongs to the old clip.
    setTranscript("");
    setMeta("");
    setError("");
  }

  async function transcribe() {
    if (!clip || busy) return;
    setError("");
    setTranscript("");
    setMeta("");
    setBusy(true);
    const started = performance.now();
    try {
      setStatus("Converting to 16 kHz WAV…");
      const { base64, seconds } = await toAsrBase64(clip.blob);

      const { output, executionTime, delayTime } = await runJob<AsrOutput>(
        "asr",
        { audio_base64: base64, latency_ms: latency },
        (p) => setStatus(p.note),
      );

      const text = (output.text ?? "").trim();
      setTranscript(text);
      const bits = [
        // The worker echoes the latency it actually used; prefer that over the
        // dropdown, which the user may have changed since submitting.
        `${output.latency_ms ?? latency} ms`,
        `audio ${(output.duration_s ?? seconds).toFixed(2)}s`,
        `inference ${(executionTime / 1000).toFixed(2)}s`,
        `queue/cold ${(delayTime / 1000).toFixed(1)}s`,
        `total ${((performance.now() - started) / 1000).toFixed(1)}s`,
      ];
      if (output.audio_rms !== undefined) {
        bits.push(
          `rms ${output.audio_rms.toFixed(4)} / peak ${(output.audio_peak ?? 0).toFixed(3)}`,
        );
      }
      if (output.gain_applied) bits.push(`gain ×${output.gain_applied.toFixed(1)}`);
      if (output.device) bits.push(output.device);
      setMeta(bits.join(" · "));

      if (!text) {
        // An empty result is a real outcome, not a crash — say which.
        setError(`Nothing transcribed — ${output.note ?? "no speech decoded"}.`);
      }
      setStatus("");
    } catch (e) {
      setStatus("");
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function toggleRecording() {
    if (recording) {
      recorderRef.current?.stop();
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      const parts: BlobPart[] = [];
      recorder.ondataavailable = (e) => e.data.size > 0 && parts.push(e.data);
      recorder.onstop = () => {
        stream.getTracks().forEach((t) => t.stop());
        setRecording(false);
        const blob = new Blob(parts, { type: recorder.mimeType || "audio/webm" });
        stageClip(blob, "Recording");
      };
      recorderRef.current = recorder;
      recorder.start();
      setRecording(true);
      setError("");
    } catch {
      setError(
        "Could not open the microphone. Check the browser's permission for this site.",
      );
    }
  }

  return (
    <div className="rounded-2xl border border-line bg-surface shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3.5">
        <h2 className="text-sm font-semibold">Speech to Text</h2>
        <label className="flex items-center gap-2 text-sm">
          <span className="text-muted">Latency</span>
          <span className="relative">
            <select
              value={latency}
              onChange={(e) => setLatency(Number(e.target.value))}
              disabled={busy}
              className="appearance-none rounded-lg border border-line bg-surface py-1.5 pl-3 pr-8 text-sm font-medium outline-none focus:ring-2 focus:ring-brand-300"
            >
              {LATENCIES.map((l) => (
                <option key={l.ms} value={l.ms}>
                  {l.label}
                </option>
              ))}
            </select>
            <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          </span>
        </label>
      </div>

      <div className="px-5 py-5">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={toggleRecording}
            disabled={busy && !recording}
            className={`flex items-center gap-2 rounded-xl px-4 py-2.5 text-sm font-semibold shadow-sm transition disabled:opacity-60 ${
              recording
                ? "bg-red-600 text-white hover:bg-red-700"
                : "border border-line bg-surface text-ink hover:bg-canvas"
            }`}
          >
            <MicIcon />
            {recording ? "Stop recording" : "Record"}
          </button>

          <button
            type="button"
            onClick={() => fileRef.current?.click()}
            disabled={busy || recording}
            className="flex items-center gap-2 rounded-xl border border-line px-4 py-2.5 text-sm font-medium transition hover:bg-canvas disabled:opacity-60"
          >
            <UploadIcon />
            Upload audio
          </button>
          <input
            ref={fileRef}
            type="file"
            accept="audio/*"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              e.target.value = "";
              if (file) stageClip(file, file.name);
            }}
          />

          {clip && (
            <button
              type="button"
              onClick={() => {
                if (clipUrlRef.current) URL.revokeObjectURL(clipUrlRef.current);
                clipUrlRef.current = null;
                setClip(null);
                setTranscript("");
                setMeta("");
                setError("");
              }}
              disabled={busy}
              className="text-sm text-muted transition hover:text-ink disabled:opacity-60"
            >
              Clear
            </button>
          )}
        </div>

        {clip ? (
          <div className="mt-4">
            <p className="mb-2 truncate text-xs text-muted">{clip.name}</p>
            <audio src={clip.url} controls className="w-full" aria-label="Your clip" />
          </div>
        ) : (
          <p className="mt-4 text-sm text-muted">
            Record or upload Haitian Creole audio, listen back, then transcribe it.
          </p>
        )}

        <button
          type="button"
          onClick={transcribe}
          disabled={!clip || busy || recording}
          className="mt-4 w-full rounded-xl bg-brand-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {busy ? "Transcribing…" : "Transcribe"}
        </button>

        <div className="mt-4 min-h-[112px] rounded-xl bg-canvas p-4">
          {transcript ? (
            <p className="whitespace-pre-wrap text-[15px] leading-relaxed">{transcript}</p>
          ) : (
            <p className="text-sm text-muted">
              The transcription appears here.
            </p>
          )}
        </div>

        <div className="mt-3 min-h-[18px] text-xs">
          {status && (
            <span className="flex items-center gap-2 text-muted">
              <SpinnerIcon /> {status}
            </span>
          )}
          {!status && error && <span className="text-red-600">{error}</span>}
          {!status && !error && meta && <span className="text-muted">{meta}</span>}
        </div>
      </div>
    </div>
  );
}
