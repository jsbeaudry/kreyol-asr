"use client";

import { useEffect, useRef, useState } from "react";

import { formatTime } from "@/lib/audio";
import { DownloadIcon, PauseIcon, PlayIcon, Skip } from "./Icons";

export type Track = { url: string; title: string; subtitle: string; filename: string };

/**
 * Bottom transport for the most recent clip. Owns a single <audio> element so
 * seeking and playback state stay in one place.
 */
export function PlayerBar({ track }: { track: Track | null }) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);

  // A new clip should start from the beginning rather than inherit the last
  // one's position.
  useEffect(() => {
    setCurrent(0);
    setDuration(0);
    setPlaying(false);
  }, [track?.url]);

  if (!track) return null;

  const seek = (to: number) => {
    const el = audioRef.current;
    if (!el) return;
    el.currentTime = Math.min(Math.max(to, 0), duration || 0);
    setCurrent(el.currentTime);
  };

  const toggle = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) void el.play();
    else el.pause();
  };

  const pct = duration > 0 ? (current / duration) * 100 : 0;

  return (
    <div className="border-t border-line bg-surface px-5 py-3.5">
      <audio
        ref={audioRef}
        src={track.url}
        preload="metadata"
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration || 0)}
        onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
      />

      <div className="flex items-center gap-3">
        <span className="w-11 shrink-0 text-xs tabular-nums text-muted">
          {formatTime(current)}
        </span>
        <input
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={current}
          onChange={(e) => seek(Number(e.target.value))}
          aria-label="Seek"
          className="w-full"
          style={{
            background: `linear-gradient(to right, var(--color-brand-500) ${pct}%, var(--color-line) ${pct}%)`,
          }}
        />
        <span className="w-11 shrink-0 text-right text-xs tabular-nums text-muted">
          {formatTime(duration)}
        </span>
      </div>

      <div className="mt-2.5 flex items-center gap-4">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold">{track.title}</p>
          <p className="truncate text-xs text-muted">{track.subtitle}</p>
        </div>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => seek(current - 10)}
            aria-label="Back 10 seconds"
            className="rounded-full p-2 text-muted transition hover:bg-canvas hover:text-ink"
          >
            <Skip back />
          </button>
          <button
            type="button"
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            className="grid h-11 w-11 place-items-center rounded-full bg-brand-600 text-white shadow-sm transition hover:bg-brand-700"
          >
            {playing ? <PauseIcon /> : <PlayIcon />}
          </button>
          <button
            type="button"
            onClick={() => seek(current + 10)}
            aria-label="Forward 10 seconds"
            className="rounded-full p-2 text-muted transition hover:bg-canvas hover:text-ink"
          >
            <Skip />
          </button>
        </div>

        <a
          href={track.url}
          download={track.filename}
          className="flex items-center gap-1.5 rounded-lg px-3 py-2 text-sm font-medium text-muted transition hover:bg-canvas hover:text-ink"
        >
          <DownloadIcon />
          <span className="hidden sm:inline">Download</span>
        </a>
      </div>
    </div>
  );
}
