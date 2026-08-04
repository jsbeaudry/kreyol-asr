import { NextResponse } from "next/server";

import { rateLimit, clientIp } from "@/lib/ratelimit";
import { RunpodError, submit } from "@/lib/runpod";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const LIMIT = 20;
const WINDOW_MS = 60_000;

// Right context -> latency, with WER from a 332-clip speaker-disjoint test set.
const LATENCIES = new Set([80, 320, 560, 1120]);

// Base64 inflates by ~4/3, and RunPod caps a payload at 20 MB. Refuse early
// rather than letting the platform reject it with something less legible.
const MAX_BASE64_CHARS = 12_000_000;

export async function POST(req: Request) {
  const limit = rateLimit(`asr:${clientIp(req)}`, LIMIT, WINDOW_MS);
  if (!limit.ok) {
    return NextResponse.json(
      { error: `Rate limit reached. Try again in ${limit.retryAfter}s.` },
      { status: 429, headers: { "Retry-After": String(limit.retryAfter) } },
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }

  const audio = typeof body.audio_base64 === "string" ? body.audio_base64 : "";
  if (!audio) {
    return NextResponse.json({ error: "Provide `audio_base64`." }, { status: 400 });
  }
  if (audio.length > MAX_BASE64_CHARS) {
    return NextResponse.json(
      { error: "That clip is too large. Keep uploads under about 8 MB of audio." },
      { status: 413 },
    );
  }

  const latency = Number(body.latency_ms ?? 320);
  if (!LATENCIES.has(latency)) {
    return NextResponse.json(
      { error: `latency_ms must be one of ${[...LATENCIES].join(", ")}.` },
      { status: 400 },
    );
  }

  try {
    const jobId = await submit("asr", { audio_base64: audio, latency_ms: latency });
    return NextResponse.json({ jobId });
  } catch (e) {
    const err = e as RunpodError;
    return NextResponse.json({ error: err.message }, { status: err.status ?? 502 });
  }
}
