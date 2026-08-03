import { NextResponse } from "next/server";

import { rateLimit, clientIp } from "@/lib/ratelimit";
import { RunpodError, submit } from "@/lib/runpod";
import { MAX_CHARS } from "@/lib/segment";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// One segment per request. The browser splits long text and submits the pieces,
// so a burst here is a long passage rather than abuse — hence a fairly generous
// window paired with the per-segment character cap.
const LIMIT = 60;
const WINDOW_MS = 60_000;

const VOICES = new Set([
  "nana",
  "deniz",
  "mako",
  "mariz",
  "klodin",
  "jan",
  "job",
  "leo",
]);

// name -> [low, high]; mirrors GEN_PARAMS in serverless-tts/handler.py. The
// worker clamps too, but rejecting here saves a round trip.
const RANGES: Record<string, [number, number]> = {
  temperature: [0.1, 2.0],
  top_p: [0.05, 1.0],
  repetition_penalty: [1.0, 2.0],
};

export async function POST(req: Request) {
  const limit = rateLimit(`tts:${clientIp(req)}`, LIMIT, WINDOW_MS);
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

  const text = typeof body.text === "string" ? body.text.trim() : "";
  if (!text) return NextResponse.json({ error: "Provide `text`." }, { status: 400 });
  if (text.length > MAX_CHARS) {
    return NextResponse.json(
      { error: `Segment is ${text.length} characters; the limit is ${MAX_CHARS}.` },
      { status: 400 },
    );
  }

  const voice = typeof body.voice === "string" ? body.voice.toLowerCase() : "nana";
  if (!VOICES.has(voice)) {
    return NextResponse.json(
      { error: `Unknown voice "${voice}". Choose one of ${[...VOICES].join(", ")}.` },
      { status: 400 },
    );
  }

  const input: Record<string, unknown> = { text, voice };
  for (const [name, [low, high]] of Object.entries(RANGES)) {
    const raw = body[name];
    if (raw === undefined || raw === null) continue;
    const value = Number(raw);
    if (!Number.isFinite(value)) {
      return NextResponse.json(
        { error: `${name} must be a number, got ${JSON.stringify(raw)}.` },
        { status: 400 },
      );
    }
    input[name] = Math.min(Math.max(value, low), high);
  }

  try {
    const jobId = await submit("tts", input);
    return NextResponse.json({ jobId }, { headers: { "X-RateLimit-Remaining": String(limit.remaining) } });
  } catch (e) {
    const err = e as RunpodError;
    return NextResponse.json({ error: err.message }, { status: err.status ?? 502 });
  }
}
