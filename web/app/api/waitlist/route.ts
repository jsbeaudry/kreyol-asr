import { appendFile, mkdir } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { rateLimit, clientIp } from "@/lib/ratelimit";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const LIMIT = 5;
const WINDOW_MS = 60 * 60_000;

// Deliberately loose: the goal is to catch typos, not to adjudicate RFC 5322.
const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

const STORE = path.join(process.cwd(), ".data", "waitlist.jsonl");

export async function POST(req: Request) {
  const limit = rateLimit(`waitlist:${clientIp(req)}`, LIMIT, WINDOW_MS);
  if (!limit.ok) {
    return NextResponse.json(
      { error: "Too many signups from this address. Try again later." },
      { status: 429 },
    );
  }

  let body: Record<string, unknown>;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Body must be JSON." }, { status: 400 });
  }

  const email = typeof body.email === "string" ? body.email.trim() : "";
  if (!EMAIL.test(email)) {
    return NextResponse.json({ error: "That does not look like an email address." }, { status: 400 });
  }
  const note = typeof body.note === "string" ? body.note.trim().slice(0, 1000) : "";
  const useCase = typeof body.useCase === "string" ? body.useCase.trim().slice(0, 80) : "";

  const record = JSON.stringify({ email, useCase, note, at: new Date().toISOString() });

  try {
    await mkdir(path.dirname(STORE), { recursive: true });
    await appendFile(STORE, `${record}\n`, "utf8");
  } catch (e) {
    // A read-only or ephemeral filesystem (most serverless hosts) will land
    // here. Log it so the signup is at least recoverable from the platform log
    // rather than silently lost, and tell the caller the truth.
    console.error("[waitlist] could not persist signup:", record, e);
    return NextResponse.json(
      {
        error:
          "Could not record that signup — this deployment has no writable store. " +
          "Point the waitlist at a database or form service before launch.",
      },
      { status: 503 },
    );
  }

  return NextResponse.json({ ok: true });
}
