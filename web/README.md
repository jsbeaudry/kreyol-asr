# Vwa.ai — Haitian Creole speech AI

Next.js front end over the two RunPod Serverless endpoints:

| service | endpoint id      |
| ------- | ---------------- |
| TTS     | `90fnsmvwgqfl6y` |
| ASR     | `9fds364d4gicy0` |

No accounts and no payments — the playground is open, and `/pricing` carries
indicative tiers plus a waitlist. Nothing on the site can be purchased.

## Run it

```bash
cp .env.example .env.local   # then put a real RUNPOD_API_KEY in it
npm install
npm run dev
```

`RUNPOD_API_KEY` **must be scoped to both endpoints**. A RunPod *restricted* key
only reaches the endpoints picked when it was created, so a key that works for
one service returns 403 for the other. The app names that case explicitly rather
than showing a bare status code, and distinguishes it from a 401 — a key that has
been revoked or rotated.

The key is read only on the server. Nothing in `app/api/*` or `lib/runpod.ts`
reaches the browser, and the RunPod job id handed back is opaque. **Never**
rename these to `NEXT_PUBLIC_*`; that ships the credential to every visitor.

## How a request flows

```
browser → POST /api/tts         → RunPod /run    → { jobId }
browser → GET  /api/job/tts/:id → RunPod /status → { status, output }
```

Submit-and-poll, not one blocking call. A cold GPU worker pulls >10 GB before it
can answer, which takes 1–2 minutes — longer than Cloudflare's proxy limit and
longer than a serverless function is allowed to run. Each HTTP request here stays
short and the browser owns the waiting.

## Long text

The TTS worker takes 120 characters per request. `lib/segment.ts` splits longer
text at sentence boundaries, packs consecutive short sentences back together (a
segment is a separate GPU round trip, so naive per-sentence splitting multiplies
the cost), and falls back to clause punctuation then word wrapping for a single
sentence too long on its own. The browser synthesises segments in order and joins
the audio with a 150 ms gap.

This is a port of the splitter in `../space/app.py`; keep the two in step.

## Layout

```
app/
  page.tsx                        landing
  pricing/page.tsx                tiers + waitlist
  docs/page.tsx                   API reference
  playground/
    page.tsx                      text to speech
    speech-to-text/page.tsx       transcription
  api/
    tts/route.ts                  submit a TTS segment
    asr/route.ts                  submit a clip
    job/[service]/[id]/route.ts   status passthrough
    waitlist/route.ts             signup
lib/
  runpod.ts    server-only client
  segment.ts   sentence splitting
  audio.ts     WAV encode/decode, resampling, joining
  client.ts    browser-side submit + poll
  ratelimit.ts per-IP buckets
```

## Before you launch

- **Rate limiting is in-process.** `lib/ratelimit.ts` keeps counters in memory, so
  they reset on redeploy and are not shared across instances. Behind autoscaling,
  move it to Redis/Upstash before advertising limits. Until accounts exist this is
  the only thing between a scraper and your GPU bill.
- **The waitlist writes to `.data/waitlist.jsonl`.** That works on a host with a
  writable disk and fails loudly on serverless. Point it at a database or a form
  service before launch — the route returns 503 rather than dropping a signup
  silently.
- **`/api/tts` and `/api/asr` are unauthenticated.** They exist for this front end.
  Do not hand them to customers until keys and quotas exist.
- `npm audit` reports 3 high-severity advisories from `next`'s own transitive
  `postcss` and `sharp`. They arrived with the scaffold; check for a patched
  `next` before shipping.
