/**
 * Server-only RunPod client.
 *
 * The API key never reaches the browser: pages call our own routes, which
 * submit the job and hand back an opaque RunPod job id. The browser then polls
 * /api/job for status.
 *
 * Jobs are submitted with /run rather than /runsync deliberately. A cold worker
 * pulls >10 GB before it can answer, and a synchronous call dies at
 * Cloudflare's ~100 s proxy limit long before that finishes — and a serverless
 * function host would time out well before even that.
 */
import "server-only";

export const TTS_ENDPOINT = process.env.RUNPOD_TTS_ENDPOINT ?? "90fnsmvwgqfl6y";
export const ASR_ENDPOINT = process.env.RUNPOD_ASR_ENDPOINT ?? "9fds364d4gicy0";

export type Service = "tts" | "asr";

export const ENDPOINTS: Record<Service, string> = {
  tts: TTS_ENDPOINT,
  asr: ASR_ENDPOINT,
};

export class RunpodError extends Error {
  readonly status: number;
  constructor(message: string, status = 502) {
    super(message);
    this.status = status;
  }
}

function apiKey(): string {
  const key = process.env.RUNPOD_API_KEY;
  if (!key) {
    throw new RunpodError(
      "RUNPOD_API_KEY is not set on the server. Copy .env.example to .env.local and add it.",
      500,
    );
  }
  return key;
}

function describe403(service: Service, endpoint: string): string {
  // A RunPod *restricted* key is scoped to the endpoints chosen when it was
  // created, so a key that works for one service can be forbidden for the
  // other. Saying so beats a bare 403.
  return (
    `RunPod refused the ${service.toUpperCase()} endpoint (${endpoint}) with 403. ` +
    `The API key is most likely a restricted key scoped to other endpoints — add ` +
    `this one to its scope in the RunPod console, or use an unrestricted key.`
  );
}

async function call(path: string, service: Service, init?: RequestInit) {
  const endpoint = ENDPOINTS[service];
  const res = await fetch(`https://api.runpod.ai/v2/${endpoint}/${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${apiKey()}`,
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (res.status === 401) {
    // Distinct from 403: the key is not merely out of scope, it is not a valid
    // key at all — typically revoked or rotated since it was pasted in.
    throw new RunpodError(
      "RunPod rejected the API key (401). It has most likely been revoked or " +
        "rotated — put the current key in RUNPOD_API_KEY and restart the server.",
      401,
    );
  }
  if (res.status === 403) throw new RunpodError(describe403(service, endpoint), 403);
  if (!res.ok) {
    const body = await res.text();
    throw new RunpodError(
      `RunPod ${service} returned ${res.status}: ${body.slice(0, 300)}`,
      502,
    );
  }
  return res.json();
}

/** Submit a job. Returns the RunPod job id for the browser to poll. */
export async function submit(service: Service, input: unknown): Promise<string> {
  const json = await call("run", service, {
    method: "POST",
    body: JSON.stringify({ input }),
  });
  const id = json?.id;
  if (typeof id !== "string") {
    throw new RunpodError(`No job id in RunPod response: ${JSON.stringify(json).slice(0, 200)}`);
  }
  return id;
}

export type JobStatus = {
  status: string;
  output?: Record<string, unknown> | null;
  error?: string | null;
  executionTime?: number;
  delayTime?: number;
};

export async function status(service: Service, jobId: string): Promise<JobStatus> {
  const json = await call(`status/${encodeURIComponent(jobId)}`, service);
  return json as JobStatus;
}

/**
 * The worker reports a rejected request by returning {"error": ...}, which
 * RunPod surfaces as a FAILED job rather than a COMPLETED one with an error
 * field. Both shapes mean the same thing to a caller.
 */
export function jobError(job: JobStatus): string | null {
  if (job.error) return String(job.error);
  const out = job.output as Record<string, unknown> | undefined | null;
  if (out && typeof out.error === "string") return out.error;
  if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(job.status)) {
    return `${job.status}: ${JSON.stringify(job.output ?? {}).slice(0, 300)}`;
  }
  return null;
}
