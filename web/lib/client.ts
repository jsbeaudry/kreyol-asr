/**
 * Browser side of the job protocol: POST to our route, then poll it.
 *
 * The server holds the RunPod key and returns only a job id, so nothing here
 * ever sees a credential.
 */

export const POLL_MS = 1500;

/**
 * A warm worker answers in seconds and a recently-used image starts in about a
 * minute, but the first request after a new image is deployed has to pull >10 GB
 * onto a fresh host — measured past 200 s and still queued. 240 s was not
 * enough, and timing out on someone's first ever request is the worst possible
 * introduction to the product.
 */
export const MAX_WAIT_MS = 420_000;

export type Progress = { state: string; waitedMs: number; note: string };

export class JobError extends Error {}

function describe(state: string, waitedMs: number): string {
  const seconds = Math.round(waitedMs / 1000);
  const label = state.toLowerCase().replace(/_/g, " ");
  if (waitedMs > 15_000) return `${label} — ${seconds}s (cold start takes ~2 min)`;
  return `${label} — ${seconds}s`;
}

async function readError(res: Response): Promise<string> {
  try {
    const json = await res.json();
    if (typeof json?.error === "string") return json.error;
    return `Request failed with ${res.status}.`;
  } catch {
    return `Request failed with ${res.status}.`;
  }
}

export async function runJob<T = Record<string, unknown>>(
  service: "tts" | "asr",
  body: unknown,
  onProgress?: (p: Progress) => void,
  signal?: AbortSignal,
): Promise<{ output: T; executionTime: number; delayTime: number }> {
  const res = await fetch(`/api/${service}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new JobError(await readError(res));

  const { jobId } = (await res.json()) as { jobId: string };
  const started = Date.now();

  for (;;) {
    if (signal?.aborted) throw new JobError("Cancelled.");
    if (Date.now() - started > MAX_WAIT_MS) {
      throw new JobError(`Timed out after ${Math.round(MAX_WAIT_MS / 1000)}s.`);
    }
    await new Promise((r) => setTimeout(r, POLL_MS));

    const statusRes = await fetch(`/api/job/${service}/${encodeURIComponent(jobId)}`, { signal });
    if (!statusRes.ok) throw new JobError(await readError(statusRes));
    const job = (await statusRes.json()) as {
      status: string;
      output?: T | null;
      error?: string;
      executionTime?: number;
      delayTime?: number;
    };

    if (job.error) throw new JobError(job.error);
    if (job.status === "COMPLETED") {
      if (!job.output) throw new JobError("The worker returned nothing.");
      return {
        output: job.output,
        executionTime: job.executionTime ?? 0,
        delayTime: job.delayTime ?? 0,
      };
    }
    const waitedMs = Date.now() - started;
    onProgress?.({ state: job.status, waitedMs, note: describe(job.status, waitedMs) });
  }
}
