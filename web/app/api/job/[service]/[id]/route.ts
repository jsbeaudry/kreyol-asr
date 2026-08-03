import { NextResponse, type NextRequest } from "next/server";

import { RunpodError, jobError, status, type Service } from "@/lib/runpod";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const SERVICES = new Set<Service>(["tts", "asr"]);

/**
 * Status passthrough. The browser polls this instead of holding one long
 * request open — a cold worker can take ~2 minutes, which outlives both
 * Cloudflare's proxy limit and a serverless function's own timeout.
 */
export async function GET(
  _req: NextRequest,
  ctx: RouteContext<"/api/job/[service]/[id]">,
) {
  const { service, id } = await ctx.params;
  if (!SERVICES.has(service as Service)) {
    return NextResponse.json({ error: `Unknown service "${service}".` }, { status: 404 });
  }

  try {
    const job = await status(service as Service, id);
    const error = jobError(job);
    if (error) {
      // A rejected request is the worker doing its job, not an outage: report
      // it as a completed-with-error rather than a transport failure.
      return NextResponse.json({ status: job.status, error }, { status: 200 });
    }
    return NextResponse.json({
      status: job.status,
      output: job.output ?? null,
      executionTime: job.executionTime ?? 0,
      delayTime: job.delayTime ?? 0,
    });
  } catch (e) {
    const err = e as RunpodError;
    return NextResponse.json({ error: err.message }, { status: err.status ?? 502 });
  }
}
