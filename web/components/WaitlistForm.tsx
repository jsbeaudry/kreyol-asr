"use client";

import { useState } from "react";

import { SpinnerIcon } from "./Icons";

const USE_CASES = [
  "Media & broadcasting",
  "Education",
  "Healthcare",
  "Government & NGO",
  "App developer",
  "Something else",
];

export function WaitlistForm() {
  const [email, setEmail] = useState("");
  const [useCase, setUseCase] = useState(USE_CASES[0]!);
  const [note, setNote] = useState("");
  const [state, setState] = useState<"idle" | "busy" | "done">("idle");
  const [error, setError] = useState("");

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setState("busy");
    try {
      const res = await fetch("/api/waitlist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, useCase, note }),
      });
      const json = (await res.json().catch(() => ({}))) as { error?: string };
      if (!res.ok) throw new Error(json.error ?? `Request failed (${res.status}).`);
      setState("done");
    } catch (err) {
      setState("idle");
      setError(err instanceof Error ? err.message : String(err));
    }
  }

  if (state === "done") {
    return (
      <div className="rounded-2xl border border-brand-200 bg-brand-50 p-6 text-center">
        <p className="font-semibold text-brand-900">You&apos;re on the list.</p>
        <p className="mt-1 text-sm text-brand-800/80">
          We&apos;ll email {email} when paid plans open up.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={submit} className="rounded-2xl border border-line bg-surface p-6 shadow-sm">
      <div className="grid gap-4 sm:grid-cols-2">
        <div className="sm:col-span-2">
          <label htmlFor="email" className="text-sm font-medium">
            Work email
          </label>
          <input
            id="email"
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@company.ht"
            className="mt-1.5 w-full rounded-xl border border-line px-3.5 py-2.5 text-sm outline-none focus:ring-2 focus:ring-brand-300"
          />
        </div>

        <div className="sm:col-span-2">
          <label htmlFor="usecase" className="text-sm font-medium">
            What would you use it for?
          </label>
          <select
            id="usecase"
            value={useCase}
            onChange={(e) => setUseCase(e.target.value)}
            className="mt-1.5 w-full rounded-xl border border-line bg-surface px-3.5 py-2.5 text-sm outline-none focus:ring-2 focus:ring-brand-300"
          >
            {USE_CASES.map((u) => (
              <option key={u}>{u}</option>
            ))}
          </select>
        </div>

        <div className="sm:col-span-2">
          <label htmlFor="note" className="text-sm font-medium">
            Anything else? <span className="font-normal text-muted">(optional)</span>
          </label>
          <textarea
            id="note"
            rows={3}
            value={note}
            onChange={(e) => setNote(e.target.value)}
            placeholder="Rough volume, languages, deadlines…"
            className="mt-1.5 w-full resize-y rounded-xl border border-line px-3.5 py-2.5 text-sm outline-none focus:ring-2 focus:ring-brand-300"
          />
        </div>
      </div>

      {error && <p className="mt-3 text-sm text-red-600">{error}</p>}

      <button
        type="submit"
        disabled={state === "busy"}
        className="mt-5 flex w-full items-center justify-center gap-2 rounded-xl bg-brand-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700 disabled:opacity-60"
      >
        {state === "busy" && <SpinnerIcon />}
        {state === "busy" ? "Sending…" : "Join the waitlist"}
      </button>
      <p className="mt-3 text-center text-xs text-muted">
        No card required. We only use your address to tell you when plans open.
      </p>
    </form>
  );
}
