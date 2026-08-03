import Link from "next/link";

import { Logo } from "@/components/Icons";
import { WaitlistForm } from "@/components/WaitlistForm";

export const metadata = { title: "Pricing" };

// Indicative only — nothing here can be purchased yet, and the page says so.
const PLANS = [
  {
    name: "Playground",
    price: "Free",
    cadence: "",
    blurb: "Open to anyone, no account needed.",
    features: [
      "Both models, full quality",
      "Up to 5 000 characters per synthesis",
      "Rate limited per IP",
      "Not for production traffic",
    ],
    cta: { label: "Open playground", href: "/playground" },
    highlight: false,
  },
  {
    name: "Starter",
    price: "$29",
    cadence: "/month",
    blurb: "For a single app or a small team.",
    features: [
      "API access with keys",
      "250 000 TTS characters / month",
      "10 hours of transcription / month",
      "Email support",
    ],
    cta: { label: "Join the waitlist", href: "#waitlist" },
    highlight: true,
  },
  {
    name: "Scale",
    price: "Custom",
    cadence: "",
    blurb: "Volume, on-prem, or a voice of your own.",
    features: [
      "Committed-use pricing",
      "Custom or cloned voices",
      "Dedicated workers, no cold starts",
      "Priority support",
    ],
    cta: { label: "Talk to us", href: "#waitlist" },
    highlight: false,
  },
];

function Check() {
  return (
    <svg viewBox="0 0 20 20" className="mt-0.5 h-4 w-4 shrink-0 text-brand-600" aria-hidden>
      <path
        d="m4.5 10.5 3.5 3.5 7.5-8"
        fill="none"
        stroke="currentColor"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

export default function PricingPage() {
  return (
    <div className="flex min-h-screen flex-col">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between px-5 py-5">
        <Link href="/" className="flex items-center gap-2.5">
          <Logo />
          <span className="text-lg font-bold tracking-tight">
            Vwa<span className="text-brand-600">.ai</span>
          </span>
        </Link>
        <Link
          href="/playground"
          className="rounded-xl bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-brand-700"
        >
          Open playground
        </Link>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-5">
        <section className="py-12 text-center sm:py-16">
          <h1 className="text-4xl font-bold tracking-tight sm:text-5xl">
            Simple pricing, once we open
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-lg text-muted">
            Paid plans are not live yet — there is nothing to buy on this page
            today. The playground is free and fully functional in the meantime, and
            the waitlist below is how you get told when billing opens.
          </p>
        </section>

        <section className="grid gap-5 lg:grid-cols-3">
          {PLANS.map((plan) => (
            <div
              key={plan.name}
              className={`flex flex-col rounded-2xl border bg-surface p-7 shadow-sm ${
                plan.highlight ? "border-brand-300 ring-2 ring-brand-200" : "border-line"
              }`}
            >
              {plan.highlight && (
                <span className="mb-3 inline-flex w-fit rounded-full bg-brand-50 px-3 py-1 text-xs font-semibold text-brand-800">
                  Planned launch tier
                </span>
              )}
              <h2 className="text-lg font-semibold">{plan.name}</h2>
              <p className="mt-3">
                <span className="text-4xl font-bold tracking-tight">{plan.price}</span>
                <span className="text-muted">{plan.cadence}</span>
              </p>
              <p className="mt-2 text-sm text-muted">{plan.blurb}</p>

              <ul className="mt-6 flex flex-1 flex-col gap-3 text-sm">
                {plan.features.map((f) => (
                  <li key={f} className="flex gap-2.5">
                    <Check />
                    <span>{f}</span>
                  </li>
                ))}
              </ul>

              <Link
                href={plan.cta.href}
                className={`mt-7 rounded-xl px-5 py-3 text-center text-sm font-semibold transition ${
                  plan.highlight
                    ? "bg-brand-600 text-white shadow-sm hover:bg-brand-700"
                    : "border border-line hover:bg-canvas"
                }`}
              >
                {plan.cta.label}
              </Link>
            </div>
          ))}
        </section>

        <section id="waitlist" className="mx-auto my-16 max-w-xl scroll-mt-8">
          <div className="mb-6 text-center">
            <h2 className="text-2xl font-bold tracking-tight">Tell us what you need</h2>
            <p className="mt-2 text-muted">
              We would rather price against real usage than guess. Tell us what you
              would run and we will come back with numbers.
            </p>
          </div>
          <WaitlistForm />
        </section>
      </main>

      <footer className="border-t border-line py-8">
        <div className="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-3 px-5 text-sm text-muted">
          <p>© {new Date().getFullYear()} Vwa.ai</p>
          <nav className="flex gap-5">
            <Link href="/playground" className="hover:text-ink">
              Playground
            </Link>
            <Link href="/docs" className="hover:text-ink">
              API
            </Link>
          </nav>
        </div>
      </footer>
    </div>
  );
}
