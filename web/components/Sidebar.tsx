"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { DocIcon, HomeIcon, Logo, MicIcon, SpeakerIcon, TagIcon } from "./Icons";

type Item = { href: string; label: string; icon: React.ReactNode; soon?: boolean };

const PLAYGROUND: Item[] = [
  { href: "/playground", label: "Text to Speech", icon: <SpeakerIcon /> },
  { href: "/playground/speech-to-text", label: "Speech to Text", icon: <MicIcon /> },
];

const PRODUCT: Item[] = [
  { href: "/pricing", label: "Pricing", icon: <TagIcon /> },
  { href: "/docs", label: "API docs", icon: <DocIcon /> },
];

function Row({ item, active }: { item: Item; active: boolean }) {
  return (
    <Link
      href={item.href}
      aria-current={active ? "page" : undefined}
      className={`flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm transition ${
        active
          ? "bg-surface font-semibold text-ink shadow-sm ring-1 ring-line"
          : "text-muted hover:bg-surface/70 hover:text-ink"
      }`}
    >
      <span className={active ? "text-brand-600" : "text-muted"}>{item.icon}</span>
      {item.label}
    </Link>
  );
}

export function Sidebar() {
  const pathname = usePathname();
  const isActive = (href: string) =>
    href === "/playground" ? pathname === href : pathname.startsWith(href);

  return (
    <aside className="hidden w-64 shrink-0 flex-col gap-6 px-4 py-6 lg:flex">
      <Link href="/" className="flex items-center gap-2.5 px-2">
        <Logo />
        <span className="text-lg font-bold tracking-tight">Vwa<span className="text-brand-600">.ai</span></span>
      </Link>

      <nav className="flex flex-col gap-1" aria-label="Main">
        <Row item={{ href: "/", label: "Home", icon: <HomeIcon /> }} active={pathname === "/"} />

        <p className="mt-5 px-3 pb-1 text-xs font-medium uppercase tracking-wide text-muted">
          Playground
        </p>
        {PLAYGROUND.map((item) => (
          <Row key={item.href} item={item} active={isActive(item.href)} />
        ))}

        <p className="mt-5 px-3 pb-1 text-xs font-medium uppercase tracking-wide text-muted">
          Product
        </p>
        {PRODUCT.map((item) => (
          <Row key={item.href} item={item} active={isActive(item.href)} />
        ))}
      </nav>

      <div className="mt-auto rounded-2xl bg-brand-50 p-4 ring-1 ring-brand-100">
        <p className="text-sm font-semibold text-brand-900">Kreyòl, done properly</p>
        <p className="mt-1 text-xs leading-relaxed text-brand-800/80">
          Models trained on Haitian Creole rather than bolted onto a multilingual
          checkpoint.
        </p>
        <Link
          href="/pricing"
          className="mt-3 inline-flex text-xs font-semibold text-brand-700 hover:underline"
        >
          See pricing →
        </Link>
      </div>
    </aside>
  );
}
