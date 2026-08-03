import Link from "next/link";

import { Sidebar } from "@/components/Sidebar";

export default function PlaygroundLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex min-w-0 flex-1 flex-col p-3 lg:py-4 lg:pl-0 lg:pr-4">
        <div className="flex min-h-0 flex-1 flex-col overflow-hidden rounded-2xl bg-surface shadow-sm ring-1 ring-line">
          {children}
        </div>
        <p className="px-2 pt-3 text-center text-xs text-muted lg:hidden">
          <Link href="/" className="hover:underline">
            Vwa.ai
          </Link>{" "}
          ·{" "}
          <Link href="/pricing" className="hover:underline">
            Pricing
          </Link>
        </p>
      </main>
    </div>
  );
}
