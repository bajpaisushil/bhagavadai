"use client";

import { useEffect, useRef } from "react";

import type { ModelInfo } from "@/lib/types";

export function AboutDialog({ open, onClose, info }: { open: boolean; onClose: () => void; info: ModelInfo | null }) {
  const closeButton = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeButton.current?.focus();
    const onKey = (event: KeyboardEvent) => event.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/40 p-0 sm:items-center sm:p-4" onClick={onClose}>
      <div
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-title"
        className="max-h-[90dvh] w-full max-w-lg overflow-y-auto rounded-t-2xl bg-white p-6 shadow-xl sm:rounded-2xl dark:bg-stone-900"
        onClick={(event) => event.stopPropagation()}
      >
        <h2 id="about-title" className="font-serif text-2xl text-stone-900 dark:text-stone-50">
          About Bhagavatam AI
        </h2>
        <p className="mt-3 text-sm leading-relaxed text-stone-700 dark:text-stone-300">
          An experimental English conversational model being trained on the Śrīmad-Bhāgavatam. Knowledge is meant
          to come from training on properly licensed texts, not from searching documents at answer time.
        </p>
        {info && (
          <dl className="mt-4 grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 rounded-xl bg-stone-50 p-4 text-sm dark:bg-stone-800/60">
            <dt className="text-stone-500">Model</dt>
            <dd className="font-mono text-xs leading-5 break-all">{info.name}</dd>
            <dt className="text-stone-500">Revision</dt>
            <dd className="font-mono text-xs leading-5">{info.revision.slice(0, 10)}</dd>
            <dt className="text-stone-500">Status</dt>
            <dd>{info.status}</dd>
            <dt className="text-stone-500">Runs on</dt>
            <dd>
              {info.device.toUpperCase()} · {info.dtype} · {(info.parameters / 1e6).toFixed(0)}M parameters
            </dd>
          </dl>
        )}
        <h3 className="mt-5 font-semibold text-stone-900 dark:text-stone-100">Please keep in mind</h3>
        <ul className="mt-2 list-disc space-y-1.5 pl-5 text-sm text-stone-700 dark:text-stone-300">
          <li>It is an AI, not a sage, teacher or religious authority, and it can be confidently wrong.</li>
          <li>Verify anything important against a trusted translation of the Śrīmad-Bhāgavatam.</li>
          <li>Citations it cannot possibly mean (e.g. canto 13) are struck through automatically.</li>
          <li>
            Training data is limited to public-domain or openly licensed sources. Bhaktivedanta Book Trust (Prabhupāda)
            translations and purports are not used.
          </li>
          <li>Conversations are kept in your browser only; the server does not store them.</li>
        </ul>
        {info?.disclosure && <p className="mt-4 text-xs leading-relaxed text-stone-500">{info.disclosure}</p>}
        <div className="mt-6 flex justify-end">
          <button
            ref={closeButton}
            onClick={onClose}
            className="rounded-lg bg-amber-700 px-4 py-2 text-sm font-medium text-white hover:bg-amber-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-amber-500"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
