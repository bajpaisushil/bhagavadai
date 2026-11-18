"use client";

import { splitReferences } from "@/lib/references";
import type { StoredMessage } from "@/lib/types";

function ReferenceChip({ text, valid }: { text: string; valid: boolean | undefined }) {
  const style =
    valid === false
      ? "border-red-300 bg-red-50 text-red-800 line-through decoration-red-500/60 dark:border-red-900 dark:bg-red-950/60 dark:text-red-200"
      : "border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/50 dark:text-amber-100";
  const title = valid === false ? "Not a valid verse reference (no such canto/chapter)" : "Śrīmad-Bhāgavatam reference";
  return (
    <span title={title} className={`mx-0.5 inline-flex items-center rounded-md border px-1.5 py-px font-mono text-[0.85em] ${style}`}>
      {text}
    </span>
  );
}

export function MessageView({ message, streaming }: { message: StoredMessage; streaming: boolean }) {
  const isUser = message.role === "user";
  const validity = new Map((message.references ?? []).map((r) => [r.text, r.valid]));
  const invalid = (message.references ?? []).filter((r) => !r.valid);

  return (
    <div className={`flex w-full ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={
          isUser
            ? "max-w-[85%] rounded-2xl rounded-br-md bg-amber-700 px-4 py-2.5 text-white shadow-sm"
            : "max-w-[92%] rounded-2xl rounded-bl-md border border-stone-200 bg-white px-4 py-3 shadow-sm dark:border-stone-800 dark:bg-stone-900"
        }
      >
        {!isUser && (
          <div className="mb-1 text-xs font-semibold uppercase tracking-wide text-amber-800 dark:text-amber-400">
            Bhagavatam AI
          </div>
        )}
        <div className="whitespace-pre-wrap break-words leading-relaxed">
          {splitReferences(message.content).map((segment, index) =>
            segment.kind === "ref" ? (
              <ReferenceChip key={index} text={segment.text} valid={validity.get(segment.text)} />
            ) : (
              <span key={index}>{segment.text}</span>
            ),
          )}
          {streaming && <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse rounded-sm bg-amber-600 align-middle" />}
          {!streaming && !isUser && !message.content && !message.error && <span className="text-stone-400">…</span>}
        </div>
        {invalid.length > 0 && (
          <p className="mt-2 text-xs text-red-700 dark:text-red-300">
            {invalid.length === 1 ? "One citation does" : `${invalid.length} citations do`} not exist in the
            Bhāgavatam&apos;s 12 cantos / 335 chapters.
          </p>
        )}
        {message.finishReason === "length" && (
          <p className="mt-2 text-xs text-stone-500">Answer cut off at the length limit.</p>
        )}
        {message.finishReason === "cancelled" && <p className="mt-2 text-xs text-stone-500">Stopped.</p>}
        {message.error && <p className="mt-2 text-sm text-red-700 dark:text-red-300">⚠ {message.error}</p>}
        {message.usage && message.usage.tokens_per_s !== null && (
          <p className="mt-2 text-[11px] text-stone-400">
            {message.usage.completion_tokens} tokens · {message.usage.tokens_per_s} tok/s
          </p>
        )}
      </div>
    </div>
  );
}
