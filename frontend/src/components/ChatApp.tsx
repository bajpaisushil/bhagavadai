"use client";

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from "react";

import { AboutDialog } from "@/components/AboutDialog";
import { MessageView } from "@/components/MessageView";
import { ApiError, getHealth, getInfo, streamChat } from "@/lib/api";
import { loadConversations, newId, saveConversations, titleFrom } from "@/lib/storage";
import type { ChatMessage, Conversation, Health, ModelInfo, StoredMessage } from "@/lib/types";

const MAX_CHARS = 4000;
const CONTEXT_MESSAGES = 12; // recent turns sent as context; keeps prompts short on slow hardware
const SUGGESTIONS = [
  "Who narrated the Śrīmad-Bhāgavatam to King Parīkṣit?",
  "Why does a person suffer even when they try to do good?",
  "What does the Bhāgavatam teach about devotion (bhakti)?",
  "Tell me the story of Dhruva.",
];

type Connection = "connecting" | "offline" | Health["status"];

function StatusBadge({ connection, info }: { connection: Connection; info: ModelInfo | null }) {
  const [label, style] =
    connection === "ready"
      ? info?.fine_tuned
        ? ["Fine-tuned model", "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-200"]
        : ["Base model · untrained", "bg-amber-100 text-amber-900 dark:bg-amber-950 dark:text-amber-200"]
      : connection === "loading" || connection === "connecting"
        ? ["Loading model…", "animate-pulse bg-stone-200 text-stone-700 dark:bg-stone-800 dark:text-stone-300"]
        : ["Model unavailable", "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-200"];
  return <span className={`rounded-full px-2.5 py-1 text-xs font-medium whitespace-nowrap ${style}`}>{label}</span>;
}

const noSubscription = () => () => {};

/** False during server rendering and hydration, true afterwards (browser-only state is safe to read). */
function useHydrated(): boolean {
  return useSyncExternalStore(
    noSubscription,
    () => true,
    () => false,
  );
}

export function ChatApp() {
  // Conversations live in localStorage, which does not exist on the server: render the session only
  // after hydration so its state can be initialised from storage without a hydration mismatch.
  return useHydrated() ? (
    <ChatSession />
  ) : (
    <div className="flex h-dvh items-center justify-center bg-stone-50 text-stone-500 dark:bg-stone-950">Loading…</div>
  );
}

function ChatSession() {
  const [conversations, setConversations] = useState<Conversation[]>(loadConversations);
  const [activeId, setActiveId] = useState<string | null>(() => conversations[0]?.id ?? null);
  const [input, setInput] = useState("");
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [connection, setConnection] = useState<Connection>("connecting");
  const [healthDetail, setHealthDetail] = useState<string | null>(null);
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const controller = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  // Persist (in this browser only) whenever a reply is complete.
  useEffect(() => {
    if (streamingId === null) saveConversations(conversations);
  }, [conversations, streamingId]);

  // Poll backend health until the model is ready (it loads in the background at server start).
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const health = await getHealth();
        if (cancelled) return;
        setConnection(health.status);
        setHealthDetail(health.detail);
        if (health.status === "ready") {
          setInfo(await getInfo());
          return;
        }
      } catch {
        if (!cancelled) setConnection("offline");
      }
      if (!cancelled) timer = setTimeout(poll, 3000);
    };
    void poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);

  const active = useMemo(() => conversations.find((c) => c.id === activeId) ?? null, [conversations, activeId]);

  const messageCount = active?.messages.length ?? 0;
  const lastContent = active?.messages.at(-1)?.content;
  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messageCount, lastContent]);

  const updateMessage = useCallback((conversationId: string, messageId: string, change: Partial<StoredMessage>) => {
    setConversations((all) =>
      all.map((c) =>
        c.id !== conversationId
          ? c
          : {
              ...c,
              updatedAt: Date.now(),
              messages: c.messages.map((m) => (m.id === messageId ? { ...m, ...change } : m)),
            },
      ),
    );
  }, []);

  const send = useCallback(
    async (text: string) => {
      const content = text.trim();
      if (!content || streamingId || connection !== "ready") return;
      setNotice(null);
      const now = Date.now();
      const userMessage: StoredMessage = { id: newId(), role: "user", content };
      const reply: StoredMessage = { id: newId(), role: "assistant", content: "" };
      const conversation: Conversation = active ?? {
        id: newId(),
        title: titleFrom(content),
        createdAt: now,
        updatedAt: now,
        messages: [],
      };
      const history: ChatMessage[] = [...conversation.messages, userMessage]
        .filter((m) => !m.error && m.content)
        .slice(-CONTEXT_MESSAGES)
        .map(({ role, content: c }) => ({ role, content: c }));
      setConversations((all) => [
        { ...conversation, updatedAt: now, messages: [...conversation.messages, userMessage, reply] },
        ...all.filter((c) => c.id !== conversation.id),
      ]);
      setActiveId(conversation.id);
      setInput("");
      setStreamingId(reply.id);
      const abort = new AbortController();
      controller.current = abort;
      let streamed = "";
      try {
        await streamChat(
          history,
          {
            onDelta: (delta) => {
              streamed += delta;
              updateMessage(conversation.id, reply.id, { content: streamed });
            },
            onDone: ({ references, usage, finishReason }) =>
              updateMessage(conversation.id, reply.id, { content: streamed.trim(), references, usage, finishReason }),
          },
          abort.signal,
        );
      } catch (error) {
        if (abort.signal.aborted) {
          updateMessage(conversation.id, reply.id, { content: streamed.trim(), finishReason: "cancelled" });
        } else {
          const message = error instanceof ApiError ? error.message : "Could not reach the model server.";
          updateMessage(conversation.id, reply.id, { error: message });
          if (error instanceof ApiError && error.status === 503) setNotice(message);
        }
      } finally {
        controller.current = null;
        setStreamingId(null);
      }
    },
    [active, connection, streamingId, updateMessage],
  );

  const newChat = () => {
    if (streamingId) return;
    setActiveId(null);
    setSidebarOpen(false);
  };

  const removeChat = (id: string) => {
    if (streamingId) return;
    setConversations((all) => all.filter((c) => c.id !== id));
    if (activeId === id) setActiveId(null);
  };

  const ready = connection === "ready";
  const canSend = ready && !streamingId && input.trim().length > 0 && input.length <= MAX_CHARS;

  return (
    <div className="flex h-dvh flex-col bg-stone-50 text-stone-900 dark:bg-stone-950 dark:text-stone-100">
      <header className="flex items-center gap-3 border-b border-stone-200 bg-white/80 px-4 py-3 backdrop-blur dark:border-stone-800 dark:bg-stone-900/80">
        <button
          className="rounded-md p-1.5 text-stone-600 hover:bg-stone-100 md:hidden dark:text-stone-300 dark:hover:bg-stone-800"
          aria-label="Toggle conversations"
          onClick={() => setSidebarOpen((o) => !o)}
        >
          ☰
        </button>
        <div className="flex min-w-0 flex-1 items-baseline gap-2">
          <h1 className="font-serif text-xl font-semibold tracking-tight">Bhagavatam AI</h1>
          <span className="hidden truncate text-sm text-stone-500 sm:inline">Conversations on the Śrīmad-Bhāgavatam</span>
        </div>
        <StatusBadge connection={connection} info={info} />
        <button
          onClick={() => setAboutOpen(true)}
          className="rounded-lg border border-stone-300 px-3 py-1.5 text-sm hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"
        >
          About
        </button>
      </header>

      <div className="relative flex min-h-0 flex-1">
        <aside
          className={`${sidebarOpen ? "flex" : "hidden"} absolute inset-y-0 left-0 z-20 w-72 flex-col border-r border-stone-200 bg-white p-3 shadow-lg md:static md:flex md:shadow-none dark:border-stone-800 dark:bg-stone-900`}
        >
          <button
            onClick={newChat}
            disabled={!!streamingId}
            className="mb-3 rounded-lg bg-amber-700 px-3 py-2 text-sm font-medium text-white hover:bg-amber-800 disabled:opacity-50"
          >
            + New conversation
          </button>
          <nav className="min-h-0 flex-1 space-y-1 overflow-y-auto" aria-label="Conversations">
            {conversations.length === 0 && <p className="px-2 text-sm text-stone-500">No conversations yet.</p>}
            {conversations.map((c) => (
              <div
                key={c.id}
                className={`group flex items-center rounded-lg ${c.id === activeId ? "bg-amber-50 dark:bg-amber-950/40" : "hover:bg-stone-100 dark:hover:bg-stone-800"}`}
              >
                <button
                  className="min-w-0 flex-1 truncate px-2.5 py-2 text-left text-sm"
                  onClick={() => {
                    setActiveId(c.id);
                    setSidebarOpen(false);
                  }}
                  disabled={!!streamingId}
                >
                  {c.title}
                </button>
                <button
                  aria-label={`Delete conversation ${c.title}`}
                  className="px-2 text-stone-400 opacity-60 hover:text-red-600 group-hover:opacity-100"
                  onClick={() => removeChat(c.id)}
                  disabled={!!streamingId}
                >
                  ×
                </button>
              </div>
            ))}
          </nav>
          <p className="mt-3 border-t border-stone-200 pt-3 text-xs text-stone-500 dark:border-stone-800">
            Saved in this browser only.
          </p>
        </aside>

        <main className="flex min-w-0 flex-1 flex-col">
          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-6" aria-live="polite">
            <div className="mx-auto flex max-w-3xl flex-col gap-4">
              {!active || active.messages.length === 0 ? (
                <div className="mt-8 text-center sm:mt-16">
                  <p className="font-serif text-3xl text-amber-800 dark:text-amber-400">॥ श्रीमद्भागवतम् ॥</p>
                  <h2 className="mt-4 font-serif text-2xl">Ask about the Śrīmad-Bhāgavatam</h2>
                  <p className="mx-auto mt-2 max-w-md text-sm text-stone-600 dark:text-stone-400">
                    Its narratives, characters and teachings. This preview runs the <strong>untrained base model</strong>
                    , so expect fluent but unreliable answers until training is complete.
                  </p>
                  <div className="mx-auto mt-6 grid max-w-xl gap-2 sm:grid-cols-2">
                    {SUGGESTIONS.map((s) => (
                      <button
                        key={s}
                        onClick={() => void send(s)}
                        disabled={!ready || !!streamingId}
                        className="rounded-xl border border-stone-200 bg-white px-4 py-3 text-left text-sm hover:border-amber-400 hover:bg-amber-50 disabled:opacity-50 dark:border-stone-800 dark:bg-stone-900 dark:hover:bg-stone-800"
                      >
                        {s}
                      </button>
                    ))}
                  </div>
                </div>
              ) : (
                active.messages.map((m) => <MessageView key={m.id} message={m} streaming={m.id === streamingId} />)
              )}
              <div ref={bottom} />
            </div>
          </div>

          <div className="border-t border-stone-200 bg-white px-4 pt-3 pb-[max(0.75rem,env(safe-area-inset-bottom))] dark:border-stone-800 dark:bg-stone-900">
            <div className="mx-auto max-w-3xl">
              {connection === "offline" && (
                <p className="mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-800 dark:bg-red-950/50 dark:text-red-200">
                  Cannot reach the model server. Start it with <code className="font-mono">make api</code> (or{" "}
                  <code className="font-mono">make demo</code>); retrying automatically…
                </p>
              )}
              {connection === "error" && (
                <p className="mb-2 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-800 dark:bg-red-950/50 dark:text-red-200">
                  The model failed to load: {healthDetail}
                </p>
              )}
              {notice && <p className="mb-2 text-sm text-amber-800 dark:text-amber-300">{notice}</p>}
              <form
                className="flex items-end gap-2"
                onSubmit={(event) => {
                  event.preventDefault();
                  void send(input);
                }}
              >
                <label htmlFor="question" className="sr-only">
                  Your question
                </label>
                <textarea
                  id="question"
                  rows={1}
                  value={input}
                  maxLength={MAX_CHARS + 100}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                      event.preventDefault();
                      void send(input);
                    }
                  }}
                  placeholder={ready ? "Ask a question… (Enter to send, Shift+Enter for a new line)" : "Waiting for the model…"}
                  className="max-h-48 min-h-11 flex-1 resize-y rounded-xl border border-stone-300 bg-stone-50 px-3.5 py-2.5 text-base focus:border-amber-500 focus:ring-2 focus:ring-amber-500/30 focus:outline-none dark:border-stone-700 dark:bg-stone-950"
                />
                {streamingId ? (
                  <button
                    type="button"
                    onClick={() => controller.current?.abort()}
                    className="h-11 rounded-xl border border-stone-300 px-4 text-sm font-medium hover:bg-stone-100 dark:border-stone-700 dark:hover:bg-stone-800"
                  >
                    Stop
                  </button>
                ) : (
                  <button
                    type="submit"
                    disabled={!canSend}
                    className="h-11 rounded-xl bg-amber-700 px-4 text-sm font-medium text-white hover:bg-amber-800 disabled:cursor-not-allowed disabled:opacity-50"
                  >
                    Send
                  </button>
                )}
              </form>
              <div className="mt-1.5 flex justify-between text-[11px] text-stone-500">
                <span>AI can be wrong. Verify important points with a trusted translation.</span>
                {input.length > MAX_CHARS * 0.8 && (
                  <span className={input.length > MAX_CHARS ? "text-red-600" : ""}>
                    {input.length}/{MAX_CHARS}
                  </span>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
      <AboutDialog open={aboutOpen} onClose={() => setAboutOpen(false)} info={info} />
    </div>
  );
}
