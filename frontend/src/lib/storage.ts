import type { Conversation } from "./types";

export const STORAGE_KEY = "bhagavatam-ai:conversations:v1";
const MAX_CONVERSATIONS = 50;

type KeyValueStore = Pick<Storage, "getItem" | "setItem">;

function defaultStore(): KeyValueStore | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null; // storage disabled (private mode, blocked site data)
  }
}

function isConversation(value: unknown): value is Conversation {
  const c = value as Conversation;
  return (
    typeof c === "object" &&
    c !== null &&
    typeof c.id === "string" &&
    typeof c.title === "string" &&
    typeof c.updatedAt === "number" &&
    Array.isArray(c.messages)
  );
}

/** Conversations stay in this browser only; nothing is sent to or stored on the server. */
export function loadConversations(store: KeyValueStore | null = defaultStore()): Conversation[] {
  try {
    const raw = store?.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter(isConversation) : [];
  } catch {
    return [];
  }
}

export function saveConversations(conversations: Conversation[], store: KeyValueStore | null = defaultStore()): void {
  try {
    const recent = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, MAX_CONVERSATIONS);
    store?.setItem(STORAGE_KEY, JSON.stringify(recent));
  } catch {
    // quota exceeded or storage unavailable: keep working in memory
  }
}

export function titleFrom(text: string): string {
  const flat = text.replace(/\s+/g, " ").trim();
  return flat.length > 60 ? `${flat.slice(0, 57)}…` : flat || "New conversation";
}

export function newId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}
