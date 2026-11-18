import { describe, expect, it } from "vitest";

import { loadConversations, saveConversations, STORAGE_KEY, titleFrom } from "./storage";
import type { Conversation } from "./types";

function memoryStore() {
  const data = new Map<string, string>();
  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => void data.set(key, value),
    data,
  };
}

const conversation = (id: string, updatedAt: number): Conversation => ({
  id,
  title: id,
  createdAt: 0,
  updatedAt,
  messages: [{ id: "m", role: "user", content: "hi" }],
});

describe("conversation storage", () => {
  it("round-trips and keeps the most recent first", () => {
    const store = memoryStore();
    saveConversations([conversation("old", 1), conversation("new", 2)], store);
    expect(loadConversations(store).map((c) => c.id)).toEqual(["new", "old"]);
  });

  it("survives corrupt or foreign data", () => {
    const store = memoryStore();
    store.setItem(STORAGE_KEY, "{not json");
    expect(loadConversations(store)).toEqual([]);
    store.setItem(STORAGE_KEY, JSON.stringify([{ id: 1 }, conversation("ok", 1)]));
    expect(loadConversations(store).map((c) => c.id)).toEqual(["ok"]);
  });

  it("does not throw when storage is unavailable", () => {
    const failing = {
      getItem: () => {
        throw new Error("denied");
      },
      setItem: () => {
        throw new Error("quota");
      },
    };
    expect(loadConversations(failing)).toEqual([]);
    expect(() => saveConversations([conversation("x", 1)], failing)).not.toThrow();
    expect(loadConversations(null)).toEqual([]);
  });

  it("derives short titles", () => {
    expect(titleFrom("  Who   narrated it?  ")).toBe("Who narrated it?");
    expect(titleFrom("x".repeat(100))).toHaveLength(58);
    expect(titleFrom("   ")).toBe("New conversation");
  });
});
