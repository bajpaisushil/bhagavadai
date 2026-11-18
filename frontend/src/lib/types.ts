export type Role = "user" | "assistant";

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface Reference {
  text: string;
  ref: string | null;
  valid: boolean;
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  seconds: number;
  tokens_per_s: number | null;
}

export interface StoredMessage extends ChatMessage {
  id: string;
  references?: Reference[];
  usage?: Usage;
  finishReason?: string;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  messages: StoredMessage[];
}

export interface ModelInfo {
  name: string;
  revision: string;
  adapter: string | null;
  fine_tuned: boolean;
  device: string;
  dtype: string;
  parameters: number;
  status: string;
  disclosure: string;
}

export interface Health {
  status: "loading" | "ready" | "error";
  busy: boolean;
  detail: string | null;
}

export type StreamEvent =
  | { type: "start" }
  | { type: "delta"; text: string }
  | { type: "done"; references: Reference[]; usage: Usage; finish_reason: string }
  | { type: "error"; message: string };
