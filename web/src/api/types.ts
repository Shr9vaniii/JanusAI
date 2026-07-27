export type Citation = {
  id?: string;
  source?: string;
  name?: string;
  chunk_type?: string;
  score?: number;
  snippet?: string;
};

export type RewriteMeta = {
  needs_rewrite?: boolean;
  standalone_query?: string;
  topic_status?: string;
  confidence?: number;
  active_topic?: string;
  reason?: string;
};

export type DecomposeMeta = {
  is_multi?: boolean;
  sub_queries?: string[];
  confidence?: number;
  reason?: string;
};

export type AskResponse = {
  request_id: string;
  session_id: string;
  question: string;
  answer: string;
  citations: Citation[];
  intent: string;
  retrieval_query: string;
  rewrite: RewriteMeta | null;
  decompose: DecomposeMeta | null;
  dual_retrieval: boolean;
  cache_hit: boolean;
  partial: boolean;
  errors: string[];
  timings: Record<string, number>;
  num_chunks: number;
  model_status: string;
};

export type HealthResponse = {
  status?: string;
  ready?: boolean;
  message?: string;
  retrieval?: { ok?: boolean; error?: string };
  cache?: { available?: boolean; status?: string; enabled?: boolean };
  generation?: { ok?: boolean; backend?: string; detail?: string };
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  meta?: AskResponse;
  error?: boolean;
};
