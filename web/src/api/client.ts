import type { AskResponse, HealthResponse } from "./types";

async function readError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    const detail = data?.detail;
    if (typeof detail === "string") return detail;
    if (detail?.message) return detail.message;
    return JSON.stringify(data);
  } catch {
    return res.statusText || `HTTP ${res.status}`;
  }
}

export async function createSession(): Promise<string> {
  const res = await fetch("/sessions", { method: "POST" });
  if (!res.ok) throw new Error(await readError(res));
  const data = await res.json();
  return data.session_id as string;
}

export async function askQuestion(body: {
  question: string;
  session_id?: string;
  bypass_cache?: boolean;
}): Promise<AskResponse> {
  const res = await fetch("/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}

export async function fetchHealth(): Promise<HealthResponse> {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(await readError(res));
  return res.json();
}
