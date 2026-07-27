import { useCallback, useEffect, useRef, useState } from "react";
import { askQuestion, createSession, fetchHealth } from "./api/client";
import type { AskResponse, ChatMessage, HealthResponse } from "./api/types";
import { Chat } from "./components/Chat";
import { Citations } from "./components/Citations";
import { HealthStrip } from "./components/HealthStrip";
import { ScenarioChips } from "./components/ScenarioChips";
import { TraceDrawer } from "./components/TraceDrawer";
import { STARTER_TOPICS, type Scenario } from "./data/scenarios";
import "./App.css";

const SESSION_KEY = "janus_ai_session_id";
const DEMO_KEY = "janus_ai_demo_mode";

function uid() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function App() {
  const [sessionId, setSessionId] = useState(
    () => localStorage.getItem(SESSION_KEY) || "",
  );
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [bypassCache, setBypassCache] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [latest, setLatest] = useState<AskResponse | null>(null);
  const [demoMode, setDemoMode] = useState(
    () => localStorage.getItem(DEMO_KEY) === "1",
  );
  const [traceOpen, setTraceOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const refreshHealth = useCallback(async () => {
    try {
      const data = await fetchHealth();
      setHealth(data);
    } catch {
      setHealth(null);
    } finally {
      setHealthLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshHealth();
    const id = window.setInterval(refreshHealth, 30000);
    return () => window.clearInterval(id);
  }, [refreshHealth]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  useEffect(() => {
    localStorage.setItem(DEMO_KEY, demoMode ? "1" : "0");
  }, [demoMode]);

  async function ensureSession(forceNew = false): Promise<string> {
    if (!forceNew && sessionId) return sessionId;
    const id = await createSession();
    setSessionId(id);
    localStorage.setItem(SESSION_KEY, id);
    return id;
  }

  async function handleNewSession() {
    setMessages([]);
    setLatest(null);
    setQuestion("");
    try {
      const id = await ensureSession(true);
      setMessages([
        {
          id: uid(),
          role: "system",
          content: `New session ${id.slice(0, 8)}… Ask a FastAPI onboarding question.`,
        },
      ]);
    } catch (e) {
      setMessages([
        {
          id: uid(),
          role: "assistant",
          content: String(e),
          error: true,
        },
      ]);
    }
  }

  async function sendQuestion(text: string, opts?: { newSession?: boolean }) {
    const q = text.trim();
    if (!q || busy) return;

    setBusy(true);
    setQuestion("");
    const thinkingId = uid();

    try {
      const sid = await ensureSession(Boolean(opts?.newSession));
      if (opts?.newSession) {
        setMessages([
          {
            id: uid(),
            role: "system",
            content: "Fresh session for this scenario.",
          },
          { id: uid(), role: "user", content: q },
          { id: thinkingId, role: "assistant", content: "Thinking…" },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { id: uid(), role: "user", content: q },
          { id: thinkingId, role: "assistant", content: "Thinking…" },
        ]);
      }

      const data = await askQuestion({
        question: q,
        session_id: sid,
        bypass_cache: bypassCache,
      });
      if (data.session_id) {
        setSessionId(data.session_id);
        localStorage.setItem(SESSION_KEY, data.session_id);
      }
      setLatest(data);
      setMessages((prev) =>
        prev.map((m) =>
          m.id === thinkingId
            ? { ...m, content: data.answer, meta: data, error: false }
            : m,
        ),
      );
    } catch (e) {
      setMessages((prev) => {
        const hasThinking = prev.some((m) => m.id === thinkingId);
        if (hasThinking) {
          return prev.map((m) =>
            m.id === thinkingId
              ? { ...m, content: String(e), error: true, meta: undefined }
              : m,
          );
        }
        return [
          ...prev,
          { id: uid(), role: "user", content: q },
          { id: uid(), role: "assistant", content: String(e), error: true },
        ];
      });
    } finally {
      setBusy(false);
      refreshHealth();
    }
  }

  function onScenario(scenario: Scenario) {
    void sendQuestion(scenario.question, { newSession: scenario.newSession });
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <p className="brand">JanusAI</p>
          <h1>Your guide to technical beginnings</h1>
          <p className="tagline">
            Ask FastAPI questions. Get cited answers — built for developers starting out.
          </p>
        </div>
        <div className="topbar-actions">
          <HealthStrip health={health} loading={healthLoading} />
          <label className="toggle">
            <input
              type="checkbox"
              checked={demoMode}
              onChange={(e) => setDemoMode(e.target.checked)}
            />
            Demo mode
          </label>
          {demoMode && (
            <button type="button" className="ghost-btn" onClick={() => setTraceOpen(true)}>
              Trace
            </button>
          )}
          <button type="button" className="ghost-btn" onClick={() => void handleNewSession()}>
            New session
          </button>
        </div>
      </header>

      <ScenarioChips disabled={busy} onSelect={onScenario} />

      <div className="workspace">
        <main className="main-col">
          <Chat messages={messages} bottomRef={bottomRef} />
          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault();
              void sendQuestion(question);
            }}
          >
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask about Depends, UploadFile, HTTPException…"
              rows={2}
              disabled={busy}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void sendQuestion(question);
                }
              }}
            />
            <div className="composer-side">
              <label className="toggle compact">
                <input
                  type="checkbox"
                  checked={bypassCache}
                  onChange={(e) => setBypassCache(e.target.checked)}
                />
                bypass cache
              </label>
              <button type="submit" disabled={busy || !question.trim()}>
                {busy ? "…" : "Ask"}
              </button>
            </div>
          </form>
          <div className="starters">
            {STARTER_TOPICS.map((t) => (
              <button
                key={t}
                type="button"
                className="starter"
                disabled={busy}
                onClick={() => void sendQuestion(t)}
              >
                {t}
              </button>
            ))}
          </div>
        </main>
        <Citations citations={latest?.citations || []} />
      </div>

      <footer className="footer">
        <span>
          Session {sessionId ? sessionId.slice(0, 8) : "—"} · same API powers CLI / MCP later
        </span>
        
      </footer>

      <TraceDrawer open={traceOpen} latest={latest} onClose={() => setTraceOpen(false)} />
    </div>
  );
}
