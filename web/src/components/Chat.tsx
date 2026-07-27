import ReactMarkdown from "react-markdown";
import type { ChatMessage } from "../api/types";

type Props = {
  messages: ChatMessage[];
  bottomRef: React.RefObject<HTMLDivElement | null>;
};

export function Chat({ messages, bottomRef }: Props) {
  return (
    <div className="chat" role="log" aria-live="polite">
      {messages.length === 0 && (
        <div className="chat-empty">
          <p>
            Ask about FastAPI APIs, patterns, and debugging. Answers are grounded in the
            onboarding corpus — or the assistant will abstain.
          </p>
        </div>
      )}
      {messages.map((m) => (
        <article
          key={m.id}
          className={`msg ${m.role}${m.error ? " error" : ""}`}
        >
          <div className="msg-label">
            {m.role === "user" ? "You" : m.role === "system" ? "System" : "Assistant"}
          </div>
          <div className="msg-body">
            {m.role === "assistant" && !m.error ? (
              <ReactMarkdown>{m.content}</ReactMarkdown>
            ) : (
              m.content
            )}
          </div>
          {m.meta && (
            <div className="msg-stats">
              <span className="pill">{m.meta.cache_hit ? "cache HIT" : "cache MISS"}</span>
              <span className="pill">{m.meta.intent}</span>
              {m.meta.timings?.total_ms != null && (
                <span className="pill">{Math.round(m.meta.timings.total_ms)} ms</span>
              )}
              {m.meta.dual_retrieval && <span className="pill">dual retrieval</span>}
              {m.meta.decompose?.is_multi && (
                <span className="pill">
                  multi×{(m.meta.decompose.sub_queries || []).length}
                </span>
              )}
              {m.meta.partial && <span className="pill warn">partial</span>}
            </div>
          )}
        </article>
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
