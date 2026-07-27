import type { AskResponse } from "../api/types";

type Props = {
  open: boolean;
  latest: AskResponse | null;
  onClose: () => void;
};

export function TraceDrawer({ open, latest, onClose }: Props) {
  if (!open) return null;

  return (
    <div className="trace-drawer" role="dialog" aria-label="Pipeline trace">
      <div className="trace-head">
        <h2>Pipeline trace</h2>
        <button type="button" className="ghost-btn" onClick={onClose}>
          Close
        </button>
      </div>
      {!latest ? (
        <p className="muted">Send a question to see rewrite, decompose, and timings.</p>
      ) : (
        <div className="trace-body">
          <section>
            <h3>Retrieval query</h3>
            <pre>{latest.retrieval_query || "—"}</pre>
          </section>
          <section>
            <h3>Rewrite</h3>
            <pre>{JSON.stringify(latest.rewrite, null, 2)}</pre>
          </section>
          <section>
            <h3>Decompose</h3>
            <pre>{JSON.stringify(latest.decompose, null, 2)}</pre>
          </section>
          <section>
            <h3>Flags</h3>
            <pre>
              {JSON.stringify(
                {
                  intent: latest.intent,
                  dual_retrieval: latest.dual_retrieval,
                  cache_hit: latest.cache_hit,
                  partial: latest.partial,
                  num_chunks: latest.num_chunks,
                  model_status: latest.model_status,
                  errors: latest.errors,
                },
                null,
                2,
              )}
            </pre>
          </section>
          <section>
            <h3>Timings (ms)</h3>
            <pre>{JSON.stringify(latest.timings, null, 2)}</pre>
          </section>
        </div>
      )}
    </div>
  );
}
