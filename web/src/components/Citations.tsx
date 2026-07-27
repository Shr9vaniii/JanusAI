import type { Citation } from "../api/types";

type Props = {
  citations: Citation[];
};

export function Citations({ citations }: Props) {
  if (!citations.length) {
    return (
      <aside className="side-panel">
        <h2>Citations</h2>
        <p className="muted">Citations from the latest answer appear here.</p>
      </aside>
    );
  }

  return (
    <aside className="side-panel">
      <h2>Citations</h2>
      <ol className="cite-list">
        {citations.map((c, i) => (
          <li key={`${c.id || c.source || i}-${i}`} className="cite-item">
            <div className="cite-head">
              <strong>{c.name || c.chunk_type || `Chunk ${i + 1}`}</strong>
              {c.score != null && (
                <span className="cite-score">{c.score.toFixed(2)}</span>
              )}
            </div>
            {c.source && <div className="cite-source">{c.source}</div>}
            {c.snippet && <p className="cite-snippet">{c.snippet}</p>}
          </li>
        ))}
      </ol>
    </aside>
  );
}
