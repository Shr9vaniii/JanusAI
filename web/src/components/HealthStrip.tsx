import type { HealthResponse } from "../api/types";

type Props = {
  health: HealthResponse | null;
  loading?: boolean;
};

function pillClass(ok: boolean | undefined, waking?: boolean) {
  if (waking) return "pill warn";
  if (ok) return "pill ok";
  if (ok === false) return "pill bad";
  return "pill";
}

export function HealthStrip({ health, loading }: Props) {
  if (loading && !health) {
    return <div className="health-strip"><span className="pill">checking…</span></div>;
  }
  if (!health) {
    return <div className="health-strip"><span className="pill bad">offline</span></div>;
  }

  const genOk = health.generation?.ok;
  const waking =
    !genOk &&
    Boolean(
      health.message?.match(/wak/i) ||
        health.generation?.detail?.match(/wak|INFERENCE|not configured/i) ||
        health.status === "starting",
    );

  const label =
    health.status === "ok"
      ? "healthy"
      : waking
        ? "model waking up"
        : health.status || "degraded";

  return (
    <div className="health-strip" aria-live="polite">
      <span className={pillClass(health.status === "ok", waking)}>{label}</span>
      <span className={pillClass(health.retrieval?.ok)}>
        retrieval {health.retrieval?.ok ? "ok" : "down"}
      </span>
      <span className={pillClass(health.cache?.available)}>
        cache {health.cache?.available ? "ok" : health.cache?.status || "off"}
      </span>
      <span className={pillClass(genOk, waking)}>
        gen {genOk ? health.generation?.backend || "ready" : waking ? "waking" : "down"}
      </span>
    </div>
  );
}
