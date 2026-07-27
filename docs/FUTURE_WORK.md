# Future work (intentionally deferred)

These were consciously kept **out** of the interview sprint to protect a working, evaluated demo. They are documented here as design sketches and interview talking points, not as pending tasks.

## 1. Related-page expansion (retrieval)

**Idea:** After the initial hybrid retrieval + rerank, follow graph-like relationships between chunks (e.g. class → its methods/attributes, doc page → sibling sections, example → the API it uses) to pull in neighboring context the lexical/dense query missed.

**Why deferred:**
- Touches the core retriever, which is currently green on the eval set. Highest regression risk this close to interviews.
- Changes the candidate set, so existing Recall@k / MRR numbers would no longer be comparable without new labeled cases.
- Flagged in the plan as exactly the kind of retrieval R&D to freeze during the sprint.

**How I'd build it later:**
- Add explicit edges in chunk metadata (`parent`, `qualified_name`, `source` page) — much of this already exists.
- After rerank, expand the top-N by 1 hop along `parent`/sibling edges, cap the added chunks, and re-rank the union.
- Gate behind a flag and add eval cases (e.g. "attributes of X" should surface attribute chunks via the class hit) before trusting the metrics.

**Risk:** Context bloat, latency increase, and diluted precision if expansion is unbounded.

## 2. Corpus sync with GitHub (freshness)

**Idea:** Keep `chroma_db_v3` + BM25 index in sync as the source repo/docs change.

**Why deferred:**
- Pure ingestion/ops concern; not needed for the interview demo.
- A *frequent/automatic* scheduler adds real moving parts: change detection, re-chunking, re-embedding, BM25 rebuild, and cache invalidation.

**How I'd build it later (incremental, low-risk first):**
1. **Manual/on-demand refresh script** — `python -m ingestion.refresh --since <git-ref>`:
   - Diff changed files, re-chunk only those, upsert into Chroma, rebuild BM25 (`get_bm25(..., force_rebuild=True)`).
   - Bump `CACHE_VERSION` (or rely on chunk-ID fingerprint) so stale answers invalidate automatically — the versioned cache keys already support this.
2. **Scheduled sync** — wrap the script in a cron/GitHub Action that runs on push to the docs repo, then redeploys/mounts the refreshed artifacts.
3. **Zero-downtime** — build into a new Chroma path and atomically swap.

**Why the cache design already helps:** answer keys hash `CACHE_VERSION | model_version | prompt_version | normalized_query | chunk_ids`, so changing corpus chunks (new IDs) or bumping a version invalidates affected answers without a full flush.

**Risk:** Embedding cost on large refreshes, and partial-rebuild bugs if change detection misses a file — the safe fallback is a full rebuild.

## 3. CLI + IDE / MCP clients (same API contract)

**Idea:** Thin clients that call the same `POST /sessions` + `POST /ask` (+ optional retrieve-only) used by the presentation UI.

**Why deferred:** Interview MVP ships the React UI + public URL; CLI/`onboard ask` and Cursor/VS Code MCP are the natural next surfaces once the contract is stable.

**How I'd build it later:**
- CLI: Typer/Click wrapper around the HTTP API (not a second orchestrator).
- MCP: tools `ask` and `retrieve_only` with the same JSON shapes as `AskResponse`.

**Interview line:** *“UI is one client; CLI and MCP are the same contract.”*

---

Both are good "what would you do next / how would you scale this" answers in an interview, which is why they're captured rather than rushed.
