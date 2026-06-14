# LIN-137 — Tune ranking to surface already-retrieved answer chunks

**Linear:** [LIN-137](https://linear.app/lin-sk-wsp/issue/LIN-137) · follow-up to LIN-136
**Project:** Exchange Connectivity Hub
**Status:** Design approved 2026-06-14

## Problem

Two HKSE eval questions have low `context_precision` (CP) even though a complete
answer chunk *is* retrieved:

| Question | Answer chunk | Symptom | 512/64 baseline CP |
|----------|--------------|---------|--------------------|
| CAS closing auction | CAS doc pg4 (full close period table) | reranker places it **5th** | 0.20 |
| Min lot size | BoardLot pg9 ("10 … 100,000 shares") | never reaches post-rerank top-5 | 0.00 |

LIN-136 proved global chunk-size tuning is a wash: bumping to 1024 lifted CAS only
because the larger chunk *out-ranked* competitors, and it regressed precise-fact
questions (tick-size, vol-control). The residual gap is therefore a **ranking**
problem, not a chunking one.

## Goal

Lift CAS + lot-size CP versus a freshly re-measured 512/64 baseline, with **no
regression > 0.05** on the 6 currently-passing HKSE questions, by tuning the
ranking stage only. Record a per-question CP table.

## How the metric actually scores (drives the whole approach)

RAGAS `context_precision` is **rank-aware**. A relevant chunk at rank 5 with four
irrelevant chunks above it scores ~0.20 *even though it is included*. Consequences:

- **CAS** (chunk at rank 5) needs to be ranked **higher** — `rerank_top_n` cannot
  help; only reranker quality can.
- **Lot-size** (chunk below rank 5) needs to **enter** the window — a larger
  `rerank_top_n` buys partial credit, but better ranking is still the real fix.
- **Ensemble weights** are the weakest lever: the chunk is already retrieved into
  the top-20 that feeds the reranker, which re-scores everything anyway.
- Raising `rerank_top_n` is **double-edged**: extra chunks can *lower* CP on the 6
  passing questions.

Therefore the primary bet is **reranker query expansion**; weights are a control we
expect to be roughly flat and will record as such.

## Current pipeline

```
query → EnsembleRetriever → top_k=20 → Voyage rerank-2.5 → top_n=5 → LLM
          BM25 0.4 / vec 0.6   (config)      (config)        (config)
          ^^^ hardcoded constants in hybrid_retriever.py
```

`rerank_top_n` and `top_k` are already config-driven. Ensemble weights are
hardcoded module constants (`_BM25_WEIGHT` / `_VECTOR_WEIGHT`).

## Changes

### 1. Config-driven ensemble weights (mirrors LIN-136 chunk-param change)

`config.yaml` → `retrieval:` gains:
```yaml
  bm25_weight: 0.4
  vector_weight: 0.6
  rerank_query_expansion: false   # off by default
```
`create_hybrid_retriever(..., bm25_weight=None, vector_weight=None)` reads config
when args are `None`; explicit args override (same contract as `top_k`). Remove the
`_BM25_WEIGHT` / `_VECTOR_WEIGHT` constants; keep the explanatory comment.

### 2. Reranker query expansion (`reranker.py`)

`rerank_documents(..., expand_query: bool | None = None)`. Resolves from
`retrieval.rerank_query_expansion` when `None`. When enabled, the query sent to
Voyage is enriched **generically** before the rerank call.

- Mechanism: a small, domain-general HKEX/markets term map applying bidirectional
  acronym ↔ phrase expansion (e.g. `closing auction ↔ CAS`,
  `board lot ↔ lot size`, `reference price ↔ fixing`). The expanded query appends
  matched synonyms to the original text.
- **Overfitting guardrail:** the map keys on *domain terminology*, never on exact
  golden-question phrasings. It must plausibly help any HKEX query. The term map
  lives in a reviewable module-level constant.
- Rejected alternative: LLM-based query rewrite — extra cost, latency, and
  nondeterminism that would muddy an already-noisy CP measurement.

### 3. CLI overrides in `run_eval.py`

One command per config; each runs in a fresh process so the global config cache is
never stale:
```
python evals/run_eval.py --exchanges HKSE --rerank-top-n 8 \
    --bm25-weight 0.3 --vector-weight 0.7 --rerank-query-expansion
```
New flags: `--rerank-top-n`, `--top-k`, `--bm25-weight`, `--vector-weight`,
`--rerank-query-expansion / --no-rerank-query-expansion`.

Implemented via a `apply_config_overrides(**kwargs)` helper in `config.py` that
deep-merges supplied values into the cached config **before** the chain is built.
Config stays the single source of truth; the CLI merely sets it for that run.

### 4. Per-question CP table (Definition of Done)

`run_eval.py` already saves per-question scores in JSON but never surfaces them.
Add:
- a printed `question → CP` table in `print_results`, and
- a `per_question_cp` array (`[{question, context_precision}]`) in the saved JSON.

No new measurement — only readable output of existing data.

### 5. Baseline re-measurement (process step, not code)

The single on-disk results file is the stale 1024 run (CAS 0.75, tick-size 0.59),
not the 512/64 baseline. RAGAS uses an LLM judge, so CP is noisy run-to-run. The
first sweep run re-measures the committed defaults
(512/64, weights 0.4/0.6, `top_n` 5, expansion off) on `--exchanges HKSE` to
capture a clean baseline. All deltas are measured against this; a passing question
counts as **regressed only if its CP drops by more than 0.05**.

## The sweep — OFAT (one factor at a time)

Not a full grid: cost control plus clean attribution of each lever's effect.

| Run | Config | Hypothesis |
|-----|--------|------------|
| baseline | top_n 5, w 0.4/0.6, expand off | reference |
| A — expand | top_n 5, w 0.4/0.6, **expand on** | primary lever: lifts CAS + lot |
| B — top_n | **top_n 8**, w 0.4/0.6, expand off | lot-size partial credit; watch the 6 passing |
| C — weights | top_n 5, **w 0.3/0.7**, expand off | likely ~flat (records a finding) |
| D — combine | best of A/B/C together | confirm gains hold, no regression |

~5–6 runs, HKSE only. Winner = best CAS + lot-size gain with ≤0.05 drop on the 6
passing questions.

## Testing (TDD; mock all external APIs — mirrors `test_chunker` / `test_hybrid_retriever`)

- **Config weights:** `create_hybrid_retriever` uses config weights by default;
  explicit args override them.
- **Query expansion:** enabled → expanded query reaches the mocked Voyage client;
  disabled → original query passed unchanged.
- **`apply_config_overrides`:** deep-merges nested keys without dropping siblings.
- **`run_eval` CLI:** parsed args map to the correct override calls (unit-test the
  mapping; do not run a live eval).
- **Per-question CP table builder:** maps `user_input → context_precision`.

## Out of scope (YAGNI)

Multi-exchange sweep · automated grid harness · retriever-architecture changes ·
LLM-based query expansion · any per-question hand-tuning of the term map.

## Definition of Done (from LIN-137)

- [ ] CAS + lot-size CP improved vs the re-measured 512/64 baseline (0.20 / 0.00).
- [ ] Zero regression (> 0.05) on the 6 passing HKSE questions.
- [ ] Per-question CP table recorded.
