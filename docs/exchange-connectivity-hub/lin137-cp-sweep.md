# LIN-137 — Reranker / Ranking Tuning: context_precision Sweep

**Date:** 2026-06-14
**Scope:** HKSE golden set (10 answerable + 5 unanswerable). Metric: RAGAS `context_precision` (rank-aware).
**Baseline config:** 512/64 chunking, EnsembleRetriever BM25 0.4 / vector 0.6, top_k 20, rerank-2.5 top_n 5, query expansion off.
**Method:** One-factor-at-a-time (OFAT) sweep via `run_eval.py` CLI overrides. Each run is one fresh process; results copied to `evals/results/lin137_<run>.json` (gitignored data artifacts).

> **Baseline provenance.** The LIN-137 issue cites a 512/64 baseline of CAS=0.20 / lot-size=0.00. This sweep **re-measured** that baseline on a clean run (the on-disk result the issue referenced was a stale 1024-chunk run). The re-measured 512/64 CAS is **0.70**, not 0.20 — the corpus is confirmed 512/64 (Chroma chunk tokens: max 477, median 331; a 1024-chunk DB would show max ≈1000). The 0.20→0.70 gap reflects RAGAS's LLM-judge variance on a borderline-ranked chunk plus the stale reference; the plan anticipated this and made re-measurement the first step. All deltas below are vs the honest re-measured 512/64 baseline.

## Per-question context_precision

★ = Definition-of-Done targets · ▸ = regression-guard set (CP ≥ 0.75 at baseline)

| Question | Baseline | A: expansion | B: top_n 8 | C: wt 0.3/0.7 |
|----------|:---:|:---:|:---:|:---:|
| ★ minimum lot size | 0.00 | 0.00 | 0.00 | 0.00 |
| order types | 0.00 | 0.00 | 0.00 | 0.00 |
| short selling | 0.45 | 0.50 | 0.45 | 0.20 |
| ★ closing auction (CAS) | 0.70 | **0.80** ↑ | 0.27 ↓↓ | 0.20 ↓↓ |
| ▸ tick size | 0.81 | **1.00** ↑ | 0.81 | 0.81 |
| ▸ market makers | 0.81 | 0.81 | 0.70 ↓ | 0.81 |
| ▸ trading hours | 0.87 | **1.00** ↑ | 0.87 | 0.87 |
| ▸ settlement | 1.00 | 1.00 | 1.00 | 1.00 |
| ▸ volatility control | 1.00 | 1.00 | 0.94 ↓ | 1.00 |
| ▸ VCM threshold | 1.00 | 1.00 | 1.00 | 1.00 |
| **Overall context_precision** | **0.663** | **0.711** ✅ | 0.603 ❌ | 0.588 ❌ |

Runs:
- **A — query expansion** (`--rerank-query-expansion`): generic HKEX synonym enrichment of the rerank query.
- **B — larger rerank window** (`--rerank-top-n 8`): keep 8 chunks after rerank instead of 5.
- **C — weight shift** (`--bm25-weight 0.3 --vector-weight 0.7`): more vector-dominant ensemble.

## Outcome

**Winner: Run A (query expansion).** Adopted as the committed default (`config.yaml: rerank_query_expansion: true`).

- CAS: 0.70 → **0.80** (+0.10) — the ranking lever works: expansion surfaces the answer chunk.
- Overall CP: 0.663 → **0.711** (+0.048).
- **Zero regression** on the 6 passing questions — three of them improved (tick size, trading hours, plus short selling outside the guard set).
- B and C both crater CAS (extra/over-weighted-vector chunks dilute the rerank) and regress passing questions — rejected.
- "Run D = combine winners" collapses to A alone (B and C produced no gains), so no additional paid run was made.

## Definition of Done — status

| DoD item | Status |
|----------|--------|
| CAS context_precision improved vs baseline | ✅ 0.70 → 0.80 |
| Lot-size context_precision improved vs baseline | ❌ 0.00 → 0.00 — **not a ranking problem** (see below) |
| Zero regression on the 6 passing questions | ✅ all held or improved |
| Per-question CP table recorded | ✅ this document |

## Why lot-size could not be moved by ranking

Inspecting Run A's retrieved contexts for the lot-size question:

```
Retrieved (all board-lot themed — expansion DID surface them to the top):
  • "CHAPTER 1 … A board lot defines the number of shares per trading unit …"
  • "CHAPTER 2: THE PROPOSAL … HKEX has considered …"
  • "… eight specific board lot units were identified as being optimal …"

Ground truth: "Board lot sizes vary by stock (10–100,000 shares), calibrated to
               HK$1,000–2,000 lot value; there is no single fixed minimum."
```

Query expansion did its job — board-lot chunks were ranked to the top. But every board-lot chunk in the corpus comes from an HKEX **consultation/proposal paper about reforming lot structure**, not a rulebook stating the *current* minimum lot size. The factual answer is not present in any retrievable chunk, so RAGAS correctly scores CP = 0.00.

**No reranker change can surface a fact the corpus does not contain.** Lot-size (and the similarly-stuck "order types", also 0.00) is a **retrieval-recall / corpus-coverage gap**, outside LIN-137's ranking remit. Tracked as a follow-up issue (ingestion / corpus coverage).
