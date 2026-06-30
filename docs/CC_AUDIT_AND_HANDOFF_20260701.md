# CC Audit & Active Codex Handoff — 2026-07-01

Author: CC (local Windows coordination/audit agent). Scope: audit verdict +
three-way sync record + the active goal handed to remote Codex for
`/data/cyx/1030/stock`. This is provenance; `goal.md` and `AGENTS.md` remain the
top steering authority.

## Three-Way Sync (verified 2026-07-01)

| side | ref | state |
|---|---|---|
| local `E:\cc_workspace\stock` | main `4743391` | clean |
| GitHub `cfy2yue/StockHome` | origin/main `4743391` | in sync |
| server `/data/cyx/1030/stock` | main `4743391` | clean, 0 dirty |

All three sides aligned; nothing to reconcile. No tmux session / long job was
running on the server at audit time.

## Audit Verdict

- **Goal**: reasonable, well-scoped, verifiable. P0 single-stock watch/review,
  P1 candidate ranking, P2 backtest as support. Honest de-promising of broad
  active-buy. Clear hard boundaries and stop rules. **Keep as-is.**
- **Direction**: sound. Strongest evidence = frozen P0 `single_stock_small_entry_watch_v3`
  (20-day +rate ~0.84, avg +1.9pp) and P1 `ranker_anchor_v2` (top1/top2 excess
  +3.5/+1.5pp) under leakage-safe paired diagnostics. Weakest = broad active-buy
  generalization (correctly de-promised). **Biggest bottleneck = rolling data
  refresh / regime drift** (re-pull every 1–3 months; current freeze 2026-07-01).
- **Doc hygiene gap**: `PROJECT_BRIEF.md` and `MEMORY.md` still state the old
  output boundary ("四类研究分级" / "不输出确定买入/卖出") that contradicts the
  current `AGENTS.md`/`RESPONSE_PROTOCOL.md`. Marked SUPERSEDED this round (CC).

## Top Optimization Directions

1. **(Active Codex goal)** Re-validate frozen P0/P1 on the latest as-of data
   block with leakage + coverage audit; confirm whether the strong-yellow MVP
   metrics still hold under the current (H2026) regime.
2. (CC, this round) Doc hygiene: superseded banners on the two stale docs.
3. (Future) P1 v2 Pro / rolling confirmation to harden the ranker.

## Ownership For Parallel Work

- **CC owns**: `goal.md`, `docs/DECISIONS.md`, `docs/HANDOFF.md`, this handoff
  doc, doc hygiene banners.
- **Codex owns**: `src/`, `scripts/`, `runs/`, `reports/`, `memory/` ledgers,
  `RUN_STATUS.md`. Codex may APPEND dated sections to `docs/PROJECT_REVIEW.md`.
- Do not edit the same code file from both sides simultaneously.

## Active Codex Goal (handed off 2026-07-01)

```
Project: stock
Server path: /data/cyx/1030/stock
Goal: Re-validate the frozen P0 (single_stock_small_entry_watch_v3) and P1
  (candidate_comparison_ranker_anchor_v2) on the latest available as-of data
  block, with a leakage + coverage audit. Report whether the strong-yellow MVP
  metrics (P0 20-day +rate & avg return; P1 top1/top2 excess) still hold under
  the current H2026 regime. Re-pull data only if the latest block is stale per
  DATA_SOURCE_POLICY. Start with /plan to form a measurable, bounded goal first.
Why now: the project's biggest risk is rolling data refresh / regime drift;
  validating it is the highest-value bounded next step and the gate for any
  P0/P1 promotion.
Read first: goal.md, AGENTS.md, docs/HANDOFF.md, docs/RESPONSE_PROTOCOL.md,
  docs/DATA_SOURCE_POLICY.md, memory/*.csv ledgers, docs/PROJECT_REVIEW.md
Codex owns: src/, scripts/, runs/, reports/, memory/ ledgers, RUN_STATUS
CC owns: goal.md, docs/DECISIONS.md, docs/HANDOFF.md, this handoff doc
Permissions: workspace-write (server repo only)
Forbidden: print/copy/commit any secret (ds_api.txt, tushare_token.txt, tokens);
  delete/move book PDFs, BookSkill sources, data/, reports/, memory/ caches;
  put future returns / GT / future events into decision-time evidence; present
  low-exposure/all-cash behavior as stock-picking skill; brokerage/auto-trade.
Success criteria: P0 (20-day +rate, avg return) and P1 (top1/top2 excess)
  recomputed on the fresh block with leakage+coverage audit; pass/fail vs the
  frozen baseline recorded in RUN_STATUS.md and proposed for docs/DECISIONS.md.
Stop rules: data unavailable or coverage insufficient → halt and report; API/cost
  exceeds the per-run plan → halt and report; any leakage detected → halt and fix
  before reporting metrics.
Expected output paths: runs/<run>/RUN_STATUS.md and
  reports/date_generalization/<run>/ validation summary.
Progress reporting: append dated lines to runs/<run>/RUN_STATUS.md; final summary
  via codex --output-last-message.
```

## Budget / Monitoring (CC)

CC/Claude daily spend gate < $80 (Codex spend separate). Low-frequency 3600s
polling of: tmux session alive, `git status`, `runs/<run>/RUN_STATUS.md`, codex
last message, short log tail. No secret printing, no large log copies.
