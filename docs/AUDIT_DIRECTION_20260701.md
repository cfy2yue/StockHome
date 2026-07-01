# StockHome Audit Direction — 2026-07-01 (user-directed, URGENT + lightweight)

Author: CC, recording the user's urgent stage direction. Stock is the priority to move
fast; keep it lightweight. Do a round of audit + a concrete optimization plan.

## Audit asks
1. **Defects / vulnerabilities**: what holes or weaknesses does the current strategy
   have? (leakage risks, over-fitting, regime-fragility, tool reliability, evidence
   quality). The 2026-07-01 finding already showed P0 does NOT hold on the latest block
   (regime drift) — dig into why and where the strategy is brittle.
2. **Goal reasonableness**: is the goal set **too high**? Re-scope to something reliably
   achievable if so.
3. **Optimization space** — especially the user's preferred direction (see below).

## Preferred optimization: fixed, trained quantitative scoring tools + ML-score decisions
The user especially wants to **train and FIX the tools** — i.e. a set of **quantitative
scoring tools** (量化打分工具) — and have the agent make decisions **based on the ML
quantitative score**. Rationale:
- **More stable training**: with a fixed scoring model, you can directly check whether
  the **score predicts accurately** (a clean supervised target), instead of an opaque
  end-to-end agent judgment.
- **More fixed / stable process**: freezing the tools makes the pipeline deterministic
  and auditable; the agent's role narrows to acting on trustworthy scores.

Alternatives / complements:
- Train **small quantitative information-aggregation / decision-assist networks** to fuse
  signals and assist the decision (lightweight, not a heavy platform).
- Do **statistical analysis** on the existing data/results to surface **new insight**
  (what actually predicts forward return; which signals are stable vs regime-fragile).

## Success intent
Reach the product goal **ASAP** with a stable, auditable pipeline: fixed scoring tools
whose predictive accuracy is directly measurable, feeding a narrow, reliable agent
decision. Keep P0/P1 honest (no leakage, no all-cash-as-skill), consistent with
`AGENTS.md` / `RESPONSE_PROTOCOL.md`.

## Boundaries (unchanged)
No auto-trade/brokerage; no future-return/GT leakage at decision time; secrets never
printed/committed; Codex owns src/scripts/runs/reports/memory + RUN_STATUS, CC owns
goal/decision/handoff docs and all git.
