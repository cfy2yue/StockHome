# StockHome Current Goal

Last slimmed: 2026-07-01.

This is the actionable steering note for `/data/cyx/1030/stock` and GitHub
repository `https://github.com/cfy2yue/StockHome`.

The full pre-slim chronological goal is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/goal.md
docs/local_archive/20260630_pre_slim/goal_20260626_pre_audit.md
```

Those archives are intentionally ignored by Git.

## Objective

Build a research-assistant workflow for A-share stock analysis. The current
product focus is:

- P0: single-stock watch/review with explicit research action suggestions;
- P1: comparison and ranking for a small user-provided candidate set;
- P2: portfolio/backtest research as a supporting tool, not the delivery
  headline.

The system may output research-oriented suggestions such as buy, trial buy,
add, hold, reduce, sell, wait, or collect more data. It must include evidence,
counter-evidence, position/risk limits, triggers, and review conditions.

## Hard Boundaries

- No brokerage integration, no automatic trading, no order placement.
- No promises of profit, certainty, or risk-free outcomes.
- Do not present low exposure or cash-heavy behavior as stock-picking skill.
- Do not read, print, copy, or commit API keys/tokens.
- Decision-time inputs must not include future returns, future events, or
  undisclosed financial information.
- Reports and prompts must not contain secrets.

## Workspace Contract

- Project root: `/data/cyx/1030/stock`.
- Remote: `https://github.com/cfy2yue/StockHome`.
- Server login: `ssh cyx-server-proxy-cfy`.
- This project does not share `/data/cyx/1030/dataset` by default.
- Local secrets such as `ds_api.txt` and `tushare_token.txt` stay ignored.

## Current Decision State

- Continue as a strong-yellow MVP for P0/P1 research assistance.
- Do not claim broad active-buy generalization or stable 20-day positive-return
  targets.
- P0 single-stock watch and P1 ranker-anchor comparison are the current user
  workflow priorities.
- P2 backtests remain useful for audit and evidence, but not as the only success
  criterion.

## Next Useful Actions

1. Keep GitHub clean: source, configs, README/AGENTS, and current docs only.
2. Before any token-consuming or data-heavy run, write the hypothesis, input
   boundary, command, cost/resource limit, expected outputs, and stop rule.
3. CC on Windows can review code/docs, refine goals, inspect GitHub, and prepare
   prompts. Server-only tasks, secret-backed API runs, and larger data jobs
   should be coordinated with Codex/SSH.
4. Avoid simultaneous edits to the same code file; append dated Markdown notes
   for parallel doc work when possible.

## CC Audit + Active Handoff (2026-07-01)

- Three-way sync verified: local = GitHub = server, all at `4743391`, clean.
- Audit verdict: goal + direction reasonable; keep P0/P1 focus. Biggest risk =
  rolling data refresh / regime drift.
- Active Codex goal: re-validate frozen P0/P1 on the latest as-of block with a
  leakage + coverage audit (H2026 regime). Details + ownership in
  `docs/CC_AUDIT_AND_HANDOFF_20260701.md`.
- Doc hygiene: `PROJECT_BRIEF.md` and `MEMORY.md` marked SUPERSEDED (old output
  boundary contradicted current `AGENTS.md`).
- Ownership: CC owns goal/decision/handoff docs; Codex owns src/scripts/runs/
  reports/memory/RUN_STATUS.

## Read First

```text
README.md
AGENTS.md
docs/START_HERE.md
docs/GIT_AND_COLLABORATION.md
docs/GITHUB_FILE_MAP.md
docs/PROJECT_REVIEW.md
docs/DECISIONS.md
```
