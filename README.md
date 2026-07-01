# StockHome

StockHome is an A-share research assistant project. It supports single-stock
research, watchlist review, multi-stock comparison, intraday review, strategy
research, and evidence-grounded report generation.

It must not place trades, connect to broker APIs, promise returns, or present
backtest/label leakage as real decision evidence.

## Entry

- Server directory: `/data/cyx/1030/stock`
- GitHub target: `https://github.com/cfy2yue/StockHome`
- SSH entry: `ssh cyx-server-cfy`, then `cd /data/cyx/1030/stock`

## Current Workflow

The active workflow is manual local audit plus manually started remote Codex
execution.

Read these first:

1. `local_goal.md`
2. `local_audit.md`
3. `local_suggestion.md`
4. `goal.md`
5. `docs/START_HERE.md`
6. `AGENTS.md`
7. `docs/GIT_AND_COLLABORATION.md`
8. `docs/GITHUB_FILE_MAP.md`
9. `docs/USER_GUIDE.md`
10. `docs/HANDOFF.md`

Legacy auto-coordination files are archived under
`docs/archive/legacy_auto_coordination_20260701/`. They are historical
evidence, not active workflow instructions.

## Product Boundary

- P0: single-stock watchlist/research with explicit action, position/risk
  limit, evidence, counter-evidence, triggers, and review condition.
- P1: multi-stock comparison as a support workflow.
- P2: portfolio/backtest strategy research, not a replacement for P0/P1 user
  advice.

Every user-facing answer should state the recommendation first, then evidence,
counter-evidence, invalidation conditions, and next review trigger.

## Safety

- Do not print or store API keys, tokens, broker credentials, or private
  environment values.
- Do not delete raw books, BookSkill sources, caches, runs, reports, or
  backtest evidence without explicit user approval.
- Do not use future returns, GT fields, or after-the-fact outcomes inside
  evidence packs.
- Do not treat low-exposure or hindsight metrics as live stock-selection
  ability.
