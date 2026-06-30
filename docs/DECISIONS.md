# Decisions

Last slimmed: 2026-07-01.

The full pre-slim chronological decision log is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/DECISIONS.md
```

## 2026-07-01: CC Audit — Validate Frozen P0/P1 On Latest Data; Mark Stale Docs

Decision: after a clean three-way sync (local = GitHub = server at `4743391`),
hand remote Codex one bounded goal — re-validate the frozen P0/P1 workflows on
the latest as-of data block with a leakage + coverage audit (H2026 regime). Mark
`PROJECT_BRIEF.md` and `MEMORY.md` SUPERSEDED where their output boundary
contradicts the current `AGENTS.md`/`RESPONSE_PROTOCOL.md`.

Reason: the project's largest risk is rolling data refresh / regime drift;
validating it is the highest-value bounded step and the gate for any promotion.
The two legacy docs still stated "do not output definite buy/sell", which no
longer matches the approved action-advice protocol.

Consequence: see `docs/CC_AUDIT_AND_HANDOFF_20260701.md` for the full goal +
ownership. CC owns goal/decision/handoff docs; Codex owns code/runs/RUN_STATUS.

## 2026-07-01: Initialize StockHome As Its Own Repo

Decision: initialize `/data/cyx/1030/stock` as the local repository for
`https://github.com/cfy2yue/StockHome`.

Reason: stock/quant work is independent from scLatent and CellClip and has
different data, security, and claim-language constraints.

Consequence: StockHome gets its own `.gitignore`, collaboration docs, and GitHub
file map.

## 2026-07-01: Keep P0/P1 As Current Delivery Focus

Decision: prioritize single-stock watch/review and small candidate-set
comparison over broad active-buy claims.

Reason: historical evidence supports a strong-yellow MVP for P0/P1, while broad
active-buy generalization remains insufficiently stable.

Consequence: docs and future plans should not frame the project as already
solving broad market active-buy selection.

## 2026-07-01: Keep Secrets And Generated Evidence Out Of Git

Decision: ignore local data, reports, runs, ledgers, memory, deliverables,
secrets, API tokens, and local archives.

Reason: the repo should be safe for GitHub and CC collaboration. Generated
outputs remain server-local unless a curated artifact is deliberately selected.

Consequence: `ds_api.txt`, `tushare_token.txt`, `.env*`, reports, logs, and
local archives are not tracked.

## 2026-07-01: Slim Publication Docs

Decision: replace large chronological Markdown logs with compact current-state
documents and preserve full versions in ignored local archive.

Reason: CC/Codex collaboration needs crisp entry docs. Full logs are useful
provenance but poor first-read material.

Consequence: GitHub shows current project state and boundaries; server archives
keep historical evidence.

## Standing Decision: Financial Claim Boundaries

StockHome is a research assistant, not an automated trading system. It may give
research-oriented action suggestions with evidence and risk controls, but must
not promise returns, bypass user confirmation, or hide uncertainty.
