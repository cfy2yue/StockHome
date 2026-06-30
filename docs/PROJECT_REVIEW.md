# Project Review

Last slimmed: 2026-07-01.

The full pre-slim review log is preserved server-local at:

```text
docs/local_archive/20260630_pre_slim/PROJECT_REVIEW.md
```

## Current Project Goal

StockHome should deliver an A-share research assistant focused on single-stock
watch/review and candidate comparison. It should provide decision support with
clear evidence and risk boundaries, not automated trading or guaranteed-return
claims.

## Current Path

- Prioritize P0 single-stock action cards and P1 candidate comparison.
- Use deterministic/ranker anchors and agent audit rather than free-form
  unbounded recommendations.
- Keep secrets, local data, reports, ledgers, and run outputs out of Git.
- Publish only source, small configs, and high-signal docs.

## Evidence Snapshot

- Historical experiments do not support claiming solved broad active-buy
  generalization.
- Existing evidence supports a strong-yellow P0/P1 MVP with careful language,
  actionability checks, and risk triggers.
- Detailed experiment logs and older goal text remain in local archives and
  server reports, not GitHub-facing current docs.

## Risks

- Financial claims are high-stakes; wording must stay research-assistant level.
- Future-field leakage can invalidate backtests and decisions.
- Token/API secrets must never enter prompts, logs, reports, or Git.
- Overly long historical docs can cause new agents to follow obsolete product
  goals or stale run instructions.

## Direction Decision

Continue with modification.

Keep P0/P1 as the delivery center, preserve the historical evidence locally, and
publish a slim collaboration-ready repo.

## Recommended Next Action

After GitHub initialization, CC should clone the repo locally and use
`docs/START_HERE.md` plus `docs/GIT_AND_COLLABORATION.md` as the first handoff
surface. New experiments or API-backed runs require a short dated plan first.

## Files To Inspect Next

```text
goal.md
docs/START_HERE.md
docs/GIT_AND_COLLABORATION.md
docs/DECISIONS.md
docs/USER_GUIDE.md
docs/WORKFLOW.md
```
