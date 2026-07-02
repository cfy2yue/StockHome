# StockHome Local Audit Request

Updated: 2026-07-02
Maintainer: remote executor overwrites when audit is requested; auditor reads automatically.

This file is the current executor-to-auditor request. It is intentionally
overwrite-friendly: each new audit request replaces the previous one. Durable
decision history belongs in `remote_decision.md`.

Status: EMPTY_TEMPLATE

## Required Sections For The Executor

- Project identity: path, branch, HEAD, dirty state.
- Final goal and hard boundaries as understood now.
- Files read: `goal.md`, `local_goal.md`, `local_audit.md`, `local_suggestion.md`,
  `remote_decision.md`, RUN_STATUS, reports, code paths.
- Recent work: commands, changed files, outputs.
- Evidence: positive, negative, suspicious, exact metrics and paths.
- Bottlenecks and likely causes.
- At least three audit questions for the auditor.
- Suggested updates to `local_goal.md`, `local_audit.md`, `local_suggestion.md`.

