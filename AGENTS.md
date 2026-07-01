# StockHome Agent Rules

Use Chinese for user-facing summaries unless the user asks otherwise.

## Authority

Read these first:

- `goal.md`
- `local_goal.md`
- `local_audit.md`
- `local_suggestion.md`
- `docs/START_HERE.md`
- `docs/RESPONSE_PROTOCOL.md`

`goal.md` and the durable final goal in `local_goal.md` define the long-horizon
target. `Exact Next Task` is the current route hypothesis and priority start,
not the whole goal. If a route fails, record evidence and optimize the route;
do not mark the goal complete unless the final acceptance target is achieved.

## Remote Execution

- Long-run toward the final target until `ACHIEVED`, hard `BLOCKED`, user
  interruption, or `LOCAL_AUDIT_REQUEST`.
- Remote Codex may make bounded `AUTONOMOUS_DECISION` choices inside the
  resource, data, credential, and safety limits.
- Remote Codex may launch subagents for independent audit, code review, signal
  triage, leakage review, or route pre-exploration. Subagents must read the same
  authority files, stay within project limits, and report evidence into
  RUN_STATUS/reports.
- Do not edit `local_goal.md`, `local_audit.md`, or `local_suggestion.md`
  during remote execution. Suggest next local updates in RUN_STATUS/reports.

## Financial Safety

- This project is an A-share research assistant, not an auto-trading system.
- Do not connect to brokers, place orders, automate trading, promise returns,
  or claim certainty.
- User-facing outputs may give actions such as buy, trial buy, add, hold,
  reduce, sell, wait, or collect more data only with evidence,
  counter-evidence, position/risk limits, invalidation, and review conditions.
- Future labels such as `return_20d`, `future_*`, `gt_status`, or outcome
  fields are offline evaluation labels only. They must not enter decision-time
  evidence, prompts, rules, or user-facing reasoning.
- Never print, copy, commit, or log API keys, tokens, `.env*`, `ds_api.txt`,
  `tushare_token.txt`, or other credentials.
- Authorized public/member/paid/standardized data sources may be used only
  through ignored local credentials or offline caches, with source type noted.

## Git And Data Hygiene

- Keep generated runs/reports/models/caches and raw/private data out of Git
  unless explicitly curated as small documentation.
- Do not delete, reset, clean, or overwrite user data, reports, books, refs, or
  caches without explicit user instruction.
- If a data/API source fails, skip or report it; do not let the whole workflow
  silently collapse.

## Stop And Report

Output `LOCAL_AUDIT_REQUEST` when the route needs local strategy optimization,
required artifacts are missing, leakage/availability controls are unclear,
resource or credential boundaries would need to change, or results cannot be
interpreted without changing the final target.
