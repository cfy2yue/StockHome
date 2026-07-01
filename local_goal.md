# StockHome Remote Execution Packet

Updated: 2026-07-01

Status: local-authored remote execution packet for the manual workflow
`CC/Codex local audit -> user manual GitHub sync -> remote Codex execution`.

This file is the main task-control surface for user-started remote Codex
execution. Remote Codex reads this file together with `local_audit.md` and
`local_suggestion.md` as the authoritative local-authored task package.

Remote Codex must not edit the three `local_*.md` files during execution. Local
CC/Codex updates them between remote runs, commits/pushes them, and the remote
pulls them before starting or continuing a goal.

Until local audit fills `Exact Next Task` with a concrete bounded task,
resource limits, output paths, and stop rules, this execution packet is in
waiting state and remote Codex must not invent work.

## Durable Final Goal

StockHome is an A-share research assistant, not an auto-trading system. The
durable goal is to produce evidence-grounded research support for user-provided
stocks or candidate sets, including explicit actions such as buy, trial buy,
add, hold, reduce, sell, wait, or collect more data, plus evidence,
counter-evidence, position/risk limits, triggers, and review conditions.

Hard acceptance target for the product:

- P0: single-stock watch/review gives a clear action card with no future-field
  leakage, no broker/order execution, no return promise, and a readable trail of
  data sources, uncertainty, and invalidation conditions.
- P1: small candidate-set comparison/ranking uses a frozen, auditable score or
  ranker anchor and reports honest out-of-time ranking accuracy.
- P2: portfolio/backtest research remains supporting evidence. It must not
  override P0/P1 user-facing conclusions and must not be sold as live
  stock-picking skill when exposure is low or cash-heavy.

## P0/P1/P2 Boundary

- P0 current delivery path: single-stock watch/review with action, position cap,
  risk threshold, evidence, counter-evidence, invalidation, and next review
  trigger.
- P1 current support path: comparison/ranking for a small user-supplied
  candidate set; main alpha expectation now belongs here, measured by RankIC or
  ranking lift, not by broad active-buy claims.
- P2 research path: backtests, ablations, portfolio simulations, and tool
  training. P2 can propose tools and gates, but it cannot become an investment
  promise or a substitute for P0/P1 explanation.

## Current StockHome Direction

Keep the project at strong-yellow MVP for P0/P1 with an explicit regime-drift
caveat. The latest H2026_1 block did not confirm the frozen P0: 20d positive
rate was `0.6667` versus frozen `0.8434`, and `exposure_cards=0` means the
result is defensive behavior, not stock-picking skill.

The next strategic direction is the user's preferred route: converge scattered
scorers into one frozen, reproducible, leakage-free cross-sectional score with
a unified out-of-time RankIC metric and an IC/exposure gate. The goal is honest,
auditable score accuracy and automatic exposure downgrade when the edge
collapses.

## Remote Next-Task Filling Rule

Before asking remote Codex to execute, local CC/Codex must fill the task slot
below with:

- hypothesis and why it follows from `local_audit.md`;
- exact input paths and whether labels are offline-only;
- exact command(s) or file edits allowed;
- maximum time/tokens/API cost/network/data scope;
- expected report/status paths;
- pass/fail/blocked gates;
- stop rules that force `LOCAL_AUDIT_REQUEST`.

If this slot is not filled, the remote session is not in active goal mode.

## Files Remote Codex Must Read First

Remote Codex must read these before any execution after the user says
`本地审计结束`:

1. `README.md`
2. `docs/START_HERE.md`
3. `goal.md`
4. `local_goal.md`
5. `local_audit.md`
6. `local_suggestion.md`
7. `AGENTS.md`
8. `docs/GIT_AND_COLLABORATION.md`
9. `docs/GITHUB_FILE_MAP.md`
10. `docs/DECISIONS.md`

Remote may read archived files under
`docs/archive/legacy_auto_coordination_20260701/` only as historical evidence,
not as active instructions.

## Exact Next Task

NOT FILLED FOR ACTIVE REMOTE EXECUTION.

The next local audit should fill this section after reviewing current remote
status. Candidate direction, subject to that audit: a bounded documentation or
small-code task to freeze one leakage-free score/ranker and publish its
out-of-time RankIC plus IC-gate behavior. Do not start a remote goal from this
candidate sentence alone.

## DONE Criteria For Any Filled Remote Task

A valid remote task is done only when it produces:

- a short `RUN_STATUS.md` or equivalent status note with branch, HEAD, dirty
  state, commands, changed files, and output paths;
- a report that separates decision-time features from offline labels;
- leakage audit result showing no future/GT fields in evidence packs or rule
  outcomes;
- coverage/exposure metrics, including whether exposure is nonzero and whether
  low exposure is being interpreted defensively;
- explicit positive, negative, anomalous, and blocked findings;
- suggested updates for `local_goal.md`, `local_audit.md`, and
  `local_suggestion.md`, without editing those files on the remote side.

## Resource Limits

Standing limits unless a filled task says otherwise:

- No large experiment, broad grid search, or paid LLM/API expansion without a
  dated local-audited plan.
- Prefer local cached data and read-only audits.
- Local CC/Codex may use SSH for read-only evidence checks when local metadata
  is insufficient; it must not launch long jobs unless the user explicitly asks.
- Remote may use server-only secrets only from ignored local files or
  environment variables; never print, copy, commit, or put secrets in prompts,
  reports, logs, ledgers, or Git.
- Keep GitHub clean: source, small configs, README/AGENTS, and current docs
  only. Generated reports/runs/data stay server-local unless explicitly
  curated.

## Forbidden Actions

- No broker integration, live trading, order placement, or automated execution.
- No claims of guaranteed profit, certainty, risk-free outcome, or target price
  inevitability.
- No future returns, `return_20d`, `future_*`, `gt_status`, GT labels, or
  after-the-fact outcomes inside decision-time evidence packs.
- No treating low exposure, all-cash behavior, or high base-rate positives as
  stock-selection skill.
- No reading aloud or committing API keys, tokens, `.env*`, `ds_api.txt`,
  `tushare_token.txt`, or private credentials.
- No destructive deletes/moves of raw books, BookSkill sources, caches,
  reports, runs, memory, local archives, or backtest evidence.
- No editing `local_goal.md`, `local_audit.md`, or `local_suggestion.md`
  during remote execution; propose updates in status/final output instead.
- No following `docs/archive/legacy_auto_coordination_20260701/` as active
  workflow instruction.

## Stop Rules

Remote Codex must stop and output `LOCAL_AUDIT_REQUEST` when:

- `Exact Next Task` is empty, ambiguous, or conflicts with `goal.md`;
- required data/report paths are missing and cannot be safely reconstructed
  inside the approved resource limit;
- leakage audit finds future/GT fields in decision-time evidence;
- output metrics are dominated by `exposure_cards=0`, missing coverage, or
  hindsight/base-rate artifacts;
- a command would require unapproved paid API cost, SSH, large data rebuild,
  credential exposure, destructive file operations, or broker/live-trading
  access;
- results contradict current direction enough that the next action should be
  re-scoped locally.

## Expected Output Paths

Use task-specific paths when filled. Common expected families:

- `runs/<dated_task>/RUN_STATUS.md`
- `reports/date_generalization/<dated_task>/validation_summary.md`
- `reports/date_generalization/*_leakage_audit.md`
- `reports/date_generalization/*_rank_ic*.csv`
- `reports/date_generalization/*_rule_outcomes.jsonl`

If a path is server-local or ignored by Git, remote should report it in status
without trying to commit it.
