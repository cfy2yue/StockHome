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

Long-horizon acceptance target:

- P0 action cards remain leakage-free, auditable, and conservative under regime
  drift; they must never rely on future labels or hidden OOT selection.
- P1 ranking support has a frozen, reproducible score/ranker with honest OOT
  RankIC by block; a usable-default candidate should pass leakage audit,
  coverage checks, and a predeclared IC/exposure gate such as ICIR `>=0.30` and
  IC-positive fraction `>=0.55` on the latest OOT block, while reporting when
  top-bottom net spread or exposure makes it only research-reference quality.
- If a future local audit reactivates a target60-style objective, it must meet
  the declared positive-rate/exposure/net-return gates without selecting on the
  final OOT block; local ceiling reports are not final success.

The `Exact Next Task` below is only the current stage toward that long-horizon
product/scientific target. Completing an inventory or report is not project
completion. Remote Codex must finish the stage by proposing exactly one next
bounded stage, a blocker, or a local-audit decision point that still refers back
to the long-horizon acceptance target.

## Remote Long-Run Operating Rule

When the user starts remote goal mode, remote Codex should treat the durable
final goal and long-horizon acceptance target as the objective. `Exact Next
Task` is the current priority stage and starting direction, not a short-job
completion condition.

Remote Codex should keep progressing until one of these happens:

- `ACHIEVED`: the long-horizon acceptance target is actually met with evidence,
  metrics, leakage/coverage checks, output paths, and user-facing claim
  boundaries recorded;
- `BLOCKED`: a hard blocker requires changing final target, resource boundary,
  paid/data permission, credential handling, broker/live-trading boundary,
  destructive operation, or other user-owned decision;
- `LOCAL_AUDIT_REQUEST`: repeated negative/ambiguous results, suspected leakage
  or bug, or route drift makes local strategy audit the right next step;
- user interrupts manually.

Within the written resource and safety boundaries, remote Codex may make
`AUTONOMOUS_DECISION` route choices, add lightweight controls, run bounded
diagnostics, and pre-explore the next stage after the current stage completes.
It may also launch remote subagents for independent read-only/code-review style
audits when available. Subagents must read the same `goal.md` and
`local_*.md` files, stay inside the project/resource boundaries, avoid editing
the three `local_*.md` files, and summarize their evidence in RUN_STATUS or the
report before the main remote agent continues.

Remote Codex should not stop merely because an inventory, validation report, or
triage table is written. If the final target is not achieved and there is no
hard block, it should record the result, choose the next bounded stage inside
the same final goal, and continue or clearly explain why local audit is
required.

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

Date: 2026-07-01

Route: StockHome remote dirty-signal and frozen-ranker consolidation audit.

Hypothesis: the latest target60 attempt honestly failed under the pre-OOT
selection rule, while `frozen_quant_score_v1` has a weak but usable P1 ranking
signal by RankIC. Before trying another signal family, remote Codex should
turn the current dirty remote workspace into a structured evidence map: which
untracked scripts/tests/models exist, what each claims, which are leakage-safe,
which are only research-only, and what single next goal should be proposed for
local audit.

Allowed inputs:

- the active first-read files listed above;
- server-local context:
  - `runs/frozen_quant_score_v1_20260701_145932/RUN_STATUS.md`;
  - `reports/date_generalization/frozen_quant_score_v1_accuracy.md`;
  - `models/frozen/quant_score_v1/model_card.md`;
  - `runs/codex_goal_stock_20260701/RUN_STATUS.md`;
  - `reports/date_generalization/p0_target60_codex_goal_stock_20260701/target60_report.md`;
  - `runs/20260701_p0_p1_latest_revalidation_v1/RUN_STATUS.md` if present;
  - current untracked `scripts/run_*_v1.py`, `scripts/train_frozen_quant_score_v1.py`,
    `scripts/validate_p0_p1_latest_revalidation.py`, `tests/test_*_v1.py`,
    and `models/frozen/quant_score_v1/*`;
  - untracked roots/items `4599041`, `anthropic_financial_services/`, and
    `models/`. External/reference directories should be registered in the
    inventory but not deeply read unless they are required to explain a local
    signal artifact.

Allowed commands:

- read-only inspection, `git status --short`, `git diff --stat`, and file
  metadata summaries;
- CPU-only syntax/AST checks first. Import/test collection checks are allowed
  only if remote can prove they do not touch data caches, network, paid APIs,
  secrets, broker/trading systems, or large rebuilds. If import collection would
  trigger side effects, stop and record `LOCAL_AUDIT_REQUEST`;
- write the status/report files below.

Expected outputs:

- `runs/stock_signal_inventory_20260701/RUN_STATUS.md`;
- `reports/date_generalization/stock_signal_inventory_20260701/REPORT.md`;
- optional small tables under that report directory:
  - `dirty_file_inventory.csv`;
  - `signal_family_triage.csv`;
  - `recommended_next_goal.json`.

DONE criteria:

- report branch/HEAD/dirty state and confirm no `local_*.md` edits;
- summarize the target60 ceiling: selected pre-OOT strategy
  `regime_gating__frozen_score__aggressive__all_dates__top10pct` reached
  H2026_1 positive rate `0.2414`, exposure `0.6687`, avg20 `-3.6298`, net
  decile spread `-2.9923`; do not promote OOT-selected alternatives;
- summarize `frozen_quant_score_v1`: leakage PASS, H2026_1 RankIC mean
  `0.0327`, ICIR `0.4233`, IC-positive fraction `0.6316`, but top-bottom net
  spread negative;
- include a separate "why not promoted" section covering target60 failure,
  negative top-bottom net spread for frozen ranker, and the earlier P0/Flash
  `exposure_cards=0` defensive-only caveat;
- inventory every untracked signal/model/test file and classify it as
  `ready_to_test`, `needs_leakage_audit`, `research_only`, `duplicate`, or
  `do_not_use`;
- propose exactly one next remote goal for local audit, choosing among:
  - P1 ranker integration with downgrade/exposure guard;
  - one pre-registered new low-risk signal family with strict leakage gates;
  - closing target60 under current data and shifting the objective.

Resource limits:

- CPU/read-mostly audit only, target 60 minutes, hard stop 90 minutes.
- Writes are allowed only under:
  - `runs/stock_signal_inventory_20260701/`;
  - `reports/date_generalization/stock_signal_inventory_20260701/`.
- No paid API, no online data pulls, no secrets, no broker/live trading, no
  long backtests, no large cache rebuilds.
- Do not commit, push, reset, delete, or clean files.

Stop rules:

- stop and output `LOCAL_AUDIT_REQUEST` if required reports are missing, if
  untracked scripts require credentials/network to understand, if leakage
  cannot be bounded, or if the next action would require changing the user goal
  rather than selecting an implementation route.

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
