# StockHome Remote Execution Packet

Updated: 2026-07-02

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
- For the current alpha-assistance track, the user's aspirational final target
  is 20-day positive-return rate above `60%`. This target is valid only when
  paired with nontrivial exposure/coverage, positive or at least non-damaging
  net spread/return evidence, leakage PASS, base-rate comparison, and no
  selection on the final OOT block.
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

The current route is a hypothesis under the same final goal, not a step in a
step-to-step decomposition. If the route fails, underperforms, or becomes
unpromising, the right action is to record the evidence, audit/optimize the
route, and continue toward the same final target or request local route
optimization. Do not mark the goal complete merely because the current stage is
done.

Remote Codex should keep progressing until one of these happens:

- `ACHIEVED`: the long-horizon acceptance target is actually met with evidence,
  metrics, leakage/coverage checks, output paths, and user-facing claim
  boundaries recorded;
- `BLOCKED`: a hard blocker requires changing final target, resource boundary,
  paid/data permission, credential handling, broker/live-trading boundary,
  destructive operation, or other user-owned decision;
- user interrupts manually.

`LOCAL_AUDIT_REQUEST` is a soft audit marker, not a stop condition. Remote
Codex may write it in RUN_STATUS/reports/`remote_decision.md` to help future
local audit, but should not mark the long goal blocked unless all reasonable
next routes require changing the final target, resource boundary, paid/data
permission, credential handling, broker/live-trading boundary, or destructive
operation.

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

Gate semantics (collaboration mode): the user's durable target (a 20-day
positive-return WIN RATE above 60%) must NOT be silently lowered or dropped by the
remote; local audit changes only how it is honestly measured and pursued (adding
exposure/after-cost/leakage guards so it cannot be gamed) and the route/strategy,
not the target itself. The routes and gates here are local SUGGESTIONS/HYPOTHESES,
not the only path; the remote is expected to have its own problem-solving ability. A
gate is a quality bar for CLAIMING success, NOT a stop condition: if a route or gate
does not work out, do NOT declare `BLOCKED` — log the honest negative in
`remote_decision.md`, find a new in-boundary strategy yourself (new decision-time
features after an available-at/lag audit, cost-aware construction, regime gating,
small models), and keep advancing; the recorded decisions are read by the next local
audit. Reserve hard `BLOCKED` only for genuine user-owned decisions (final
target/resource/paid-data/credential/broker boundary, or a destructive operation).

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

State update (2026-07-02 local audit, verified by SSH): the prior stage — the
signal inventory — is DONE remotely (server-local `runs/stock_signal_inventory_20260701/`
+ `reports/date_generalization/stock_signal_inventory_20260701/`). Remote then
autonomously pre-registered and evaluated 7 new signal families; ALL failed
strict target60 (best H2026_1 win rate ties at `0.6000` with negative net decile
spread). The `frozen_quant_score_v1` model already exists, is leakage PASS, but
is weak and single-block-dependent (only H2026_1 is clearly positive; H2024_2 and
H2025_2 RankIC are negative) and its after-cost top-bottom decile spread is
NEGATIVE. Root cause (from `feature_rank_ic_audit.csv` and
`supervised_ranker_experiment_v2.md`): the project's implicit alpha is
cross-sectional reversal, which collapsed to ~zero RankIC on H2026_1, and after a
1.5% round-trip cost essentially no model has a positive top-decile net spread.

So the frozen-one-score goal is effectively already delivered at v1; what remains
is NOT training more models but: relabel the frozen score honestly as an
observe-grade P1 rank aid, wire it as the agent's ranking anchor behind a
downgrade/exposure guard, and stop chasing a strict raw `>0.60` win rate that the
current feature set cannot reach after costs without selection tricks.

Re-scope note for the user's 60% target (advisory; `goal.md` body is NOT edited):
"20 日正收益率 > 60%" means a 20-trading-day forward-return WIN RATE above 60%
(not a 60% cumulative return; ground truth `return_20d = close[t+20]/close[t]-1`).
Recommend the accepted acceptance form become: on the latest OOT block, an
exposure-gated candidate whose win rate clears its block base rate AND whose
after-cost (>=1.5% round-trip) top-decile net spread is non-negative AND whose
RankIC/ICIR gate passes AND with no final-OOT selection. A raw >0.60 win rate
alone is not accepted (gameable by base rate + tiny exposure).

Lightweight execution is preferred: first fix one auditable quant score/ranker as
the agent's decision anchor with a guard, then let P0/P1 reasoning use that score
with evidence, counter-evidence, and downgrade gates. Small quantitative
information-aggregation or decision-support networks stay deferred until an
available-at/lag audit clears clean decision-time inputs (news/event and
peer-cohesion families are currently `needs_leakage_audit`).

## Remote Next-Task Filling Rule

Before asking remote Codex to execute, local CC/Codex must fill the task slot
below with:

- hypothesis and why it follows from `local_audit.md`;
- exact input paths and whether labels are offline-only;
- exact command(s) or file edits allowed;
- maximum time/tokens/API cost/network/data scope;
- expected report/status paths;
- pass/fail/blocked gates;
- hard stop rules and soft pivot rules. Soft pivots should be recorded in
  `remote_decision.md` and followed by continued safe work.

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
7. `remote_decision.md`
8. `AGENTS.md`
9. `docs/GIT_AND_COLLABORATION.md`
10. `docs/GITHUB_FILE_MAP.md`
11. `docs/DECISIONS.md`

Remote may read archived files under
`docs/archive/legacy_auto_coordination_20260701/` only as historical evidence,
not as active instructions.

## Exact Next Task

Date: 2026-07-02

Route: P1 frozen-ranker integration with an honest downgrade/exposure guard, plus
an after-cost reality check. This supersedes the completed signal-inventory task
(done remotely as `stock_signal_inventory_20260701`) and the completed 7-family
target60 search (done as `p0_target60_new_signal_family_20260701`, all failed).
Do NOT re-run the inventory or start another new signal family this round.

Prior-stage results confirmed by local audit (do not redo, just build on):
`frozen_quant_score_v1` is leakage PASS but weak/single-block (H2026_1 RankIC
`0.0327`, ICIR `0.4233`, ic_pos `0.6316`; H2024_2 and H2025_2 RankIC negative) and
its top-bottom decile NET spread is negative (`-2.5158pp`). target60 and all 7
autonomous families failed strict `>0.60` (ceiling ties at `0.6000` with negative
net spread). Root cause: implicit reversal alpha collapsed on H2026_1 and after a
1.5% round-trip cost no model has a positive top-decile net spread
(`supervised_ranker_experiment_v2.md`).

Hypothesis: the frozen ranker can be a useful P1 candidate-comparison ANCHOR only
if a downgrade guard automatically suppresses active-buy / high-exposure language
whenever any of {H2026_1 (or latest-block) RankIC, ICIR, coverage, after-cost
net decile spread} weakens. The deliverable is a guarded, honest ranking aid, not
a return promise and not a strict-`>0.60` chaser.

Sub-goals (all read/CPU-only, offline caches only):

1. After-cost reality check: for the frozen score AND the reversal_composite
   baseline, recompute per-block (esp. H2026_1) top-decile and top-minus-bottom
   NET decile spread at round-trip cost 1.5% (and a conservative flat 1.5% floor),
   using existing offline `return_20d` labels for evaluation only. Confirm or
   refute the negative-net-spread finding and report whether ANY decision-time
   feature family reaches a non-negative H2026_1 net top-decile spread.
2. Downgrade guard spec + wiring: define/verify a deterministic guard that maps
   {latest-block RankIC, ICIR, coverage, net-spread sign, active_exposure} to a
   tool grade in {`active_ok`, `observe_only`, `suppress`} and to user-facing
   language (no active-buy / no high-exposure wording when suppressed). Wire the
   frozen score through `src/agent_training/quant_tool_context.py` (sanitize) and
   `date_regime_gate.py` (exposure) so the agent's P1 comparison consumes only the
   guarded, sanitized score summary (no future/GT fields).
3. Honest relabel proposal: recommend changing
   `models/frozen/quant_score_v1/model_card.md` `usable_in_agent_default` from
   `true` to `observe_only`/`limited_rank_reference` given the negative net spread
   and single-block edge (propose in report; do not edit local_*.md).
4. Optional (only if 1-3 done and time remains): one small candidate-set P1 dry
   run on a user-style handful of stocks showing the guarded ranking output with
   evidence, counter-evidence, exposure downgrade, and coverage — NO paid LLM/Pro
   unless a later local task explicitly authorizes it.

Allowed inputs:

- the active first-read files listed above;
- server-local context:
  - `models/frozen/quant_score_v1/model_card.md`, `feature_list.json`,
    `train_blocks.json`, `model.joblib`;
  - `reports/date_generalization/frozen_quant_score_v1_accuracy.md` (+ `.csv`);
  - `reports/date_generalization/supervised_ranker_experiment_v2.md` (+ its
    step/variant/aggregate CSVs);
  - `reports/date_generalization/feature_rank_ic_audit.csv`;
  - `reports/date_generalization/stock_signal_inventory_20260701/REPORT.md`,
    `signal_family_triage.csv`, `recommended_next_goal.json`;
  - `reports/date_generalization/p0_target60_new_signal_family_20260701/rolling_autonomy_summary_20260701.md`;
  - `reports/date_generalization/20260701_p0_p1_latest_revalidation_v1_round2_flash_confirm/validation_summary.md`;
  - the joined offline cache
    `data/date_generalization_cache/market_5000/joined_ground_truth_combined_news.csv`
    (read-only, for the after-cost recompute; labels are offline-eval only);
  - modules `src/agent_training/quant_tool_context.py`, `date_regime_gate.py`,
    `evidence_pack.py`, `decision_card.py`; `scripts/audit_feature_rank_ic.py`,
    `scripts/run_supervised_ranker_experiment.py`,
    `scripts/audit_kline_peer_chip_turnover_cost.py`.

Allowed commands:

- read-only inspection, `git status --short`, `git diff --stat`, file metadata;
- CPU-only pandas/sklearn recompute over the offline joined cache for the
  after-cost spread and guard-grade tables; loading the frozen `model.joblib` for
  scoring is allowed (it is a local artifact, no network);
- CPU-only syntax/AST checks; running the existing `tests/test_date_regime_gate_exposure_guard.py`,
  `tests/test_quant_agent_tools.py`, `tests/test_evidence_pack_leakage_audit.py`
  is allowed only if it provably touches no network/paid API/secrets/large rebuild;
  if a test would trigger side effects, skip it and record why in `remote_decision.md`;
- write the status/report files below.

Expected outputs:

- `runs/p1_ranker_guard_integration_20260702/RUN_STATUS.md`;
- `reports/date_generalization/p1_ranker_guard_integration_20260702/validation_summary.md`;
- small tables under that report dir:
  - `after_cost_net_spread_by_block.csv`;
  - `ranker_guard_grade_table.csv`;
  - optional `p1_candidate_dryrun.csv` (only if sub-goal 4 runs).

DONE criteria:

- report branch/HEAD/dirty state and confirm no `local_*.md` edits;
- after-cost net decile spread reported per block incl. H2026_1 for frozen score
  and reversal_composite, with an explicit yes/no on whether any decision-time
  family clears a non-negative H2026_1 net top-decile spread;
- a deterministic downgrade-guard grade table mapping metrics -> {active_ok,
  observe_only, suppress} and the corresponding user-facing language rule;
- proof (leakage audit `PASS`, hits=0) that the guarded score summary passed to
  the agent contains NO future/GT fields (return_5/10/20d, future_*, gt_status,
  positive_20d);
- coverage/exposure reported; any low/zero exposure explicitly called
  defensive/no-action, never stock-picking skill;
- a "why not promoted" section: negative after-cost net spread, single-block
  frozen-score edge, target60/7-family failure, P0 `exposure_cards=0`;
- a recommendation on relabeling the model card to `observe_only` (proposed, not
  applied to local_*.md);
- exactly one proposed next stage (e.g. available-at/lag audit of the
  `needs_leakage_audit` news/peer families, OR a small decision-support network
  only after that audit, OR formally closing strict target60), tied back to the
  long-horizon acceptance target.

Resource limits:

- CPU/read-mostly, target 75 minutes, hard stop 120 minutes.
- Writes allowed only under:
  - `runs/p1_ranker_guard_integration_20260702/`;
  - `reports/date_generalization/p1_ranker_guard_integration_20260702/`.
- The guard-integration sub-goals above are offline (existing caches + local
  `model.joblib`) and do not require network. Paid APIs ARE provisioned and
  authorized for this project when a task needs them: tushare (A-share market data)
  and the ds/DeepSeek LLM, credentials under `/data/cyx/1030/api` — load keys from
  there or env, never print/commit them, keep cost reasonable. No broker/live
  trading, no GPU, no long backtests, no large cache rebuilds this task.
- Do not commit, push, reset, delete, or clean files.

Stop rules:

- stop the current subtask, record a `SOFT_BLOCK` or `ROUTE_PIVOT` in
  `remote_decision.md`, and continue with another safe sub-goal if a required
  report/cache is missing or a leakage boundary cannot be bounded;
- mark hard `BLOCKED` only if every reasonable next route requires changing the
  user goal, paid/data permission, credential handling, broker/live-trading
  boundary, destructive operations, or an unapproved large rebuild.

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

- Paid APIs are provisioned for this project: tushare (A-share market data) and the
  ds/DeepSeek LLM, credentials under `/data/cyx/1030/api`. The remote may use them
  when a task needs fresh data or LLM reasoning; load keys only from
  `/data/cyx/1030/api` or env and never print/commit them. Keep cost reasonable —
  no unbounded batch loops or broad grid search without a dated plan.
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

Remote Codex must hard-stop only when:

- `Exact Next Task` is empty, ambiguous, or conflicts with `goal.md`;
- required data/report paths are missing and all safe reconstruction or
  alternative diagnostic routes are exhausted inside the approved resource
  limit;
- leakage audit finds future/GT fields in decision-time evidence;
- output metrics are dominated by `exposure_cards=0`, missing coverage, or
  hindsight/base-rate artifacts;
- a command would require clearly-excessive or unbounded paid API cost beyond
  normal provisioned tushare/ds usage, a large data rebuild, credential exposure,
  destructive file operations, or broker/live-trading access;
- results contradict current direction and all safe autonomous route pivots have
  been tried or ruled out. Weak metrics alone are not hard block; record them in
  `remote_decision.md` and continue.

## Expected Output Paths

Use task-specific paths when filled. Common expected families:

- `runs/<dated_task>/RUN_STATUS.md`
- `reports/date_generalization/<dated_task>/validation_summary.md`
- `reports/date_generalization/*_leakage_audit.md`
- `reports/date_generalization/*_rank_ic*.csv`
- `reports/date_generalization/*_rule_outcomes.jsonl`

If a path is server-local or ignored by Git, remote should report it in status
without trying to commit it.
