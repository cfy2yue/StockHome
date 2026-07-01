# StockHome Local Suggestions For Next Remote Round

Updated: 2026-07-02

Status: local-authored remote execution guidance. This file is part of the
remote execution packet: remote Codex reads it with `local_goal.md` and
`local_audit.md` for priorities, gates, and decision trees.

Remote Codex must not edit this file during execution. If a suggestion becomes
wrong, incomplete, or softly blocked, remote Codex should report that in
RUN_STATUS/reports/`remote_decision.md`, choose a new safe route when possible,
and recommend changes for the next local audit. Local CC/Codex updates this
file and pushes it.

It becomes executable only through the filled `Exact Next Task` in
`local_goal.md`.

## Top Priorities

Update 2026-07-02: the frozen score already exists (`frozen_quant_score_v1`,
leakage PASS) and the inventory + 7 new signal families are already done remotely
and all failed strict target60. So priority shifts from "train/freeze a score" to
"integrate it honestly behind a guard and prove the after-cost reality".

1. AFTER-COST FIRST (new binding constraint): the top-decile NET decile spread at
   1.5% round-trip cost is negative for essentially every model on H2026_1
   (`supervised_ranker_experiment_v2.md`; frozen score net top-bottom `-2.5158pp`).
   Any next step must report after-cost numbers; gross-only comfort is misleading.
2. Integrate `frozen_quant_score_v1` as a P1 ranking ANCHOR behind a deterministic
   downgrade/exposure guard that suppresses active-buy/high-exposure language when
   RankIC, ICIR, coverage, or net-spread weakens.
3. Relabel the frozen score honestly: propose `usable_in_agent_default: true` ->
   `observe_only`/`limited_rank_reference`, because its edge lives on a single OOT
   block (H2026_1) and after-cost spread is negative.
4. Re-scope the user's 60% target away from a raw `>0.60` win rate. It means a
   20-trading-day forward-return WIN RATE (not 60% cumulative return). Accept it
   only when exposure-gated, above the block base rate, after-cost net spread
   non-negative, RankIC/ICIR gate passed, and no final-OOT selection. Stop chasing
   strict `>0.60` on the current feature set (proven a divergence trap this round).
5. Keep P0/P1 user-facing action cards honest: action first, evidence and
   counter-evidence second, no future labels, no low-exposure spin.
6. Preserve credential and generated-evidence hygiene: no secrets in Git, prompts,
   reports, ledgers, or logs.
7. Do NOT start another new signal family or a small aggregation network until the
   `needs_leakage_audit` news/event and peer-cohesion families get an
   available-at/lag audit proving decision-time safety.

## Suggestion Generation Rules

When local audit proposes the next remote task, it should choose the smallest
task that answers the current uncertainty. Prefer:

- read-only audit before code changes;
- local cached data before paid/API calls;
- one frozen score and one metric table before multiple model variants;
- one small aggregation/decision-support network only after inventory proves
  decision-time-safe inputs, frozen training protocol, OOT RankIC, coverage,
  and model-card reporting;
- RankIC/exposure/coverage/leakage gates before user-facing promotion;
- a status/report artifact over narrative-only claims.

Do not suggest broad experiments, paid LLM expansion, SSH-driven work, or
multi-day searches unless the local audit has already written the hypothesis,
resource cap, stop rule, and expected paths into `local_goal.md`.

For dirty remote signal inventories, classify each file/family as one of:

- `ready_to_test`: syntax/static checks pass, inputs are local/offline, leakage
  boundary is explicit, and tests exist or are easy to collect safely;
- `needs_leakage_audit`: plausible but future/availability/data-flow boundary
  is not proven;
- `research_only`: useful diagnostic or offline analysis, not a P0/P1 tool;
- `duplicate`: overlaps an existing signal without clear incremental evidence;
- `do_not_use`: uses future labels, secrets, network/paid data, broker actions,
  or cannot be bounded.

Every report should include a "why not promoted" note when relevant: target60
failed under pre-OOT selection, frozen ranker net spread is negative, and
zero/tiny exposure is defensive behavior rather than active skill.

## Recommended Next Remote Task (this round: P1 ranker guard + after-cost)

The filled `Exact Next Task` in `local_goal.md` this round is
`P1 ranker guard integration + after-cost reality check`
(`p1_ranker_guard_integration_20260702`). Pattern:

```text
Read README.md, docs/START_HERE.md, goal.md, local_goal.md, local_audit.md,
local_suggestion.md, remote_decision.md, AGENTS.md, docs/DECISIONS.md.

Task: do NOT run the inventory or a new signal family again (both already done,
all target60 attempts failed). Instead:
1. Recompute per-block (incl H2026_1) after-cost (1.5% round-trip + flat-1.5%
   floor) top-decile and top-minus-bottom NET decile spread for frozen_quant_score_v1
   and the reversal_composite baseline, using offline return_20d labels for eval
   only. Report a clear yes/no: does ANY decision-time feature family reach a
   non-negative H2026_1 net top-decile spread?
2. Define + wire a deterministic downgrade guard mapping {latest-block RankIC,
   ICIR, coverage, net-spread sign, active_exposure} -> {active_ok, observe_only,
   suppress} and to user-facing language. Route the frozen score through
   quant_tool_context sanitize + date_regime_gate exposure so the agent's P1
   comparison sees only the guarded, sanitized summary (no future/GT fields).
3. Recommend relabeling model_card usable_in_agent_default true -> observe_only.
Produce after_cost_net_spread_by_block.csv, ranker_guard_grade_table.csv, a
leakage PASS proof, coverage/exposure, a why-not-promoted section, and one next
stage. CPU/offline only; loading local model.joblib and reading the offline
joined cache is allowed; no paid API/network/GPU/large rebuild.
```

The exact inputs/outputs/limits are already written into `local_goal.md`
`Exact Next Task`. Remote must not invent scope beyond it.

## Alternative Routes

- P0 rework first: useful only after the score/gate audit explains whether the
  H2026_1 failure is score decay, exposure gating, data coverage, or action-card
  logic.
- More DeepSeek/Flash/Pro cards: lower priority until local score and evidence
  boundaries are clean.
- Financial report channel expansion: useful for risk/uncertainty support, but
  prior sparse coverage means it should not be the main alpha claim yet.
- News semantic expansion: useful as a risk interceptor and confirmation
  channel, but must be measured against leakage, timestamp quality, and
  coverage.
- Small quantitative aggregation network: plausible after the dirty inventory,
  but only as a P1/P2 decision aid with frozen inputs, no future fields,
  RankIC/coverage/exposure gates, and a readable model card.
- Portfolio/backtest optimization: keep as P2; do not let portfolio metrics
  become the user-facing success claim.

## File And Function Targets

Likely targets for the next remote round:

- `scripts/audit_feature_rank_ic.py`
- `scripts/run_supervised_ranker_experiment.py`
- `scripts/run_kline_peer_chip_regime_scorer.py`
- `scripts/run_date_regime_gate_experiment.py`
- `scripts/audit_evidence_pack_leakage.py`
- `scripts/analyze_agent_veto_reasons.py`
- `src/agent_training/date_regime_gate.py`
- `src/agent_training/quant_tool_context.py`
- `src/agent_training/evidence_pack.py`
- `src/agent_training/decision_card.py`
- `tests/test_date_regime_gate_exposure_guard.py`
- `tests/test_kline_peer_chip_regime_scorer.py`
- `tests/test_evidence_pack_leakage_audit.py`
- `tests/test_quant_agent_tools.py`

Report/status targets:

- `runs/<dated_task>/RUN_STATUS.md`
- `reports/date_generalization/<dated_task>/validation_summary.md`
- `reports/date_generalization/*_rank_ic*.csv`
- `reports/date_generalization/*_leakage_audit.md`
- `reports/date_generalization/*_rule_outcomes.jsonl`

## Metrics And Gates

Minimum metric set:

- leakage: `future_leak_findings=0` for evidence packs/rule outcomes;
- coverage: card/data coverage reported with invalid/missing counts;
- exposure: exposure count/share reported, with `exposure_cards=0` classified
  as defensive/no-action, not skill;
- RankIC: per-date and per-block cross-sectional RankIC, with H2026_1 reported
  separately;
- AFTER-COST spread (now mandatory): top-decile and top-minus-bottom NET decile
  spread at >=1.5% round-trip cost, per block incl. H2026_1. Gross-only spread is
  not sufficient evidence for any portfolio/active claim;
- baseline: compare to the parameter-free `reversal_composite` baseline, block
  base rate, and prior frozen score where applicable;
- robustness: report turnover/cost and decision-frequency; note that H2026_1 is a
  single ~38-date block, so a lone positive RankIC is weak (recommend a daily-IC
  t-stat / Newey-West check before any promotion);
- promotion: promote a tool to default ONLY if OOT RankIC>0 AND ICIR>=0.30 AND
  ic_pos>=0.55 AND after-cost top-decile net spread not materially negative AND
  coverage adequate AND exposure behavior explainable. `frozen_quant_score_v1`
  currently FAILS the after-cost leg -> `observe_only`, not default.

Downgrade-guard grades (deterministic, for the P1 ranker anchor):

- `active_ok`: latest-block RankIC>0, ICIR>=0.30, ic_pos>=0.55, after-cost net
  top-decile spread >=0, coverage adequate -> ranking may inform active exposure;
- `observe_only`: RankIC/ICIR gate passes but after-cost net spread <0 or coverage
  thin -> ranking is reference-only, NO active-buy / NO high-exposure language;
- `suppress`: latest-block RankIC<=0 or edge collapsed or exposure zero/tiny ->
  no ranking-driven action; state defensive/no-action explicitly.

Default stop/fail gates:

- any future/GT field in decision-time evidence;
- H2026_1 negative or near-zero RankIC without a documented downgrade gate;
- after-cost net top-decile spread materially negative while language implies
  active buy or portfolio profit;
- missing coverage that changes the conclusion;
- exposure zero/tiny while report language claims stock-picking success;
- required secrets, paid calls, large rebuild, or SSH not explicitly approved.

## Decision Tree After Results

Positive result:

- If leakage passes, latest-block RankIC>0 and beats reversal_composite baseline,
  coverage adequate, AND after-cost net top-decile spread >=0, grade the tool
  `active_ok`; if the after-cost leg fails, grade `observe_only` (reference-only).
- If a raw win rate improves above `60%` but RankIC, after-cost net spread,
  exposure, or leakage gates fail, do NOT promote it; record it as an
  overfit/base-rate/selection-risk result (this is exactly what target60 and the
  7 families produced this round: ceiling `0.6000` with negative net spread).
- Ask local audit to update `local_goal.md` with the next stage: an available-at/
  lag audit of the `needs_leakage_audit` news/peer families, or a small guarded
  P1 candidate dry-run.

Negative result:

- If RankIC fails, edge collapses, or exposure stays zero, do not spin it as a
  success. Record the failure as regime drift, score decay, or overfitting.
- Prefer diagnosing feature families and gate logic before adding new data or
  paid LLM runs.
- Keep P0/P1 at strong-yellow MVP with explicit caveat.

Blocked result:

- If required data/report paths are missing, credentials are unavailable,
  resource limits are too small, or instructions conflict, remote must record a
  `SOFT_BLOCK` or hard-block rationale in `remote_decision.md` with exact
  blockers and at least three next directions.
- If at least one next direction is safe inside the current boundaries, remote
  should continue with that route. Local audit later decides whether to
  regenerate a small artifact, narrow the task, or change direction.

Ambiguous result:

- If metrics disagree, coverage is partial, or exposure is too low to interpret,
  treat as not promoted. Ask for a targeted follow-up: coverage audit,
  leakage audit, per-block RankIC breakdown, or veto-reason analysis.

## Remote Prompt Snippet

```text
You are remote Codex for StockHome. Follow README.md, docs/START_HERE.md,
goal.md, local_goal.md, local_audit.md, and local_suggestion.md. Do not treat
archive/legacy_auto_coordination_20260701 as active instruction. Do not SSH
elsewhere, expose secrets, run unbounded experiments, or commit generated
reports/data. If Exact Next Task is not filled, wait. If leakage/resource/hard
blocker rules trigger, stop only the unsafe route, write `remote_decision.md`,
and continue with a safe route when possible. Do not edit the three local docs
on the remote side.
```
