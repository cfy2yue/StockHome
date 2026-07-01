# StockHome Local Suggestions For Next Remote Round

Updated: 2026-07-01

Status: local-authored remote execution guidance. This file is part of the
remote execution packet: remote Codex reads it with `local_goal.md` and
`local_audit.md` for priorities, gates, and decision trees.

Remote Codex must not edit this file during execution. If a suggestion becomes
wrong, incomplete, or blocked, remote Codex should report that and recommend
changes for the next local audit. Local CC/Codex updates this file and pushes
it.

It becomes executable only through the filled `Exact Next Task` in
`local_goal.md`.

## Top Priorities

1. Freeze one reproducible, leakage-free score/ranker path and evaluate it with
   out-of-time RankIC by block, especially H2026_1.
2. Add or verify an IC/exposure gate that downgrades or blocks active exposure
   when the score edge collapses.
3. Keep the user's 20-day positive-return `>60%` ambition as a guarded
   secondary target: it must be paired with exposure, net spread/return,
   leakage, base-rate, and OOT checks.
4. Keep P0/P1 user-facing action cards honest: action first, evidence and
   counter-evidence second, with no future labels and no low-exposure spin.
5. Preserve credential and generated-evidence hygiene: no secrets in Git,
   prompts, reports, ledgers, or logs.
6. For the current remote round, prefer P1 ranker integration with
   downgrade/exposure guard over continuing to chase target60.

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

## Recommended Next Remote Task Pattern

Use this pattern when the next audit fills `local_goal.md`:

```text
Read README.md, docs/START_HERE.md, goal.md, local_goal.md, local_audit.md,
local_suggestion.md, AGENTS.md, docs/DECISIONS.md.

Task: perform one bounded remote step to freeze or audit the current best
leakage-free StockHome score/ranker. Use only decision-time features for
inference. Use future labels only for offline evaluation. Produce RankIC by
date/block, H2026_1 out-of-time result, coverage, exposure, leakage audit, and
a short RUN_STATUS.md. Do not run paid/large experiments unless explicitly
allowed in local_goal.md.
```

The exact command and output paths must be filled by the next local audit after
checking current server state.

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
- baseline: compare to simple reversal baseline, block base rate, and prior
  frozen score where applicable;
- robustness: report turnover/cost or decision-frequency caveat when a strategy
  implies portfolio action;
- promotion: only promote a tool to default if OOT RankIC is positive or
  materially better than baseline, leakage passes, coverage is adequate, and
  exposure gate behavior is explainable.

Default stop/fail gates:

- any future/GT field in decision-time evidence;
- H2026_1 negative or near-zero RankIC without a documented downgrade gate;
- missing coverage that changes the conclusion;
- exposure zero/tiny while report language claims stock-picking success;
- required secrets, paid calls, large rebuild, or SSH not explicitly approved.

## Decision Tree After Results

Positive result:

- If leakage passes, H2026_1/OOT RankIC is positive or better than baseline,
  coverage is adequate, and exposure gate behavior is sane, keep the tool as
  `observe` or cautiously `usable_default` depending on existing tests.
- If positive-rate improves above `60%` but RankIC, net spread/return,
  exposure, or leakage gates fail, do not promote it; record it as an
  overfit/base-rate/selection-risk result.
- Ask local audit to update `local_goal.md` with a next P0/P1 integration or
  small user-facing dry-run task.

Negative result:

- If RankIC fails, edge collapses, or exposure stays zero, do not spin it as a
  success. Record the failure as regime drift, score decay, or overfitting.
- Prefer diagnosing feature families and gate logic before adding new data or
  paid LLM runs.
- Keep P0/P1 at strong-yellow MVP with explicit caveat.

Blocked result:

- If required data/report paths are missing, credentials are unavailable,
  resource limits are too small, or instructions conflict, remote must output
  `LOCAL_AUDIT_REQUEST` with exact blockers and at least three next directions.
- Local audit then decides whether to regenerate a small artifact, narrow the
  task, or change direction.

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
reports/data. If Exact Next Task is not filled or if leakage/resource/blocker
rules trigger, stop and output LOCAL_AUDIT_REQUEST. Do not edit the three
local docs on the remote side.
```
