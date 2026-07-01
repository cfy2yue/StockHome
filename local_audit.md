# StockHome Local Audit Notes

Updated: 2026-07-01

Status: local-authored remote execution evidence map. This document is part of
the remote execution packet: remote Codex reads it before executing
`local_goal.md`, while local CC/Codex updates it between remote runs.

Remote Codex must not edit this file during execution. New evidence, blockers,
or corrections should be written to RUN_STATUS/reports/final output and
recommended back to local CC/Codex for the next update.

## Remote Status Summary

No new remote `LOCAL_AUDIT_REQUEST` was provided for this documentation pass.
This pass used current repository docs and source/report indexes only. The
latest recorded remote-result facts are from `goal.md` and `docs/DECISIONS.md`:

- H2026_1 latest-block P0 revalidation did not confirm the frozen P0.
- Round-2 Flash confirmation had full card coverage (`ok_cards=24/24`) and
  leakage PASS (`future_leak_findings=0`).
- P0 20d positive rate was `0.6667` versus frozen `0.8434`; avg20 was
  `+0.5563pp` versus frozen `+1.9139pp`.
- `exposure_cards=0`, so this is defensive behavior, not evidence of active
  stock-picking skill.

## Durable Goal Interpretation

The project can make research-oriented action suggestions, but only as a
research assistant with evidence, counter-evidence, position/risk limits,
triggers, and review conditions. It must not place orders, connect to brokers,
promise returns, or leak future labels into decision-time evidence.

The current product scope is:

- P0: single-stock watch/review action cards.
- P1: small candidate-set comparison/ranking.
- P2: backtest/portfolio/tool research as supporting evidence only.

The current scientific target is not "prove broad 20d buy signal works." It is
"freeze a reproducible, leakage-free score/ranker and report honest out-of-time
ranking accuracy plus an exposure/IC gate."

The user's current alpha preference is still concrete: try to reach a 20-day
positive-return rate above `60%`, with strategy guidance. Local audit should
preserve that as the aspirational final target, but never as a standalone gate:
it must be interpreted with exposure, net return/spread, OOT block choice,
base-rate comparison, and leakage/available-at checks.

## Data Flow The Auditor Must Understand

StockHome has four practical input streams:

- Market/price/volume stream: `mootdx/pytdx`, BaoStock, AKShare/efinance, and
  local caches. `mootdx/pytdx` is quote-protocol data, not exchange-direct raw
  tick data.
- News/announcement stream: official disclosure when available, public
  aggregators such as AKShare/Eastmoney as supplements, and authorized offline
  paid/standardized caches. Reports must separate fact, interpretation,
  speculation, and unverified items.
- Financial report/event stream: official disclosure or authorized offline
  caches with `ann_date`/available-at boundaries. Missing disclosure date or
  available-at means the row cannot enter walk-forward decision evidence.
- Quant/current-data stream: valuation, financial, industry/peer, K-line,
  chip/cost, news questionnaire, and ML score fields. Model-estimated items
  such as capital flow must be labeled `model_estimate`.

Decision-time evidence packs are assembled in `src/agent_training/evidence_pack.py`.
Quant tool outcomes are sanitized in `src/agent_training/quant_tool_context.py`.
Offline labels such as future returns may be used only in audits/training
reports, not in evidence packs or remote prompts.

## Metric And Backtest Flow

Important evaluation concepts:

- `return_20d` and related GT/future fields are offline labels only.
- RankIC is the preferred accuracy metric for P1/ranker work. It should be
  reported by date/block and out-of-time block, especially H2026_1.
- Positive 20d rate alone is risky because high base rates and low exposure can
  create fake comfort.
- Exposure metrics must be read with performance metrics. `exposure_cards=0`
  is a defensive/no-action result, not a positive stock-selection result.
- Cost/turnover and decision frequency matter for portfolio/backtest P2
  summaries; high turnover gross lift is not enough.

Key scripts and modules to inspect before tasking remote:

- `scripts/audit_feature_rank_ic.py`
- `scripts/audit_evidence_pack_leakage.py`
- `scripts/run_supervised_ranker_experiment.py`
- `scripts/run_kline_peer_chip_regime_scorer.py`
- `scripts/run_date_regime_gate_experiment.py`
- `scripts/audit_kline_peer_chip_turnover_cost.py`
- `scripts/analyze_agent_veto_reasons.py`
- `scripts/summarize_financial_report_ablation.py`
- `src/agent_training/date_regime_gate.py`
- `src/agent_training/evidence_pack.py`
- `src/agent_training/quant_tool_context.py`
- `src/agent_training/decision_card.py`
- `src/world_model/financial_report_channel.py`
- `src/backtest/scoring.py`
- `src/backtest/engine.py`

Key tests to inspect when changing behavior:

- `tests/test_evidence_pack_leakage_audit.py`
- `tests/test_agent_training_phase1.py`
- `tests/test_quant_agent_tools.py`
- `tests/test_date_regime_gate_exposure_guard.py`
- `tests/test_kline_peer_chip_regime_scorer.py`
- `tests/test_kline_peer_chip_turnover_cost.py`
- `tests/test_backtest_light.py`
- `tests/test_financial_report_channel.py`
- `tests/test_financial_safety_hygiene_gates.py`

## Existing Key Conclusions

- P0 latest block is not confirmed. Keep the strong-yellow MVP only with a
  regime-drift caveat.
- The older frozen P0 appears tied to cross-sectional reversal behavior; this
  edge collapsed in H2026_1 according to the recorded audit direction.
- P1/ranker-anchor comparison is a more honest place to measure alpha via
  ranking accuracy than P0 absolute timing.
- Financial report events are useful as high-trust support/risk/uncertainty
  channels, but prior coverage was sparse and not yet proof of independent
  alpha.
- News and announcement channels may act more as risk interceptors and
  confirmation context than as standalone positive-return sources.
- Generated reports/runs are mostly server-local/ignored; absence in the local
  GitHub clone is not proof they never existed. When a result becomes important
  to the next task, local audit should inspect it by SSH if needed and sync
  high-signal metadata into Git-tracked docs or the next `local_*.md` packet.

## Risk Checklist

Must check these risks before approving any remote task:

- Leakage: no future returns, `future_*`, `return_20d`, `gt_status`, outcome
  labels, or after-the-fact event fields in evidence packs, rule outcomes,
  prompts, or user-facing report reasoning.
- Future availability: financial/news/announcement rows must have disclosure or
  available-at dates. Same-day data without timestamp should be conservatively
  available no earlier than the next natural day.
- Low exposure: if exposure is zero or tiny, treat results as defensive/no
  decision. Do not call this stock-picking ability.
- Coverage: report ok/invalid/missing cards, data-source failures, field
  missingness, and whether coverage changes the conclusion.
- Base-rate artifact: compare to block/date baselines and rank metrics, not only
  raw positive rate.
- Regime drift: H2026_1 failure means new claims must survive fresh or latest
  block checks.
- Credential safety: never print, quote, copy, commit, or put secrets in
  prompts/reports/logs; paid data must come from ignored local secrets or
  offline caches only.
- Trading-advice boundary: outputs may recommend research actions with risk
  limits, but must not connect to brokers, place orders, promise returns, or
  state certainty.
- Archive confusion: `docs/archive/legacy_auto_coordination_20260701/` is
  historical evidence only.

## Must-Check Report Paths

When available on the remote/server workspace, inspect:

- `runs/20260701_p0_p1_latest_revalidation_v1/RUN_STATUS.md`
- `reports/date_generalization/20260701_p0_p1_latest_revalidation_v1/validation_summary.md`
- `reports/date_generalization/20260701_p0_p1_latest_revalidation_v1_round2_flash_confirm/validation_summary.md`
- `reports/date_generalization/feature_rank_ic_audit.csv`
- `reports/date_generalization/quant_tool_rule_outcomes.jsonl`
- `reports/latest/multisource_data_smoke.md`
- `reports/date_generalization/financial_report_channel_coverage.md`

If any of these are absent in the local clone, do not recreate them during a
documentation audit. Ask remote to summarize or regenerate only under a bounded
filled task.

## User Ideas Considered

The user's preferred optimization is to train and freeze quantitative scoring
tools, then have the agent make decisions based on the ML/quant score. Local
judgment: this is the right direction if kept lightweight, deterministic,
leakage-free, and measured by out-of-time RankIC plus exposure gating. It is
better than an opaque end-to-end agent judgment or chasing a raw 20d positive
rate target.

The 20d `>60%` target is useful as an intuitive user-facing ambition, but it is
too noisy and gameable as the only hard metric. It can pass only when the score
also has nontrivial exposure/coverage, no final-OOT selection, leakage PASS,
and at least neutral-to-positive net spread or return behavior.

Complementary ideas:

- small information-aggregation networks are acceptable only if reproducible
  and auditable;
- statistical analysis of existing results is high-value and low-risk;
- paid/LLM expansion should wait until a frozen local metric justifies it.

## Local Checks Run In This Pass

Read-only/small checks only:

- listed repository root and `docs/`;
- read `README.md`, `goal.md`, `local_goal.md`, `local_audit.md`,
  `local_suggestion.md`, `docs/START_HERE.md`, `docs/DECISIONS.md`,
  `docs/AUDIT_DIRECTION_20260701.md`, `docs/PROJECT_REVIEW.md`,
  `docs/GITHUB_FILE_MAP.md`, `docs/DATA_FLOW.md`, `docs/DATA_SOURCE_POLICY.md`,
  `docs/RESPONSE_PROTOCOL.md`, and `AGENTS.md`;
- grep-searched for P0/P1, RankIC, leakage, exposure, H2026, score, and
  validation paths under docs/src/scripts/tests;
- skimmed the heads of `evidence_pack.py`, `quant_tool_context.py`,
  `date_regime_gate.py`, `audit_feature_rank_ic.py`,
  `audit_evidence_pack_leakage.py`, `run_supervised_ranker_experiment.py`, and
  `run_kline_peer_chip_regime_scorer.py`.

No experiments, tests, or remote/API calls were run in this pass. Future local
audits may use SSH for read-only evidence checks when the local clone lacks
server-local metadata.

## First-Round Local Audit Update - 2026-07-01

User context: first manual local-audit round for all three projects; no new
remote `LOCAL_AUDIT_REQUEST` was provided. Local audit used SSH read-only
checks on `/data/cyx/1030/stock` for server-local metrics and dirty state.

Remote sync/state:

- Historical snapshot HEAD when this first-round evidence was pulled:
  `d9e5e09`. Current remote execution-packet commit after catchup is
  `1a59862`.
- Remote dirty state: 21 untracked entries. Key roots/items:
  `4599041`, `anthropic_financial_services/`, `models/`,
  `scripts/run_p0_target60_walkforward.py`,
  `scripts/train_frozen_quant_score_v1.py`,
  `scripts/validate_p0_p1_latest_revalidation.py`, seven new
  `scripts/run_*_v1.py` signal-family scripts, and matching `tests/test_*_v1.py`
  files.
- These files must be inventoried and classified before any local docs treat
  them as project source. External/reference directories should be registered
  but not deeply read unless needed to explain a local signal artifact.

Remote evidence verified:

- `runs/frozen_quant_score_v1_20260701_145932/RUN_STATUS.md` finished with
  leakage PASS and gate PASS under its ICIR/IC-positive rules.
- `reports/date_generalization/frozen_quant_score_v1_accuracy.md`:
  H2026_1 RankIC mean `0.0327`, ICIR `0.4233`, IC-positive fraction `0.6316`,
  AUC `0.5253`, precision at top decile `0.3757`, and top-bottom decile net
  spread `-2.5158pp`. Local interpretation: usable only as a limited P1 ranking
  reference after human review; not a P0 buy signal or return-positive promise.
- `reports/date_generalization/p0_target60_codex_goal_stock_20260701/target60_report.md`:
  selected-by-pre-OOT strategy
  `regime_gating__frozen_score__aggressive__all_dates__top10pct` reached
  H2026_1 positive rate `0.2414`, active exposure `0.6687`, avg20 `-3.6298`,
  net decile spread `-2.9923`. The report correctly marks `DECISION_NEEDED`
  because using another H2026_1 row would be OOT selection.
- Earlier P0/Flash `exposure_cards=0` remains a defensive/no-action result,
  not evidence of stock-picking skill.
- Older docs such as `docs/USER_GUIDE.md`, `docs/HANDOFF.md`, and
  `docs/PROJECT_ENTRY.md` may still describe earlier P0/P1 capabilities more
  optimistically. Treat them as historical or superseded for current execution
  when they conflict with `goal.md`, `local_goal.md`, this audit, or
  `local_suggestion.md`.

Local subagent review:

- StockHome independent subagent agreed the first remote goal is appropriate:
  inventory and triage the dirty signal/model/test workspace, not another
  target60 chase.
- Required tightening from subagent has been applied to `local_goal.md`: only
  the named run/report output directories may be written; import/test
  collection must not trigger data/network/credential/large-compute side
  effects; the untracked roots above must be covered.

## Open Questions For Next Local Audit

- Which latest server report is authoritative if report paths differ from the
  GitHub clone?
- Has the one-frozen-score direction already been implemented remotely, or is
  it still only a decision note?
- What minimum H2026_1 RankIC/exposure gate should be required before any tool
  becomes default P0/P1 context?
- Which generated artifacts, if any, should be summarized into Git-tracked docs
  without committing raw reports/data?
