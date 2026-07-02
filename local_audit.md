# StockHome Local Audit Notes

Updated: 2026-07-02 (round 3: hot-rank available-at + after-cost re-audit)

Status: local-authored remote execution evidence map. This document is part of
the remote execution packet: remote Codex reads it before executing
`local_goal.md`, while local CC/Codex updates it between remote runs.

Remote Codex must not edit this file during execution. New evidence, blockers,
or corrections should be written to RUN_STATUS/reports/final output and
recommended back to local CC/Codex for the next update.

## Remote Status Summary

Round 3 (2026-07-02): a real remote `LOCAL_AUDIT_REQUEST` WAS provided. The
authoritative round-3 facts (ranker-guard done, broker closed, hk_hold partial,
hot-rank source-boundary, after-cost as the binding bottleneck) are in the
"Third-Round Local Audit Update" section below with exact SSH-verified paths. The
older summary here is retained as historical baseline:

The latest recorded remote-result facts are from `goal.md` and `docs/DECISIONS.md`:

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

## Second-Round Local Audit Update - 2026-07-02

Context: this is a from-scratch full local audit. No remote `LOCAL_AUDIT_REQUEST`
was provided this round (the previously mislabeled block was scLatent content).
Local audit = deep local clone read + read-only SSH verification of the current
server state + user ideas. SSH host `cyx-server-cfy`, path `/data/cyx/1030/stock`.

### Remote sync/state (verified 2026-07-02)

- Remote HEAD = `3346794` = local HEAD `334679491c5a458e1fa03964c5c1988b4da87a69`.
  Local clone is in sync with the remote tracked tree and GitHub.
- Remote dirty state: 21 untracked entries, unchanged in kind from the prior
  round: `4599041`, `anthropic_financial_services/`, `models/`, eight
  `scripts/run_*_v1.py` signal-family scripts, `scripts/run_p0_target60_walkforward.py`,
  `scripts/train_frozen_quant_score_v1.py`,
  `scripts/validate_p0_p1_latest_revalidation.py`, and matching `tests/test_*_v1.py`.
- Key finding: the prior `Exact Next Task` (signal inventory) has ALREADY been
  executed remotely and is server-local (git-ignored), so the local docs were
  a full round behind the real experiment state. New server-local products found:
  - `runs/stock_signal_inventory_20260701/RUN_STATUS.md` + `reports/date_generalization/stock_signal_inventory_20260701/REPORT.md`
    (inventory DONE, recommends next goal `P1 ranker integration with downgrade/exposure guard`,
    stops at `LOCAL_AUDIT_REQUEST`);
  - `runs/codex_goal_stock_newsignal_20260701/RUN_STATUS.md` +
    `reports/date_generalization/p0_target60_new_signal_family_20260701/` with a
    `rolling_autonomy_summary_20260701.md`: remote AUTONOMOUSLY pre-registered
    and evaluated 7 new signal families, ALL failed strict target60;
  - `reports/date_generalization/supervised_ranker_experiment_v2.md` (the single
    most strategically important artifact this round, see below).
- `remote_decision.md` exists on the server but only contains the template +
  `2026-07-02 initialized`; no remote goal-run decision entries yet.

### 20d-positive-return-rate DEFINITION verification (this round's headline)

The user's target "20 日正收益率 > 60%" was traced to its exact code/report meaning:

- `goal.md` itself does NOT contain a 60% hard target. It only says: do NOT claim
  "stable 20-day positive-return targets" (goal.md line 52-53). So 60% is a
  user-stated aspiration carried in `local_*.md`, not a durable goal-file gate.
- The operative metric everywhere in code/reports is `positive_20d_rate` =
  the fraction of decision rows whose 20-trading-day forward return is positive
  (a HIT RATE / win rate), NOT a 60% cumulative return. Ground truth is computed
  in `src/backtest/ground_truth.py`:
  `return_20d = round((close[idx+20]/close[idx] - 1)*100, 2)` and the target
  scripts count `positive_20d_rate = mean(return_20d > 0)`.
- The strict gate used by the remote target60 route is `positive_20d_rate > 0.60`
  (strict `>`, so a tied `0.6000` does NOT pass), plus `active_exposure >= 0.50`
  and leakage `PASS`. Source: `runs/codex_goal_stock_newsignal_20260701/RUN_STATUS.md`
  Hard Boundaries block.
- CONCLUSION on the ambiguity the local audit was asked to resolve: the target is
  the WIN-RATE reading, not a 60% cumulative-return reading. The win-rate reading
  is legitimate and publication-defensible ONLY when reported alongside (a) base
  rate of the block, (b) active_exposure, (c) net (after-cost) return / decile
  spread, and (d) no selection on the final OOT block. A raw 60% win rate on its
  own is gameable (high market base rate + tiny exposure fakes it). The 60%
  cumulative-return reading, if anyone ever intends it, is not realistic as a
  per-name 20-day expectation and should be explicitly rejected.

### Achievability verdict for the 60% win-rate target (evidence-backed)

On the latest OOT block H2026_1, the 60% win-rate target is currently NOT
achieved and the recent evidence argues it is not reachable with the present
feature set without leakage/selection tricks:

- target60 walk-forward (`p0_target60_codex_goal_stock_20260701/target60_report.md`):
  the pre-OOT-selected winner `regime_gating__frozen_score__aggressive__all_dates__top10pct`
  hit H2026_1 `positive_20d_rate=0.2414`, `active_exposure=0.6687`,
  `avg_return_20d=-3.6298`, `net_decile_spread=-2.9923`. Leakage PASS. The report
  correctly refuses to promote any better-looking H2026_1 row as that would be
  OOT selection.
- 7 autonomous new signal families (`rolling_autonomy_summary_20260701.md`): best
  achievable H2026_1 win rate across all families was a TIE at `0.6000` (fails
  strict `>0.60`), and every one of those tied configs has NEGATIVE net decile
  spread (e.g. news_event_catalyst_v1 `0.6000 / net -1.2912`,
  chip_pressure_absorption_v1 `0.6000 / net -1.8534`). Remote stop-rule fired:
  no remaining local-only route avoids completed-family reuse, OOT selection, or
  recent-loser reversal.
- So the honest present ceiling on H2026_1 is ~0.60 WITH negative after-cost
  spread, i.e. even the ceiling is not a real positive-return edge.

### Root-cause: the project's implicit alpha is cross-sectional REVERSAL, and it collapsed in H2026_1

Direct evidence from `reports/date_generalization/feature_rank_ic_audit.csv` and
`reports/date_generalization/supervised_ranker_experiment_v2.md`:

- Nearly all price/peer momentum features have NEGATIVE forward-20d RankIC
  averaged over blocks: `kline_return_20d` ALL_meanIC `-0.0604`,
  `corr_peer_avg_return_20d` `-0.0496`, `kline_return_5d` `-0.0438`,
  `corr_peer_relative_return_20d` `-0.0431`. Negative momentum IC = reversal is
  the edge.
- `supervised_ranker_experiment_v2`: a parameter-free `reversal_composite`
  baseline dominates all ML models OOS (OOS mean RankIC `0.1109`, ICIR `0.6353`,
  ic_pos_rate `0.7517`) versus best multi-feature model `0.0507`. This confirms
  the alpha is reversal, not a learned nonlinearity.
- BUT on final OOT H2026_1, `reversal_composite` collapses to RankIC `0.0054`
  (~zero) while gbdt/logistic tick up to `0.043 / 0.0364`. Many negative-IC
  momentum features flip sign in the H2026_1 column of the IC audit (e.g.
  `corr_peer_positive_breadth_20d` H2026_1 raw `+0.0453`). So the H2026_1 P0/score
  failure is signal-regime coupling (reversal turned off), not a bug.
- This also explains `frozen_quant_score_v1`'s fragility: it is negative RankIC in
  H2024_2 (`-0.106`) and H2025_2 (`-0.019`), near-zero in H2025_1 (`+0.010`), and
  only clearly positive in H2026_1 (`+0.0327`). Its "gate_pass_candidate" verdict
  rests almost entirely on the single final OOT block passing ICIR>=0.30 /
  ic_pos>=0.55 while the earlier blocks fail. This is close to single-block
  survival and should be treated as weak/observe, not a robust default.

### Cost/turnover is the real wall (highest-value new finding)

In `supervised_ranker_experiment_v2`, with round-trip cost modeled at 1.5%
(`net = gross - turnover*1.5%`), the after-cost top-k spread (`net_turnover`,
`net_flat`) is NEGATIVE for essentially every model in both the OOS and the
H2026_1 tables (e.g. reversal_composite OOS gross_topk `+0.6396` but net_turnover
`+0.0976` / net_flat `-0.8604`; H2026_1 net_turnover `-1.837`). The frozen score's
own top-bottom NET decile spread is `-2.5158pp` on H2026_1. Interpretation: even
where a gross ranking edge exists pre-cost, turnover eats it. Any target that
implies portfolio action MUST report after-cost numbers; the gross-only comfort
is misleading. This is the single strongest argument that chasing target60 on the
current feature set is a divergence trap.

### Backtest methodology audit (leakage / cost / statistics)

Positive controls found (the pipeline is more disciplined than typical):

- Explicit future-field blacklists exist in three independent places and agree:
  `src/agent_training/evidence_pack.py::FUTURE_RESULT_FIELDS`,
  `src/agent_training/quant_tool_context.py::FUTURE_RESULT_FIELDS`,
  `src/agent_training/date_regime_gate.py::FUTURE_FIELD_BLACKLIST`
  (return_5/10/20d, future_*, gt_status, positive_20d, top_decile_flag, etc.).
- `scripts/train_frozen_quant_score_v1.py` is methodologically clean: single fixed
  model family (HistGradientBoostingClassifier), single feature group `wide_safe`,
  label = `return_20d>0` used offline only, `assert_no_future_fields` import from
  `audit_p0_decision_stack_v1`, leakage check raises on any hit, walk-forward
  `_rolling_split`, and the docstring/`split_boundary` state H2026_1 is never used
  to pick hyperparameters or thresholds.
- Time blocks are fixed calendar halves H2023_1..H2026_1 (`date_regime_gate.py`),
  H2026_1 is the frozen final OOT, exposure gate presets are fit on 2023-2025 only.
- Leakage audits report `PASS / hits=0` on every recent run (frozen score,
  target60, all 7 families, P0 revalidation round-2). No future/GT field was found
  in evidence packs or rule outcomes.

Remaining methodology risks / gaps to watch:

- Selection-on-final-OOT risk is understood and guarded in the target60/family
  reports (pre-OOT selection, refusing to promote H2026_1-best rows), but the
  frozen score's usable_in_agent_default=true relies on the final block being the
  only strong one; this is a soft form of surviving on one OOT block and should be
  downgraded to observe/limited.
- Financial-report channel: as-of 90-day GT match rate is only `0.0231`
  (`docs/DATA_FLOW.md`), high-risk group stats cover as few as 3 stocks; not yet
  independent alpha, correctly framed as risk/uncertainty channel only.
- News/announcement availability: same-day items without an intraday timestamp are
  conservatively made available next natural day (documented), but the news/event
  catalyst and peer-cohesion families still need explicit available-at/lag proof
  before any of their fields enter decision-time evidence (inventory marks these
  `needs_leakage_audit`).
- Statistical significance: H2026_1 has ~38 decision dates / ~18.7k rows per block;
  a single-block RankIC of `+0.033` with ICIR `0.42` is not a large-sample proof,
  especially against three earlier blocks that fail. No formal significance test
  (e.g. Newey-West t on daily IC) is reported; recommend adding a simple IC t-stat.
- `src/backtest/engine.py` + `ground_truth.py` compute forward returns by integer
  index offset on daily close; this is fine for a fixed universe but does not model
  slippage, price limits (涨跌停), suspension, or liquidity/tradability filters at
  fill time. The v2 ranker report adds a flat 1.5% round-trip cost but not limit/
  liquidity constraints; treat portfolio-level numbers as upper bounds.

### User "quant scoring tool" idea — current-state audit

There IS already a frozen quant score: `models/frozen/quant_score_v1/` (a
serialized `model.joblib` + `model_card.md` + `feature_list.json` +
`train_blocks.json`), trained by `scripts/train_frozen_quant_score_v1.py`, scored
by walk-forward RankIC. It is leakage PASS and its training/eval/freeze protocol
is sound. So the user's "train a scoring tool and freeze it" idea is largely
ALREADY IMPLEMENTED at v1. What is missing is not a new model but: (1) honest
accuracy is weak and single-block-dependent; (2) after-cost spread is negative;
(3) it is not yet wired as the agent's decision anchor with a downgrade guard; and
(4) there is no small decision-support network yet (correctly gated behind an
available-at audit of news/peer/financial features). The right next step is
INTEGRATION + GUARD + honest re-labelling of the frozen score as an observe-grade
P1 rank aid, not training more models.

### Local/remote checks run this round

- Read locally: `goal.md`, all three `local_*.md`, `README.md`, `AGENTS.md`,
  `docs/START_HERE.md`, `docs/DECISIONS.md`, `docs/PROJECT_REVIEW.md`,
  `docs/DATA_FLOW.md`, `docs/RESPONSE_PROTOCOL.md`; source modules
  `evidence_pack.py`, `date_regime_gate.py`, `quant_tool_context.py`,
  `backtest/scoring.py`, `backtest/engine.py`, `backtest/ground_truth.py`;
  Glob over `src/`, `scripts/` (163), `tests/` (128), `config/` (20).
- SSH read-only on `/data/cyx/1030/stock`: HEAD/status/log; `models/frozen/quant_score_v1/model_card.md`;
  `reports/date_generalization/frozen_quant_score_v1_accuracy.md`;
  `reports/date_generalization/p0_target60_codex_goal_stock_20260701/target60_report.md`;
  `runs/stock_signal_inventory_20260701/RUN_STATUS.md` + inventory REPORT.md +
  recommended_next_goal.json; `runs/codex_goal_stock_newsignal_20260701/RUN_STATUS.md`;
  `rolling_autonomy_summary_20260701.md`; `supervised_ranker_experiment_v2.md`;
  head of `feature_rank_ic_audit.csv`; `scripts/train_frozen_quant_score_v1.py` head;
  `remote_decision.md`.
- No experiments, training, GPU, paid API, or writes to the remote were run.

## Third-Round Local Audit Update - 2026-07-02 (hot-rank available-at + after-cost re-audit)

Context: remote provided a real `LOCAL_AUDIT_REQUEST` this round. Local audit did
read-only SSH verification at HEAD `2a5a2d4` (in sync with local; large dirty
overnight worktree present, NOT pulled). User decision: run tracks (a) hot-rank
A-share available-at audit + (c) after-cost protocol re-audit in parallel, and
local audit must also propose stronger strategies.

### Remote sync/state (verified 2026-07-02, round 3)

- Remote HEAD = `2a5a2d4` = local HEAD. In sync with GitHub tracked tree.
- Remote dirty worktree is LARGE (overnight autonomous work). Tracked modifications:
  `README.md`, `docs/HANDOFF.md`/`PROJECT_ENTRY.md`/`PROJECT_REVIEW.md`/`START_HERE.md`/
  `USER_GUIDE.md`, `goal.md`, `remote_decision.md`, `scripts/build_tushare_cache.py`,
  `scripts/run_agent_strategy_training_rounds.py`, `scripts/run_full_channel_ablation_round.py`,
  and ~10 `src/agent_training/*.py` modules + ~12 `tests/test_*.py`.
- Many NEW untracked scripts this round (all round-3 audit/build/run families):
  `scripts/audit_broker_recommend_full_month_coverage.py`,
  `scripts/audit_candidate_family_available_at_lag.py`,
  `scripts/audit_hk_hold_availability_cadence.py`,
  `scripts/audit_tushare_specialty_lag_coverage.py`,
  `scripts/inventory_tushare_specialty_endpoints.py`,
  `scripts/build_broker_recommend_conservative_join_audit.py`,
  `scripts/build_hk_hold_d1_join_quality_audit.py`,
  `scripts/build_hk_hold_nonempty_calendar_join_rerun.py`,
  `scripts/build_live_tushare_feature_matrix_{plan,preview}.py`,
  `scripts/build_live_tushare_lag_preflight.py`,
  `scripts/run_after_close_cost_aware_small_model_scout.py`,
  `scripts/run_after_close_low_frequency_mutation.py`,
  `scripts/run_after_close_regime_exclusion_mutation.py`,
  `scripts/run_after_close_turnover_guard_mutation.py`,
  `scripts/run_after_close_score_persistence_family.py`, plus many `audit_*`/`build_*`
  scripts and untracked `models/`, `anthropic_financial_services/`, `4599041`.

### Dirty worktree ownership judgment (round 3)

- The new `audit_*`/`build_*`/`run_after_close_*` scripts + matching `tests/test_*`
  are ACTIVE SOURCE for live source-boundary audits (broker/hk_hold/hot-rank) and
  after-close cost/turnover/regime mutations. They should eventually be curated
  into Git as small source (they are the round-3 experiment machinery). Do NOT
  discard them; they are the record of the autonomous exploration.
- `models/`, generated `reports/`/`runs/`, `anthropic_financial_services/`, and
  `4599041` remain SERVER-LOCAL evidence (git-ignored); their absence in the local
  clone is not proof they never existed. Register them, do not deep-read.
- The tracked-file modifications to docs + `goal.md` + `remote_decision.md` +
  `src/agent_training/*` are remote's own in-flight edits; local audit treats them
  as evidence of active work, does NOT pull/reset, and keeps `goal.md` (durable)
  and the three `local_*.md` as the authoritative local-owned surfaces.

### New-source result matrix (exact paths, verified by SSH)

THREE numbers must be audited SEPARATELY and never conflated: raw endpoint rows
!= A-share joinable rows (after suffix/market filter + universe join) != decision-
universe match rate. Every source below is reported in those terms.

- P1 ranker-guard integration (DONE) —
  `reports/date_generalization/p1_ranker_guard_integration_20260702/validation_summary.md`:
  - frozen_quant_score_v1 H2026_1 RankIC `0.0327`, ICIR `0.4290`, ic_pos `0.6316`,
    top-vs-pool net turnover `-1.2092`, top-bottom net `-1.7033`, guard `observe_only`.
  - reversal_composite H2026_1 RankIC `0.0054`, ICIR `0.0545`, ic_pos `0.5263`,
    top-vs-pool net `-1.8370`, top-bottom net `-2.5959`, guard `suppress`.
  - ONE non-negative row in the family scan: `quality_momentum_accumulation_v1`
    H2026 net `+1.2098`, but `not_promoted` (direct target60 `0.4444`, needs
    available-at/lag audit before feature reuse).
  - Exposure guard summary: H2026_1 mean exposure scale `0.3404` (25 abstain /
    12 half / 10 deploy) — research-only, does NOT override the negative after-cost
    guard. Leakage audit: forbidden future/result fields in sanitized output = 0.
  - Metric-protocol addendum: historical rows in `after_cost_net_spread_by_block.csv`
    use the serialized final model per block, while `frozen_quant_score_v1_accuracy.csv`
    is the authoritative walk-forward protocol (historical_mismatch_count `3`,
    H2026 `protocol_consistent`). Do NOT cite optimistic serialized-model history as
    promotion evidence.
- broker_recommend (CLOSED as direct sparse selector) —
  `reports/date_generalization/p1_broker_recommend_sparse_label_gate_20260702/` +
  `remote_decision.md` (2026-07-02 AUTONOMOUS_DECISION):
  - PRE_H2026: universe 16950, active 547 (rate `0.0323`), active positive `0.4552`
    vs base `0.4827` (lift `-0.0274`), after-cost spread `-2.2895pp`, gate FALSE.
  - H2026_1: universe 2455, active 60 (rate `0.0244`), active positive `0.4000` vs
    base `0.4713`, after-cost spread `-4.0266pp`, diagnostic gate FALSE.
  - by_block: after-cost spread negative in 6 of 7 blocks (only H2024_1 `+1.44`).
  - Status `BROKER_RECOMMEND_SPARSE_LABEL_GATE_NOT_PROMOTED`; auxiliary-context
    feature only. Useful NEGATIVE evidence.
- hk_hold (NOT a clean close; partially joinable, coverage too holey now) —
  `reports/date_generalization/p1_hk_hold_d1_join_quality_audit_20260702/`:
  - source_rows `41326` -> feature_rows `11312` -> mean overall match rate `0.3916`;
    min_block_match_rate `0.0`; contains_labels `False`; modeling_allowed_now `False`.
  - cadence probe (`p1_hk_hold_availability_cadence_20260702`): probe_dates 15,
    nonempty 13, sparse_blocks `['H2024_2','H2025_2']`. Zero coverage was snapshot
    cadence, not pure code mapping. Still not modelable yet.
- hot-rank ths_hot / dc_hot (source-boundary only; the target of track (a)) —
  `reports/date_generalization/p1_tushare_specialty_lag_coverage_20260702/`
  (`specialty_lag_coverage_by_endpoint.csv`):
  - `ths_hot`: has_stock_identifier True, `has_rank_time=True`, nonempty blocks
    H2024_1/H2024_2/H2025_1/H2025_2 (4), `coverage_ok_for_next_cache_design=False`,
    decision `insufficient_sample_coverage_for_next_cache_design`.
  - `dc_hot`: has_stock_identifier True, `has_rank_time=True`, nonempty blocks
    H2024_2/H2025_1/H2025_2 (3), `coverage_ok=False`, same insufficient-coverage
    decision.
  - Both mix A-share / HK / US / concept-board rows -> a market/`data_type`/
    `ts_code`-suffix filter + rank_time/D+1 audit is required before any label use.
    No dedicated ths_hot/dc_hot report dir yet — track (a) is exactly that audit.
- moneyflow_hsgt — same by_endpoint CSV: `has_stock_identifier=False` ->
  `regime_context_only_not_stock_selector`. Not a cross-sectional selector; usable
  as a REGIME/TIMING overlay signal (supports the regime-gating strategy).
- broker_recommend / hk_hold in that probe: `has_rank_time=False` -> need an
  explicit release-time policy before modeling.

### COMMON FAILURE MODE (the real bottleneck this round)

After-cost net spread is the FIRST bottleneck and it is negative across the frozen
score, the reversal baseline, the broker sparse selector, and most blocks. Every
newly explored source / small model / sparse selector goes negative once the 1.5%
round-trip cost is subtracted. Gross-only comfort is misleading. This is why the
user asked to re-audit the after-cost protocol (track c): confirm whether the cost
convention is fair or artificially pinning strategies dead, and whether a lower-
turnover / monthly-rebalance / netting convention flips any block non-negative.

### Remote's Coordination Note to local CC (must be acted on)

The ranker-guard `validation_summary.md` contains an explicit note: remote will not
mechanically follow the three local docs when new evidence shows a route conflict;
it owns independent decision space, logs divergences in `remote_decision.md`, and
asks local CC for "concise systemic feedback when near-end audit patterns repeatedly
miss exposure, after-cost, availability/lag, or promotion-boundary issues." Local
audit ACCEPTS this: this round the packet foregrounds (1) the three-number
availability separation, (2) after-cost as the binding gate, (3) explicit
promotion-boundary language, and (4) anti-stop / initiative principles so remote is
never nudged to stop on a negative after-cost result.

### Local/remote checks run this round

- Read locally: `goal.md`, all three `local_*.md` (pre-round-3 versions).
- SSH read-only on `/data/cyx/1030/stock`: HEAD/status (large dirty worktree);
  `remote_decision.md` tail (broker close decision); ranker-guard
  `validation_summary.md` + `RUN_STATUS.md`; broker gate `summary_gates.csv` +
  `by_block.csv`; hk_hold `p1_hk_hold_d1_join_quality_audit_20260702/validation_summary.md`
  + `p1_hk_hold_availability_cadence_20260702/validation_summary.md`; specialty
  `p1_tushare_specialty_lag_coverage_20260702/validation_summary.md` +
  `specialty_lag_coverage_by_endpoint.csv`; script/report inventory greps.
- No experiments, training, GPU, paid API, or writes to the remote were run. Some
  deep report bodies (full CSVs) were only head/tail-sampled — flagged below.

## Open Questions For Next Local Audit

- Should `frozen_quant_score_v1` be relabelled from `usable_in_agent_default=true`
  to `observe_only`, given its edge lives on a single OOT block and net spread is
  negative? (Local audit leans yes.)
- After-cost is the binding constraint: is there ANY decision-time feature family
  whose net (post-1.5%) top-decile spread is positive on H2026_1? If not, the
  honest product claim is ranking-reference only, never active-buy.
- Do the news/event and peer-cohesion families have provable available-at/lag
  semantics, or must they stay out of decision-time evidence permanently?
- Should the user's 60% win-rate target be formally re-scoped in `local_goal.md`
  to "OOT RankIC + after-cost spread + exposure-gated win rate" so remote stops
  chasing a strict `>0.60` raw rate? (Local audit leans yes; done this round.)
- Which of the 21 untracked server items should be curated into Git as small
  source (the `*_v1.py` signal scripts + tests look like real source), and which
  stay server-local (reports/runs/models/`4599041`/external refs)?

## Prior Open Questions (resolved this round)

- "Has the one-frozen-score direction been implemented remotely?" -> YES,
  `models/frozen/quant_score_v1/` exists and is leakage-clean; it is weak/
  single-block, not a robust default.
- "Which minimum H2026_1 gate before a tool becomes default?" -> proposed:
  H2026_1 RankIC>0 AND ICIR>=0.30 AND positive-fraction>=0.55 AND after-cost
  top-decile spread not materially negative AND exposure explainable; otherwise
  `observe_only`. frozen v1 fails the after-cost leg, so `observe_only`.
