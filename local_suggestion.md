# StockHome Local Suggestions For Next Remote Round

Updated: 2026-07-02 (round 3: hot-rank available-at + after-cost re-audit)

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

## Round-3 Strategy Portfolio (accurate screening + user guidance)

Product goal: find a strategy that delivers ACCURATE stock screening and ACCURATE
guidance for the user watching stocks. after-cost alpha is the real wall — but a
negative result is NOT a stop; carry a PORTFOLIO of parallel strategies and keep
advancing. Pick / interleave / spawn subagents; a wall on one => move to another.

Parallel strategy directions:

1. hot-rank A-share available-at audit (a): data_type / market filter, rank_time / D+1
   policy, no-label source contract before any label work.
2. after-cost protocol re-audit (c): is 1.5% round-trip / turnover / exposure aligned
   with the user's real holding horizon and cost? Test whether the cost model is
   "self-strangling" a real edge.
3. Cost-aware construction: lower turnover, monthly rebalance, position netting and
   sizing so a real edge can survive cost.
4. Frozen scorer as a RANKING PRIOR (not a direct selector) inside an exposure-gated,
   cost-aware selection.
5. Regime / timing overlay: reversal alpha collapsed in H2026 — test regime-conditional
   deployment (turn the signal off in adverse regimes) rather than always-on.
6. Signal breadth with available-at discipline: fundamentals as-of, analyst revisions,
   industry / peer flows, northbound / holdings — each gated by an available-at audit.
7. Product-value decoupling: deliver the honest P0/P1 research-assistant value (action
   cards, evidence, counter-evidence, risk / position guidance, ds-LLM reasoning) that
   helps the user WATCH stocks accurately, even before an after-cost selector edge
   exists — this directly serves "accurate guidance".
8. Risk-management framing: accurate guidance may be risk-aware position / entry-exit
   advice (drawdown control, invalidation triggers), not only cross-sectional selection.

Standing gates (claim-bars, not stop conditions): leakage PASS (no future/GT fields),
after-cost net spread reported, exposure / coverage honest (low/zero exposure is not
skill), 60% target NOT lowered (only measured honestly). gate-fail => log + switch
strategy, never `BLOCKED`.

## Top Priorities

Update 2026-07-02 (round 3): the P1 ranker-guard integration is DONE (frozen score
downgraded to `observe_only`, reversal to `suppress`, both after-cost negative on
H2026_1). The direct `broker_recommend` sparse selector is CLOSED. So priority
shifts to (a) a hot-rank A-share available-at contract, (c) an after-cost protocol
re-audit, and continuously shipping P0/P1 product value — with remote exercising
initiative and never stopping on a negative after-cost result.

0. ANTI-STOP / INITIATIVE (overrides everything below): a failing strategy or a
   negative after-cost result is NOT `BLOCKED`. Remote should think, launch
   read-only subagents to audit/optimize the route, switch strategy, and keep
   advancing the durable 60% goal, logging decisions in `remote_decision.md`. Hard
   `BLOCKED` is only for user-owned decisions (goal/threshold change, new
   credentials, destructive ops, broker/live trading).
1. AFTER-COST IS THE REAL BOTTLENECK, and its PROTOCOL is now itself under audit
   (track c). The flat 1.5%-every-rebalance + 1.5% floor made the frozen score, the
   reversal baseline, AND the broker selector negative on H2026_1 and most blocks.
   Re-audit whether this convention is fair or artificially pinning strategies dead
   (turnover accounting, double-counted floor, rebalance cadence vs a realistic
   monthly/low-churn product). Do not silently weaken the gate — propose the change
   with evidence.
2. HOT-RANK AVAILABLE-AT CONTRACT FIRST, no label, no model (track a). `ths_hot`
   and `dc_hot` carry `has_rank_time=True` but thin coverage and mixed markets.
   Require a market/`data_type`/`ts_code`-suffix filter + rank_time/D+1 audit that
   separates the THREE numbers (raw endpoint rows / A-share joinable rows / decision-
   universe match rate). Only after filter + rank_time + adequate A-share coverage
   does a source become a label candidate — otherwise D+1-contract or CLOSED.
3. Do NOT suggest a direct `broker_recommend` sparse selector again (closed:
   after-cost `-2.29pp` / `-4.03pp`). It stays auxiliary-context only. Do NOT re-run
   target60, the 7 failed families, or a direct broker selector.
4. Keep `frozen_quant_score_v1` as `observe_only`; propose the model-card relabel
   (`usable_in_agent_default: true -> observe_only`). Prefer using it as a RANKING
   PRIOR inside a cost-aware, exposure-gated selection, not as a direct selector.
5. Re-scope the 60% target the same way (WIN RATE, exposure-gated, above base rate,
   after-cost non-negative, RankIC/ICIR gate, no final-OOT selection). The target
   itself is DURABLE; only the honest measurement/route changes.
6. Ship P0/P1 product value INDEPENDENT of after-cost alpha: an honest research
   assistant (single-stock action card + guarded candidate ranking with evidence,
   counter-evidence, exposure downgrade, coverage) is valuable even with no
   after-cost selector edge. Decouple these two deliverables.
7. Credential/evidence hygiene: no secrets in Git, prompts, reports, ledgers, logs.
8. New label candidates still require an available-at/lag audit before any field
   enters decision-time evidence (news/event, peer-cohesion, hot-rank, hk_hold).

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

## Recommended Next Remote Task (this round: hot-rank available-at + after-cost re-audit)

The filled `Exact Next Task` in `local_goal.md` this round is
`p1_hotrank_availableat_and_aftercost_reaudit_20260702` — a PARALLEL long-running
stage running tracks (a) and (c) plus continuous product value. Pattern:

```text
Read README.md, docs/START_HERE.md, goal.md, local_goal.md, local_audit.md,
local_suggestion.md, remote_decision.md, AGENTS.md, docs/DECISIONS.md.

Do NOT re-run inventory / target60 / the 7 failed families / a direct broker
sparse selector (all done or closed). After-cost failing is NOT blocked — log it,
switch strategy, continue. Run in parallel:

(a) hot-rank A-share available-at audit (NO label, NO model) for ths_hot + dc_hot:
    filter by data_type/market and ts_code suffix; report THREE separate numbers
    per block (raw endpoint rows / A-share joinable rows / decision-universe match
    rate); audit rank_time vs the trade decision and anchor to D+1 if intraday/
    after-close; emit available_at_policy, coverage_by_block, source_semantic_
    contract; verdict usable / D+1 / CLOSED per source.
(c) after-cost protocol re-audit: check turnover accounting, whether the flat-1.5%
    floor double-counts, and rebalance cadence vs a realistic monthly/low-churn
    product; recompute the frozen score after-cost net spread under monthly/lower-
    turnover/netting; state whether the 1.5%-every-rebalance protocol is over-strict
    and recommend the accepted cost/holding-period convention.
Also keep shipping the honest P0/P1 research assistant regardless of after-cost
alpha, and try at least one strategy from the exploration list below, logging the
choice in remote_decision.md.
Produce the dated run/report dirs, the (a)/(c) tables, a leakage PASS, and one
next stage. tushare specialty pulls (metadata/coverage only in track a) and
ds/DeepSeek reasoning are authorized under /data/cyx/1030/api; keep cost bounded.
```

The exact inputs/outputs/limits are in `local_goal.md` `Exact Next Task`. Remote
may adapt the strategy list and record divergences in `remote_decision.md`, but
must not silently weaken any gate or claim.

## Additional Strategy Exploration (local audit's proactive proposals — parallel)

Local audit was explicitly asked to propose stronger strategies. Offer these as
PARALLEL hypotheses; remote picks/adapts/replaces and logs choices in
`remote_decision.md`. None is a stop condition; each is a route to keep advancing.

1. Low-turnover / monthly rebalance + netting. The whole after-cost failure is a
   turnover problem: gross edges exist but 1.5%-every-rebalance eats them. Test a
   monthly (or lower-frequency) rebalance with position netting and report whether
   any block flips non-negative after cost. This is the single most direct attack
   on the bottleneck and matches a likely lower-churn product intent.
2. Frozen scorer as a RANKING PRIOR, not a selector. Instead of buying the top
   decile of the frozen score directly (which loses after cost), use the score as a
   prior inside a cost-aware, exposure-gated selection: only trade when the ranking
   edge exceeds the incurred turnover cost, cap turnover, and let the exposure guard
   downgrade to observe/suppress otherwise. Measure after-cost net spread of the
   cost-aware selection, not the raw decile.
3. Regime-conditional / timing overlay. The implicit reversal alpha collapsed on
   H2026_1 — so gate it by regime rather than trading it unconditionally. Build a
   regime detector (volatility/breadth/`moneyflow_hsgt` which is `regime_context_
   only` and available) and only deploy the reversal/frozen ranking in regimes where
   it historically held; abstain otherwise. Report per-regime after-cost spread.
4. Decouple product value from beating cost. Even with no after-cost selector edge,
   the honest P0/P1 research assistant (action-first decision cards with evidence,
   counter-evidence, position/risk limits, triggers, review conditions, explicit
   uncertainty and downgrade) is a real deliverable. Ship and harden it; treat
   "an after-cost-positive selector" as a separate, still-open research goal.

Priority for this round: (1) low-turnover/monthly + (2) ranking-prior-in-cost-aware
selection are the two to run first (they attack the after-cost wall directly and
are offline/CPU-cheap); (3) regime overlay and (4) product-value decoupling run in
parallel as they mature.

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
