# CC Audit & Active Codex Handoff — 2026-07-01 (PRODUCT GOAL: 20d +rate > 60%)

Author: CC. This is the FINAL product target set by the user. Pursue it with honest,
leakage-safe, anti-overfit guardrails. Builds on the frozen quant score v1
(`models/frozen/quant_score_v1/`) and the P0 reversal-regime audit
(`docs/CC_AUDIT_AND_HANDOFF_20260701_frozen_score.md`, DECISIONS 2026-07-01).

## GOAL (concrete DONE condition — do NOT stop until met OR a documented honest ceiling)
Achieve **20-day positive-return rate > 0.60** from the research-assistant strategy, on a
**leakage-safe, walk-forward, out-of-sample** evaluation (report per block; H2026_1 is the
final OOT), subject to ALL of:
- **Genuine stock-picking, not defensive cash**: active_exposure must be >= a
  pre-declared floor (e.g. >= 0.50 of the comparable full-exposure baseline). A 0.60 rate
  achieved with near-zero exposure (`exposure_cards≈0`) does NOT count (that's the H2026_1
  loophole).
- **No OOT selection**: NEVER tune hyperparameters/thresholds/strategy choice on the OOT
  block. All choices are made on earlier blocks (walk-forward); OOT is read once.
- **Reproducible**: the winning strategy config is serialized and re-runnable.
Report the 20d +rate, active_exposure, and avg return per block, with leakage=pass.

## Strategy guidance (pre-declared set to iterate through — CC's direction)
The audit shows the only orthogonal edge is **cross-sectional reversal** (buy relative
losers), and it is **regime-dependent** (strong in some blocks, collapsed in H2026_1). So
the most promising path to a HONEST >60% is **regime-conditioned deployment** rather than
brute-force feature chasing. Iterate through, in order, stopping when the DONE condition is
honestly met:
1. **Regime gating (highest priority)**: deploy active positions only when the reversal
   edge is live — condition on the frozen score's rolling IC/ICIR and regime-breadth
   features (`regime_prior_positive_breadth_20d`, etc.); when the edge is dead (IC≈0), fall
   back to defensive/observe (but that block then fails the exposure floor, so it must be
   handled as "no valid signal", not counted as a win). Find the regime rule that keeps
   exposure up WHEN the edge is live and still clears 60% there.
2. **P1 cross-sectional ranking as the core** (reversal is cross-sectional): long the
   top-decile by the frozen score, size by score × IC-confidence; P1 already shows Top2
   excess +2.67pp. Combine with P0 single-stock only as a filter.
3. **Model/feature**: HGB on `wide_safe` or `kline_peer_chip_news_fin` (the two yellow
   feature groups); keep the frozen-score v1 as the base, improve only via walk-forward.
4. **Portfolio/risk**: turnover-aware net returns (after 1.5%); avoid the decile-spread
   going net-negative (H2026_1 was −2.5%/20d net).

## Read first (server paths)
- docs/CC_AUDIT_AND_HANDOFF_20260701_frozen_score.md, docs/DECISIONS.md, goal.md, AGENTS.md
- models/frozen/quant_score_v1/ (base score) + reports/date_generalization/frozen_quant_score_v1_accuracy.{md,csv}
- reports/date_generalization/feature_rank_ic_audit.csv (regime-dependence evidence)
- scripts/audit_p0_action_label_scorer_v1.py, scripts/run_lightweight_ml_channel_experiment.py (`_rolling_split`), scripts/audit_feature_rank_ic.py (per_date_rank_ic)
- src/agent_training/quant_tool_context.py, src/agent_training/evidence_pack.py (consumption + IC gate)
- data/date_generalization_cache/... (materialized offline data)

## Codex owns
scripts/ (strategy improvements + regime gating), models/frozen/, runs/, reports/, RUN_STATUS.
## CC owns
goal.md/DECISIONS/HANDOFF; promote decision; ALL git.

## Permissions
sandbox = workspace-write; CPU-only (or <=1 GPU); model gpt-5.5; effort high. Read only
materialized local offline caches; **no online/paid data pulls.**

## Forbidden
- ANY OOT-block selection to hit 60% (immediate STOP if a bar needs it — that's overfitting).
- Counting a 60% that rests on near-zero exposure (defensive) as a win.
- Future/GT leakage (recursive blacklist + assert; leakage must PASS).
- Auto-trade/brokerage; secret print/commit; **git commit/push (CC finalizes).**

## Do NOT stop until (goal-mode intent)
Keep iterating through the pre-declared strategy set until the DONE condition is honestly
met on the leakage-safe OOT eval. Only stop when EITHER (a) DONE is met, OR (b) the whole
pre-declared set is exhausted and the honest achievable ceiling is documented (with the
per-block 20d +rate + exposure + why 60% is not honestly reachable on the latest regime) —
then write a `DECISION NEEDED` block for CC. Do NOT stop after a shallow first pass; do NOT
fabricate or overfit to reach the number.

## Success criteria (DONE)
20d +rate > 0.60 on leakage-safe walk-forward OOT (H2026_1 reported), active_exposure >=
floor, no OOT selection, reproducible config serialized, leakage=pass. Report per-block
table + the winning strategy spec + a proposed DECISIONS entry.

## Expected outputs
runs/<run>/RUN_STATUS.md (per-block 20d +rate/exposure/avg-return, leakage=pass, DONE-or-ceiling
verdict) + the serialized winning config + reports/date_generalization/p0_target60_<run>/.

## Progress format
One line per strategy tried: `[strategy] block: 20d+rate / exposure / net-decile-spread`.
Final summary via --output-last-message: DONE (>0.60 honest) or the documented ceiling.
