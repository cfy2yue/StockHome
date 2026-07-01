# CC Audit & Active Codex Handoff — 2026-07-01 (frozen quant score v1)

Author: CC (audit by CC subagent; finalized by CC). `goal.md`/`AGENTS.md` remain top
authority. CC owns audit/direction/this doc/git; Codex EXECUTES this goal only.

## Audit basis (why this goal)
- **P0's H2026_1 failure is signal–regime coupling, not a bug.** P0's implicit alpha is
  cross-sectional **reversal** (buy recent losers): all price/peer-momentum features have
  NEGATIVE forward-20d RankIC (`reports/date_generalization/feature_rank_ic_audit.csv`:
  kline_return_20d meanIC −0.076, ICIR −0.48). That reversal edge **collapsed in
  H2026_1** (kline_return_20d IC −0.145→−0.017; corr_peer flipped +0.008). The frozen
  baseline 0.8434 was a high-base-rate artifact; `exposure_cards=0` means P0 went pure
  defensive → 0.6667 is NOT stock-picking skill.
- **Biggest debt: 6+ scorers, none serialized/frozen** (run-once-discard). The user's
  "fixed scoring tool" does not exist as an artifact yet. `quant_tool_context.py`
  consumption channel is ready but no usable score feeds it.
- Fix = converge to ONE frozen score + one leakage-free OOT accuracy metric (RankIC) + an
  IC-gate that auto-downgrades exposure when the edge collapses (exactly the H2026 mode).

## Goal-reasonableness (re-scope)
Current goal (strong-yellow MVP + drift caveat) is honest. TOO HIGH = any implicit
"broad 20d +rate ≥0.60/0.65" — chasing it is a divergence trap (the only orthogonal edge,
reversal, is |IC|~0.05 and H2026-collapsed). Re-scope success to: deliver a FROZEN,
reproducible score whose OOT accuracy is clearly measured+published (honest numbers even
if IC is small); score = narrow decision aid (IC-gated); push main alpha to P1 (ranking),
not P0 (absolute timing).

## GOAL (Codex executes exactly this — one bounded step)
Freeze a SINGLE cross-sectional quant scoring tool v1 and publish its leakage-free
time-out RankIC accuracy. Using existing local offline data, train and **serialize** one
cross-sectional ranking score (HGB, `wide_safe` feature group), evaluate its forward-20d
RankIC/ICIR under walk-forward leakage-free splits with **H2026_1 as final OOT**, and
produce RUN_STATUS + accuracy table + an agent IC-gate record. Do ONLY this one artifact;
no open-ended audit, no architecture search, no config-chasing for a return number.

## Why now
See audit basis. Converging the scaffold into one frozen artifact + one OOT accuracy
metric is the shortest verifiable step to the product goal, and it structurally fixes the
P0 drift failure (gate auto-downgrades when the signal collapses).

## Read first (exact paths)
- `scripts/audit_p0_action_label_scorer_v1.py` (HGB fit / build_matrix / score_model — reuse)
- `scripts/run_lightweight_ml_channel_experiment.py` (`_rolling_split`; PRICE_CORE/NEWS/FINANCIAL/REGIME/TUSHARE_PEER feature defs)
- `scripts/audit_feature_rank_ic.py` (`per_date_rank_ic`, industry-neutral, decile spread — REUSE as the accuracy metric)
- `src/agent_training/quant_tool_context.py`, `src/agent_training/evidence_pack.py` (consumption channel, SAFE fields, leakage blacklist)
- `reports/date_generalization/feature_rank_ic_audit.csv`, `.../quant_tool_rule_outcomes.jsonl`
- `data/date_generalization_cache/market_5000/task_labels_v1.csv` + feature CSVs

## Codex owns
New `scripts/train_frozen_quant_score_v1.py`; frozen artifact
`models/frozen/quant_score_v1/` (joblib + `feature_list.json` + `model_card.md` +
`train_blocks.json`); accuracy table CSV + `reports/date_generalization/frozen_quant_score_v1_accuracy.md`;
one appended record in `quant_tool_rule_outcomes.jsonl` (confidence = final-OOT ICIR);
`runs/<run>/RUN_STATUS.md`.

## CC owns
goal.md / DECISIONS.md / HANDOFF.md updates; promote/no-promote decision; ALL git.

## Permissions
sandbox = workspace-write; CPU-only (HGB needs no GPU); model gpt-5.5; effort high.
Read only already-materialized local offline caches. **No online/paid data pulls.**

## Forbidden
- No future/result fields in features (reuse the recursive future-key blacklist + assert).
- **NEVER select any hyperparam/threshold on the H2026_1 OOT block** (walk-forward only).
- No auto-trade/brokerage; no secret print/commit; **no git commit/push (CC finalizes).**

## Success criteria (measurable)
1. Reproducible frozen model artifact (reload → score, no retrain).
2. Accuracy table per block: RankIC mean, ICIR, IC>0 fraction — H2026_1 listed separately.
3. Top-decile vs bottom-decile 20d spread (gross + net after 1.5% turnover) per block.
4. Classifier head: time-out AUC + precision@top-decile.
5. `quant_tool_rule_outcomes.jsonl` new record; `confidence` = final-OOT ICIR; the gate
   (usable_in_agent_default=true only if ICIR≥threshold AND IC>0 fraction≥0.55) written out.
6. Leakage audit PASS (0 hits).

## Stop rules / anti-spin
Train exactly ONE frozen score; no architecture search, no config-sweeping for a return
number. If H2026_1 OOT |RankIC| < 0.02 or ICIR not meaningful → do NOT keep tuning;
record honestly "latest-block signal insufficient", mark the tool `research_only/observe`,
write a `DECISION NEEDED` block for CC, and STOP (don't burn budget). If reaching any bar
requires selecting on the OOT block → STOP immediately (that's overfitting).

## Expected outputs
The artifacts above + `runs/<run>/RUN_STATUS.md` (leakage=pass, per-block RankIC, gate verdict).

## Progress format
One line per step `[step] status | key metric`; end with `RUN_STATUS: DONE / DECISION
NEEDED` + one-sentence conclusion (did we get a usable/gateable frozen score). No git ops.
