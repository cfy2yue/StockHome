# StockHome Remote Execution Packet

Updated: 2026-07-03 (round 7: STRATEGIC PIVOT to new signal classes — event-driven PEAD first; cross-sectional price/volume factor-source family CLOSED for new additions; methodology discipline unchanged)

RESEARCH NATURE (binding): StockHome is a quant-finance METHODOLOGY RESEARCH
project. The research question is whether certain public-market signals carry
generalizable, statistically significant predictive power under strictly
leakage-free, cost-inclusive historical evaluation. All outputs are research
findings with evidence, counter-evidence, uncertainty, and failure boundaries —
NOT investment advice, no live trading/orders, no return promises. Negative
results are first-class deliverables. Standing safety gates: research framing
only, no broker/live trading, no secret leakage, no future/GT leakage, low/zero
exposure is never presented as predictive skill.

Status: local-authored remote execution packet for the manual workflow
`CC/Codex local audit -> user manual GitHub sync -> remote Codex execution`.

## Highest-Priority Operating Principles (READ FIRST, this round)

These override any narrower wording below if they ever conflict.

1. Remote's #1 priority is to KEEP ADVANCING the durable final goal. A failing
   strategy, an unmet gate, or a dead data source is NOT `BLOCKED`. Remote is
   expected to exercise initiative: think for itself, launch read-only/code-review
   subagents to audit and OPTIMIZE the project route, switch strategy, and keep
   going. Record every route/decision point in `remote_decision.md`. Do not bounce
   back to local for help on anything you can decide inside the boundaries.
2. There is NO predefined hard-block condition, and this CANCELS every `BLOCKED` /
   "hard block" / "hard-stop" statement elsewhere in this file. The goal pauses ONLY
   when (a) the remote ITSELF concludes it is genuinely blocked after exhausting the
   in-boundary routes it can think of, recording the full reasoning in
   `remote_decision.md`, or (b) the user manually interrupts for local audit. "Needs
   more data / a new source / more compute / a user-owned decision" is NOT an
   auto-stop — pursue it yourself (authorized tushare/ds data + small-model training)
   and, if you truly think the user is needed, keep doing everything else meanwhile
   and just note it. Only broker / live-trading remains a genuine hard safety line.
   Do not stop and wait.
3. No unfounded resource limits. tushare (A-share market data) and the ds/DeepSeek
   LLM are ALREADY provisioned and authorized under `/data/cyx/1030/api` (read keys
   only from there or env, never print/commit, keep cost reasonable). Any data pull
   or small-model training genuinely needed to reach the goal may be pursued
   autonomously. Standing safety gates remain: no broker/live trading, no
   future/GT leakage, low/zero exposure is never stock-picking skill, no secret
   leakage.
4. The durable final goal and its threshold are DURABLE. A gate is the bar for
   CLAIMING success, NOT a stop condition. After-cost net spread is the current
   real bottleneck, but a negative after-cost result does NOT mean stop — log it,
   pick a new in-boundary strategy, and continue.

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

State update (2026-07-02 round 3 local audit, verified by SSH at HEAD `2a5a2d4`):
the P1 ranker-guard integration stage is DONE remotely
(`runs/p1_ranker_guard_integration_20260702/` +
`reports/date_generalization/p1_ranker_guard_integration_20260702/`). Outcome:
frozen score downgraded to `observe_only`, `reversal_composite` to `suppress`;
both stay after-cost NEGATIVE on H2026_1 (frozen top-vs-pool net `-1.209`,
top-bottom net `-1.703`; reversal `-1.837` / `-2.596`). Prior stages also confirmed
done: signal inventory, 7-family target60 (all failed, ceiling `0.6000` w/ negative
net spread), `frozen_quant_score_v1` leakage PASS but weak/single-block.

New-source exploration this round (all no-label / no-model, tushare authorized):

- `broker_recommend`: sparse contract READY (36 pre-H2026 months, 6 blocks) but
  CLOSED as a direct sparse selector — PRE_H2026 active positive `0.4552` vs base
  `0.4827`, after-cost spread `-2.29pp`; H2026_1 `0.4000` vs `0.4713`, after-cost
  `-4.03pp`. Auxiliary-context only (`remote_decision.md` 2026-07-02 decision).
- `hk_hold`: NOT a clean close — D+1 no-label join audit
  (`p1_hk_hold_d1_join_quality_audit_20260702`) shows source_rows `41326` ->
  feature_rows `11312` -> mean overall match rate `0.3916`, but
  min_block_match_rate `0.0` and sparse_blocks `['H2024_2','H2025_2']`. Partially
  joinable, coverage too holey for modeling now (`modeling_allowed_now=False`).
- Hot-rank (`ths_hot`, `dc_hot`): from `p1_tushare_specialty_lag_coverage_20260702`
  they DO carry `has_rank_time=True` (an available-at timestamp exists), but
  `coverage_ok_for_next_cache_design=False` — only 3-4 pre-H2026 blocks sampled
  nonempty (dc_hot: H2024_2/H2025_1/H2025_2; ths_hot: H2024_1/H2024_2/H2025_1/
  H2025_2). They also mix A-share / HK / US / concept-board rows, so a
  market/`data_type`/`ts_code`-suffix filter plus rank_time/D+1 audit is required
  before any label use.
- `moneyflow_hsgt`: `has_stock_identifier=False` -> `regime_context_only`, not a
  stock selector. Useful as a regime/timing overlay signal, not for cross-section.

COMMON FAILURE MODE (the real bottleneck): after-cost net spread is NEGATIVE
across the frozen score, reversal baseline, broker sparse selector, and most
blocks — multiple new sources / small models / sparse selectors all go negative
once the 1.5% round-trip cost is subtracted. This is the true wall.

ONE non-negative row to keep on radar (not yet promotable): the guard scan found
`quality_momentum_accumulation_v1` H2026 net spread `+1.2098`, but its direct
target60 win rate is `0.4444` and it needs an available-at/lag audit before any
feature reuse. It is `not_promoted` now.

So: do NOT chase a strict raw `>0.60` win rate on the current feature set, do NOT
re-run target60 / the 7 families / a direct broker selector. [Round-4 note: the
after-cost protocol re-audit (c) is now DONE remotely and its verdict accepted —
see "Round-4 State Update" below; the active advance directions are Tracks
S/A/U/F/P in the round-4 `Exact Next Task`.] Keep delivering P0/P1 product value
regardless of after-cost alpha.

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

## Round-4 State Update (2026-07-02, read-only SSH verified at HEAD 465ef0d)

- Remote overnight autonomous run (at `2a5a2d4`, before the round-3 packet
  landed) produced ~157 dated 20260702 run dirs and an 8085-line
  `remote_decision.md` (server-side; the tracked local copy lags).
- Round-3 track (c) is ALREADY DONE autonomously:
  `after_cost_protocol_and_block_assumption_audit_20260702`. Accepted verdict:
  the 1.5% cost protocol is conservative but NOT the main bottleneck — H2026
  GROSS top-vs-pool is already negative for additive/logistic/GBDT/reversal
  routes; repeated after-cost failure is primarily signal failure. The current
  evaluation is a signal diagnostic, not a tradable backtest. Keep 1.5% as the
  claim gate; sensitivity curves are research information only. Do NOT redo.
- Remote issued `local_audit_request_after_cost_route_exhaustion_20260702`:
  six after-close mutation routes (cost-aware small model, sticky turnover
  guard, turnover stability, regime exclusion, low-frequency execution, score
  persistence) all fail stability and/or H2026 diagnostics. All CLOSED; the
  closed-route rule covers them.
- `next_source_boundary_prefilter_20260702`: no inspected source family may
  proceed to modeling without a local whitelist. The semantic evidence-pack
  channel is `FEATURE_SOURCE_CANDIDATE_WITH_CAVEATS` (15947 lag-PASS no-label
  rows, 114 fields) but modeling is NOT allowed yet.
- Round-3 track (a) hot-rank available-at audit has NO output dir yet — it
  carries over as Track A of the round-4 task.
- New binding methodology findings (see `local_audit.md` round 4): H2026_1 is
  SEMI-CONTAMINATED by 20+ diagnostic reads (multiple-testing risk, no FDR
  control anywhere); no NW t-stat / bootstrap CI protocol exists; the
  survivorship/universe/label-alignment question has never been audited.
  Promotion claims therefore may ONLY come from the pre-registered
  freeze-and-forward window (Track F), never from H2026_1 alone.

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

Date: 2026-07-03 (round 7) — CONTINUE the running StockHome session with this route.

Task ID: `p1_round7_event_signal_pivot_pead_availability_audit_20260703`.

### Round-7 STRATEGIC PIVOT (user decision, binding)

The user has decided: the exhausted routes are all the SAME FLAVOR of
cross-sectional price/volume/chip/factor sources (reversal, value/PB, margin,
CYQ, index_weight, industry, pledge, hot-rank, broker_recommend,
stk_holdertrade, ...). Their common wall — confirmed by M1/M2 as a genuine
result, not a cost/unit/mask artifact — is that the GROSS edge itself is weak
or negative. Therefore:

1. NO NEW same-flavor cross-sectional price/volume/chip/factor sources may be
   added as scout candidates. The closed-route list is generalized to the whole
   family flavor. (Finishing the already-preregistration-approved
   `stk_holdertrade` scout per round-6 spec is allowed as closure work — it is
   itself an ann_date event source and its clean positive/negative verdict is
   wanted either way.)
2. The project pivots to NEW SIGNAL CLASSES with genuinely different
   information content, starting with EVENT-DRIVEN signals (PEAD /
   earnings-surprise drift) — ranked and justified in `local_audit.md`
   round 7 and `local_suggestion.md` "Round-7 New-Signal-Class Portfolio".
3. Methodology discipline is UNCHANGED and non-negotiable: PIT available-at
   audit first, no-label-first (no scout before a PASS availability/coverage
   audit + preregistration), BH-FDR multiple-testing control, after-cost claim
   gate @1.5pp round-trip, H2026 diagnostic-only, promotion ONLY via Track-F
   pre-registered single-use forward window, negative results first-class.
4. The gross-edge meta-finding stays a binding claim boundary and is every new
   scout's prior.

### Round-7 state baseline (SSH-verified 2026-07-03 at remote HEAD `3131411`)

- Round-6 route (`p1_round6_stk_holdertrade_prereg_tiny_scout_20260703`) was
  issued; if the remote has not yet completed it, FINISH IT FIRST exactly as
  specified in the superseded round-6 section below (prereg -> in_de taxonomy
  audit -> tiny scout -> admission -> close-or-Track-S-candidate). It is
  bounded and cheap; its verdict feeds the event-class portfolio.
- Data-availability facts verified by local read-only SSH (2026-07-03):
  - `data/date_generalization_cache/tushare_pro/tables/` holds 21 cached
    endpoint dirs incl. `forecast`, `express`, `anns_d`.
  - `forecast` fields confirmed: `ts_code, ann_date, end_date, type,
    p_change_min, p_change_max, net_profit_min, net_profit_max,
    last_parent_net, first_ann_date, summary, change_reason, update_flag` —
    a native `ann_date` PIT anchor AND a self-contained surprise definition
    (forecast midpoint vs `last_parent_net`) that needs NO analyst consensus.
  - `express` fields confirmed: `ann_date` + yoy fundamentals
    (`yoy_net_profit`, `diluted_eps`, `yoy_sales`, ...) — same anchor.
  - CAVEAT: the local forecast/express caches are THIN (a 200-code probe;
    sampled per-code CSVs are header-only; the round-5 live probe observed
    0 rows in its window). A bounded ann_date-window backfill is required
    and is authorized below.
  - `anns_d` (announcement list) is ALREADY CACHED LARGE: 737 daily files,
    2023-01-01 -> 2026-01-09, ~420MB, fields `ann_date, ts_code, name,
    title, url` — raw material for a title-level event taxonomy and a later
    LLM extraction pipeline (ds/DeepSeek authorized).
  - Analyst-revision endpoints (`report_rc` or similar) appear in NO
    inventory and NO script — availability UNVERIFIED (possible credit-tier
    gate); needs a single bounded probe.
  - `/data/cyx/1030/api` confirmed to hold `tushare_token.txt`, `ds_api.txt`
    (keys never printed/committed).

### Round-7 route (event-signal pivot: PEAD availability audit + two bounded probes)

A negative result is NEVER `BLOCKED`; log and continue. CPU/offline preferred;
bounded tushare pulls authorized as scoped below; keys from
`/data/cyx/1030/api`, never printed/committed; research findings only.

0. FINISH ROUND-6 stk_holdertrade first if not already done (see superseded
   spec below). Record its admission verdict in the registry either way.
1. PEAD no-label available-at/coverage audit (the ranked #1 new class; SAME
   rigorous pipeline shape as stk_holdertrade: available-at -> no-label
   contract -> matrix preview -> audit -> prereg -> tiny pre-H2026 scout ->
   admission; NO label join and NO scout inside this round):
   a. Bounded backfill of `forecast` + `express` by ann_date windows covering
      2023-01 -> 2026-06 (full A-share market; these are small event tables —
      expected tens of thousands of rows total; no unbounded loops; cache
      under the existing tushare_pro cache convention).
   b. Available-at policy audit: `ann_date` is the ONLY anchor; audit
      `first_ann_date` vs `ann_date` semantics and `update_flag` (revisions
      must use their OWN later ann_date, never the first announcement's);
      after-close disclosure -> D+1 anchoring per the standing 15:00 rule;
      `end_date` is report-period, NEVER availability.
   c. Type taxonomy value audit (the in_de lesson): audit raw `type` values
      (预增/预减/扭亏/续盈/略增/略减/首亏/续亏...) on a small sample BEFORE
      trusting any direction feature; record the mapping.
   d. Three-number separation per block (raw event rows / A-share joinable
      rows / decision-universe match rate), event counts + unique codes per
      block, >= 5 pre-H2026 nonempty blocks required.
   e. No-label feature contract + matrix preview + audit: candidate features
      are event-anchored `_d1` aggregates, e.g. surprise magnitude
      (net_profit midpoint vs last_parent_net), p_change midpoint, type
      direction class, express yoy deltas, days-since-event; text fields
      (`summary`, `change_reason`, `perf_summary`) are EXCLUDED from features
      in this round (they belong to the later LLM route).
2. Parallel bounded probe (2a) anns_d title taxonomy (no-label, no-model):
   sample the cached daily files (e.g. 2 days/month across the range), build a
   keyword/rule first-pass event-type frequency table (解禁/回购/增持减持/中标
   合同/重组并购/问询函/诉讼/股权激励/业绩类...) with per-block coverage;
   OPTIONALLY spot-check <= 500 titles with ds/DeepSeek classification
   (temperature 0, prompt text archived in the report, output = category
   only, bounded cost) to measure rule-vs-LLM agreement. Output = a taxonomy
   + frequency + coverage report that decides whether the LLM event-extraction
   pipeline (ranked #2) is chartered next round. No label joins.
3. Parallel bounded probe (2b) analyst-revision availability: ONE bounded
   probe of the analyst endpoints (`report_rc` or the current tushare
   equivalent) — does the token tier return rows? Fields? An
   ann-date/report-date availability proxy? Verdict: `available ->
   queue_for_availability_audit` or `closed_credit_tier` /
   `closed_no_timestamp`. No backfill this round.
4. Event-study evaluation-framework spec (document only, no big refactor):
   write `event_study_framework_spec.md` defining event-aligned CAR(0,20)
   vs a matched benchmark (pool mean and/or size-matched), event-clustered
   standard errors, overlap handling, and how the existing selected-vs-pool
   admission gates map onto sparse event signals (the broker_recommend lesson:
   sparse events are DILUTED in daily cross-sectional ranking; event-aligned
   measurement is the native metric, and BOTH views must be reported at
   admission). This spec becomes part of the PEAD scout preregistration next
   round.
5. Close: choose exactly ONE next route and log it — expected default:
   (a) PEAD audit PASS -> next round = PEAD scout preregistration + tiny
   pre-H2026 scout under the event-study spec; (b) PEAD audit FAIL -> close
   it, queue the next event source (share_float 解禁 / dividend / 
   stk_holdernumber / block_trade from the existing admission queue) for its
   availability audit; (c) taxonomy probe strong + PEAD weak -> charter the
   LLM title-extraction pipeline audit instead. M5 standardized action-card
   output stays ALWAYS-ON.

Gates (pass/fail for THIS round's audits; claim bars, not stop conditions):

- PEAD availability audit PASS = ann_date policy proven (incl. update_flag /
  first_ann_date semantics + D+1 anchor) AND >= 5 pre-H2026 nonempty blocks
  AND decision-universe match rate >= 0.5 AND type taxonomy audited AND
  no-label matrix audit failed-checks = []. Only then is a preregistered tiny
  scout allowed NEXT round.
- Probe verdicts are recorded either way; a closed probe is a result.
- NO label join, NO model, NO H2026/forward read anywhere in this round;
  Track-F stays 0.

Resource limits (this round): same envelope as round 6, plus (i) bounded
tushare backfill of forecast/express by ann_date window (event tables only;
no minute/tick data; no unbounded loops), (ii) ds/DeepSeek <= 500 title
classifications at temperature 0 with archived prompt and bounded cost,
(iii) ONE analyst-endpoint probe call family. No GPU, no model training,
no new cross-sectional factor sources.

Expected outputs:

- `runs/p1_round7_event_signal_pivot_20260703/RUN_STATUS.md`;
- `reports/date_generalization/p1_pead_forecast_express_availability_audit_20260703/`
  (available_at_policy, type taxonomy audit, three-number coverage tables,
  no-label feature contract + matrix preview + audit);
- `reports/date_generalization/p1_anns_d_title_taxonomy_probe_20260703/`
  (taxonomy, frequency/coverage tables, optional ds agreement spot-check,
  archived prompt);
- `reports/date_generalization/p1_analyst_endpoint_availability_probe_20260703/`;
- `reports/date_generalization/event_study_framework_spec_20260703.md`;
- registry + `remote_decision.md` appended (pivot acknowledgment, round-6
  closure verdict, probe verdicts, ONE chosen next route).

DONE: round-6 stk_holdertrade verdict recorded; PEAD availability audit
complete with explicit PASS/FAIL against the gates above; both probes
verdicted; event-study spec written; exactly ONE next route chosen and
logged; leakage PASS; Track-F 0; forward unread; H2026 untouched; no label
joins anywhere.

## Superseded Round-6 Task (finish first if incomplete; otherwise evidence only)

Date: 2026-07-03 (round 6) — CONTINUE the running StockHome session with this route.

Task ID: `p1_round6_stk_holdertrade_prereg_tiny_scout_20260703`.

### Round-6 state baseline (SSH-verified 2026-07-03 at remote HEAD `3131411`)

Round-5 is DONE and its results are ACCEPTED by local audit:

- M1-M5 measurement foundation COMPLETE (`ROUND5_MEASUREMENT_FOUNDATION_DONE`,
  `reports/date_generalization/p1_round5_measurement_foundation_20260702/`).
  M1 verdict: the round-4 pre/H2026 scale gap is NOT a unit-conversion bug —
  both are pp under the canonical convention (`canonical_metric_cost_spec.md`,
  86-row registry re-expressed). M2 verdict: mask-v2 decomposed; 2 best
  candidates re-run with ONLY hard execution exclusion; `revived = False`.
  Together M1+M2 CONFIRM the negatives are genuine: not cost/unit artifacts,
  not mask-veto artifacts. M3: H2026 stays diagnostic-only, forward window
  UNREAD. M4: `negative_methods_findings.md` written; chosen route =
  `negative_methods_plus_track_p_product_value`. M5: standardized action card
  implemented in `src/agent_training/decision_card.py` + example artifacts.
- `index_weight`: CLOSED as negative scout (`INDEX_WEIGHT_SOURCE_CLOSED_
  NEGATIVE_SCOUT`; selected 0.4550 vs pool 0.4951, lift -0.0400, after-cost
  net @1.5pp `-2.06`; Track-F 0). Closure CONFIRMED. The effective_date-vs-
  publication PIT caveat is real but immaterial: the leak-optimistic version is
  already negative, so the honest version can only be worse. Do NOT reopen.
- `pledge_stat`: CLOSED fail-closed on availability (`..._REVIEW_AVAILABILITY_
  BLOCK`; only period-end `end_date`, no disclosure timestamp -> not decision-
  time available). CONFIRMED correct. Reopen ONLY if an ann_date-style
  disclosure-timestamp source is found or a named audit bug is identified.
- `stk_holdertrade` (current frontier): the remote ALREADY completed the
  detail-only rebuild the round-5 audit required — source-aggregate no-label
  matrix preview (`..._SOURCE_AGGREGATE_MATRIX_PREVIEW_READY`, 48 rows, 35
  feature-ready, 6 pre-H2026 blocks, source rows read-not-persisted 352,
  errors 0, H2026 rows 0) AND its audit (`..._AUDIT_PASS_NO_MODEL`, failed
  checks `[]`, `preregistration allowed next: True`). PIT hygiene verified by
  local audit: all 8 features are `_d1`-anchored on `ann_date`; no
  `holder_name` persisted (audit check `raw_holder_not_persisted=True`); no
  `begin_date`/`close_date` in features; contract on disk = 8 features /
  5 gates, consistent with the matrix.
- M5 action card: `ROUND5_NEXT_STAGE_ACTION_CARD_READY`, cards `74`
  (disk-authoritative; an earlier `72` was stale stdout), Track-F 0; handoff
  `REMOTE_HANDOFF_INTEGRITY_GUARD_PASS` (artifact rows 11 / review rows 0).
- Closed families stay closed (do NOT re-tune): pairwise baseline, raw
  margin/CYQ, industry static residual, risk-veto, limit_list_d, index_weight,
  pledge_stat current alpha route, plus all earlier closures.

### ELEVATED META-FINDING (now a binding claim boundary)

Across MANY independent PIT-safe source families (pairwise, margin/CYQ,
broker_recommend, cyq chips, repurchase, index_weight, frozen score, reversal,
6 after-close mutations, live-tushare tiny scouts), the dominant failure mode
is that the GROSS edge itself is unstable or negative — NOT that cost eats a
real edge, NOT unit bugs (M1), NOT mask vetoes (M2). This is the project's
central methodology result so far and strengthens the negative-methods
deliverable (M4): under leakage-free, cost-inclusive, FDR-corrected,
PIT-audited evaluation, none of the inspected public-source families has shown
tradeable A-share 20d alpha. Every new scout inherits this as its prior; a
single-source positive must clear Track-S significance + Track-F forward
confirmation before any claim.

### Round-6 single focused route (stk_holdertrade preregistration + tiny scout)

A negative result is NEVER `BLOCKED`; log and continue. CPU/offline; bounded
tushare only per existing envelope; keys from `/data/cyx/1030/api`, never
printed/committed; research findings only.

1. PREREGISTRATION FIRST (before any label join): write
   `stk_holdertrade_tiny_pre_h2026_scout_preregistration` declaring the exact
   8 aggregate `_d1` features (ann_date-anchored; no holder_name; no
   begin_date/close_date; no duplicate counting; no H2026/forward), the scout
   universe/blocks (the 6 pre-H2026 blocks of the PASS aggregate matrix), and
   verbatim admission gates: selected-vs-pool win-rate lift, RankIC, after-cost
   net spread @1.5pp non-negative, Track-S registry entry + significance
   language. Resolve the standing contract caveat
   `in_de_taxonomy_requires_value_audit` INSIDE the preregistration: audit the
   raw `in_de` value taxonomy (增持/减持 coding) on a small sample BEFORE
   trusting the increase/decrease direction features; record the mapping.
2. Run the tiny pre-H2026 scout exactly as preregistered (offline labels
   eval-only; H2026 rows 0; forward unread; Track-F stays 0).
3. Admission audit against the preregistered gates (same shape as
   `p1_index_weight_tiny_scout_admission_20260703`).
4. If gates FAIL -> close `stk_holdertrade` as a negative scout (mirror the
   index_weight closure), append to the registry and `remote_decision.md`,
   and continue — NOT blocked. If gates PASS -> it becomes a Track-S registry
   candidate ONLY; promotion still requires Track-F forward preregistration;
   no claim from pre-H2026 or H2026 alone.
5. Then choose exactly ONE next route and log it: (a) name ONE genuinely new
   independent PIT source with a plausible ann_date-style available-at
   mechanism and run its no-label available-at/coverage audit; or (b)
   consolidate the negative-methods report (M4, now including the elevated
   meta-finding) + M5/Track-P product value as the standing deliverable.
   M5 standardized action-card output stays ALWAYS-ON.

Hygiene sub-items (bounded, report-only):

- remote_decision.md narrative still cites the OLD contract shape `7 features /
  4 gates` while the on-disk rewritten artifacts are `8 / 5`: append a dated
  CORRECTION entry to `remote_decision.md` (do not rewrite history) stating the
  authoritative artifact paths.
- stdout-vs-disk status mismatches (`..._REVIEW_OR_EMPTY` seen in stdout vs
  `..._READY` on disk; cards 72 vs 74): grep confirmed NO on-disk
  REVIEW_OR_EMPTY residue in the stk_holdertrade dirs — treat disk
  summary/RUN_STATUS as authoritative, note the stale-stdout root cause (rerun/
  concurrency) in the decision log, and prefer re-reading disk artifacts over
  trusting captured stdout in future stages.

Expected outputs:

- `runs/p1_round6_stk_holdertrade_prereg_tiny_scout_20260703/RUN_STATUS.md`;
- `reports/date_generalization/p1_stk_holdertrade_tiny_scout_preregistration_20260703/`
  (prereg doc + in_de taxonomy audit), `.../p1_stk_holdertrade_tiny_pre_h2026_scout_20260703/`,
  `.../p1_stk_holdertrade_tiny_scout_admission_20260703/`;
- registry + `remote_decision.md` appended (decision, correction entry, one
  chosen next route).

DONE: prereg written BEFORE label join; in_de taxonomy audited; scout run as
preregistered; admission verdict recorded (pass -> Track-S candidate only /
fail -> closed negative); exactly ONE next route chosen and logged; hygiene
correction entry appended; leakage PASS; Track-F 0; forward unread; H2026
untouched.

## Superseded Round-5 Task (executed 2026-07-02/03; M1-M5 DONE, results accepted — kept for spec reference)

Date: 2026-07-02 (round 5) — CONTINUE the running StockHome session with this route.

Task ID: `p1_round5_measurement_foundation_and_negative_methods_20260702`.

ONE focused route this round (no parallel weak scouts — honoring the multiple-testing
churn risk the remote flagged). It answers the remote's round-4 audit questions (metric
units, mask-v2 semantics, coverage, H2026/Track-F boundary, negative-methods pivot) and
SETTLES THE MEASUREMENT FOUNDATION before any further signal search — because two of the
remote's own findings mean the current negatives may be partly ARTIFACTUAL and must be
resolved first: (i) a metric/cost UNIT inconsistency (pre top-vs-pool net `-0.0734` vs
H2026 diagnostic net `-2.90` is a ~40x scale gap — one is almost certainly a fraction and
the other percentage points, or an aggregation/annualization mismatch), and (ii) mask-v2
may be misused as a HARD alpha veto (`not_claim_safe_v2` pre rate `0.587`, mostly
`unknown_namechange_active_interval 0.565`) when most of it is a CLAIM caveat, not a
tradability exclusion — excluding ~56% of names as "unknown" could be destroying signal.
Fixing these can REVIVE a weak candidate or firmly confirm the negative; either way it
re-grounds every number. A negative after-cost result is NEVER `BLOCKED`; log and
continue. CPU/offline only; bounded tushare only for metadata; keys from
`/data/cyx/1030/api`, never printed/committed; research findings only, never advice.

Round-4 results this route builds on (all NEGATIVE after-cost — CLOSED, do NOT re-tune):
pairwise baseline `liquid_large_top100/low_pb_value` weak-positive statistically (pre
RankIC 0.071) but economically NEGATIVE (pre top 20d rate 0.428, top-vs-pool net -0.073);
margin/CYQ strict pre BH-FDR survivors = 0; industry static residual = diagnostic-only (no
PIT membership); risk-veto = not alpha. Register these as closed; do not re-tune.

- M1 metric/cost-unit reconciliation (audit q1): produce ONE canonical accounting spec
  (units = percentage points; explicit turnover-cost convention) for RankIC, pairwise
  accuracy, turnover, top-vs-pool net, `return_20d`/`fwd_ret_20d`. Reconcile the -0.073 vs
  -2.90 scale gap and re-express the Track-S registry numbers in canonical units; every
  later report MUST cite this spec.
- M2 mask-v2 semantics decomposition (audit q2): split `not_claim_safe_v2` into disjoint
  (a) HARD execution exclusion (suspended / limit-locked / delisted at decision date),
  (b) CLAIM caveat (report-level uncertainty, e.g. unknown_namechange — KEEP in selection,
  flag in reporting), (c) SOFT missingness (impute/handle). Re-run the best pre-H2026
  candidates using ONLY (a) as an exclusion; measure whether after-cost results MATERIALLY
  change (RankIC delta > 0.01 or net delta > 0.5pp). This can revive a weak signal or
  confirm the negative.
- M3 H2026 / Track-F contamination lock (audit q4): confirm H2026 stays diagnostic-only
  and the forward window (post 2026-06-23) unread; Track-F pre-registration remains the
  ONLY promotion path; registry marks H2026_1 semi-contaminated.
- M4 negative-methods / guardrail deliverable (audit q5): with corrected M1/M2 numbers,
  write the honest methodology finding — under leakage-safe, cost-inclusive, BH-FDR-
  corrected, PIT/survivorship-audited evaluation, do these signal families carry tradeable
  A-share 20d alpha? If NOTHING survives -> that IS the publishable negative-methods /
  failure-map result (a legitimate methodology contribution; the durable 60% target is NOT
  dropped, it is honestly reported as unreached in the current feature universe). If M1/M2
  REVIVE a candidate -> route it to Track-F pre-registration; do NOT claim on H2026.

- M5 Track-P product-output standardization (ALWAYS-ON, parallel; explicit user priority
  2026-07-02): make the P0/P1 user-facing output STANDARDIZED, PROFESSIONAL, and
  DECISION-ORIENTED — a clear action card built from TABLES with explicit OPERATIONAL
  THRESHOLDS and JUDGMENT CRITERIA, and REUSE everything the backtest workflow learned
  (reversal, frozen score, margin/CYQ, hot-rank, regime context, mask/PIT hygiene) as
  HONEST EVIDENCE inputs, not as buy triggers. Every signal is reported with its honest
  verdict (pre-H2026 RankIC / NW-t / after-cost sign / regime dependence) so the user sees
  WHY. Backtest strategies that FAILED as pure selectors are NOT wasted — they become the
  analytical LENS (context + counter-evidence + regime flags) with their failure reported.
  Safety boundary unchanged: guard grades gate ALL action language, no future/GT fields, no
  return promise, low/zero exposure = defensive. The exact card template + threshold table +
  evidence->stance judgment rule are in `local_suggestion.md` "Round-5 Product Output Spec";
  implement in `src/agent_training` card/decision builders and ship a worked, leakage-PASS
  example `reports/date_generalization/p1_round5_measurement_foundation_20260702/action_card_example.md`.

Deferred (audit q3): financial-PIT cache expansion — DEFER (best `fina_indicator` coverage
0.458 < 0.50; quarterly fundamentals are low-EV for 20d after-cost alpha). Revisit only
with a specific fundamental-momentum hypothesis, and then a no-label coverage audit FIRST
(never a direct label scout).

Decision output (exactly ONE next route, tied to the durable target + Track-F path):
- if M2 correction revives a candidate above the significance gate -> next route = write
  it into Track-F forward pre-registration (single-use forward window);
- elif a GENUINELY NEW independent source family is identified (name ONE with a plausible
  decision-time-available mechanism, e.g. northbound HSGT holdings-change flow, block-trade
  / large-order flow, shareholder-count change, or index-membership rebalancing events) ->
  next route = that source's available-at / no-label coverage audit;
- else -> consolidate on the negative-methods report + Track-P product value as the
  standing deliverable; keep the forward window reserved for a future candidate.

Expected outputs:
- `runs/p1_round5_measurement_foundation_20260702/RUN_STATUS.md`;
- `reports/date_generalization/p1_round5_measurement_foundation_20260702/`:
  `canonical_metric_cost_spec.md`, `registry_recomputed_canonical_units.csv`,
  `mask_v2_semantics_decomposition.csv`, `mask_corrected_candidate_deltas.csv`,
  `negative_methods_findings.md`, `next_route_decision.md`;
- append the decision + any revived candidate to `remote_decision.md`.

DONE: metric spec written + registry re-expressed in canonical units; mask-v2 decomposed +
corrected deltas measured; H2026/forward boundary confirmed intact; negative-methods
finding written; exactly ONE next route chosen and logged. Leakage PASS; research framing;
every negative logged-and-continued in `remote_decision.md`.

## Superseded Round-4 Task (executed 2026-07-02; all tracks negative after-cost — do NOT re-run; kept for Track A/F spec reference)

Date: 2026-07-02 (round 4)

Task ID: `p1_round4_stats_hardening_pit_sources_20260702`.

This is a PARALLEL, LONG-RUNNING stage and the local ANSWER to remote's
`local_audit_request_after_cost_route_exhaustion_20260702`: local audit chooses
a PORTFOLIO — option 1 (new lag-safe PIT sources, audit-first) + option 3
EXTENDED (statistics + universe methodology hardening; the protocol audit
itself is accepted as done) + option 2 always-on (product value). It supersedes
`p1_hotrank_availableat_and_aftercost_reaudit_20260702` (its track (c) is done;
its track (a) carries over as Track A below — the full track (a) spec in the
superseded section further down REMAINS the authoritative detail). All closed
routes stay closed. A negative result in any track is logged in
`remote_decision.md` and work continues on the others; after-cost or gate
failure is NEVER `BLOCKED`.

Hypothesis chain (from `local_audit.md` round 4): the binding risks are now
(1) multiple-testing contamination of H2026_1 (~20+ diagnostic reads, no FDR
control), (2) a missing significance protocol (overlapping 20d labels,
date-clustered effective N, no NW t / bootstrap CI), (3) an un-audited
survivorship/universe/label-alignment question, and (4) availability contracts
for new PIT sources. Fixing (1)-(3) re-grounds every number the project has;
(4) is the only sanctioned path to new signal families.

### Track S: statistical inference hardening + multiple-testing registry

CPU/offline only, labels offline-eval only.

1. Build `route_hypothesis_registry.csv`: one row per route/family/config ever
   evaluated against H2026_1 (7 target60 families, frozen score, reversal,
   broker, hk_hold, 6 after-close mutations, semantic scout, margin detail,
   regime probes, ...). Columns: family, selection protocol, headline metric,
   H2026_1 usage (diagnostic/promotion), evidence path. Source: server
   `remote_decision.md` + run/report dirs.
2. Recompute honest significance for `frozen_quant_score_v1` and
   `reversal_composite`: daily RankIC series -> Newey-West t-stat (lag >= 20)
   per block and pooled pre-H2026; moving-block bootstrap 90% CI for after-cost
   net spread; win rate vs block base rate with date-clustered binomial/
   bootstrap CI (effective N = decision dates, not rows).
3. Apply BH-FDR q=0.10 across the registry headline metrics; report survivors
   (expect ~zero — that is itself the publishable finding).
4. Write into the report: H2026_1 is SEMI-CONTAMINATED for promotion; all
   future promotion claims route through Track F.

Gates: a "signal exists" research claim requires pre-H2026 pooled NW t >= 2.0
AND BH-FDR survival AND after-cost bootstrap CI lower bound > -0.5pp. Anything
less is recorded as noise-compatible.

### Track A: hot-rank A-share available-at audit (carried over, NO label/model)

Execute the round-3 track (a) spec exactly (see superseded section below):
A-share filter with THREE-number separation (raw endpoint rows / A-share
joinable rows / decision-universe match rate) per block, rank_time vs decision
close with D+1 anchoring, `available_at_policy` + `coverage_by_block` +
`source_semantic_contract`, verdict usable / D+1-contract / CLOSED per source.
Gate to become a later label candidate: availability proven AND >= 5 nonempty
pre-H2026 blocks AND decision-universe match rate >= 0.5.

### Track U: survivorship / universe / label-alignment audit (new)

1. Determine whether decision-universe membership is point-in-time: are names
   later delisted/ST/long-suspended present at historical decision dates, or is
   the universe a current-listing snapshot? Quantify affected name counts per
   block.
2. Audit `src/backtest/ground_truth.py` index-offset labels: `close[idx+20]`
   skips suspension gaps and stretches the horizon. Measure the calendar/
   trading-day span distribution of all labels; report the fraction spanning
   > 30 trading days.
3. If feasible offline, rerun the frozen-score block accuracy table on
   span-corrected labels and report deltas.

Gates: bias quantified; material shift = RankIC delta > 0.01 or net spread
delta > 0.5pp on any block -> annotate all prior conclusions + re-score the
Track-S registry. If PIT membership cannot be reconstructed, record "universe
is survivor-biased; positive historical metrics are upper bounds" as a standing
claim boundary.

### Track F: freeze-and-forward pre-registration (write-only now, no evaluation)

Write `reports/date_generalization/preregistration_h2026_forward_v1.md`
declaring AT MOST 3 frozen candidates with exact configs, e.g.:

- frozen scorer as a RANKING PRIOR inside a turnover-capped monthly selection;
- a low-turnover / monthly-rebalance netting variant;
- a regime-gated variant using `moneyflow_hsgt` regime context (only if D4-style
  permutation evidence exists pre-H2026).

Declare metrics and gates verbatim: forward-window RankIC NW t >= 2.0;
exposure-gated win rate above base rate with date-clustered CI excluding zero;
after-cost (1.5%) net spread >= 0; leakage PASS; nontrivial exposure. The
window = as-of data accruing AFTER the current GT boundary (2026-06-23), is
SINGLE-USE, and is burned once read; log the doc hash + date in
`remote_decision.md`. NO forward evaluation in this stage.

### Track P: product-value track (always on, unchanged from round 3)

Keep shipping the honest P0/P1 research assistant independent of after-cost
alpha: action cards with evidence, counter-evidence, position/risk limits,
triggers, review conditions; guard grades gate all action language; low/zero
exposure explicitly defensive. Research question "honest information
presentation" stays decoupled from "statistically beating cost".

### Round-4 allowed inputs / commands / limits

Same envelope as round 3 (see superseded section): read-mostly CPU work over
offline caches; bounded tushare pulls ONLY for Track A metadata/coverage;
ds/DeepSeek only for Track P reasoning; keys only from `/data/cyx/1030/api` or
env, never printed/committed; writes only under the dated run/report dirs and
`remote_decision.md`; no commit/push/reset/clean; no GPU; no large rebuilds.

Expected outputs:

- `runs/p1_round4_stats_hardening_pit_sources_20260702/RUN_STATUS.md`;
- `reports/date_generalization/p1_round4_stats_hardening_pit_sources_20260702/validation_summary.md`;
- Track S: `route_hypothesis_registry.csv`, `ic_newey_west_by_block.csv`,
  `net_spread_bootstrap_ci.csv`, `multiple_testing_fdr_summary.md`;
- Track A: the round-3 track (a) tables (`hotrank_ashare_available_at_policy.csv`,
  `hotrank_coverage_by_block.csv`, `hotrank_source_semantic_contract.md`);
- Track U: `universe_survivorship_audit.md`, `label_alignment_distribution.csv`,
  optional `frozen_score_corrected_label_deltas.csv`;
- Track F: `preregistration_h2026_forward_v1.md` (hash-logged);
- Track P: optional `p1_candidate_dryrun.csv` (guarded, no future/GT fields).

DONE criteria (stage ends by proposing exactly one next bounded stage):

- branch/HEAD/dirty state reported; no `local_*.md` edits;
- Track S: registry + NW/bootstrap tables + FDR survivor count + the H2026_1
  contamination statement written;
- Track A: three-number tables + available_at_policy + usable/D+1/CLOSED
  verdict per source;
- Track U: PIT verdict + label-span distribution + (if run) corrected deltas;
- Track F: pre-registration doc exists and is hash-logged BEFORE any forward
  read; confirm the forward window untouched;
- leakage PASS (hits=0) for anything decision-time; low/zero exposure labeled
  defensive; every negative logged-and-continued in `remote_decision.md`;
- exactly one proposed next stage tied back to the durable 60% win-rate target
  (measured honestly) and to the Track-F promotion path.

Stop rules: unchanged from round 3 — soft `ROUTE_PIVOT`/`SOFT_BLOCK` with
continued safe work for missing caches/unboundable subtasks; hard `BLOCKED`
only for user-owned decisions (target/resource/credential/broker/destructive).

## Superseded Round-3 Task (evidence only — do NOT execute as written; Track (a) spec below is still referenced by Track A above)

Date: 2026-07-02 (round 3)

Task ID: `p1_hotrank_availableat_and_aftercost_reaudit_20260702`.

This is a PARALLEL, LONG-RUNNING stage, not a short job. It supersedes the
completed `p1_ranker_guard_integration_20260702` (frozen score already downgraded
to `observe_only`) and all completed inventory / 7-family / direct-broker-selector
routes. Do NOT re-run any of those. Under the highest-priority principles above,
remote runs tracks (a) and (c) in parallel and keeps P0/P1 product value moving,
exercising initiative and pulling subagents as needed. after-cost NOT passing is
NOT `BLOCKED` — log it, switch strategy, continue.

Prior-stage facts confirmed by local audit (build on, do not redo): frozen score
`observe_only`, reversal `suppress`, both after-cost NEGATIVE on H2026_1; broker
sparse selector CLOSED (after-cost `-2.29pp` / `-4.03pp`); hk_hold partially
joinable (mean match `0.3916`, min-block `0.0`); hot-rank `ths_hot`/`dc_hot` have
`has_rank_time=True` but thin coverage and mixed markets; `moneyflow_hsgt` is
`regime_context_only`; `quality_momentum_accumulation_v1` is the ONE non-negative
H2026 net-spread row (`+1.2098`) but `not_promoted` pending an available-at/lag
audit.

### Track (a): hot-rank A-share available-at audit (NO label, NO model)

For `ths_hot` and `dc_hot` (source dir already probed under
`p1_tushare_specialty_lag_coverage_20260702`), produce a decision-time-safety
contract WITHOUT building labels or models:

1. Filter to A-share only: partition rows by `data_type`/market and by `ts_code`
   suffix (`.SH`/`.SZ`/`.BJ` A-share vs `.HK`/US/concept-board), and report the
   drop at each stage. Explicitly separate THREE numbers per block: raw endpoint
   rows -> A-share joinable rows (after suffix/market filter, joined to the
   decision universe) -> decision-universe match rate. These three are NOT the
   same and must never be conflated.
2. rank_time / D+1 audit: using the existing `has_rank_time=True` timestamp,
   verify whether the rank is available strictly BEFORE the decision close. If the
   timestamp is intraday/after-close, anchor availability to D+1 (next natural
   trading day) and report `coverage_by_block` under that D+1 anchor.
3. Emit: an `available_at_policy` (per source: usable same-day vs D+1 vs closed),
   `coverage_by_block` (A-share joinable, by H2023_1..H2026_1), and a
   `source_semantic_contract` (what each row means, market mix, rank semantics).
4. Decision rule: if A-share coverage is too thin OR rank_time is later than the
   trade decision -> mark D+1-contract or CLOSED and do NOT admit into labels.
   Only if a source passes filter + rank_time + coverage does it become a
   `needs_leakage_audit -> ready_to_test` label candidate for a LATER stage.

### Track (c): after-cost protocol RE-AUDIT (is the cost model over-strict?)

The current protocol subtracts a flat 1.5% round-trip cost every rebalance
(`net = gross - turnover*1.5%`) and a flat 1.5% floor. Re-audit whether this
wording is CONSISTENT with the product's real holding period and cost, or whether
it artificially kills strategies:

1. Turnover/exposure accounting: check how turnover is measured (per-rebalance
   vs annualized), whether the flat-1.5% floor double-counts a cost already in the
   turnover term, and whether the rebalance cadence assumed matches a realistic
   product cadence.
2. Holding-period sensitivity: recompute the frozen score's after-cost net spread
   under LOWER-turnover / MONTHLY-rebalance / netting assumptions (offline labels
   for eval only). The user's likely intent is a lower-churn, monthly-rebalanced
   product; report whether monthly netting flips any block from negative to
   non-negative after cost.
3. Verdict: state clearly whether the 1.5%-every-rebalance protocol is
   "pinning strategies dead" vs a fair floor, and recommend the cost/turnover/
   holding-period convention that should be the accepted after-cost gate going
   forward. Do not silently weaken the gate — propose the change with evidence.

### Product-value track (always on, independent of after-cost alpha)

Keep advancing the honest P0/P1 research-assistant deliverable (single-stock
watch action card + small candidate-set guarded ranking with evidence,
counter-evidence, exposure downgrade, coverage) EVEN IF no after-cost alpha
exists yet. Product value (a useful, honest research assistant) is DECOUPLED from
"a selector that beats cost". Deliver product value regardless.

### Autonomous strategy exploration (remote is expected to think, not just execute)

Local audit proposes these as PARALLEL hypotheses (see `local_suggestion.md` for
detail); remote should pick, adapt, or replace them and record choices in
`remote_decision.md`:

- low-turnover / monthly rebalance + netting to survive cost;
- use the frozen scorer as a RANKING PRIOR inside a cost-aware, exposure-gated
  selection, not as a direct selector;
- regime-conditional / timing overlay (reversal alpha died in H2026 — gate it by
  regime; `moneyflow_hsgt` is available as a regime-context signal);
- decouple product value from beating cost (ship the honest research assistant).

Allowed inputs:

- the active first-read files listed above;
- server-local context (read-only): the frozen model card/joblib and its accuracy
  report; `supervised_ranker_experiment_v2.md`; `feature_rank_ic_audit.csv`; the
  ranker-guard report dir `p1_ranker_guard_integration_20260702/`; the specialty
  probe dirs `p1_tushare_specialty_lag_coverage_20260702/` and
  `p1_tushare_specialty_endpoint_inventory_20260702/`; the hk_hold audit dirs; the
  broker gate dir; the offline joined ground-truth cache under
  `data/date_generalization_cache/` (labels offline-eval only);
- tushare specialty endpoints under `/data/cyx/1030/api` MAY be pulled for the
  hot-rank source-boundary audit (metadata/coverage only — no label persistence in
  track (a)); ds/DeepSeek MAY be used for P0/P1 product-value reasoning. Load keys
  only from `/data/cyx/1030/api` or env; never print/commit.

Allowed commands:

- read-only inspection, `git status --short`, `git diff --stat`, file metadata;
- CPU-only pandas recompute over offline caches for the after-cost re-audit and
  the hot-rank coverage matrix; loading the local `model.joblib` is allowed;
- bounded tushare specialty pulls for track (a) source-boundary audit (small,
  metadata/coverage only, no unbounded loops);
- CPU-only syntax/AST checks; existing tests only if they provably touch no
  network/paid API/secret/large rebuild, else skip and record why;
- launching remote read-only/code-review subagents for route audit/optimization;
- write the status/report files below.

Expected outputs (create the dated dirs as needed):

- `runs/p1_hotrank_availableat_and_aftercost_reaudit_20260702/RUN_STATUS.md`;
- `reports/date_generalization/p1_hotrank_availableat_and_aftercost_reaudit_20260702/validation_summary.md`;
- track (a) tables: `hotrank_ashare_available_at_policy.csv`,
  `hotrank_coverage_by_block.csv`, `hotrank_source_semantic_contract.md`;
- track (c) tables: `after_cost_protocol_reaudit.md`,
  `after_cost_sensitivity_by_holding_period.csv`;
- optional product-value artifact: `p1_candidate_dryrun.csv` (guarded ranking on
  a user-style handful; no future/GT fields).

DONE criteria for this stage (it ends by proposing exactly one next stage, not by
declaring the project done):

- report branch/HEAD/dirty state and confirm no `local_*.md` edits;
- track (a): the THREE-number separation (raw rows / A-share joinable / match
  rate) per source per block, an available_at_policy, coverage_by_block, and a
  source_semantic_contract; an explicit usable / D+1 / CLOSED verdict per source;
- track (c): a written verdict on whether the 1.5%-every-rebalance protocol is
  over-strict, a monthly/low-turnover after-cost sensitivity table, and a
  recommended accepted cost/holding-period convention;
- at least one product-value or strategy-exploration result logged in
  `remote_decision.md` (e.g. ranking-prior-in-cost-aware-selection, regime
  overlay, or the shipped honest assistant), even if after-cost alpha is still
  absent;
- leakage PASS (hits=0) for any decision-time output; low/zero exposure explicitly
  called defensive/no-action;
- exactly one proposed next bounded stage tied back to the durable 60% win-rate
  target and after-cost reality.

Resource limits:

- CPU/read-mostly; this is a long-running parallel stage, so pace it — target one
  full (a)+(c) pass per session, then continue. No single unbounded loop.
- Writes allowed under the two dated dirs above and `remote_decision.md`.
- Paid APIs ARE provisioned and authorized: tushare (A-share market data) and
  ds/DeepSeek LLM, credentials under `/data/cyx/1030/api` — bounded pulls for the
  source-boundary audit and product-value reasoning are allowed; load keys only
  from there or env, never print/commit, keep cost reasonable. No broker/live
  trading, no GPU, no large cache rebuilds without a dated plan.
- Do not commit, push, reset, delete, or clean files.

Stop rules:

- soft: stop the current subtask, record a `SOFT_BLOCK` or `ROUTE_PIVOT` in
  `remote_decision.md`, and continue another safe sub-goal / strategy if a report
  or cache is missing or a boundary cannot be bounded. after-cost failing is a
  soft pivot, NEVER a hard block;
- hard `BLOCKED` only for genuine user-owned decisions: changing the final
  goal/threshold, new credentials/permissions, destructive operations, or
  broker/live trading.

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
