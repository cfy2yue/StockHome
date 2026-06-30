"""Single-stock watch / mine-sweep product MVP (rev+chip_core opportunity tiers + risk flags).

Time-safe: decision_date and prior features only. Labels never enter evidence.
Research-only four grades: 继续深挖 / 放入观察 / 暂时剔除 / 信息不足.
"""
from __future__ import annotations

import glob
import gzip
import io
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.agent_training.decision_card import ALLOWED_RESEARCH_GRADES, ALLOWED_SIMULATED_ACTIONS, normalize_action_weight
from src.agent_training.dual_mode_round import TIME_BLOCKS

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "data" / "date_generalization_cache" / "market_5000"
CYQ_DIR = ROOT / "data" / "date_generalization_cache" / "tushare_pro" / "tables" / "cyq_perf"
DAILY_DIR = ROOT / "data" / "date_generalization_cache" / "tushare_pro" / "tables" / "daily"

FINAL_OOT_BLOCK = "H2026_1"
POLICY_VERSION = "single_stock_review_mvp_v1"
OPPORTUNITY_TIERS = ("强", "中", "弱", "无")
TIER_QUANTILES_SEARCH = (0.55, 0.65, 0.75, 0.85)
MEDIUM_QUANTILE = 0.55
WEAK_QUANTILE = 0.35
MIN_CALIB_SAMPLES = 80

CHIP_CORE_COLS = [
    "lower_support",
    "chip_concentration",
    "cost_band_width",
    "upper_overhang",
    "winner_rate_pct",
    "neg_winner_rate",
]
CHIP_CLOSE_COLS = ["cost_position", "neg_cost_position", "price_vs_median_cost", "chip_range_position"]

RISK_FLAG_DEFS: dict[str, str] = {
    "chase_high": "获利盘偏高且价近筹码区间高位（追高风险）",
    "extreme_cost_high": "价显著高于筹码下沿（成本位极端偏高）",
    "extreme_cost_low": "价显著低于筹码下沿（破位/陷阱风险）",
    "upper_overhang_rally": "上方筹码悬空叠加近期大涨",
    "volume_stall": "放量滞涨（短涨后波动放大但近端回落）",
    "deep_drawdown_accel": "深回撤叠加近期加速下跌",
    "peer_weakness": "同行/相关股显著弱于市场",
    "chip_loosen_rally": "大涨后获利盘偏高（筹码松动风险）",
}


def _norm(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.lstrip("\ufeff") for c in out.columns]
    return out


def _norm_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = _norm(df)
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.date.astype(str)
    return out


def _z(s: pd.Series) -> pd.Series:
    sd = s.std()
    return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0


def _safe(value: Any) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return v if not math.isnan(v) else float("nan")


def load_trade_date_table(table_dir: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(table_dir / "trade_date_*.csv")))
    if not files:
        raise FileNotFoundError(f"no cached files in {table_dir}")
    frames: list[pd.DataFrame] = []
    for path in files:
        try:
            d = pd.read_csv(path)
        except Exception:
            continue
        if "ts_code" not in d.columns:
            continue
        frames.append(_norm(d))
    out = pd.concat(frames, ignore_index=True)
    out["code"] = out["ts_code"].astype(str).str[:6]
    out["date"] = pd.to_datetime(out["trade_date"].astype(str), format="%Y%m%d", errors="coerce").dt.date.astype(str)
    return out


def build_cyq_features(cyq: pd.DataFrame, daily: pd.DataFrame | None) -> pd.DataFrame:
    df = cyq.copy()
    for c in [
        "cost_5pct", "cost_15pct", "cost_50pct", "cost_85pct", "cost_95pct",
        "weight_avg", "winner_rate", "his_low", "his_high",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    span = (df["cost_95pct"] - df["cost_5pct"]).replace(0, np.nan)
    df["winner_rate_pct"] = df["winner_rate"]
    df["neg_winner_rate"] = -df["winner_rate"]
    df["chip_concentration"] = (df["cost_85pct"] - df["cost_15pct"]) / df["cost_50pct"].clip(lower=1e-6)
    df["cost_band_width"] = span / df["cost_50pct"].clip(lower=1e-6)
    df["upper_overhang"] = (df["cost_95pct"] - df["cost_50pct"]) / df["cost_50pct"].clip(lower=1e-6)
    df["lower_support"] = (df["cost_50pct"] - df["cost_5pct"]) / df["cost_50pct"].clip(lower=1e-6)

    if daily is not None and not daily.empty:
        d = daily[["date", "code", "close"]].copy()
        d["close"] = pd.to_numeric(d["close"], errors="coerce")
        df = df.merge(d, on=["date", "code"], how="left")
        df["cost_position"] = (df["close"] - df["cost_5pct"]) / span
        df["price_vs_median_cost"] = (df["close"] - df["cost_50pct"]) / df["cost_50pct"].clip(lower=1e-6)
        df["neg_cost_position"] = -df["cost_position"]
        hi_lo = (df["his_high"] - df["his_low"]).replace(0, np.nan)
        df["chip_range_position"] = (df["close"] - df["his_low"]) / hi_lo
    return df


def load_reversal_frame(labels: pd.DataFrame) -> pd.DataFrame:
    kl = _norm_keys(
        pd.read_csv(
            io.StringIO(gzip.open(BASE / "daily_kline_multiscale_features.csv.gz", "rt").read()),
            usecols=["date", "code", "kline_return_20d", "kline_return_60d", "kline_return_3d",
                     "kline_return_5d", "kline_drawdown_60d", "kline_volatility_ratio_3_20"],
        )
    )
    cp = _norm_keys(pd.read_csv(BASE / "corr_peer_kline_features.csv", usecols=["date", "code", "corr_peer_avg_return_20d"]))
    keys = labels[["date", "code"]].drop_duplicates()
    rev = keys.merge(kl, on=["date", "code"], how="inner").merge(cp, on=["date", "code"], how="inner")
    parts = [
        -rev.groupby("date")[c].transform(_z)
        for c in ["kline_return_20d", "kline_return_60d", "corr_peer_avg_return_20d"]
    ]
    rev["reversal_composite"] = sum(parts) / len(parts)
    return rev


def composite_equal_z(frame: pd.DataFrame, cols: list[str], out_col: str) -> pd.DataFrame:
    work = frame.copy()
    avail = [c for c in cols if c in work.columns]
    if not avail:
        work[out_col] = np.nan
        return work
    zparts = [work.groupby("date")[c].transform(_z) for c in avail]
    work[out_col] = sum(zparts) / len(zparts)
    return work


def compute_risk_flags(frame: pd.DataFrame, *, enhanced: bool = True) -> pd.DataFrame:
    """Rule-based risk flags from chip / price-volume (time-safe)."""
    out = frame.copy()
    wr = pd.to_numeric(out.get("winner_rate_pct"), errors="coerce")
    crp = pd.to_numeric(out.get("chip_range_position"), errors="coerce")
    cp = pd.to_numeric(out.get("cost_position"), errors="coerce")
    uo = pd.to_numeric(out.get("upper_overhang"), errors="coerce")
    r20 = pd.to_numeric(out.get("kline_return_20d"), errors="coerce")
    r5 = pd.to_numeric(out.get("kline_return_5d"), errors="coerce")
    r3 = pd.to_numeric(out.get("kline_return_3d"), errors="coerce")
    dd60 = pd.to_numeric(out.get("kline_drawdown_60d"), errors="coerce")
    vol_ratio = pd.to_numeric(out.get("kline_volatility_ratio_3_20"), errors="coerce")
    peer = pd.to_numeric(out.get("corr_peer_avg_return_20d"), errors="coerce")

    flags: dict[str, pd.Series] = {}
    flags["chase_high"] = (wr >= 75) & (crp >= 0.85)
    flags["extreme_cost_high"] = cp >= 0.92
    flags["extreme_cost_low"] = cp <= 0.08
    flags["upper_overhang_rally"] = (uo >= 0.25) & (r20 >= 10)
    flags["volume_stall"] = (r20 >= 8) & (r3 <= 0) & (vol_ratio >= 1.3)
    flags["deep_drawdown_accel"] = (dd60 <= -25) & (r5 <= -3)
    flags["peer_weakness"] = peer <= -5

    if enhanced:
        flags["chip_loosen_rally"] = (wr >= 70) & (r20 >= 15)
    else:
        flags["chip_loosen_rally"] = pd.Series(False, index=out.index)

    flag_cols = list(flags.keys())
    for name, series in flags.items():
        out[f"risk_flag_{name}"] = series.fillna(False).astype(int)

    out["risk_flag_count"] = out[[f"risk_flag_{c}" for c in flag_cols]].sum(axis=1)
    out["risk_flag_names"] = out.apply(
        lambda r: ";".join([c for c in flag_cols if int(r.get(f"risk_flag_{c}", 0)) == 1]),
        axis=1,
    )
    # baseline score: only structural flags (no chip_loosen)
    base_cols = [f"risk_flag_{c}" for c in flag_cols if c != "chip_loosen_rally"]
    out["risk_flag_count_baseline"] = out[base_cols].sum(axis=1)
    return out


@dataclass
class TierThresholds:
    strong: float
    medium: float
    weak: float
    regime_signal_mean: float = 0.0
    regime_abstain: bool = False

    def tier_for_score(self, score: float) -> str:
        if math.isnan(score):
            return "无"
        if score >= self.strong:
            return "强"
        if score >= self.medium:
            return "中"
        if score >= self.weak:
            return "弱"
        return "无"


@dataclass
class RiskThresholds:
    flag_count_min: int = 2
    single_critical_flags: tuple[str, ...] = ("deep_drawdown_accel", "extreme_cost_low")


def _selection_score(metrics: dict[str, Any]) -> float:
    if not metrics.get("sample_count"):
        return float("-inf")
    return (
        float(metrics["avg_return_20d"])
        + 10 * float(metrics["positive_20d_rate"])
        - 7 * float(metrics["loss_gt5_rate"])
    )


def choose_strong_threshold(validation: pd.DataFrame, score_col: str = "rev_chip_core_score") -> float:
    """Validation-only threshold search (not cross-section TopK)."""
    scores = pd.to_numeric(validation[score_col], errors="coerce").dropna()
    if scores.empty:
        return 999.0
    best: tuple[float, float] | None = None
    for quantile in TIER_QUANTILES_SEARCH:
        thr = float(scores.quantile(quantile))
        sel = validation[validation[score_col] >= thr]
        m = side_metrics(sel, {"base_pos": 0, "base_mean_ret": 0, "base_loss_gt5": 0})
        if m["sample_count"] < MIN_CALIB_SAMPLES:
            continue
        score = _selection_score(m)
        if best is None or score > best[0]:
            best = (score, thr)
    if best is None:
        return float(scores.quantile(0.75))
    return best[1]


def calibrate_tier_thresholds(validation: pd.DataFrame, score_col: str = "rev_chip_core_score") -> TierThresholds:
    scores = pd.to_numeric(validation[score_col], errors="coerce")
    valid = scores.dropna()
    if len(valid) < MIN_CALIB_SAMPLES:
        return TierThresholds(strong=999.0, medium=999.0, weak=999.0, regime_abstain=True)

    strong_thr = choose_strong_threshold(validation, score_col)
    regime_mean = float(valid.mean())
    regime_abstain = regime_mean < -0.15

    return TierThresholds(
        strong=strong_thr,
        medium=float(valid.quantile(MEDIUM_QUANTILE)),
        weak=float(valid.quantile(WEAK_QUANTILE)),
        regime_signal_mean=regime_mean,
        regime_abstain=regime_abstain,
    )


def map_research_grade(
    *,
    opportunity_tier: str,
    risk_flag_count: int,
    risk_flag_names: str,
    data_missing: bool,
    regime_abstain: bool,
    confidence: float,
    risk_thr: RiskThresholds = RiskThresholds(),
) -> tuple[str, str, float]:
    """Map tiers + risk to research_grade, simulated_action, weight."""
    critical = any(f in risk_flag_names for f in risk_thr.single_critical_flags)
    if data_missing:
        return "信息不足", "信息不足不动作", 0.0
    if risk_flag_count >= risk_thr.flag_count_min or critical:
        return "暂时剔除", "降低研究暴露", normalize_action_weight("降低研究暴露", 0.05)
    if regime_abstain and opportunity_tier in ("强", "中"):
        return "放入观察", "保持观察", normalize_action_weight("保持观察", 0.1)
    if opportunity_tier == "强" and confidence >= 0.45:
        return "继续深挖", "增加研究暴露", normalize_action_weight("增加研究暴露", 0.7)
    if opportunity_tier in ("强", "中"):
        return "放入观察", "保持观察", normalize_action_weight("保持观察", 0.15)
    if opportunity_tier == "弱":
        return "放入观察", "保持观察", normalize_action_weight("保持观察", 0.08)
    if confidence < 0.25 or regime_abstain:
        return "信息不足", "信息不足不动作", 0.0
    return "放入观察", "保持观察", normalize_action_weight("保持观察", 0.05)


def compute_confidence(
    row: pd.Series | dict[str, Any],
    opportunity_tier: str,
    risk_flag_count: int,
    *,
    regime_abstain: bool,
) -> float:
    score = _safe(row.get("rev_chip_core_score"))
    tier_bonus = {"强": 0.35, "中": 0.22, "弱": 0.10, "无": 0.0}.get(opportunity_tier, 0.0)
    base = 0.25 + tier_bonus
    if not math.isnan(score):
        base += max(-0.15, min(0.25, score * 0.08))
    base -= 0.12 * risk_flag_count
    if regime_abstain:
        base -= 0.15
    missing = row.get("data_missing_flags") or row.get("missing_flags")
    if missing and str(missing).strip() not in {"", "none", "无"}:
        base -= 0.2
    return round(max(0.0, min(1.0, base)), 4)


def build_review_frame() -> tuple[pd.DataFrame, list[str]]:
    """Load labels + kline + chip cache and compute rev+chip_core scores."""
    labels = _norm_keys(pd.read_csv(BASE / "task_labels_v1.csv"))
    rev = load_reversal_frame(labels)

    cyq = load_trade_date_table(CYQ_DIR)
    daily = load_trade_date_table(DAILY_DIR)
    chip = build_cyq_features(cyq, daily)

    core_avail = [c for c in CHIP_CORE_COLS if c in chip.columns]
    close_avail = [c for c in CHIP_CLOSE_COLS if c in chip.columns]
    chip_sub = chip[["date", "code"] + core_avail + close_avail]

    merged = labels.merge(rev, on=["date", "code"], how="inner")
    merged = merged.merge(chip_sub, on=["date", "code"], how="inner")
    merged = composite_equal_z(
        merged,
        ["reversal_composite"] + core_avail,
        "rev_chip_core_score",
    )
    merged = compute_risk_flags(merged, enhanced=True)
    merged["positive_20d"] = pd.to_numeric(merged["return_20d"], errors="coerce").gt(0).astype(float)
    merged["loss_gt5_flag"] = (pd.to_numeric(merged["return_20d"], errors="coerce") <= -5).astype(float)

    notes = [
        f"chip_core={core_avail}",
        f"chip_close={close_avail}",
        f"cyq_dates={cyq['date'].nunique()}",
    ]
    return merged, notes


def review_row(
    row: pd.Series | dict[str, Any],
    tier_thr: TierThresholds,
    *,
    valid_block: str = "",
    train_blocks: str = "",
) -> dict[str, Any]:
    """Build product decision card for one (code, decision_date) row."""
    score = _safe(row.get("rev_chip_core_score"))
    tier = tier_thr.tier_for_score(score)
    risk_count = int(row.get("risk_flag_count") or 0)
    risk_names = str(row.get("risk_flag_names") or "")
    missing_flags: list[str] = []
    if math.isnan(score):
        missing_flags.append("rev_chip_core_missing")
    for col in CHIP_CORE_COLS:
        if col in row and pd.isna(pd.to_numeric(row.get(col), errors="coerce")):
            missing_flags.append(f"missing_{col}")
    data_missing = bool(missing_flags) and math.isnan(score)

    confidence = compute_confidence(
        row, tier, risk_count, regime_abstain=tier_thr.regime_abstain,
    )
    grade, action, weight = map_research_grade(
        opportunity_tier=tier,
        risk_flag_count=risk_count,
        risk_flag_names=risk_names,
        data_missing=data_missing,
        regime_abstain=tier_thr.regime_abstain,
        confidence=confidence,
    )

    flag_text = ";".join(
        [RISK_FLAG_DEFS.get(n, n) for n in risk_names.split(";") if n]
    ) or "无显著排雷旗标"

    card = {
        "type": "single_stock_review_card",
        "agent_policy_version": POLICY_VERSION,
        "variant": "rev_plus_chip_core_mvp",
        "step": 0,
        "train_blocks": train_blocks,
        "valid_block": valid_block,
        "decision_date": row.get("date"),
        "code": str(row.get("code")).zfill(6),
        "name": row.get("name") or "脱敏标的",
        "task_mode": "single_stock",
        "research_grade": grade,
        "simulated_action": action,
        "simulated_weight_change": weight,
        "python_signal_summary": (
            f"rev_chip_core={score:.4f}; tier={tier}; "
            f"reversal={_safe(row.get('reversal_composite')):.3f}; "
            f"regime_abstain={tier_thr.regime_abstain}"
        ),
        "news_signal_summary": "single_stock_review: news channel not in MVP scoring",
        "book_skill_evidence": "none_in_mvp",
        "memory_experience_used": "regime_abstain" if tier_thr.regime_abstain else "none",
        "counter_evidence": flag_text if risk_count else "无强反证",
        "final_agent_reasoning_summary": (
            f"机会分级={tier}；排雷旗标={risk_count}个；"
            f"置信={confidence}；regime均值={tier_thr.regime_signal_mean:.3f}"
        ),
        "confidence_level": confidence,
        "data_missing_flags": ";".join(missing_flags) if missing_flags else "none",
        "error_reflection": "",
        "research_only": True,
        "not_investment_instruction": True,
        "opportunity_tier": tier,
        "risk_flag_count": risk_count,
        "risk_flag_names": risk_names,
        "rev_chip_core_score": round(score, 6) if not math.isnan(score) else None,
        "tier_thresholds": {
            "strong": tier_thr.strong,
            "medium": tier_thr.medium,
            "weak": tier_thr.weak,
        },
    }
    if card["research_grade"] not in ALLOWED_RESEARCH_GRADES:
        raise ValueError(f"invalid grade: {card['research_grade']}")
    if card["simulated_action"] not in ALLOWED_SIMULATED_ACTIONS:
        raise ValueError(f"invalid action: {card['simulated_action']}")
    return card


def select_risk_flagged(
    frame: pd.DataFrame,
    *,
    enhanced: bool = True,
    flag_count_min: int = 2,
    mode: str = "exclude",
) -> pd.DataFrame:
    """Select rows flagged for risk review or exclusion.

    mode=review: flag_count>=1 (enhanced) for recall-oriented mine-sweep.
    mode=exclude: flag_count>=2 or critical single flags for 暂时剔除 mapping.
    """
    col = "risk_flag_count" if enhanced else "risk_flag_count_baseline"
    if mode == "review" and enhanced:
        flagged = frame[frame[col] >= 1].copy()
    else:
        flagged = frame[frame[col] >= flag_count_min].copy()
    critical = frame["risk_flag_names"].str.contains("deep_drawdown_accel|extreme_cost_low", regex=True, na=False)
    if mode == "exclude":
        flagged = pd.concat([flagged, frame[critical]], ignore_index=True).drop_duplicates(["date", "code"])
    return flagged


def block_base_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    vals = pd.to_numeric(frame["return_20d"], errors="coerce").dropna()
    if vals.empty:
        return {"n": 0, "base_pos": np.nan, "base_loss_gt5": np.nan, "base_mean_ret": np.nan}
    return {
        "n": int(len(vals)),
        "base_pos": round(float((vals > 0).mean()), 4),
        "base_loss_gt5": round(float((vals <= -5).mean()), 4),
        "base_mean_ret": round(float(vals.mean()), 4),
    }


def side_metrics(selected: pd.DataFrame, base: dict[str, Any]) -> dict[str, Any]:
    vals = pd.to_numeric(selected.get("return_20d"), errors="coerce").dropna()
    if vals.empty:
        return {
            "sample_count": 0,
            "positive_20d_rate": np.nan,
            "avg_return_20d": np.nan,
            "loss_gt5_rate": np.nan,
            "delta_pos_vs_base": np.nan,
            "delta_mean_vs_base": np.nan,
            "delta_loss_vs_base": np.nan,
        }
    pos = float((vals > 0).mean())
    avg = float(vals.mean())
    loss = float((vals <= -5).mean())
    return {
        "sample_count": int(len(vals)),
        "positive_20d_rate": round(pos, 4),
        "avg_return_20d": round(avg, 4),
        "loss_gt5_rate": round(loss, 4),
        "delta_pos_vs_base": round(pos - float(base["base_pos"]), 4) if pd.notna(base["base_pos"]) else np.nan,
        "delta_mean_vs_base": round(avg - float(base["base_mean_ret"]), 4) if pd.notna(base["base_mean_ret"]) else np.nan,
        "delta_loss_vs_base": round(loss - float(base["base_loss_gt5"]), 4) if pd.notna(base["base_loss_gt5"]) else np.nan,
    }


def risk_recall_precision(pool: pd.DataFrame, flagged: pd.DataFrame) -> dict[str, float]:
    bad = pool[pd.to_numeric(pool["return_20d"], errors="coerce") <= -5]
    if bad.empty:
        return {"risk_recall": np.nan, "risk_precision": np.nan}
    if flagged.empty:
        return {"risk_recall": 0.0, "risk_precision": np.nan}
    flagged_keys = set(zip(flagged["date"].astype(str), flagged["code"].astype(str)))
    hit = bad.apply(lambda r: (str(r["date"]), str(r["code"])) in flagged_keys, axis=1).sum()
    recall = float(hit / len(bad))
    flagged_loss = flagged[pd.to_numeric(flagged["return_20d"], errors="coerce") <= -5]
    precision = float(len(flagged_loss) / len(flagged)) if len(flagged) else np.nan
    return {"risk_recall": round(recall, 4), "risk_precision": round(precision, 4)}


def example_cards_for_block(
    frame: pd.DataFrame,
    tier_thr: TierThresholds,
    *,
    valid_block: str,
    train_blocks: str,
) -> list[dict[str, Any]]:
    """Pick anonymized example cards covering each research grade."""
    cards: list[dict[str, Any]] = []
    seen: set[str] = set()
    for _, row in frame.sample(n=min(len(frame), 500), random_state=42).iterrows():
        card = review_row(row, tier_thr, valid_block=valid_block, train_blocks=train_blocks)
        grade = card["research_grade"]
        if grade in seen:
            continue
        card["name"] = f"示例-{grade}"
        card["code"] = f"XX{len(seen)+1:02d}"
        cards.append(card)
        seen.add(grade)
        if len(seen) >= 4:
            break
    return cards
