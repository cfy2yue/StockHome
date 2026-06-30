"""One-shot permission smoke for behavioral/capital-flow Tushare interfaces.

Does NOT print token. First stdout line: A股研究Agent.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.tushare_pro_adapter import TushareCacheConfig, TushareProAdapter  # noqa: E402

TRADE_DATE = "20260105"
SAMPLE_TS = "000001.SZ"


def _nonempty_cols(df) -> dict[str, int]:
    out: dict[str, int] = {}
    if df is None or df.empty:
        return out
    for col in df.columns:
        s = df[col]
        nn = int(s.notna().sum())
        if nn == 0:
            continue
        # treat empty strings as null
        if s.dtype == object:
            nn = int(s.astype(str).str.strip().replace({"": None, "None": None, "nan": None}).notna().sum())
        if nn > 0:
            out[str(col)] = nn
    return out


def _probe(adapter: TushareProAdapter, name: str, **params) -> dict:
    result = {"interface": name, "params": params, "status": "ok", "rows": 0, "nonempty_cols": {}, "error": ""}
    try:
        df = adapter.call(name, **params)
        result["rows"] = len(df)
        result["nonempty_cols"] = _nonempty_cols(df)
        if result["rows"] == 0:
            result["status"] = "empty"
        elif not result["nonempty_cols"]:
            result["status"] = "all_null"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    return result


def main() -> None:
    print("A股研究Agent")
    config = TushareCacheConfig()
    adapter = TushareProAdapter(config)

    probes = [
        ("cyq_perf", {"ts_code": SAMPLE_TS, "trade_date": TRADE_DATE}),
        ("cyq_perf", {"trade_date": TRADE_DATE}),
        ("margin_detail", {"trade_date": TRADE_DATE}),
        ("moneyflow_hsgt", {"trade_date": TRADE_DATE}),
        ("moneyflow_hsgt", {"start_date": TRADE_DATE, "end_date": TRADE_DATE}),
        ("hk_hold", {"trade_date": TRADE_DATE}),
        ("hk_hold", {"ts_code": SAMPLE_TS, "start_date": "20260101", "end_date": TRADE_DATE}),
    ]

    print(f"trade_date={TRADE_DATE} sample_ts={SAMPLE_TS}")
    print("=" * 72)
    for name, params in probes:
        r = _probe(adapter, name, **params)
        usable = r["status"] == "ok" and r["rows"] > 0 and bool(r["nonempty_cols"])
        tag = "USABLE" if usable else r["status"].upper()
        print(f"\n[{tag}] {name} params={params}")
        print(f"  rows={r['rows']}")
        if r["error"]:
            print(f"  error={r['error']}")
        if r["nonempty_cols"]:
            cols = sorted(r["nonempty_cols"].items(), key=lambda x: -x[1])
            print(f"  nonempty_cols ({len(cols)}): " + ", ".join(f"{c}({n})" for c, n in cols[:20]))
            if len(cols) > 20:
                print(f"    ... +{len(cols)-20} more")


if __name__ == "__main__":
    main()
