from __future__ import annotations

import pandas as pd


def write_candidate_matrix(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame([{"股票": "信息不足", "研究分级": "信息不足"}])
    df.to_excel(path, index=False)
