from __future__ import annotations

import pandas as pd


def decision_dates(daily: pd.DataFrame) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(daily["date"]).dropna().sort_values().unique())
    if dates.empty:
        return []
    selected: list[pd.Timestamp] = []
    by_week = {}
    for value in dates:
        ts = pd.Timestamp(value)
        iso = ts.isocalendar()
        by_week.setdefault((int(iso.year), int(iso.week)), []).append(ts)
    for week_dates in by_week.values():
        selected.extend(_nearest_weekday(week_dates, 1))
        selected.extend(_nearest_weekday(week_dates, 4))
    return sorted(set(selected))


def _nearest_weekday(week_dates: list[pd.Timestamp], weekday: int) -> list[pd.Timestamp]:
    if not week_dates:
        return []
    exact = [d for d in week_dates if d.weekday() == weekday]
    if exact:
        return exact[:1]
    before = [d for d in week_dates if d.weekday() < weekday]
    after = [d for d in week_dates if d.weekday() > weekday]
    if before:
        return [before[-1]]
    if after:
        return [after[0]]
    return []

