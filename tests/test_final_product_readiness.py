from __future__ import annotations

import pandas as pd

import scripts.audit_final_product_readiness as readiness


def test_summarize_p0_selects_one_preferred_variant(tmp_path, monkeypatch) -> None:
    metrics = pd.DataFrame(
        [
            {
                "variant": "full_agent_with_opportunity_tool",
                "task_mode": "single_stock",
                "decision_cards": 6,
                "cash_adjusted_avg_return_20d": 0.1,
                "cash_adjusted_positive_20d_rate": 0.5,
                "active_exposure": 0.08,
                "invalid_outputs": 0,
            },
            {
                "variant": "full_agent_without_opportunity_tool",
                "task_mode": "single_stock",
                "decision_cards": 6,
                "cash_adjusted_avg_return_20d": 0.3,
                "cash_adjusted_positive_20d_rate": 0.7,
                "active_exposure": 0.06,
                "invalid_outputs": 0,
            },
        ]
    )
    metrics.to_csv(tmp_path / "p0_metrics.csv", index=False)
    monkeypatch.setattr(readiness, "REPORT_DIR", tmp_path)
    monkeypatch.setattr(readiness, "P0_METRIC_FILES", {"p0_acceptance_multiblock_3panel_flash": "p0_metrics.csv"})

    summary = readiness.summarize_p0()

    assert int(summary.loc[0, "cards"]) == 6
    assert summary.loc[0, "selected_variant"] == "full_agent_without_opportunity_tool"
    assert summary.loc[0, "cash_avg20"] == 0.3


def test_final_readiness_scan_includes_current_p0_delivery_artifacts() -> None:
    scan_files = set(readiness.SCAN_FILES)

    expected = {
        "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_evidence_pack.jsonl",
        "p0_small_entry_pps_q017_userop72_fresh3_key_ablation_flash_v1_decision_ledger.jsonl",
        "p0_small_entry_general_channel_fresh3_key4_flash_v1_evidence_pack.jsonl",
        "p0_small_entry_general_channel_fresh3_key4_flash_v1_decision_ledger.jsonl",
        "p0_action_label_tool_flash_preflight_v2_pair_flash_evidence_pack.jsonl",
        "p0_action_label_tool_flash_preflight_v2_pair_flash_decision_ledger.jsonl",
    }

    assert expected <= scan_files


def test_final_readiness_boundary_scan_includes_mainline_and_case_memory() -> None:
    scan_files = {path.name for path in readiness.BOUNDARY_SCAN_FILES}

    assert "product_delivery_mainline_20260630.md" in scan_files
    assert "20h_product_execution_checklist_20260630.md" in scan_files
    assert "external_agent_audit_20260630.md" in scan_files
    assert "external_agent_audit_20260630_fermat.md" in scan_files
    assert "user_actionability_contract_audit_v1.md" in scan_files
    assert "user_intake_router_audit_v1.md" in scan_files
    assert "latest_rolling_product_risk_register_v1.md" in scan_files
    assert "p0_news_channel_policy_audit_v1.md" in scan_files
    assert "p0_news_channel_case_memory_v1.md" in scan_files
    assert "tool_adoption_contract_audit_v1.md" in scan_files
    assert "p0_user_operation_case_memory_ledger.csv" in scan_files
    assert "p0_news_channel_case_memory_ledger.csv" in scan_files
    assert "p0_user_operation_case_memory_v1.md" in scan_files


def test_summarize_p0_case_memory_requires_retrievable_p0_cases(tmp_path, monkeypatch) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    ledger = memory / "p0_user_operation_case_memory_ledger.csv"
    rows = [
        "case_id,source_round,task_mode,case_bucket,case_pattern,visible_conditions,countermeasure,status,source_ref",
    ]
    for index in range(20):
        rows.append(
            "P0CASE-{index:03d},audit,single_stock_watch,false_positive_buy,"
            "single_stock small_entry_branch news_missing_or_empty financial_missing_or_no_event peer_weak_or_lagging,"
            "news_missing_or_empty;financial_missing_or_no_event;peer_weak_or_lagging;small_entry_branch,"
            "cap position and require second check before increasing exposure,open,panel|H2026_1|2026-04-07|000892|case".format(index=index)
        )
    ledger.write_text("\n".join(rows), encoding="utf-8")

    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "P0_CASE_MEMORY_LEDGER", ledger)

    summary = readiness.summarize_p0_case_memory()

    assert summary.loc[0, "status"] == "ok"
    assert int(summary.loc[0, "rows"]) == 20
    assert int(summary.loc[0, "p0_cases_in_top5"]) >= 1
    assert "P0CASE-" in summary.loc[0, "applicable_cases"]


def test_summarize_p0_news_case_memory_requires_retrievable_news_cases(tmp_path, monkeypatch) -> None:
    memory = tmp_path / "memory"
    memory.mkdir()
    ledger = memory / "p0_news_channel_case_memory_ledger.csv"
    rows = [
        "case_id,source_round,task_mode,case_bucket,case_pattern,visible_conditions,countermeasure,status,source_ref",
    ]
    for index in range(12):
        rows.append(
            "P0NEWS-{index:03d},audit,single_stock_watch,missing_news_risk_false_veto,"
            "single_stock news_channel news missing no hard warning false veto news_missing_no_hard_warning financial_missing_or_no_event peer_weak_or_lagging small_entry_branch,"
            "news_missing_no_hard_warning;financial_missing_or_no_event;peer_weak_or_lagging;small_entry_branch,"
            "Do not blindly zero because news is missing; keep a low observation position or explicit re-entry trigger,open,panel|H2026_1|2026-04-03|000980|case".format(index=index)
        )
    ledger.write_text("\n".join(rows), encoding="utf-8")

    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "P0_NEWS_CASE_MEMORY_LEDGER", ledger)

    summary = readiness.summarize_p0_news_case_memory()

    assert summary.loc[0, "status"] == "ok"
    assert int(summary.loc[0, "rows"]) == 12
    assert int(summary.loc[0, "p0news_cases_in_top10"]) >= 1
    assert "P0NEWS-" in summary.loc[0, "applicable_cases"]


def test_boundary_scan_allows_negative_safety_language(tmp_path, monkeypatch) -> None:
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    safe_doc = report_dir / "safe.md"
    unsafe_doc = report_dir / "unsafe.md"
    safe_doc.write_text("系统不自动下单，不接券商，不保证收益；会做自动下单边界 scan。", encoding="utf-8")
    unsafe_doc.write_text("系统可以自动下单。", encoding="utf-8")

    monkeypatch.setattr(readiness, "REPORT_DIR", report_dir)
    monkeypatch.setattr(readiness, "SCAN_FILES", [])
    monkeypatch.setattr(readiness, "BOUNDARY_SCAN_FILES", [safe_doc, unsafe_doc])

    scan = readiness.scan_artifacts()

    by_file = {row["file"]: row["status"] for _, row in scan.iterrows()}
    assert by_file[str(safe_doc)] == "ok"
    assert by_file[str(unsafe_doc)] == "fail"


def test_summarize_rolling_preflight_reads_ready_gate(tmp_path, monkeypatch) -> None:
    gates_path = tmp_path / "rolling_gates.csv"
    pd.DataFrame(
        [
            {"gate": "P0_latest_sample_plan", "status": "pass"},
            {"gate": "P0_latest_dryrun_evidence", "status": "pass"},
            {"gate": "P1_rolling_newdata_preflight", "status": "pass_cross_sector_only"},
            {"gate": "rolling_confirmation_next_step", "status": "ready_for_bounded_flash"},
        ]
    ).to_csv(gates_path, index=False)
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "ROLLING_PREFLIGHT_GATES", gates_path)

    summary = readiness.summarize_rolling_preflight()

    assert summary.loc[0, "status"] == "pass"
    assert summary.loc[0, "next_step_status"] == "ready_for_bounded_flash"
    assert summary.loc[0, "p1_preflight"] == "pass_cross_sector_only"


def test_summarize_user_intake_router_requires_clarification_for_ambiguous(tmp_path, monkeypatch) -> None:
    detail_path = tmp_path / "user_intake_detail.csv"
    pd.DataFrame(
        [
            {"case_id": "single_buy", "status": "pass", "actual_ask": False},
            {"case_id": "single_risk", "status": "pass", "actual_ask": False},
            {"case_id": "multi_compare", "status": "pass", "actual_ask": False},
            {"case_id": "live_watch", "status": "pass", "actual_ask": False},
            {"case_id": "strategy_research", "status": "pass", "actual_ask": False},
            {"case_id": "ambiguous", "status": "pass", "actual_ask": True},
        ]
    ).to_csv(detail_path, index=False)
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "USER_INTAKE_ROUTER_DETAIL", detail_path)

    summary = readiness.summarize_user_intake_router()

    assert summary.loc[0, "status"] == "pass"
    assert int(summary.loc[0, "rows"]) == 6
    assert bool(summary.loc[0, "ambiguous_ask"])
    assert int(summary.loc[0, "clear_routes_without_ask"]) == 5


def test_summarize_latest_rolling_risk_register_reads_logged_not_complete(tmp_path, monkeypatch) -> None:
    gates_path = tmp_path / "latest_risk_gates.csv"
    pd.DataFrame(
        [
            {"gate": "P0_latest_flash_confirmation", "status": "not_confirmed_zero_exposure"},
            {"gate": "P1_latest_flash_confirmation", "status": "partial_sorting_smoke_not_confirmation"},
            {"gate": "latest_rolling_product_risk_register", "status": "logged_not_complete"},
        ]
    ).to_csv(gates_path, index=False)
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "LATEST_ROLLING_RISK_GATES", gates_path)

    summary = readiness.summarize_latest_rolling_risk_register()

    assert summary.loc[0, "status"] == "pass"
    assert summary.loc[0, "p0_latest_status"] == "not_confirmed_zero_exposure"
    assert summary.loc[0, "overall_status"] == "logged_not_complete"


def test_summarize_tool_adoption_contract_reads_pass_summary(tmp_path, monkeypatch) -> None:
    summary_path = tmp_path / "tool_adoption_summary.csv"
    pd.DataFrame(
        [
            {
                "status": "pass",
                "rows": 12,
                "fail_rows": 0,
                "p0_operation_tool_rows": 8,
                "p0_operation_override_without_hard_counter": 0,
                "p1_anchor_rows": 4,
                "p1_anchor_change_without_hard_counter": 0,
                "forbidden_eval_hit_rows": 0,
            }
        ]
    ).to_csv(summary_path, index=False)
    monkeypatch.setattr(readiness, "ROOT", tmp_path)
    monkeypatch.setattr(readiness, "TOOL_ADOPTION_CONTRACT_SUMMARY", summary_path)

    summary = readiness.summarize_tool_adoption_contract()

    assert summary.loc[0, "status"] == "pass"
    assert int(summary.loc[0, "rows"]) == 12
    assert int(summary.loc[0, "p0_operation_tool_rows"]) == 8
    assert int(summary.loc[0, "p1_anchor_rows"]) == 4
