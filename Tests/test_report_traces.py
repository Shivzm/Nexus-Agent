from datetime import datetime, timedelta, timezone

import pytest

from evals.report_traces import build_report, save_report


def make_run(
    run_id,
    trace_id,
    name,
    run_type,
    start,
    duration,
    parent_run_id=None,
    prompt_tokens=None,
    completion_tokens=None,
    prompt_cost=None,
    completion_cost=None,
    total_cost=None,
):
    return {
        "id": run_id,
        "trace_id": trace_id,
        "parent_run_id": parent_run_id,
        "name": name,
        "run_type": run_type,
        "start_time": start,
        "end_time": start + timedelta(seconds=duration),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
        "error": None,
    }


def test_build_report_aggregates_trace_latency_tokens_and_reported_cost():
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    runs = [
        make_run("root", "trace", "LangGraph", "chain", start, 3.0),
        make_run("agent", "trace", "agent", "chain", start, 2.0, "root"),
        make_run(
            "llm",
            "trace",
            "ChatGroq",
            "llm",
            start,
            1.0,
            "agent",
            100,
            20,
            0.001,
            0.002,
        ),
    ]

    summary, queries = build_report(runs)

    assert summary["queries"] == 1
    assert summary["p50_latency_seconds"] == 3
    assert summary["total_tokens"] == 120
    assert summary["cost_usd"] == pytest.approx(0.003)
    assert summary["cost_source"] == "langsmith"
    assert summary["nodes"]["agent"]["p95_latency_seconds"] == 2
    assert summary["nodes"]["agent"]["total_tokens"] == 120
    assert queries[0]["node_usage"]["agent"]["cost_usd"] == pytest.approx(0.003)


def test_build_report_estimates_cost_only_when_both_rates_are_given():
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    runs = [
        make_run("root", "trace", "agent", "chain", start, 1),
        make_run("llm", "trace", "model", "llm", start, 0.5, "root", 1000, 500),
    ]

    summary, _ = build_report(
        runs,
        input_usd_per_million=2,
        output_usd_per_million=4,
    )

    assert summary["cost_usd"] == pytest.approx(0.004)
    assert summary["cost_source"] == "estimated"


def test_build_report_marks_cost_unavailable_when_no_rate_or_provider_cost():
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    runs = [
        make_run("root", "trace", "agent", "chain", start, 1),
        make_run("llm", "trace", "model", "llm", start, 0.5, "root", 10, 5),
    ]

    summary, queries = build_report(runs)

    assert summary["cost_usd"] is None
    assert summary["cost_source"] == "unavailable"
    assert queries[0]["cost_coverage"] == "0/1"


def test_build_report_marks_mixed_cost_sources_accurately():
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    runs = [
        make_run("root", "trace", "agent", "chain", start, 1),
        make_run("llm-1", "trace", "model", "llm", start, 0.4, "root", 100, 50),
        make_run("llm-2", "trace", "model", "llm", start, 0.4, "root", total_cost=0.01),
    ]

    summary, queries = build_report(runs, input_usd_per_million=2, output_usd_per_million=4)

    assert summary["cost_source"] == "mixed"
    assert queries[0]["cost_source"] == "mixed"


def test_build_report_rejects_partial_or_invalid_price_inputs():
    with pytest.raises(ValueError, match="provided together"):
        build_report([], input_usd_per_million=1)
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_report([], input_usd_per_million=-1, output_usd_per_million=1)


def test_save_report_writes_query_details(tmp_path):
    summary = {"queries": 0, "nodes": {}}
    json_path, csv_path = save_report(summary, [], tmp_path)

    assert json_path.is_file()
    assert csv_path.read_text(encoding="utf-8").startswith("query_id,")
