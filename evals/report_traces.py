"""Report LangSmith latency, token usage, and cost by query and graph node."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

DEFAULT_NODES = ("agent", "tools", "retrieve", "grade", "rewrite", "web_search")


def _value(run: Any, name: str, default: Any = None) -> Any:
    if isinstance(run, dict):
        return run.get(name, default)
    return getattr(run, name, default)


def _numeric(run: Any, name: str) -> float | None:
    value = _value(run, name)
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _duration_seconds(run: Any) -> float | None:
    start = _value(run, "start_time")
    end = _value(run, "end_time")
    if start is None or end is None:
        return None
    duration = (end - start).total_seconds()
    return duration if duration >= 0 else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _cost_for_run(
    run: Any,
    input_usd_per_million: float | None,
    output_usd_per_million: float | None,
) -> tuple[float | None, str | None]:
    if input_usd_per_million is not None and output_usd_per_million is not None:
        prompt = _numeric(run, "prompt_tokens")
        completion = _numeric(run, "completion_tokens")
        if prompt is not None and completion is not None:
            cost = (
                prompt * input_usd_per_million + completion * output_usd_per_million
            ) / 1_000_000
            return cost, "estimated"

    prompt_cost = _numeric(run, "prompt_cost")
    completion_cost = _numeric(run, "completion_cost")
    total_cost = _numeric(run, "total_cost")
    if prompt_cost is not None or completion_cost is not None:
        return (prompt_cost or 0.0) + (completion_cost or 0.0), "langsmith"
    if total_cost is not None:
        return total_cost, "langsmith"
    return None, None


def build_report(
    runs: Iterable[Any],
    nodes: Iterable[str] = DEFAULT_NODES,
    input_usd_per_million: float | None = None,
    output_usd_per_million: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate a flat LangSmith run list into query and per-node metrics."""
    if (input_usd_per_million is None) != (output_usd_per_million is None):
        raise ValueError("Both input and output token prices must be provided together")
    for label, price in (
        ("input", input_usd_per_million),
        ("output", output_usd_per_million),
    ):
        if price is not None and (not math.isfinite(price) or price < 0):
            raise ValueError(f"{label} token price must be finite and non-negative")

    run_list = list(runs)
    node_names = {name.casefold(): name for name in nodes}
    by_id: dict[str, Any] = {}
    by_trace: dict[str, list[Any]] = defaultdict(list)
    for run in run_list:
        run_id = str(_value(run, "id", ""))
        trace_id = str(_value(run, "trace_id", run_id))
        if run_id:
            by_id[run_id] = run
        by_trace[trace_id].append(run)

    def node_for(run: Any, trace_runs: list[Any]) -> str | None:
        parent_id = _value(run, "parent_run_id")
        seen: set[str] = set()
        while parent_id is not None:
            parent_key = str(parent_id)
            if parent_key in seen:
                break
            seen.add(parent_key)
            parent = by_id.get(parent_key)
            if parent is None:
                break
            name = str(_value(parent, "name", "")).casefold()
            if name in node_names:
                return node_names[name]
            parent_id = _value(parent, "parent_run_id")

        own_name = str(_value(run, "name", "")).casefold()
        if own_name in node_names:
            return node_names[own_name]
        for root in trace_runs:
            if _value(root, "parent_run_id") is None:
                root_name = str(_value(root, "name", "")).casefold()
                if root_name in node_names:
                    return node_names[root_name]
        return None

    query_rows: list[dict[str, Any]] = []
    node_latency: dict[str, list[float]] = defaultdict(list)
    node_tokens: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "input_tokens": 0.0,
            "output_tokens": 0.0,
            "total_tokens": 0.0,
            "cost_usd": 0.0,
            "cost_calls": 0.0,
        }
    )
    node_queries: dict[str, set[str]] = defaultdict(set)
    node_cost_coverage: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    node_cost_sources: dict[str, set[str]] = defaultdict(set)

    for trace_id, trace_runs in by_trace.items():
        roots = [run for run in trace_runs if _value(run, "parent_run_id") is None]
        if not roots:
            continue
        root = min(roots, key=lambda run: _value(run, "start_time", datetime.max.replace(tzinfo=timezone.utc)))
        root_id = str(_value(root, "id", trace_id))
        query_latency = _duration_seconds(root)
        total_input = 0.0
        total_output = 0.0
        total_cost = 0.0
        known_cost_calls = 0
        llm_calls = 0
        trace_cost_sources: set[str] = set()
        trace_node_latency: dict[str, float] = defaultdict(float)
        trace_node_usage: dict[str, dict[str, float]] = defaultdict(
            lambda: {
                "input_tokens": 0.0,
                "output_tokens": 0.0,
                "total_tokens": 0.0,
                "cost_usd": 0.0,
                "cost_calls": 0.0,
            }
        )
        total_token_count = 0.0

        for run in trace_runs:
            run_type = str(_value(run, "run_type", "")).casefold()
            if run_type in ("llm", "chat_model", "chatmodel"):
                input_tokens = _numeric(run, "prompt_tokens")
                output_tokens = _numeric(run, "completion_tokens")
                total_tokens = _numeric(run, "total_tokens")
                cost, cost_source = _cost_for_run(
                    run, input_usd_per_million, output_usd_per_million
                )
                llm_calls += 1
                if input_tokens is not None:
                    total_input += input_tokens
                if output_tokens is not None:
                    total_output += output_tokens
                if total_tokens is not None:
                    total_token_count += total_tokens
                else:
                    total_token_count += (input_tokens or 0.0) + (output_tokens or 0.0)
                if cost is not None:
                    total_cost += cost
                    known_cost_calls += 1
                    if cost_source is not None:
                        trace_cost_sources.add(cost_source)
                node = node_for(run, trace_runs)
                if node is not None:
                    usage = trace_node_usage[node]
                    usage["input_tokens"] += input_tokens or 0.0
                    usage["output_tokens"] += output_tokens or 0.0
                    usage["total_tokens"] += (
                        total_tokens
                        if total_tokens is not None
                        else (input_tokens or 0.0) + (output_tokens or 0.0)
                    )
                    if cost is not None:
                        usage["cost_usd"] += cost
                        usage["cost_calls"] += 1
                    node_queries[node].add(root_id)
                    coverage = node_cost_coverage[node]
                    coverage[0] += 1
                    coverage[1] += int(cost is not None)
                    if cost_source is not None:
                        node_cost_sources[node].add(cost_source)

            name = str(_value(run, "name", "")).casefold()
            node = node_names.get(name)
            duration = _duration_seconds(run)
            if node is not None and duration is not None:
                trace_node_latency[node] += duration
                node_queries[node].add(root_id)

        for node, duration in trace_node_latency.items():
            node_latency[node].append(duration)
        for node, usage in trace_node_usage.items():
            aggregate = node_tokens[node]
            aggregate["input_tokens"] += usage["input_tokens"]
            aggregate["output_tokens"] += usage["output_tokens"]
            aggregate["total_tokens"] += usage["total_tokens"]
            aggregate["cost_usd"] += usage["cost_usd"]
            aggregate["cost_calls"] += usage["cost_calls"]

        query_rows.append({
            "query_id": root_id,
            "started_at": _value(root, "start_time").isoformat() if _value(root, "start_time") else "",
            "latency_seconds": query_latency,
            "input_tokens": int(total_input),
            "output_tokens": int(total_output),
            "total_tokens": int(total_token_count),
            "cost_usd": total_cost if known_cost_calls else None,
            "cost_source": (
                next(iter(trace_cost_sources))
                if len(trace_cost_sources) == 1
                else "mixed"
                if trace_cost_sources
                else "unavailable"
            ),
            "cost_coverage": f"{known_cost_calls}/{llm_calls}",
            "error": bool(_value(root, "error")),
            "node_latency_seconds": dict(trace_node_latency),
            "node_usage": {
                node: {
                    "input_tokens": int(usage["input_tokens"]),
                    "output_tokens": int(usage["output_tokens"]),
                    "total_tokens": int(usage["total_tokens"]),
                    "cost_usd": usage["cost_usd"] if usage["cost_calls"] else None,
                }
                for node, usage in trace_node_usage.items()
            },
        })

    query_latencies = [
        row["latency_seconds"] for row in query_rows if row["latency_seconds"] is not None
    ]
    query_costs = [row["cost_usd"] for row in query_rows if row["cost_usd"] is not None]
    query_cost_sources = {row["cost_source"] for row in query_rows if row["cost_usd"] is not None}
    nodes_summary = {}
    for node in sorted(set(node_latency) | set(node_tokens)):
        latencies = node_latency[node]
        usage = node_tokens[node]
        calls, calls_with_cost = node_cost_coverage[node]
        nodes_summary[node] = {
            "queries": len(node_queries[node]),
            "p50_latency_seconds": _percentile(latencies, 0.50),
            "p95_latency_seconds": _percentile(latencies, 0.95),
            "input_tokens": int(usage["input_tokens"]),
            "output_tokens": int(usage["output_tokens"]),
            "total_tokens": int(usage["total_tokens"]),
            "cost_usd": usage["cost_usd"] if usage["cost_calls"] else None,
            "cost_source": (
                next(iter(node_cost_sources[node]))
                if len(node_cost_sources[node]) == 1
                else "mixed"
                if node_cost_sources[node]
                else "unavailable"
            ),
            "cost_coverage": f"{calls_with_cost}/{calls}",
        }

    summary = {
        "queries": len(query_rows),
        "errors": sum(row["error"] for row in query_rows),
        "p50_latency_seconds": _percentile(query_latencies, 0.50),
        "p95_latency_seconds": _percentile(query_latencies, 0.95),
        "input_tokens": sum(row["input_tokens"] for row in query_rows),
        "output_tokens": sum(row["output_tokens"] for row in query_rows),
        "total_tokens": sum(row["total_tokens"] for row in query_rows),
        "cost_usd": sum(query_costs) if query_costs else None,
        "cost_source": (
            next(iter(query_cost_sources))
            if len(query_cost_sources) == 1
            else "mixed"
            if query_cost_sources
            else "unavailable"
        ),
        "cost_coverage": (
            f"{sum(int(row['cost_coverage'].split('/')[0]) for row in query_rows)}/"
            f"{sum(int(row['cost_coverage'].split('/')[1]) for row in query_rows)}"
        ),
        "nodes": nodes_summary,
    }
    return summary, query_rows


def save_report(summary: dict[str, Any], queries: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"traces-{timestamp}.json"
    csv_path = output_dir / f"traces-{timestamp}.csv"
    json_path.write_text(
        json.dumps({"summary": summary, "queries": queries}, indent=2) + "\n",
        encoding="utf-8",
    )
    columns = [
        "query_id",
        "started_at",
        "latency_seconds",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "cost_source",
        "cost_coverage",
        "error",
        "node_latency_seconds",
        "node_usage",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=columns)
        writer.writeheader()
        for query in queries:
            writer.writerow({
                **query,
                "node_latency_seconds": json.dumps(query["node_latency_seconds"]),
                "node_usage": json.dumps(query["node_usage"]),
            })
    return json_path, csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default=os.getenv("LANGSMITH_PROJECT", "nexus-agent"))
    parser.add_argument("--hours", type=float, default=24)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    parser.add_argument(
        "--input-usd-per-million",
        type=float,
        help="Optional input-token price; supply this and output price together to estimate cost.",
    )
    parser.add_argument(
        "--output-usd-per-million",
        type=float,
        help="Optional output-token price; supply this and input price together to estimate cost.",
    )
    parser.add_argument(
        "--nodes",
        default=",".join(DEFAULT_NODES),
        help="Comma-separated LangGraph node names to report (default: %(default)s).",
    )
    args = parser.parse_args(argv)
    if args.hours <= 0 or not math.isfinite(args.hours):
        parser.error("--hours must be finite and greater than zero")
    if args.limit < 1:
        parser.error("--limit must be at least one")
    if (args.input_usd_per_million is None) != (args.output_usd_per_million is None):
        parser.error("Provide both token price arguments or neither")
    if not args.project.strip():
        parser.error("--project must not be empty")

    from langsmith import Client

    client = Client()
    runs = list(client.list_runs(
        project_name=args.project,
        start_time=datetime.now(timezone.utc) - timedelta(hours=args.hours),
        limit=args.limit,
        select=[
            "id",
            "trace_id",
            "parent_run_id",
            "name",
            "run_type",
            "start_time",
            "end_time",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "prompt_cost",
            "completion_cost",
            "total_cost",
            "error",
        ],
    ))
    summary, queries = build_report(
        runs,
        nodes=(node.strip() for node in args.nodes.split(",") if node.strip()),
        input_usd_per_million=args.input_usd_per_million,
        output_usd_per_million=args.output_usd_per_million,
    )
    json_path, csv_path = save_report(summary, queries, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    main()
