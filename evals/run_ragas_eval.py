"""Run a dense retrieval and Ragas baseline on the reviewed question set."""

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evals.run_eval import load_cases, make_target


def retrieval_scores(sources: list[str], expected_source: str) -> dict[str, float]:
    ranks = [
        rank
        for rank, source in enumerate(sources[:5], start=1)
        if source == expected_source
    ]
    return {
        "hit_at_5": float(bool(ranks)),
        "mrr": 1 / ranks[0] if ranks else 0.0,
    }


def build_evaluation_rows(
    cases: list[dict[str, Any]],
    llm: Any,
    max_retries: int,
) -> list[dict[str, Any]]:
    target = make_target(llm, search_config="dense", max_retries=max_retries)
    rows = []
    for case in cases:
        output = target({"question": case["question"]})
        rows.append({
            "id": case.get("id", ""),
            "question": case["question"],
            "ground_truth": case["ground_truth"],
            "answer": output["answer"],
            "contexts": output["contexts"],
            "sources": output["sources"],
            **retrieval_scores(output["sources"], case["source"]),
        })
    return rows


def normalize_ragas_scores(scores: list[dict[str, Any]]) -> list[dict[str, float]]:
    return [
        {
            "faithfulness": float(score["faithfulness"]),
            "context_recall": float(score["context_recall"]),
            "context_precision": float(score["llm_context_precision_with_reference"]),
        }
        for score in scores
    ]


def run_ragas(rows: list[dict[str, Any]], llm: Any, max_retries: int) -> list[dict[str, float]]:
    from ragas import EvaluationDataset, evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithReference, LLMContextRecall
    from ragas.run_config import RunConfig

    dataset = EvaluationDataset.from_list([
        {
            "user_input": row["question"],
            "retrieved_contexts": row["contexts"],
            "response": row["answer"],
            "reference": row["ground_truth"],
        }
        for row in rows
    ])
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            LLMContextRecall(),
            LLMContextPrecisionWithReference(),
        ],
        llm=LangchainLLMWrapper(llm),
        run_config=RunConfig(max_workers=1, max_retries=max_retries),
        raise_exceptions=True,
    )
    return normalize_ragas_scores(result.scores)


def save_results(
    rows: list[dict[str, Any]],
    ragas_scores: list[dict[str, float]],
    output_dir: Path,
    model: str,
) -> tuple[Path, Path, dict[str, Any]]:
    if not rows:
        raise ValueError("Cannot save an empty evaluation")
    if len(rows) != len(ragas_scores):
        raise ValueError("Ragas score count does not match the evaluated question count")

    details = [
        {
            **row,
            "contexts": json.dumps(row["contexts"], ensure_ascii=False),
            "sources": json.dumps(row["sources"], ensure_ascii=False),
            **scores,
        }
        for row, scores in zip(rows, ragas_scores, strict=True)
    ]
    score_names = ("faithfulness", "context_recall", "context_precision")
    summary = {
        name: sum(float(scores[name]) for scores in ragas_scores) / len(ragas_scores)
        for name in score_names
    }
    summary.update({
        "hit_at_5": sum(row["hit_at_5"] for row in rows) / len(rows),
        "mrr": sum(row["mrr"] for row in rows) / len(rows),
        "questions": len(rows),
        "model": model,
        "config": "dense",
    })

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csv_path = output_dir / f"ragas-dense-{timestamp}.csv"
    json_path = output_dir / f"ragas-dense-{timestamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as result_file:
        writer = csv.DictWriter(result_file, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return csv_path, json_path, summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/reviewed_dataset.jsonl"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument(
        "--model",
        default=os.getenv("EVAL_MODEL", "openai/gpt-oss-20b"),
        help="Groq model for answer generation and Ragas judging.",
    )
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/results"))
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")
    if args.max_retries < 0:
        parser.error("--max-retries cannot be negative")

    from langchain_groq import ChatGroq

    cases = load_cases(args.dataset, limit=args.count)
    llm = ChatGroq(model=args.model, temperature=0)
    rows = build_evaluation_rows(cases, llm, args.max_retries)
    scores = run_ragas(rows, llm, args.max_retries)
    csv_path, json_path, summary = save_results(rows, scores, args.output_dir, args.model)
    print(json.dumps(summary, indent=2))
    print(f"Per-question results: {csv_path}")
    print(f"Aggregate results: {json_path}")


if __name__ == "__main__":
    main()
