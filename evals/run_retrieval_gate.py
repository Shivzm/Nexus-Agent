"""Run a deterministic retrieval-quality gate without LLM calls or private corpus data."""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from evals.run_eval import load_cases

DEFAULT_DATASET = Path("evals/ci_retrieval_dataset.jsonl")


def calculate_metrics(
    cases: list[dict[str, Any]],
    retrieve_fn: Callable[..., list[dict[str, Any]]],
    top_k: int = 5,
) -> dict[str, float]:
    if not cases:
        raise ValueError("Cannot evaluate an empty dataset")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    hits = []
    reciprocal_ranks = []
    for case in cases:
        results = retrieve_fn(case["question"], top_k=top_k, search_config="dense")
        ranks = [
            rank
            for rank, result in enumerate(results, start=1)
            if result.get("metadata", {}).get("source") == case["source"]
        ]
        hits.append(float(bool(ranks)))
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)

    return {
        "hit_at_5": sum(hits) / len(hits),
        "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks),
        "questions": len(cases),
    }


def evaluate_thresholds(
    metrics: dict[str, float],
    min_hit_at_5: float,
    min_mrr: float,
) -> tuple[bool, list[str]]:
    for name, threshold in (("min_hit_at_5", min_hit_at_5), ("min_mrr", min_mrr)):
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError(f"{name} must be between 0 and 1")

    failures = []
    if metrics["hit_at_5"] < min_hit_at_5:
        failures.append(f"hit@5 {metrics['hit_at_5']:.3f} is below {min_hit_at_5:.3f}")
    if metrics["mrr"] < min_mrr:
        failures.append(f"MRR {metrics['mrr']:.3f} is below {min_mrr:.3f}")
    return not failures, failures


def run_fixed_corpus_gate(cases: list[dict[str, Any]]) -> dict[str, float]:
    import chromadb
    from unittest.mock import patch

    from src.rag.lexical import clear_bm25_index
    from src.rag.retriever import retrieve
    from src.rag.store import get_embedder

    client = chromadb.EphemeralClient()
    collection = client.create_collection(f"retrieval-gate-{uuid4().hex}")
    documents = ["\n\n".join(case["reference_contexts"]) for case in cases]
    collection.add(
        ids=[f"ci-{index}" for index in range(len(cases))],
        documents=documents,
        metadatas=[{"source": case["source"]} for case in cases],
        embeddings=get_embedder().encode(documents).tolist(),
    )

    try:
        with (
            patch("src.rag.retriever.get_collection", return_value=collection),
            patch("src.rag.lexical.get_collection", return_value=collection),
        ):
            clear_bm25_index()
            return calculate_metrics(cases, retrieve)
    finally:
        clear_bm25_index()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--min-hit-at-5", type=float, default=0.75)
    parser.add_argument("--min-mrr", type=float, default=0.50)
    args = parser.parse_args(argv)
    if args.count < 1:
        parser.error("--count must be at least 1")

    cases = load_cases(args.dataset, limit=args.count)
    metrics = run_fixed_corpus_gate(cases)
    passed, failures = evaluate_thresholds(metrics, args.min_hit_at_5, args.min_mrr)
    report = {
        **metrics,
        "thresholds": {"hit_at_5": args.min_hit_at_5, "mrr": args.min_mrr},
        "passed": passed,
        "failures": failures,
    }
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
