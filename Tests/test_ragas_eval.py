import json

import pytest

from evals.run_ragas_eval import normalize_ragas_scores, retrieval_scores, save_results


def test_retrieval_scores_calculates_hit_at_five_and_reciprocal_rank():
    assert retrieval_scores(["a.pdf", "b.pdf", "target.pdf"], "target.pdf") == {
        "hit_at_5": 1.0,
        "mrr": pytest.approx(1 / 3),
    }
    assert retrieval_scores(["a.pdf"], "target.pdf") == {
        "hit_at_5": 0.0,
        "mrr": 0.0,
    }


def test_normalize_ragas_scores_uses_comparable_context_precision_key():
    assert normalize_ragas_scores([{
        "faithfulness": 0.9,
        "context_recall": 0.8,
        "llm_context_precision_with_reference": 0.7,
    }]) == [{
        "faithfulness": 0.9,
        "context_recall": 0.8,
        "context_precision": 0.7,
    }]


def test_save_results_writes_per_question_and_aggregate_metrics(tmp_path):
    rows = [{
        "id": "q1",
        "question": "Question?",
        "ground_truth": "Answer.",
        "answer": "Answer.",
        "contexts": ["Evidence."],
        "sources": ["paper.pdf"],
        "hit_at_5": 1.0,
        "mrr": 1.0,
    }]
    scores = [{
        "faithfulness": 0.9,
        "context_recall": 0.8,
        "context_precision": 0.7,
    }]

    csv_path, json_path, summary = save_results(rows, scores, tmp_path, "test-model")

    assert csv_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        **scores[0],
        "hit_at_5": 1.0,
        "mrr": 1.0,
        "questions": 1,
        "model": "test-model",
        "config": "dense",
    }
    assert summary["questions"] == 1


def test_save_results_rejects_empty_or_mismatched_results(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_results([], [], tmp_path, "test-model")
    with pytest.raises(ValueError, match="score count"):
        save_results([{}], [], tmp_path, "test-model")
