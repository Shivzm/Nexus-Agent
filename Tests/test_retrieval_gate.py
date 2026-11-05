import pytest

from evals.run_retrieval_gate import calculate_metrics, evaluate_thresholds


def test_calculate_metrics_uses_expected_source_rank_and_fixed_top_k():
    cases = [
        {"question": "first", "source": "a.pdf"},
        {"question": "second", "source": "b.pdf"},
        {"question": "missing", "source": "c.pdf"},
    ]
    results = {
        "first": [
            {"metadata": {"source": "other.pdf"}},
            {"metadata": {"source": "a.pdf"}},
        ],
        "second": [{"metadata": {"source": "b.pdf"}}],
        "missing": [{"metadata": {"source": "other.pdf"}}],
    }
    calls = []

    def retrieve_fn(question, top_k, search_config):
        calls.append((question, top_k, search_config))
        return results[question]

    assert calculate_metrics(cases, retrieve_fn) == {
        "hit_at_5": pytest.approx(2 / 3),
        "mrr": pytest.approx(0.5),
        "questions": 3,
    }
    assert calls == [(case["question"], 5, "dense") for case in cases]


@pytest.mark.parametrize(
    ("metrics", "thresholds", "passed"),
    [
        ({"hit_at_5": 0.8, "mrr": 0.6}, (0.75, 0.5), True),
        ({"hit_at_5": 0.7, "mrr": 0.6}, (0.75, 0.5), False),
        ({"hit_at_5": 0.8, "mrr": 0.4}, (0.75, 0.5), False),
    ],
)
def test_evaluate_thresholds(metrics, thresholds, passed):
    actual, failures = evaluate_thresholds(metrics, *thresholds)

    assert actual is passed
    assert bool(failures) is (not passed)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_evaluate_thresholds_rejects_invalid_values(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        evaluate_thresholds({"hit_at_5": 1.0, "mrr": 1.0}, threshold, 0.5)
