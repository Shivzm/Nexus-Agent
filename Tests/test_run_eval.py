import json

import pytest

from evals.run_eval import (
    load_cases,
    make_judge_evaluator,
    make_retrieval_evaluator,
    make_target,
    parse_judge_scores,
    run_evaluation,
    save_results,
)


def test_load_cases_limits_and_validates_dataset(tmp_path):
    path = tmp_path / "reviewed.jsonl"
    cases = [
        {
            "question": f"Question {index}",
            "ground_truth": "Answer",
            "source": "paper.pdf",
            "reference_contexts": ["Evidence"],
        }
        for index in range(3)
    ]
    path.write_text("".join(json.dumps(case) + "\n" for case in cases), encoding="utf-8")

    assert len(load_cases(path, limit=2)) == 2
    with pytest.raises(ValueError, match="limit must be at least 1"):
        load_cases(path, limit=0)


def test_target_generates_answer_and_returns_retrieval_data():
    class Response:
        content = "A retrieval method."

    class FakeLLM:
        def invoke(self, prompt):
            assert "What is described?" in prompt
            return Response()

    def fake_retrieve(question, top_k, search_config):
        assert top_k == 5
        assert search_config == "hybrid"
        return [{
            "content": "The paper describes a retrieval method.",
            "metadata": {"source": "papers/example.pdf"},
        }]

    target = make_target(FakeLLM(), retrieve_fn=fake_retrieve, search_config="hybrid")

    assert target({"question": "What is described?"}) == {
        "answer": "A retrieval method.",
        "contexts": ["The paper describes a retrieval method."],
        "sources": ["papers/example.pdf"],
    }


def test_retrieval_evaluator_returns_hit_at_five_and_mrr():
    evaluator = make_retrieval_evaluator()

    assert evaluator(
        {"sources": ["other.pdf", "paper.pdf"]},
        {"source": "paper.pdf"},
    ) == [
        {"key": "hit_at_5", "score": 1.0},
        {"key": "mrr", "score": 0.5},
    ]


def test_judge_parser_validates_all_scores():
    scores = parse_judge_scores(
        '{"faithfulness":0.8,"context_recall":0.7,"context_precision":0.9}'
    )

    assert scores == {
        "faithfulness": 0.8,
        "context_recall": 0.7,
        "context_precision": 0.9,
    }
    with pytest.raises(ValueError, match="between 0 and 1"):
        parse_judge_scores(
            '{"faithfulness":2,"context_recall":0.7,"context_precision":0.9}'
        )


def test_judge_evaluator_uses_rate_limit_retry_and_returns_scores():
    class Response:
        content = '{"faithfulness":1,"context_recall":0.5,"context_precision":0}'

    class FakeLLM:
        def invoke(self, prompt):
            assert "Retrieved contexts" in prompt
            return Response()

    evaluator = make_judge_evaluator(FakeLLM())

    results = evaluator(
        {"answer": "answer", "contexts": ["context"]},
        {
            "question": "question",
            "ground_truth": "reference",
        },
    )

    assert [(result["key"], result["score"]) for result in results] == [
        ("faithfulness", 1.0),
        ("context_recall", 0.5),
        ("context_precision", 0.0),
    ]


def test_run_evaluation_is_local_and_single_worker():
    class FakeClient:
        def evaluate(self, target, **kwargs):
            self.kwargs = kwargs
            self.target = target
            return "experiment"

    client = FakeClient()
    cases = [{
        "question": "q",
        "ground_truth": "a",
        "source": "paper.pdf",
        "reference_contexts": ["context"],
    }]
    result = run_evaluation(cases, llm=object(), client=client)

    assert result == "experiment"
    assert client.kwargs["max_concurrency"] == 1
    assert client.kwargs["upload_results"] is False
    assert callable(client.target)
    with pytest.raises(ValueError, match="max_workers=1"):
        run_evaluation(cases, llm=object(), max_workers=2)


def test_real_langsmith_local_runner_runs_without_uploading():
    from langsmith import Client

    class Response:
        def __init__(self, content):
            self.content = content

    class FakeLLM:
        def invoke(self, prompt):
            if prompt.startswith("Score this retrieval-augmented answer"):
                return Response(
                    '{"faithfulness":1,"context_recall":1,"context_precision":1}'
                )
            return Response("Supported answer.")

    case = {
        "question": "What is supported?",
        "ground_truth": "Supported answer.",
        "source": "paper.pdf",
        "reference_contexts": ["Supported answer."],
    }

    experiment = run_evaluation(
        [case],
        llm=FakeLLM(),
        client=Client(),
        retrieve_fn=lambda question, top_k, search_config: [{
            "content": "Supported answer.",
            "metadata": {"source": "paper.pdf"},
        }],
    )
    results = list(experiment)

    scores = {result.key: result.score for result in results[0]["evaluation_results"]["results"]}
    assert scores == {
        "hit_at_5": 1.0,
        "mrr": 1.0,
        "faithfulness": 1.0,
        "context_recall": 1.0,
        "context_precision": 1.0,
    }


def test_save_results_writes_csv_and_summary(tmp_path):
    class Result:
        def __init__(self, key, score):
            self.key = key
            self.score = score

    class Example:
        inputs = {"question": "q"}
        outputs = {"ground_truth": "a"}

    class Run:
        outputs = {"answer": "a", "sources": ["paper.pdf"]}

    class Experiment:
        def __iter__(self):
            return iter([{
                "example": Example(),
                "run": Run(),
                "evaluation_results": {"results": [
                    Result("hit_at_5", 1.0),
                    Result("mrr", 1.0),
                ]},
            }])

    csv_path, json_path, summary = save_results(Experiment(), tmp_path, "test-model")

    assert csv_path.is_file()
    assert json.loads(json_path.read_text(encoding="utf-8")) == {
        "hit_at_5": 1.0,
        "mrr": 1.0,
        "questions": 1,
        "model": "test-model",
        "config": "dense",
    }
    assert summary["questions"] == 1
