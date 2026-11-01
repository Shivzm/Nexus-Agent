"""End-to-end: a daily-limit hit stops the run cleanly and a rerun reuses finished calls."""
import json

import httpx
import pytest
from groq import RateLimitError

from evals.build_dataset import DailyLimitReached
from evals.llm_cache import CachedLLM
from evals.run_eval import RunGuard, run_evaluation

TPD_MSG = "on tokens per day (TPD): Limit 200000, Used 199000, Requested 3000. Please try again in 11m40s."


class Reply:
    def __init__(self, content):
        self.content = content


def fake_retrieve(question, top_k, search_config):
    return [{"content": f"evidence for {question}", "metadata": {"source": "paper.pdf"}}]


def make_cases(n):
    return [
        {"question": f"q{i}", "ground_truth": "a", "source": "paper.pdf", "reference_contexts": ["e"]}
        for i in range(n)
    ]


class FlakyLLM:
    """Answers and judges fine for `budget` calls, then reports the daily quota is gone."""

    def __init__(self, budget=None):
        self.budget, self.calls = budget, 0

    def invoke(self, prompt):
        if self.budget is not None and self.calls >= self.budget:
            response = httpx.Response(429, headers={"retry-after": "700"}, request=httpx.Request("POST", "https://x"))
            raise RateLimitError("limited", response=response, body={"error": {"message": TPD_MSG}})
        self.calls += 1
        if "Return only a JSON object" in prompt or "Score this" in prompt:
            return Reply(json.dumps({"faithfulness": 1, "context_recall": 1, "context_precision": 1}))
        return Reply("an answer")


def run(llm, cases, guard):
    from langsmith import Client

    return run_evaluation(cases, llm, retrieve_fn=fake_retrieve, guard=guard, client=Client())


def test_daily_limit_stops_early_then_rerun_only_pays_for_the_rest(tmp_path):
    cases = make_cases(4)
    db = tmp_path / "cache.sqlite"

    # run 1: quota allows 5 calls = 2 full questions (answer+judge each) + 1 answer
    first_inner = FlakyLLM(budget=5)
    guard = RunGuard()
    run(CachedLLM(first_inner, db, "m"), cases, guard)
    assert isinstance(guard.error, DailyLimitReached)
    assert first_inner.calls == 5           # nothing sent after the limit tripped

    # run 2: quota reset; only the unfinished work should hit the API
    second_inner = FlakyLLM(budget=None)
    guard2 = RunGuard()
    cached = CachedLLM(second_inner, db, "m")
    run(cached, cases, guard2)
    assert guard2.error is None
    assert cached.hits == 5
    assert second_inner.calls == 3          # 4 questions x 2 calls - 5 cached


def test_debug_retrieval_reports_empty_and_wrong_source_rows():
    from evals.run_eval import debug_retrieval

    answers = {
        "q0": ([0.4, 0.9], ["paper.pdf", "x.pdf"]),   # scores 0.71, 0.53 -> both pass, right source
        "q1": ([1.6, 1.9], ["paper.pdf", "x.pdf"]),   # scores 0.38, 0.34 -> nothing passes
        "q2": ([0.5, 0.8], ["x.pdf", "y.pdf"]),       # passes, but wrong source
    }
    lines = []
    rows = debug_retrieval(
        make_cases(3), top_k=2, score_threshold=0.5,
        query_fn=lambda q, k: answers[q], out=lines.append,
    )
    assert [r["passed"] for r in rows] == [2, 0, 2]
    assert [r["source_in_topk"] for r in rows] == [True, True, False]
    text = "\n".join(lines)
    assert "EMPTY" in text and "WRONG-SRC" in text
    assert "1 with zero chunks above threshold, 1 retrieved but wrong source" in text


def _threshold_retriever(calls):
    """Mimics retrieve(): the default threshold filters everything, threshold=0 keeps the closest chunks."""

    def retrieve_fn(question, top_k, search_config, score_threshold=None):
        calls.append((top_k, score_threshold))
        if score_threshold == 0.0:
            return [{"content": f"close chunk {i}", "metadata": {"source": "paper.pdf"}} for i in range(top_k)]
        return []

    return retrieve_fn


def test_fallback_is_off_by_default_and_matches_shipped_behaviour():
    from evals.run_eval import make_target

    calls = []
    target = make_target(FlakyLLM(), retrieve_fn=_threshold_retriever(calls))
    out = target({"question": "q"})
    assert out["contexts"] == [] and out["sources"] == []
    assert calls == [(5, None)]            # one plain retrieval, no second attempt


def test_fallback_top_k_retries_without_threshold_only_when_empty():
    from evals.run_eval import make_target

    calls = []
    target = make_target(FlakyLLM(), retrieve_fn=_threshold_retriever(calls), fallback_k=2)
    out = target({"question": "q"})
    assert len(out["contexts"]) == 2
    assert calls == [(5, None), (2, 0.0)]  # normal attempt first, then the fallback
