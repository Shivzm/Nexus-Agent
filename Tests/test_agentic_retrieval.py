import json

import pytest

from src.rag.agentic_retrieval import build_agentic_retrieval_graph, parse_context_grade


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeGrader:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse(next(self.responses))


def grade(relevant, rewritten_query=""):
    return json.dumps({"relevant": relevant, "rewritten_query": rewritten_query})


def test_context_grade_parser_validates_shape_and_rewrite():
    assert parse_context_grade(grade(False, "rephrased")) == (False, "rephrased")
    with pytest.raises(ValueError, match="boolean"):
        parse_context_grade('{"relevant":"yes","rewritten_query":"q"}')
    with pytest.raises(ValueError, match="rewritten query"):
        parse_context_grade(grade(False))


def test_relevant_context_finishes_without_retry_or_web_search():
    grader = FakeGrader([grade(True)])
    retrieval_queries = []
    web_queries = []
    graph = build_agentic_retrieval_graph(
        web_search_fn=web_queries.append,
        retrieve_fn=lambda query: retrieval_queries.append(query) or [{"content": "evidence"}],
        grader=grader,
    )

    result = graph.invoke({"question": "Original?", "query": "Original?", "attempts": 0})

    assert retrieval_queries == ["Original?"]
    assert web_queries == []
    assert result["relevant"] is True
    assert "evidence" in grader.prompts[0]


def test_weak_context_rewrites_once_then_accepts_relevant_result():
    grader = FakeGrader([grade(False, "specific query"), grade(True)])
    retrieval_queries = []
    web_queries = []
    graph = build_agentic_retrieval_graph(
        web_search_fn=web_queries.append,
        retrieve_fn=lambda query: retrieval_queries.append(query) or [{"content": query}],
        grader=grader,
    )

    result = graph.invoke({"question": "Original?", "query": "Original?", "attempts": 0})

    assert retrieval_queries == ["Original?", "specific query"]
    assert web_queries == []
    assert result["relevant"] is True
    assert result["attempts"] == 1


def test_weak_context_after_retry_falls_back_to_web():
    grader = FakeGrader([grade(False, "specific query"), grade(False, "another query")])
    retrieval_queries = []
    web_queries = []
    graph = build_agentic_retrieval_graph(
        web_search_fn=lambda query: web_queries.append(query) or "web evidence",
        retrieve_fn=lambda query: retrieval_queries.append(query) or [],
        grader=grader,
    )

    result = graph.invoke({"question": "Original?", "query": "Original?", "attempts": 0})

    assert retrieval_queries == ["Original?", "specific query"]
    assert web_queries == ["Original?"]
    assert result["web_results"] == "web evidence"
    assert result["relevant"] is False


def test_identical_rewrite_falls_back_without_repeating_same_search():
    grader = FakeGrader([grade(False, "Original?")])
    retrieval_queries = []
    web_queries = []
    graph = build_agentic_retrieval_graph(
        web_search_fn=lambda query: web_queries.append(query) or "web evidence",
        retrieve_fn=lambda query: retrieval_queries.append(query) or [],
        grader=grader,
    )

    result = graph.invoke({"question": "Original?", "query": "Original?", "attempts": 0})

    assert retrieval_queries == ["Original?"]
    assert web_queries == ["Original?"]
    assert result["web_results"] == "web evidence"


def test_negative_rewrite_limit_is_rejected():
    with pytest.raises(ValueError, match="cannot be negative"):
        build_agentic_retrieval_graph(web_search_fn=lambda _: "", grader=FakeGrader([]), max_rewrites=-1)
