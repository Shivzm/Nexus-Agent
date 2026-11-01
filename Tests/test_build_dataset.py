import pytest

from groq import RateLimitError
from httpx import Request, Response
from langchain_core.documents import Document

from evals.build_dataset import (
    generate_dataset,
    invoke_with_rate_limit_retry,
    parse_candidate,
    select_chunks,
)


def make_rate_limit_error(message="try again in 250ms", headers=None):
    response = Response(429, request=Request("POST", "https://api.groq.com"))
    if headers:
        response.headers.update(headers)
    return RateLimitError(
        "rate limited",
        response=response,
        body={"error": {"message": message}},
    )


def test_invoke_retries_groq_rate_limit_and_uses_server_delay():
    class FakeLLM:
        calls = 0

        def invoke(self, prompt):
            self.calls += 1
            if self.calls == 1:
                raise make_rate_limit_error(headers={"retry-after": "3"})
            return "success"

    llm = FakeLLM()
    delays = []

    result = invoke_with_rate_limit_retry(llm, "prompt", sleep_fn=delays.append)

    assert result == "success"
    assert llm.calls == 2
    assert delays == [3.0]


def test_invoke_stops_after_retry_budget():
    class FakeLLM:
        def invoke(self, prompt):
            raise make_rate_limit_error()

    delays = []
    with pytest.raises(RuntimeError, match="persisted after 1 retries"):
        invoke_with_rate_limit_retry(
            FakeLLM(),
            "prompt",
            max_retries=1,
            sleep_fn=delays.append,
        )

    assert delays == [1]


def test_select_chunks_is_reproducible_and_spreads_sources():
    chunks = [
        Document(page_content=f"{source}-{index}", metadata={"source": source})
        for source in ("a.pdf", "b.pdf", "c.pdf")
        for index in range(3)
    ]

    selected = select_chunks(chunks, count=3, seed=7)

    assert [chunk.metadata["source"] for chunk in selected] == [
        chunk.metadata["source"] for chunk in select_chunks(chunks, count=3, seed=7)
    ]
    assert {chunk.metadata["source"] for chunk in selected} == {"a.pdf", "b.pdf", "c.pdf"}


def test_generate_dataset_keeps_grounding_and_one_based_page():
    class FakeLLM:
        def invoke(self, prompt):
            assert "Source: papers/example.pdf" in prompt
            class Response:
                content = '{"question":" What is the main result? ","ground_truth":" The method improves retrieval. "}'

            return Response()

    chunks = [
        Document(
            page_content="The method improves retrieval.",
            metadata={"source": "papers/example.pdf", "source_file": "example.pdf", "page": 2},
        )
    ]

    rows = generate_dataset(chunks, FakeLLM())

    assert rows == [{
        "question": "What is the main result?",
        "ground_truth": "The method improves retrieval.",
        "source": "papers/example.pdf",
        "source_file": "example.pdf",
        "page": 3,
        "reference_contexts": ["The method improves retrieval."],
    }]


def test_parse_candidate_accepts_groq_markdown_response():
    response = (
        "**Question:** According to the excerpt, how many milliseconds are spent on query encoding?\n\n"
        "**Ground‑truth answer:** 13 milliseconds."
    )

    candidate = parse_candidate(response)

    assert candidate.question == "According to the excerpt, how many milliseconds are spent on query encoding?"
    assert candidate.ground_truth == "13 milliseconds."


def test_parse_candidate_accepts_fenced_json():
    candidate = parse_candidate(
        '```json\n{"question":"What changed?","ground_truth":"Retrieval improved."}\n```'
    )

    assert candidate.question == "What changed?"
    assert candidate.ground_truth == "Retrieval improved."


def test_parse_candidate_rejects_unrecognized_response():
    with pytest.raises(ValueError, match="neither a JSON object nor labeled"):
        parse_candidate("I could not produce a question.")
