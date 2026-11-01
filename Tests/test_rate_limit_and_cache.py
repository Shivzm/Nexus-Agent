import httpx
import pytest
from groq import RateLimitError

from evals.build_dataset import (
    DailyLimitReached,
    _parse_wait,
    _rate_limit_delay,
    invoke_with_rate_limit_retry,
    limit_kind,
)
from evals.llm_cache import CachedLLM

TPM_MSG = (
    "Rate limit reached for model `openai/gpt-oss-20b` on tokens per minute (TPM): "
    "Limit 8000, Used 6100, Requested 3200. Please try again in 2.25s."
)
TPD_MSG = (
    "Rate limit reached for model `openai/gpt-oss-20b` on tokens per day (TPD): "
    "Limit 200000, Used 197500, Requested 3200. Please try again in 11m40.32s."
)


def rate_limit_error(message, retry_after=None):
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.groq.com"))
    return RateLimitError("rate limited", response=response, body={"error": {"message": message}})


class Reply:
    def __init__(self, content):
        self.content = content


@pytest.mark.parametrize(
    "message, seconds",
    [
        ("Please try again in 250ms.", 0.25),
        ("Please try again in 2.25s.", 2.25),
        ("Please try again in 11m40.32s.", 700.32),
        ("Please try again in 1h2m3s.", 3723.0),
    ],
)
def test_parse_wait_handles_compound_durations(message, seconds):
    assert _parse_wait(message) == pytest.approx(seconds)


def test_limit_kind_and_delay():
    assert limit_kind(rate_limit_error(TPM_MSG)) == "TPM"
    assert limit_kind(rate_limit_error(TPD_MSG)) == "TPD"
    assert _rate_limit_delay(rate_limit_error(TPD_MSG), 0) == pytest.approx(700.32)


def test_per_minute_limit_still_sleeps_and_retries():
    calls, delays = [], []

    class LLM:
        def invoke(self, prompt):
            calls.append(prompt)
            if len(calls) == 1:
                raise rate_limit_error(TPM_MSG)
            return Reply("ok")

    assert invoke_with_rate_limit_retry(LLM(), "p", sleep_fn=delays.append).content == "ok"
    assert delays == [2.25]


def test_daily_limit_fails_fast_without_sleeping():
    delays = []

    class LLM:
        def invoke(self, prompt):
            raise rate_limit_error(TPD_MSG, retry_after="700")

    with pytest.raises(DailyLimitReached, match="rerun the same command"):
        invoke_with_rate_limit_retry(LLM(), "p", sleep_fn=delays.append)
    assert delays == []


def test_cache_hits_skip_the_model_and_keys_include_fingerprint(tmp_path):
    calls = []

    class LLM:
        def invoke(self, prompt):
            calls.append(prompt)
            return Reply(f"answer to {prompt}")

    path = tmp_path / "cache.sqlite"
    first = CachedLLM(LLM(), path, "model-a")
    assert first.invoke("q1").content == "answer to q1"
    assert first.invoke("q1").content == "answer to q1"
    assert (first.hits, first.misses, len(calls)) == (1, 1, 1)

    again = CachedLLM(LLM(), path, "model-a")  # new process, same file
    assert again.invoke("q1").content == "answer to q1"
    assert len(calls) == 1

    other = CachedLLM(LLM(), path, "model-b")  # different model must miss
    other.invoke("q1")
    assert len(calls) == 2


def test_cache_does_not_store_empty_or_failed_responses(tmp_path):
    class LLM:
        def invoke(self, prompt):
            return Reply("   ")

    cached = CachedLLM(LLM(), tmp_path / "c.sqlite", "m")
    cached.invoke("q")
    cached.invoke("q")
    assert cached.hits == 0 and cached.misses == 2
