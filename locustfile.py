"""HTTP load profile for cache-aware API endpoints.

Run with:
    uv run --group load locust -f locustfile.py --host http://127.0.0.1:8000
"""

import json
import os
import threading
import uuid

from locust import HttpUser, constant_pacing, task


def _load_questions() -> list[str]:
    raw = os.getenv("LOAD_TEST_QUESTIONS")
    if not raw:
        return [
            "What is retrieval augmented generation?",
            "What is retrieval-augmented generation?",
            "What is 23 multiplied by 17?",
        ]
    questions = json.loads(raw)
    if not isinstance(questions, list) or not questions or any(
        not isinstance(question, str) or not question.strip()
        for question in questions
    ):
        raise ValueError("LOAD_TEST_QUESTIONS must be a non-empty JSON list of strings")
    return questions


class ChatApiUser(HttpUser):
    wait_time = constant_pacing(float(os.getenv("LOAD_TEST_PACING_SECONDS", "3")))
    questions = _load_questions()
    _question_index = 0
    _question_lock = threading.Lock()

    def on_start(self) -> None:
        self.authorization = os.getenv("API_BEARER_TOKEN")

    @classmethod
    def _next_question(cls) -> str:
        with cls._question_lock:
            question = cls.questions[cls._question_index % len(cls.questions)]
            cls._question_index += 1
            return question

    def _headers(self) -> dict[str, str]:
        if self.authorization:
            return {"Authorization": f"Bearer {self.authorization}"}
        return {}

    def _payload(self) -> dict[str, str]:
        return {
            "thread_id": f"load-{uuid.uuid4().hex}",
            "question": self._next_question(),
        }

    @task(4)
    def chat(self) -> None:
        with self.client.post(
            "/chat",
            json=self._payload(),
            headers=self._headers(),
            name="/chat",
            catch_response=True,
        ) as response:
            cache_status = response.headers.get("X-Semantic-Cache", "unknown")
            response.request_meta["name"] = f"/chat [{cache_status}]"
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
                return
            try:
                body = response.json()
            except ValueError:
                response.failure("Invalid JSON response")
                return
            if not body.get("answer") or not isinstance(body.get("sources"), list):
                response.failure("Response must include answer and sources")
                return
            response.success()

    @task(1)
    def chat_stream(self) -> None:
        with self.client.post(
            "/chat/stream",
            json=self._payload(),
            headers=self._headers(),
            name="/chat/stream",
            catch_response=True,
        ) as response:
            cache_status = response.headers.get("X-Semantic-Cache", "unknown")
            response.request_meta["name"] = f"/chat/stream [{cache_status}]"
            if response.status_code != 200:
                response.failure(f"HTTP {response.status_code}")
            elif "event: done" not in response.text:
                response.failure("Stream did not complete")
            else:
                response.success()
