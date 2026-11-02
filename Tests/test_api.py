from pathlib import Path

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from api.app import create_app
from src.agents.base import AgentState


def build_test_graph(checkpointer):
    def respond(state):
        turn = sum(message.type == "human" for message in state["messages"])
        return {
            "messages": [
                ToolMessage(
                    content="[1] arxiv/example.pdf, p.2 (score 0.9)\nEvidence",
                    tool_call_id=f"call-{turn}",
                    name="search_documents",
                ),
                AIMessage(content=f"answer-{turn}"),
            ]
        }

    builder = StateGraph(AgentState)
    builder.add_node("agent", respond)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)
    return builder.compile(checkpointer=checkpointer)


def test_chat_returns_answer_sources_and_persists_threads(tmp_path: Path):
    checkpoint_path = tmp_path / "nested" / "checkpoints.sqlite"
    app = create_app(checkpoint_path, graph_factory=build_test_graph)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

        first = client.post(
            "/chat",
            json={"thread_id": "conversation-1", "question": "first"},
        )
        assert first.status_code == 200
        assert first.json() == {
            "thread_id": "conversation-1",
            "answer": "answer-1",
            "sources": [{"source": "arxiv/example.pdf", "page": 2}],
        }

        second = client.post(
            "/chat",
            json={"thread_id": "conversation-1", "question": "second"},
        )
        assert second.json()["answer"] == "answer-2"

        isolated = client.post(
            "/chat",
            json={"thread_id": "conversation-2", "question": "separate"},
        )
        assert isolated.json()["answer"] == "answer-1"

    reopened_app = create_app(checkpoint_path, graph_factory=build_test_graph)
    with TestClient(reopened_app) as client:
        resumed = client.post(
            "/chat",
            json={"thread_id": "conversation-1", "question": "after restart"},
        )

    assert resumed.json()["answer"] == "answer-3"


def test_chat_stream_emits_answer_and_sources(tmp_path: Path):
    app = create_app(tmp_path / "stream.sqlite", graph_factory=build_test_graph)

    with TestClient(app) as client:
        response = client.post(
            "/chat/stream",
            json={"thread_id": "stream-1", "question": "hello"},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert 'event: token\ndata: {"text": "answer-1"}' in response.text
    assert '"sources": [{"source": "arxiv/example.pdf", "page": 2}]' in response.text
    assert "event: done" in response.text


def test_chat_rejects_invalid_request(tmp_path: Path):
    app = create_app(tmp_path / "invalid.sqlite", graph_factory=build_test_graph)

    with TestClient(app) as client:
        blank_question = client.post(
            "/chat",
            json={"thread_id": "thread-1", "question": ""},
        )
        invalid_thread = client.post(
            "/chat",
            json={"thread_id": "bad thread", "question": "hello"},
        )

    assert blank_question.status_code == 422
    assert invalid_thread.status_code == 422


def test_chat_requires_configured_bearer_token(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("API_BEARER_TOKEN", "local-test-token")
    app = create_app(tmp_path / "auth.sqlite", graph_factory=build_test_graph)

    with TestClient(app) as client:
        unauthorized = client.post(
            "/chat",
            json={"thread_id": "thread-1", "question": "hello"},
        )
        authorized = client.post(
            "/chat",
            headers={"Authorization": "Bearer local-test-token"},
            json={"thread_id": "thread-1", "question": "hello"},
        )

    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
