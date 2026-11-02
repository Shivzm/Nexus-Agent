"""FastAPI endpoints with SQLite-persisted ReAct conversation threads."""

import json
import hmac
import logging
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator, Annotated

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from src.agents.react_agent import build_react_agent

logger = logging.getLogger(__name__)
SOURCE_PATTERN = re.compile(r"\[\d+\]\s+(.+?),\s+p\.(\d+)")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20_000)
    thread_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        )
    return ""


def _response_from_state(state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = state.get("messages", [])
    last_human = max(
        (index for index, message in enumerate(messages) if getattr(message, "type", None) == "human"),
        default=-1,
    )
    answer = next(
        (
            _text_content(message.content)
            for message in reversed(messages[last_human + 1 :])
            if isinstance(message, AIMessage) and _text_content(message.content).strip()
        ),
        "",
    )
    sources = []
    seen_sources = set()
    for message in messages[last_human + 1 :]:
        if not isinstance(message, ToolMessage):
            continue
        for source, page in SOURCE_PATTERN.findall(_text_content(message.content)):
            key = (source.strip(), int(page))
            if key not in seen_sources:
                seen_sources.add(key)
                sources.append({"source": key[0], "page": key[1]})
    return {"thread_id": thread_id, "answer": answer, "sources": sources}


def _authorize(authorization: Annotated[str | None, Header()] = None) -> None:
    expected_token = os.getenv("API_BEARER_TOKEN")
    if expected_token and not hmac.compare_digest(
        authorization or "",
        f"Bearer {expected_token}",
    ):
        raise HTTPException(status_code=401, detail="Invalid or missing bearer token")


def create_app(
    checkpoint_path: Path | None = None,
    graph_factory=build_react_agent,
) -> FastAPI:
    db_path = checkpoint_path or Path(os.getenv("API_CHECKPOINT_DB", "data/api_checkpoints.sqlite"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            app.state.graph = graph_factory(checkpointer=checkpointer)
            yield

    app = FastAPI(
        title="Nexus-Agent API",
        description="Streaming chat API with persistent SQLite-backed conversation threads.",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat")
    def chat(
        request: ChatRequest,
        _: Annotated[None, Depends(_authorize)],
    ) -> dict[str, Any]:
        config = {"configurable": {"thread_id": request.thread_id}}
        state = app.state.graph.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config,
        )
        return _response_from_state(state, request.thread_id)

    @app.post("/chat/stream")
    def chat_stream(
        request: ChatRequest,
        _: Annotated[None, Depends(_authorize)],
    ) -> StreamingResponse:
        config = {"configurable": {"thread_id": request.thread_id}}

        def events() -> Iterator[str]:
            for update in app.state.graph.stream(
                {"messages": [HumanMessage(content=request.question)]},
                config=config,
                stream_mode="updates",
            ):
                for delta in update.values():
                    for message in delta.get("messages", []):
                        if isinstance(message, AIMessage):
                            text = _text_content(message.content)
                            if text:
                                yield f"event: token\ndata: {json.dumps({'text': text}, ensure_ascii=False)}\n\n"
            state = app.state.graph.get_state(config).values
            result = _response_from_state(state, request.thread_id)
            yield f"event: done\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app


app = create_app()
