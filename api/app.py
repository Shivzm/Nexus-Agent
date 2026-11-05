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
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import BaseModel, Field

from config.settings import settings
from src.agents.react_agent import build_react_agent
from src.rag.semantic_cache import SemanticResponseCache

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
    cache_factory=None,
) -> FastAPI:
    db_path = checkpoint_path or Path(os.getenv("API_CHECKPOINT_DB", "data/api_checkpoints.sqlite"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with SqliteSaver.from_conn_string(str(db_path)) as checkpointer:
            app.state.graph = graph_factory(checkpointer=checkpointer)
            app.state.semantic_cache = (
                cache_factory()
                if cache_factory is not None
                else _create_semantic_cache()
                if settings.api_semantic_cache_enabled
                else None
            )
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
    ) -> JSONResponse:
        config = {"configurable": {"thread_id": request.thread_id}}
        cached, cache_status = _get_cache_hit(app, request, config)
        if cached is not None:
            cached["thread_id"] = request.thread_id
            cached.pop("cache_similarity", None)
            return JSONResponse(cached, headers={"X-Semantic-Cache": cache_status})
        state = app.state.graph.invoke(
            {"messages": [HumanMessage(content=request.question)]},
            config=config,
        )
        response = _response_from_state(state, request.thread_id)
        if _is_cacheable(state):
            _put_cache_entry(app, request.question, response)
        return JSONResponse(response, headers={"X-Semantic-Cache": cache_status})

    @app.post("/chat/stream")
    def chat_stream(
        request: ChatRequest,
        _: Annotated[None, Depends(_authorize)],
    ) -> StreamingResponse:
        config = {"configurable": {"thread_id": request.thread_id}}
        cached, cache_status = _get_cache_hit(app, request, config)

        def events() -> Iterator[str]:
            if cached is not None:
                cached.pop("cache_similarity", None)
                cached["thread_id"] = request.thread_id
                yield f"event: token\ndata: {json.dumps({'text': cached['answer']}, ensure_ascii=False)}\n\n"
                yield f"event: done\ndata: {json.dumps(cached, ensure_ascii=False)}\n\n"
                return
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
            if _is_cacheable(state):
                _put_cache_entry(app, request.question, result)
            yield f"event: done\ndata: {json.dumps(result, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Semantic-Cache": cache_status,
            },
        )

    return app


def _create_semantic_cache() -> SemanticResponseCache:
    from src.rag.store import get_embedder

    cache_namespace = "|".join((
        "semantic-response-v1",
        settings.rag_collection,
        settings.rag_embedding_model,
        settings.rag_retrieval_config,
        settings.react_model,
        settings.default_model,
        settings.rag_reranker_model,
        str(settings.rag_score_threshold),
    ))
    return SemanticResponseCache(
        path=settings.api_semantic_cache_db,
        embed=lambda text: get_embedder().encode([text])[0].tolist(),
        namespace=cache_namespace,
        threshold=settings.api_semantic_cache_threshold,
        max_entries=settings.api_semantic_cache_max_entries,
    )


def _get_cache_hit(
    app: FastAPI,
    request: ChatRequest,
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    cache = getattr(app.state, "semantic_cache", None)
    if cache is None:
        return None, "disabled"
    current = app.state.graph.get_state(config).values
    if current.get("messages"):
        return None, "bypass"
    response = cache.get(request.question)
    return response, "hit" if response is not None else "miss"


def _is_cacheable(state: dict[str, Any]) -> bool:
    for message in state.get("messages", []):
        if any(call.get("name") == "search_web" for call in getattr(message, "tool_calls", [])):
            return False
        if (
            isinstance(message, ToolMessage)
            and "Web fallback results" in _text_content(message.content)
        ):
            return False
    return True


def _put_cache_entry(app: FastAPI, question: str, response: dict[str, Any]) -> None:
    cache = getattr(app.state, "semantic_cache", None)
    if cache is not None:
        cache.put(question, response)


app = create_app()
