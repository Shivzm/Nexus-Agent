"""SQLite-backed semantic response cache for stateless API requests."""

import hashlib
import json
import math
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any, Callable

import numpy as np


class SemanticResponseCache:
    def __init__(
        self,
        path: Path | str,
        embed: Callable[[str], list[float]],
        namespace: str,
        threshold: float = 0.96,
        max_entries: int = 2000,
    ) -> None:
        if not math.isfinite(threshold) or not 0 < threshold <= 1:
            raise ValueError("Semantic cache threshold must be in (0, 1]")
        if max_entries < 1:
            raise ValueError("Semantic cache max_entries must be at least one")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.embed = embed
        self.namespace = namespace
        self.threshold = threshold
        self.max_entries = max_entries
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
        self._embedding_lock = threading.Lock()
        with closing(self._connect()) as db, db:
            db.execute(
                """
                CREATE TABLE IF NOT EXISTS semantic_responses (
                    namespace TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    question TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (namespace, question_hash)
                )
                """
            )
            db.execute(
                "CREATE INDEX IF NOT EXISTS semantic_responses_created "
                "ON semantic_responses(namespace, created_at)"
            )

    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        db.execute("PRAGMA journal_mode=WAL")
        return db

    def _vector(self, question: str) -> np.ndarray:
        with self._embedding_lock:
            vector = np.asarray(self.embed(question), dtype=np.float32)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError("Semantic cache embedder must return a finite, non-empty vector")
        norm = float(np.linalg.norm(vector))
        if norm == 0:
            raise ValueError("Semantic cache embedder returned a zero vector")
        return vector / norm

    def get(self, question: str) -> dict[str, Any] | None:
        vector = self._vector(question)
        with self._lock, closing(self._connect()) as db:
            entries = db.execute(
                """
                SELECT embedding, response_json
                FROM semantic_responses
                WHERE namespace = ?
                """,
                (self.namespace,),
            ).fetchall()

        best_response = None
        best_score = -1.0
        for embedding_bytes, response_json in entries:
            candidate = np.frombuffer(embedding_bytes, dtype=np.float32)
            if candidate.shape != vector.shape:
                continue
            score = float(np.dot(vector, candidate))
            if score > best_score:
                best_score = score
                best_response = response_json

        if best_response is not None and best_score >= self.threshold:
            with self._lock:
                self.hits += 1
            response = json.loads(best_response)
            response["cache_similarity"] = best_score
            return response

        with self._lock:
            self.misses += 1
        return None

    def put(self, question: str, response: dict[str, Any]) -> None:
        vector = self._vector(question)
        normalized_question = " ".join(question.casefold().split())
        question_hash = hashlib.sha256(normalized_question.encode("utf-8")).hexdigest()
        response_json = json.dumps(response, ensure_ascii=False)
        with self._lock, closing(self._connect()) as db, db:
            db.execute(
                """
                INSERT INTO semantic_responses
                    (namespace, question_hash, question, embedding, response_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, question_hash) DO UPDATE SET
                    question = excluded.question,
                    embedding = excluded.embedding,
                    response_json = excluded.response_json,
                    created_at = excluded.created_at
                """,
                (
                    self.namespace,
                    question_hash,
                    question,
                    vector.tobytes(),
                    response_json,
                    time.time(),
                ),
            )
            db.execute(
                """
                DELETE FROM semantic_responses
                WHERE namespace = ? AND question_hash IN (
                    SELECT question_hash FROM semantic_responses
                    WHERE namespace = ?
                    ORDER BY created_at DESC
                    LIMIT -1 OFFSET ?
                )
                """,
                (self.namespace, self.namespace, self.max_entries),
            )


def clear_semantic_cache(path: Path | str) -> None:
    """Invalidate cached answers after the indexed corpus changes."""
    cache_path = Path(path)
    if not cache_path.exists():
        return
    with closing(sqlite3.connect(cache_path, timeout=30)) as db, db:
        db.execute("DELETE FROM semantic_responses")
