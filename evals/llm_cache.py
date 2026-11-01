"""Tiny on-disk cache for LLM calls so reruns and resumed evals cost zero tokens."""

import hashlib
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any


class CachedLLM:
    """Wrap any object with ``invoke(prompt) -> response.content`` with a SQLite cache.

    The key covers the model fingerprint and the full prompt, so changing the model,
    reasoning effort, token cap or retrieved contexts always misses. Only successful
    text responses are stored, so a crash never leaves a bad entry behind.
    """

    def __init__(self, llm: Any, path: Path | str, fingerprint: str):
        self.llm = llm
        self.fingerprint = fingerprint
        self.hits = 0
        self.misses = 0
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, content TEXT NOT NULL)")
        self._db.commit()

    def _key(self, prompt: str) -> str:
        return hashlib.sha256(json.dumps([self.fingerprint, prompt]).encode()).hexdigest()

    def invoke(self, prompt: str) -> Any:
        key = self._key(prompt)
        row = self._db.execute("SELECT content FROM cache WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return SimpleNamespace(content=row[0])
        response = self.llm.invoke(prompt)
        self.misses += 1
        if isinstance(response.content, str) and response.content.strip():
            self._db.execute("INSERT OR REPLACE INTO cache (key, content) VALUES (?, ?)", (key, response.content))
            self._db.commit()
        return response
