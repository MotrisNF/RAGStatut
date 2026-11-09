"""Search by meaning with vectors from Gemini Embedding.

The chunks were embedded once and their vectors are saved on disk. A search
only embeds the question, and a question already embedded is read from the
same file, so it costs nothing. The free quota is a hard limit: every call
is written to a ledger before it is sent, nothing is ever retried, and
whenever Gemini cannot be used the search falls back to the local retriever.
"""
from collections.abc import Callable
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import urllib.error
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
from numpy.typing import NDArray

from .base_retriever import Retriever
from .bm25_index import BM25Index
from .exceptions import RagError
from .models import Chunk, SearchHit


class QuotaGuard:
    """Refuses any call that could go over the free quota.

    The API does not report the remaining quota, so the calls are counted
    here, in a ledger shared with the script that embeds the corpus.

    Attributes:
        DAY_LIMIT: Calls per quota day (the real limit is 1000).
        MINUTE_REQUESTS: Calls per minute (the real limit is 100).
        MINUTE_TOKENS: Estimated tokens per minute (the real limit is
            30000).
        WINDOW: Seconds counted as a minute, with some margin.
        TIMEZONE: The quota day ends at midnight in this zone.
    """

    DAY_LIMIT: int = 900
    MINUTE_REQUESTS: int = 40
    MINUTE_TOKENS: int = 15000
    WINDOW: int = 62
    TIMEZONE: ZoneInfo = ZoneInfo("America/Los_Angeles")

    def __init__(self, ledger: Path,
                 clock: Callable[[], float] = time.time,
                 day_limit: int | None = None,
                 minute_requests: int | None = None,
                 minute_tokens: int | None = None) -> None:
        """Remember where the calls are written.

        Args:
            ledger: The JSON Lines file with one entry per call.
            clock: Gives the current time, in seconds.
            day_limit: Calls per day; ``DAY_LIMIT`` by default.
            minute_requests: Calls per minute; ``MINUTE_REQUESTS`` by
                default.
            minute_tokens: Tokens per minute; ``MINUTE_TOKENS`` by default.
        """
        self._ledger = ledger
        self._clock = clock
        self._day_limit = day_limit or self.DAY_LIMIT
        self._minute_requests = minute_requests or self.MINUTE_REQUESTS
        self._minute_tokens = minute_tokens or self.MINUTE_TOKENS

    def _entries(self) -> list[dict[str, object]]:
        """Every call written so far; none if the file does not exist.

        Raises:
            OSError: The file exists but cannot be read.
            ValueError: A line is not valid JSON.
        """
        if not self._ledger.exists():
            return []
        with self._ledger.open(encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def reserve(self, tokens: int) -> bool:
        """Write a call in the ledger, if it fits in every limit.

        It is written before the call is sent, so a call that fails also
        counts. An unreadable ledger refuses the call: without knowing
        what was spent, nothing can be spent.

        Args:
            tokens: Estimated tokens of the call.

        Returns:
            Whether the call can be sent.
        """
        now = self._clock()
        day = datetime.fromtimestamp(now, self.TIMEZONE).strftime("%Y-%m-%d")
        try:
            entries = self._entries()
            today = sum(entry["day"] == day for entry in entries)
            recent = [entry for entry in entries
                      if now - float(str(entry["t"])) < self.WINDOW]
            spent = sum(int(str(entry["tokens"])) for entry in recent)
            if (today >= self._day_limit
                    or len(recent) >= self._minute_requests
                    or spent + tokens > self._minute_tokens):
                return False
            self._ledger.parent.mkdir(parents=True, exist_ok=True)
            with self._ledger.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"day": day, "t": now,
                                       "tokens": tokens}) + "\n")
        except (OSError, ValueError, KeyError, TypeError):
            return False
        return True


class GeminiRetriever(Retriever):
    """Rank the chunks by similarity with Gemini Embedding vectors.

    Only files with a vector for every chunk are searched this way. Any
    other search goes to ``fallback`` without calling the API, so a corpus
    without saved vectors behaves exactly as before.

    Attributes:
        MODEL: The embedding model.
        DIMENSIONS: Length of the vectors.
        URL: Endpoint that embeds one text.
        API_KEY_ENV: Environment variable with the key.
        VECTORS_NAME: File with the saved vectors.
        LEDGER_NAME: File with the calls made.
        CHARS_PER_TOKEN: For estimating tokens; low on purpose, so the
            estimate is above the real count.
        TIMEOUT: Seconds to wait for an answer.
    """

    MODEL: str = "gemini-embedding-2"
    DIMENSIONS: int = 768
    URL: str = ("https://generativelanguage.googleapis.com/v1beta/models/"
                f"{MODEL}:embedContent")
    API_KEY_ENV: str = "GEMINI_API_KEY"
    VECTORS_NAME: str = "vectors.jsonl"
    LEDGER_NAME: str = "ledger.jsonl"
    CHARS_PER_TOKEN: int = 3
    TIMEOUT: int = 20

    def __init__(self, bm25: BM25Index, fallback: Retriever, folder: Path,
                 embed: Callable[[str], list[float]] | None = None,
                 guard: QuotaGuard | None = None) -> None:
        """Load the saved vectors of the indexed chunks.

        Args:
            bm25: The index with the chunks; chunk ids are shared.
            fallback: Used whenever Gemini cannot answer.
            folder: Folder with the vectors and the ledger.
            embed: Turns a text into a vector; by default, the API. The
                tests replace it.
            guard: The quota limits; by default, ``folder``'s ledger.
        """
        self._fallback = fallback
        self._vectors_path = folder / self.VECTORS_NAME
        self._guard = guard or QuotaGuard(folder / self.LEDGER_NAME)
        self._embed = embed
        self._saved = self._read_vectors()
        self._disabled = False
        self._warned = False

        chunks = bm25.chunks
        self._chunks = chunks
        vectors = [self._saved.get(self.key(self.document_text(chunk)))
                   for chunk in chunks]
        self._names = [PurePosixPath(chunk.file_path).stem
                       for chunk in chunks]
        self._ids = [i for i, vector in enumerate(vectors)
                     if vector is not None]
        self._matrix: NDArray[np.float32] = (
            np.stack([self._unit(vector) for vector in vectors
                      if vector is not None])
            if self._ids else
            np.zeros((0, self.DIMENSIONS), dtype=np.float32))
        missing = {name for name, vector in zip(self._names, vectors)
                   if vector is None}
        self._covered = set(self._names) - missing

    @classmethod
    def key(cls, text: str) -> str:
        """Key of a text in the vectors file."""
        return hashlib.sha256(
            f"{cls.MODEL}|{cls.DIMENSIONS}|{text}".encode()).hexdigest()

    @staticmethod
    def document_text(chunk: Chunk) -> str:
        """The text embedded for a chunk, as the model asks for it."""
        return f"title: {chunk.title or 'none'} | text: {chunk.text}"

    @staticmethod
    def query_text(query: str) -> str:
        """The text embedded for a question, as the model asks for it."""
        return f"task: search result | query: {query}"

    @staticmethod
    def _unit(vector: NDArray[np.float32]) -> NDArray[np.float32]:
        """The vector scaled to length 1."""
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def _read_vectors(self) -> dict[str, NDArray[np.float32]]:
        """The saved vectors by key; broken lines are ignored."""
        saved: dict[str, NDArray[np.float32]] = {}
        try:
            with self._vectors_path.open(encoding="utf-8") as file:
                for line in file:
                    try:
                        row = json.loads(line)
                        saved[row["key"]] = np.array(row["v"],
                                                     dtype=np.float32)
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            return {}
        return saved

    def _call_api(self, text: str) -> list[float]:
        """Embed one text with the API, once.

        Args:
            text: The text to embed.

        Returns:
            The vector.

        Raises:
            RagError: The call failed. The key is never in the message.
        """
        body = json.dumps({
            "model": f"models/{self.MODEL}",
            "content": {"parts": [{"text": text}]},
            "output_dimensionality": self.DIMENSIONS}).encode()
        request = urllib.request.Request(
            self.URL, data=body,
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": os.environ[self.API_KEY_ENV]})
        try:
            with urllib.request.urlopen(request,
                                        timeout=self.TIMEOUT) as response:
                values: list[float] = json.load(response)["embedding"][
                    "values"]
        except urllib.error.HTTPError as e:
            raise RagError(f"HTTP {e.code}") from e
        except (OSError, ValueError, KeyError) as e:
            raise RagError(type(e).__name__) from e
        return values

    def _warn(self, reason: str) -> None:
        """Say once that the local search is being used."""
        if not self._warned:
            self._warned = True
            print(f"WARNING: Gemini not used ({reason}); using the local "
                  "search.", file=sys.stderr)

    def _save(self, key: str, vector: NDArray[np.float32]) -> None:
        """Keep a new vector, so that question never costs again."""
        self._saved[key] = vector
        try:
            with self._vectors_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps({"key": key,
                                       "v": vector.tolist()}) + "\n")
        except OSError:
            self._warn("cannot save the vector")

    def _query_vector(self, query: str) -> NDArray[np.float32] | None:
        """The vector of a question: saved if possible, else one call.

        Args:
            query: The question.

        Returns:
            A vector of length 1, or ``None`` when it cannot be obtained
            (no key, no quota, or a previous call failed).
        """
        text = self.query_text(query)
        key = self.key(text)
        if key in self._saved:
            return self._unit(self._saved[key])
        if self._disabled:
            return None
        embed = self._embed
        if embed is None:
            if not os.environ.get(self.API_KEY_ENV):
                self._warn(f"{self.API_KEY_ENV} is not set")
                return None
            embed = self._call_api
        if not self._guard.reserve(len(text) // self.CHARS_PER_TOKEN + 1):
            self._warn("quota limit reached")
            return None
        try:
            values = embed(text)
        except RagError as e:
            # Never retried: this process stops calling the API.
            self._disabled = True
            self._warn(str(e))
            return None
        vector = np.array(values, dtype=np.float32)
        self._save(key, vector)
        return self._unit(vector)

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return the ``k`` chunks most similar to the question.

        Args:
            query: The question.
            k: Maximum number of results.
            sources: Names of the files to search, without folder or
                extension; ``None`` searches them all.

        Returns:
            Up to ``k`` hits, best first, with their similarity; the
            hits of ``fallback`` when Gemini cannot answer.
        """
        chosen = set(self._names) if sources is None else set(sources)
        if not chosen or not chosen <= self._covered:
            return self._fallback.search(query, k, sources)
        vector = self._query_vector(query)
        if vector is None:
            return self._fallback.search(query, k, sources)

        rows = [row for row, chunk_id in enumerate(self._ids)
                if self._names[chunk_id] in chosen]
        scores = self._matrix[rows] @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [SearchHit(chunk=self._chunks[self._ids[rows[i]]],
                          score=float(scores[i]),
                          chunk_id=self._ids[rows[i]])
                for i in order]
