"""Answer cache: a repeated question answered without loading the model.

Generation is deterministic (greedy decoding, ``do_sample=False``), so the
same question with the same context always gives the same answer, and a
cached answer is exactly the one that would be generated. Checking the
cache never imports ``src.generator`` (nor PyTorch): a hit is instant.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

from pydantic import ValidationError

from .models import AnswerCacheData, Chunk


class AnswerCache:
    """Answers saved by question and context, between commands.

    Unlike ``CachedRetriever`` it wraps nothing: the commands ask it for
    an answer and, on a miss, generate it and ``put`` it. Like it, the
    cache is a help, never a requirement: a missing, broken or foreign
    file means an empty cache, and a file that cannot be written only
    prints a warning.

    Attributes:
        FILE_NAME: Name of the cache file inside the index folder.
        GENERATOR_FILE: Source of the generator; its text, not the
            module, gives the namespace.
    """

    FILE_NAME = "answer_cache.json"
    GENERATOR_FILE = Path(__file__).parent / "generator.py"

    @classmethod
    def file_path(cls, index_dir: Path) -> Path:
        """Where the answer cache is saved.

        Args:
            index_dir: Folder of the index, next to the query caches.

        Returns:
            ``index_dir/answer_cache.json``.
        """
        return index_dir / cls.FILE_NAME

    @classmethod
    def namespace(cls) -> str:
        """Identify the generator that produced the answers.

        The model, the prompt, the token budget and the decoding all live
        in ``generator.py``, so any change to it gives another namespace
        and the old answers are ignored. The file is read as text, never
        imported, so PyTorch is not loaded.

        Returns:
            The sha256 of the text of ``GENERATOR_FILE``, in hexadecimal.

        Raises:
            OSError: ``generator.py`` cannot be read (a broken install).
        """
        text = cls.GENERATOR_FILE.read_text(encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def key(question: str, chunks: list[Chunk]) -> str:
        """Key of an answer: everything that goes into the prompt.

        The text of each chunk is part of the key, not only its location,
        so an edited file never gives a stale answer. The chunks are kept
        in order, because their order in the prompt changes the answer.

        Args:
            question: The question, as given.
            chunks: The context chunks, in the order given to the model.

        Returns:
            The sha256 of a JSON with the question and the path, range
            and text of each chunk, in hexadecimal (64 characters).
        """
        data = {"question": question,
                "chunks": [[c.file_path, c.first_character_index,
                            c.last_character_index, c.text]
                           for c in chunks]}
        # sort_keys: the same content always gives the same text.
        text = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def __init__(self, path: Path) -> None:
        """Read the saved answers of the current generator.

        Args:
            path: The JSON file the cache is read from and written to
                (``file_path(index_dir)``).
        """
        self._path = path
        self._namespace = self.namespace()
        self._entries = self._read()
        self._changed = False

    def _read(self) -> dict[str, str]:
        """Load the saved answers, if they can be reused.

        Returns:
            The saved entries, or an empty dict when the file is missing,
            unreadable, not valid, or from another namespace.
        """
        try:
            text = self._path.read_text(encoding="utf-8")
            data = AnswerCacheData.model_validate_json(text)
        except (OSError, ValidationError, UnicodeDecodeError):
            return {}
        if data.namespace != self._namespace:
            return {}
        return data.entries

    def get(self, key: str) -> str | None:
        """Return the cached answer of a key.

        Args:
            key: From ``key(question, chunks)``.

        Returns:
            The answer, or ``None`` when it is not cached.
        """
        return self._entries.get(key)

    def put(self, key: str, answer: str) -> None:
        """Remember an answer; ``flush`` saves it.

        Args:
            key: From ``key(question, chunks)``.
            answer: The generated answer.
        """
        self._entries[key] = answer
        self._changed = True

    def flush(self) -> None:
        """Save the cache if an answer was added.

        The file is written to a ``.tmp`` file first and then moved over
        the old one (``os.replace``), so an interrupted write never leaves
        a half-written cache. If it cannot be written, a warning is
        printed and the command goes on.
        """
        if not self._changed:
            return
        data = AnswerCacheData(namespace=self._namespace,
                               entries=self._entries)
        tmp = self._path.with_suffix(".tmp")

        try:
            tmp.write_text(data.model_dump_json(), encoding="utf-8")
            os.replace(tmp, self._path)
            self._changed = False
        except OSError as e:
            print("WARNING: cannot save the answer cache in "
                  f"'{self._path}': {e}", file=sys.stderr)
            return
