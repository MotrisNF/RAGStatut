"""Split corpus documents into chunks small enough to retrieve."""
from abc import ABC, abstractmethod
import re

from tqdm import tqdm

from .models import Document, Chunk


class BaseChunker(ABC):
    """Common base of the chunkers: size check and batch chunking."""

    def __init__(self, max_chunk_size: int) -> None:
        """Store the maximum chunk size.

        Args:
            max_chunk_size: Maximum length of a chunk, in characters.

        Raises:
            TypeError: ``max_chunk_size`` is not an integer.
            ValueError: ``max_chunk_size`` is lower than 1.
        """
        if (not isinstance(max_chunk_size, int)
                or isinstance(max_chunk_size, bool)):
            raise TypeError(
                f"max_chunk_size must be a valid integer. Got {max_chunk_size}"
            )

        if max_chunk_size <= 0:
            raise ValueError(
                f"max_chunk_size must be at least 1. Got {max_chunk_size}"
            )

        self._max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split one document into chunks.

        Args:
            doc: The document to split.

        Returns:
            Its chunks, in file order.
        """

    def chunk_documents(self, docs: list[Document]) -> list[Chunk]:
        """Split every document, with a progress bar.

        Args:
            docs: The documents to split.

        Returns:
            The chunks of all the documents, in order.
        """
        chunk_list: list[Chunk] = []
        # No bar for an empty list (an incremental index that only
        # changed files of the other type).
        for doc in tqdm(docs, desc="Chunking files", unit="file",
                        disable=not docs):
            chunk_list.extend(self.chunk_document(doc))

        return chunk_list


class FixedSizeChunker(BaseChunker):
    """Cut text into chunks of at most ``max_chunk_size`` characters.

    Each cut is moved back to the last separator found in the second
    half of the chunk, so chunks tend to end at a blank line, a line or
    a word instead of in the middle of one.

    Attributes:
        SEPARATORS: Preferred cut points, best first.
        CUT_BEFORE: Separators that START the next chunk instead of
            ending the current one. None here: a blank line, a newline or
            a space stays at the end of the chunk it closes.
    """

    SEPARATORS: tuple[str, ...] = ("\n\n", "\n", " ")
    CUT_BEFORE: frozenset[str] = frozenset()

    def _find_cut(self, text: str, start: int, end: int, size: int) -> int:
        """Position where the chunk ``text[start:end]`` should end.

        A separator of ``CUT_BEFORE`` (e.g. a Python definition in
        ``PyChunker``) is cut right after its leading newline, so it
        starts the next chunk instead of ending this one. Here
        ``CUT_BEFORE`` is empty: every cut goes after the separator.

        Args:
            text: The whole document.
            start: Start of the chunk.
            end: Furthest allowed end.
            size: Maximum chunk size.

        Returns:
            For the best separator found in the second half of the chunk:
            the position just before the definition (``CUT_BEFORE``) or
            just after the separator (the others). ``end`` if there is
            none.
        """
        for separator in self.SEPARATORS:
            position = text.rfind(separator, start, end)

            if position > start + size // 2:
                if separator not in self.CUT_BEFORE:
                    return position + len(separator)
                else:
                    return position + 1

        return end

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split one document into consecutive chunks.

        Chunks that are only whitespace are dropped.

        Args:
            doc: The document to split.

        Returns:
            Its chunks, in file order, with their character ranges.
        """
        text = doc.text
        size = self._max_chunk_size
        chunk_list: list[Chunk] = []

        start = 0

        while start < len(text):
            end = min(start + size, len(text))

            if end < len(text):
                end = self._find_cut(text, start, end, size)

            chunk = text[start:end]
            if chunk.strip():
                chunk_list.append(Chunk(
                    file_path=doc.file_path,
                    first_character_index=start,
                    last_character_index=end,
                    text=chunk
                ))

            start = end

        return chunk_list


class PyChunker(FixedSizeChunker):
    """Cut Python code preferably right before a definition.

    Chunks then hold whole classes, functions or methods when they fit,
    instead of starting in the middle of one.

    Attributes:
        SEPARATORS: Preferred cut points, best first: top-level
            definitions (decorators first, so they stay with what they
            decorate), then methods (indented four spaces), then the
            generic blank line, newline and space.
        CUT_BEFORE: The definition separators. The cut goes right after
            their leading newline, so ``@``, ``class`` or ``def`` begins
            the next chunk.
        DEFINITION_STARTS: Line starts (after the indentation) of a
            definition or a decorator: a cut right before one of them is
            moved back over the decorators above it.
    """

    SEPARATORS: tuple[str, ...] = (
        "\n@",
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n    @",
        "\n    def ",
        "\n    async def ",
        "\n\n",
        "\n",
        " ",
    )
    CUT_BEFORE: frozenset[str] = frozenset({
        "\n@",
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\n    @",
        "\n    def ",
        "\n    async def ",
    })
    DEFINITION_STARTS: tuple[str, ...] = ("@", "def ", "async def ",
                                          "class ")

    def _find_cut(self, text: str, start: int, end: int, size: int) -> int:
        """Cut like ``FixedSizeChunker``, then keep decorators with their
        definition.

        When the cut lands right before a definition or a decorator, it is
        moved back over the single-line decorators just above it
        (``@a\\n@b\\ndef f():``), so they start the next chunk with what
        they decorate. A decorator in the first half of the window is not
        a cut candidate by itself, so without this step the cut would
        fall between it and its ``def``.

        Args:
            text: The whole document.
            start: Start of the chunk.
            end: Furthest allowed end.
            size: Maximum chunk size.

        Returns:
            The position where the chunk should end; always after
            ``start``, so the chunking always advances. If the decorators
            reach back to ``start`` (the whole chunk would be decorators),
            the cut is not moved.
        """
        cut = super()._find_cut(text, start, end, size)
        if text[cut - 1] != "\n":
            return cut
        line_end = text.find("\n", cut)
        line = text[cut:line_end if line_end != -1 else len(text)]
        if not line.lstrip(" ").startswith(self.DEFINITION_STARTS):
            return cut

        moved = cut
        while True:
            previous = max(text.rfind("\n", start, moved - 1) + 1, start)
            if not text[previous:moved].lstrip(" ").startswith("@"):
                return moved
            if previous == start:
                return cut
            moved = previous


class MdChunker(FixedSizeChunker):
    """Cut Markdown and plain text preferably right after a fenced code
    block closes.

    This is the "Markdown / text" strategy of the subject: ``index`` uses
    it for ``.md`` and ``.txt`` files. A code block (``` ... ```) is kept
    whole when it fits, instead of being split in the middle of a command
    or a snippet; text without code blocks is cut like
    ``FixedSizeChunker`` (blank line, newline, space). Cutting before
    headings was measured too and lost ~4 points of docs recall@5 (see
    README → Performance), so headings are not separators.

    Attributes:
        SEPARATORS: Preferred cut points, best first: the end of a code
            block (``\\n```\\n``), then a blank line, a newline and a
            space.
        CUT_BEFORE: Empty: every cut goes after its separator. In the
            vLLM docs, almost every opening fence carries a language
            (```bash, ```python): ``\\n```\\n`` matches the closing ones,
            so the block stays in the chunk it closes.
    """

    SEPARATORS: tuple[str, ...] = (
        "\n```\n",
        "\n\n",
        "\n",
        " ",
        )
    CUT_BEFORE: frozenset[str] = frozenset()


class ArticleChunker(MdChunker):
    """Cut Markdown and text like ``MdChunker``, but never across a legal
    article.

    Every article starts a new chunk, a long article is split inside
    itself, and all its chunks keep the article's heading in ``title``, so
    a paragraph that does not name its topic can still be found. A text
    without articles gets exactly the chunks of ``MdChunker``.

    Attributes:
        HEADING: Start of an article (``Artículo 22.``, ``Artículo 38
            bis.``). A reference such as ``Artículo 37.3`` does not match.
        MAX_TITLE: Maximum length of a title, in characters.
    """

    HEADING: re.Pattern[str] = re.compile(
        r"^Art[ií]culo \d+(?: bis| ter)?\.(?=\s|$)", re.MULTILINE)
    MAX_TITLE: int = 120

    @classmethod
    def _title(cls, article: str) -> str:
        """Heading of an article, without its number on a line of its own.

        Some agreements write ``Artículo 4.`` and the name on the next
        line; both are joined.

        Args:
            article: The text of the article, from its heading on.

        Returns:
            The heading, at most ``MAX_TITLE`` characters.
        """
        lines = [line.strip() for line in article.splitlines()
                 if line.strip()]
        title = lines[0]
        if cls.HEADING.fullmatch(title) and len(lines) > 1:
            title = f"{title} {lines[1]}"
        return title[:cls.MAX_TITLE]

    def chunk_document(self, doc: Document) -> list[Chunk]:
        """Split one document, article by article.

        Args:
            doc: The document to split.

        Returns:
            Its chunks, in file order, with their character ranges in the
            whole document. The text before the first article has no
            title.
        """
        starts = sorted({0, *(found.start()
                              for found in self.HEADING.finditer(doc.text)),
                         })
        ends = starts[1:] + [len(doc.text)]
        chunk_list: list[Chunk] = []
        for start, end in zip(starts, ends):
            part = doc.text[start:end]
            title = self._title(part) if self.HEADING.match(part) else ""
            piece = Document(file_path=doc.file_path, text=part)
            for chunk in super().chunk_document(piece):
                chunk_list.append(chunk.model_copy(update={
                    "first_character_index":
                        chunk.first_character_index + start,
                    "last_character_index":
                        chunk.last_character_index + start,
                    "title": title,
                }))
        return chunk_list
