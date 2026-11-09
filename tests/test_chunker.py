"""Tests for ``src.chunker``."""
import random
import re

import pytest

from src.chunker import BaseChunker, FixedSizeChunker, MdChunker, PyChunker
from src.models import Chunk, Document, MinimalSource
from src.reader import CorpusReader
from conftest import needs_corpus

REAL_RAW = CorpusReader.PROJECT_ROOT / "data" / "raw"


class Invariants:
    """Properties every chunking of a document must satisfy."""

    @staticmethod
    def check(doc: Document, chunks: list[Chunk], size: int) -> None:
        """Assert all invariants at once.

        - every chunk belongs to the document and fits in ``size``;
        - its offsets slice exactly its text from the document;
        - chunks are in order, never overlap, never blank;
        - whatever is not covered by a chunk is only whitespace.
        """
        position = 0
        for chunk in chunks:
            start = chunk.first_character_index
            end = chunk.last_character_index
            assert chunk.file_path == doc.file_path
            assert 0 < end - start <= size
            assert doc.text[start:end] == chunk.text
            assert chunk.text.strip()
            assert start >= position
            assert doc.text[position:start].strip() == ""
            position = end
        assert doc.text[position:].strip() == ""


def make_doc(text: str, path: str = "data/raw/a.md") -> Document:
    """Build a document without touching the disk."""
    return Document(file_path=path, text=text)


class TestBaseChunker:
    """Validation and behaviour shared by every chunker."""

    def test_is_abstract(self) -> None:
        """``BaseChunker`` cannot be created: ``chunk_document`` is
        abstract.
        """
        with pytest.raises(TypeError):
            BaseChunker(10)  # type: ignore[abstract]

    @pytest.mark.parametrize("size", ["abc", "5", 2.5, 2.0, None, True,
                                      False])
    def test_size_wrong_type(self, size: object) -> None:
        """Anything but a real ``int`` is a ``TypeError``."""
        with pytest.raises(TypeError, match="valid integer"):
            FixedSizeChunker(size)  # type: ignore[arg-type]

    @pytest.mark.parametrize("size", [0, -1, -2000])
    def test_size_too_small(self, size: int) -> None:
        """Zero or negative is a ``ValueError`` (it would loop forever)."""
        with pytest.raises(ValueError, match="at least 1"):
            FixedSizeChunker(size)

    @pytest.mark.parametrize("size", [1, 2, 2000, 10**9])
    def test_size_valid(self, size: int) -> None:
        """Positive integers are accepted."""
        FixedSizeChunker(size)

    def test_chunk_documents_joins_in_order(self) -> None:
        """All documents are chunked and results keep document order."""
        docs = [make_doc("a " * 30, "data/raw/1.md"),
                make_doc("b " * 30, "data/raw/2.md"),
                make_doc("", "data/raw/3.md")]
        chunker = FixedSizeChunker(20)
        chunks = chunker.chunk_documents(docs)
        expected = [c for d in docs for c in chunker.chunk_document(d)]
        assert chunks == expected
        paths = [c.file_path for c in chunks]
        assert paths == sorted(paths)
        assert "data/raw/3.md" not in paths

    def test_chunk_documents_empty_list(self) -> None:
        """No documents, no chunks."""
        assert FixedSizeChunker(10).chunk_documents([]) == []

    def test_chunks_never_mix_files(self) -> None:
        """Each chunk comes from exactly one document."""
        docs = [make_doc("x" * 25, f"data/raw/{i}.py") for i in range(3)]
        for chunk in FixedSizeChunker(10).chunk_documents(docs):
            path = chunk.file_path
            text = next(d.text for d in docs if d.file_path == path)
            start = chunk.first_character_index
            end = chunk.last_character_index
            assert text[start:end] == chunk.text


class TestSimpleCases:
    """Small documents with an obvious answer."""

    def test_empty(self) -> None:
        """An empty document has no chunks."""
        assert FixedSizeChunker(10).chunk_document(make_doc("")) == []

    @pytest.mark.parametrize("text", [" ", "\n", "\n\n\n", "  \t \n "])
    def test_blank(self, text: str) -> None:
        """Only whitespace: no chunks."""
        assert FixedSizeChunker(10).chunk_document(make_doc(text)) == []

    def test_shorter_than_size(self) -> None:
        """A small document is one chunk with the whole text."""
        doc = make_doc("hello world\n")
        [chunk] = FixedSizeChunker(100).chunk_document(doc)
        assert (chunk.first_character_index,
                chunk.last_character_index) == (0, len(doc.text))
        assert chunk.text == doc.text

    def test_exactly_size(self) -> None:
        """A document of exactly ``size`` characters is one chunk."""
        doc = make_doc("abcdefghij")
        chunks = FixedSizeChunker(10).chunk_document(doc)
        assert [c.text for c in chunks] == ["abcdefghij"]

    def test_offsets_are_exclusive(self) -> None:
        """``last_character_index`` is exclusive, like Python slicing."""
        doc = make_doc("abcdefghij")
        chunks = FixedSizeChunker(4).chunk_document(doc)
        assert [(c.first_character_index, c.last_character_index)
                for c in chunks] == [(0, 4), (4, 8), (8, 10)]

    def test_chunk_is_a_source(self) -> None:
        """A chunk can be used wherever a ``MinimalSource`` is expected."""
        [chunk] = FixedSizeChunker(10).chunk_document(make_doc("abc"))
        assert isinstance(chunk, MinimalSource)

    def test_text_not_modified(self) -> None:
        """Leading spaces and newlines are kept: offsets stay exact."""
        doc = make_doc("   indented\n\n  code\n")
        [chunk] = FixedSizeChunker(100).chunk_document(doc)
        assert chunk.text == doc.text


class TestSeparators:
    """Where the cut is made: paragraph, line, space or hard cut."""

    def test_prefers_paragraph(self) -> None:
        """A paragraph break past half the window wins over lines."""
        text = "a" * 12 + "\n\n" + "b" * 3 + "\n" + "c" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text == "a" * 12 + "\n\n"

    def test_falls_back_to_line(self) -> None:
        """Without a paragraph break, cut after the last newline."""
        text = "a" * 8 + "\n" + "b" * 6 + "\n" + "c" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text == "a" * 8 + "\n" + "b" * 6 + "\n"

    def test_falls_back_to_space(self) -> None:
        """Without newlines, cut after the last space."""
        text = "word " * 10
        chunks = FixedSizeChunker(12).chunk_document(make_doc(text))
        assert chunks[0].text == "word word "
        assert all(c.text.endswith(" ") for c in chunks)

    def test_hard_cut(self) -> None:
        """No separator at all: cut exactly at ``size``."""
        text = "x" * 25
        chunks = FixedSizeChunker(10).chunk_document(make_doc(text))
        assert [len(c.text) for c in chunks] == [10, 10, 5]

    def test_separator_stays_in_chunk(self) -> None:
        """The whole separator ends the chunk; nothing is lost."""
        text = "a" * 12 + "\n\n" + "b" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text.endswith("\n\n")
        assert chunks[1].first_character_index == 14

    def test_early_paragraph_ignored(self) -> None:
        """A paragraph break in the first half is skipped for a later
        newline, so chunks are not tiny.
        """
        text = "a" * 3 + "\n\n" + "b" * 10 + "\n" + "c" * 20
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert chunks[0].text == "a" * 3 + "\n\n" + "b" * 10 + "\n"

    def test_early_newline_gives_hard_cut(self) -> None:
        """Only separators in the first half: hard cut at ``size``."""
        text = "ab\n" + "x" * 30
        chunks = FixedSizeChunker(10).chunk_document(make_doc(text))
        assert len(chunks[0].text) == 10

    def test_last_chunk_not_cut(self) -> None:
        """The final piece is not split at a separator, even one past
        half of the window.
        """
        text = "a" * 15 + "\nend"
        chunks = FixedSizeChunker(20).chunk_document(make_doc(text))
        assert [c.text for c in chunks] == [text]


class TestExtremeSizes:
    """Sizes that could break the loop."""

    @pytest.mark.parametrize("text", ["abc", "a\nb\n\nc d", "\n\na b",
                                      "x" * 50])
    def test_size_one(self, text: str) -> None:
        """Size 1 terminates and every chunk has one character."""
        doc = make_doc(text)
        chunks = FixedSizeChunker(1).chunk_document(doc)
        Invariants.check(doc, chunks, 1)
        assert all(len(c.text) == 1 for c in chunks)

    def test_size_two_with_separators(self) -> None:
        """Size 2 with separators everywhere still terminates."""
        doc = make_doc("\n \n\n \n a\n\nb ")
        Invariants.check(doc, FixedSizeChunker(2).chunk_document(doc), 2)

    def test_huge_size(self) -> None:
        """A size far bigger than the text gives one chunk."""
        doc = make_doc("some text\n" * 5)
        assert len(FixedSizeChunker(10**9).chunk_document(doc)) == 1


class TestRandomTexts:
    """Invariants on many generated texts (fixed seeds, reproducible)."""

    PIECES = ["word", "x", "longwordwithoutspaces", " ", "  ", "\n", "\n\n",
              "\t", "é", "日本", "🚀", "def f():", "# Title"]

    @pytest.mark.parametrize("seed", range(40))
    @pytest.mark.parametrize("size", [1, 3, 7, 16, 50, 200])
    def test_invariants(self, seed: int, size: int) -> None:
        """Every invariant holds for random mixes of words and spaces."""
        rng = random.Random(seed)
        text = "".join(rng.choice(self.PIECES)
                       for _ in range(rng.randint(0, 150)))
        doc = make_doc(text)
        Invariants.check(doc, FixedSizeChunker(size).chunk_document(doc),
                         size)


@pytest.fixture(scope="module")
def docs() -> list[Document]:
    """Read the real corpus once for the whole module."""
    return CorpusReader(REAL_RAW).read().documents


@needs_corpus
class TestRealCorpus:
    """The whole vLLM corpus, chunked like ``index`` does."""

    @pytest.mark.parametrize("size", [2000, 500])
    def test_invariants(self, docs: list[Document], size: int) -> None:
        """Every chunk of every file satisfies the invariants."""
        chunker = FixedSizeChunker(size)
        for doc in docs:
            Invariants.check(doc, chunker.chunk_document(doc), size)

    def test_max_length_for_moulinette(self, docs: list[Document]) -> None:
        """No chunk exceeds the moulinette's 2000 characters."""
        chunks = FixedSizeChunker(2000).chunk_documents(docs)
        assert max(c.last_character_index - c.first_character_index
                   for c in chunks) <= 2000
        assert len(chunks) > len(docs)


FENCE = "```"


class TestMdChunker:
    """Markdown: prefer to cut right after a code block closes."""

    def test_is_fixed_size_chunker(self) -> None:
        """Same machinery as FixedSizeChunker, only other separators."""
        assert issubclass(MdChunker, FixedSizeChunker)
        assert MdChunker.SEPARATORS[0] == "\n" + FENCE + "\n"
        assert MdChunker.CUT_BEFORE == frozenset()

    def test_cuts_after_closing_fence(self) -> None:
        """The chunk ends with the closing fence; the block stays whole."""
        text = ("Intro text.\n" + FENCE + "bash\npip install vllm\n"
                + FENCE + "\nAfter the block there is more text.")
        chunks = MdChunker(55).chunk_document(make_doc(text))
        assert chunks[0].text.endswith(FENCE + "\n")
        assert FENCE + "bash\npip install vllm\n" + FENCE in chunks[0].text

    def test_fence_beats_blank_line(self) -> None:
        """A closing fence is preferred over a later blank line (both in
        the second half of the window).
        """
        text = ("a" * 20 + "\n" + FENCE + "py\nx = 1\n" + FENCE + "\n"
                + "b" * 5 + "\n\n" + "c" * 40)
        chunks = MdChunker(50).chunk_document(make_doc(text))
        assert chunks[0].text.endswith(FENCE + "\n")
        assert "b" not in chunks[0].text

    def test_opening_fence_with_language_not_a_cut(self) -> None:
        """``\\n```bash`` is not a separator: no cut right after it."""
        text = "intro\n" + FENCE + "bash\n" + "echo hi\n" * 10
        for chunk in MdChunker(40).chunk_document(make_doc(text)):
            assert not chunk.text.endswith(FENCE + "bash\n")

    def test_headings_are_not_separators(self) -> None:
        """Headings were measured worse: they are plain text here."""
        assert not any(sep.lstrip("\n").startswith("#")
                       for sep in MdChunker.SEPARATORS)

    def test_falls_back_to_blank_line(self) -> None:
        """Without code blocks it behaves like FixedSizeChunker."""
        text = "word " * 30 + "\n\n" + "more " * 30
        doc = make_doc(text)
        assert (MdChunker(100).chunk_document(doc)
                == FixedSizeChunker(100).chunk_document(doc))

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("size", [1, 7, 40, 300])
    def test_invariants_random(self, seed: int, size: int) -> None:
        """Exact offsets, no gaps, never above the max, on random text
        full of fences, headings and blank lines.
        """
        rng = random.Random(seed)
        pieces = ["word", " ", "\n", "\n\n", FENCE, FENCE + "bash",
                  "# Title", "## Sub", "pip install x", "`code`"]
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 120)))
        doc = make_doc(text)
        Invariants.check(doc, MdChunker(size).chunk_document(doc), size)


@needs_corpus
class TestMdChunkerRealCorpus:
    """MdChunker on every Markdown file of the vLLM corpus."""

    def test_invariants(self, docs: list[Document]) -> None:
        """Every chunk satisfies the invariants, at 2000 and 500."""
        for size in (2000, 500):
            chunker = MdChunker(size)
            for doc in docs:
                if doc.file_path.endswith(".md"):
                    Invariants.check(doc, chunker.chunk_document(doc), size)

    def test_most_chunks_keep_code_blocks_closed(self,
                                                 docs: list[Document]) -> None:
        """Few chunks end inside an open code block (only blocks longer
        than the chunk size have to be cut).
        """
        inside = total = 0
        for doc in docs:
            if not doc.file_path.endswith(".md"):
                continue
            for chunk in MdChunker(2000).chunk_document(doc):
                total += 1
                end = chunk.last_character_index
                inside += doc.text[:end].count(FENCE) % 2
        assert inside / total < 0.15


DEFINITION = re.compile(r"( {4})?(@|class |def |async def )")


class TestPyChunker:
    """Python: prefer to cut right before a definition."""

    def test_is_fixed_size_chunker(self) -> None:
        """Same machinery as FixedSizeChunker, only other separators."""
        assert issubclass(PyChunker, FixedSizeChunker)

    def test_cut_before_are_separators(self) -> None:
        """Every ``CUT_BEFORE`` entry is a separator, and they all rank
        above the generic ones (which stay cut-after).
        """
        separators = PyChunker.SEPARATORS
        assert PyChunker.CUT_BEFORE <= set(separators)
        generic = separators.index("\n\n")
        assert set(separators[:generic]) == PyChunker.CUT_BEFORE
        assert not {"\n\n", "\n", " "} & PyChunker.CUT_BEFORE

    def test_top_level_before_methods(self) -> None:
        """Top-level definitions are tried before indented ones."""
        separators = PyChunker.SEPARATORS
        methods = [separators.index(s) for s in PyChunker.CUT_BEFORE
                   if s.startswith("\n    ")]
        top_level = [separators.index(s) for s in PyChunker.CUT_BEFORE
                     if not s.startswith("\n    ")]
        assert (len(top_level), len(methods)) == (4, 3)
        assert max(top_level) < min(methods)

    @pytest.mark.parametrize("definition", [
        "def foo():\n", "async def foo():\n", "class Foo:\n",
        "@decorator\ndef foo():\n",
    ])
    def test_cuts_before_top_level(self, definition: str) -> None:
        """The definition starts the next chunk; the newline before it
        stays at the end of the previous one.
        """
        text = "a" * 36 + "\n" + definition + "    return 1\n" + "b" * 50
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[0].text == "a" * 36 + "\n"
        assert chunks[1].text.startswith(definition)

    @pytest.mark.parametrize("definition", [
        "    def run(self):\n", "    async def run(self):\n",
        "    @property\n    def run(self):\n",
    ])
    def test_cuts_before_method(self, definition: str) -> None:
        """Methods (four spaces) start the next chunk, indentation
        included.
        """
        text = "class A:\n" + "a" * 36 + "\n" + definition + "b" * 50
        chunks = PyChunker(80).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith(definition)

    def test_definition_beats_later_blank_line(self) -> None:
        """A definition wins even if a blank line comes after it."""
        text = "a" * 35 + "\nclass A:\n    x = 1\n\n" + "b" * 50
        chunks = PyChunker(60).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith("class A:")

    def test_top_level_beats_later_method(self) -> None:
        """A function is cut before, not inside at its nested def."""
        text = ("a" * 36 + "\ndef top():\n    def inner():\n"
                + "        pass\n" + "b" * 50)
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith("def top():")

    def test_decorator_stays_with_def(self) -> None:
        """``\\n@`` ranks first: the cut goes before the decorator, not
        between it and its ``def``.
        """
        text = "a" * 36 + "\n@decorator\ndef f():\n    pass\n" + "b" * 50
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith("@decorator\ndef f():")

    def test_decorator_in_first_half_kept(self) -> None:
        """A decorator in the first half of the window is not a cut
        candidate, but the cut chosen before its ``def`` moves back over
        it.
        """
        text = ("a" * 30 + "\n@decorator(" + "x" * 10 + ")\ndef f():\n"
                + "b" * 80)
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[0].text == "a" * 30 + "\n"
        assert chunks[1].text.startswith("@decorator(" + "x" * 10
                                         + ")\ndef f():")

    def test_stacked_decorators_kept(self) -> None:
        """All the decorators above a definition go with it, also when
        the cut was chosen at the last one.
        """
        text = ("a" * 30 + "\n@first\n@second\n@third\ndef f():\n"
                + "b" * 80)
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith("@first\n@second\n@third\ndef f")

    def test_method_decorator_kept(self) -> None:
        """Indented decorators stay with their method too."""
        text = ("class A:\n" + "a" * 25 + "\n    @staticmethod\n"
                + "    def run():\n" + "b" * 80)
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith("    @staticmethod\n    def run")

    def test_decorated_class_kept(self) -> None:
        """Decorated classes keep their decorator."""
        text = ("a" * 30 + "\n@dataclass(frozen=True)\nclass A:\n"
                + "b" * 80)
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[1].text.startswith("@dataclass(frozen=True)\nclass")

    def test_decorators_from_chunk_start_not_moved(self) -> None:
        """If the decorators reach back to the chunk start, the cut stays
        before the ``def``: moving it would leave an empty chunk and the
        chunking would never advance.
        """
        text = "@deco\n" * 8 + "def f():\n" + "b" * 80
        doc = make_doc(text, "a.py")
        chunks = PyChunker(70).chunk_document(doc)
        Invariants.check(doc, chunks, 70)
        assert chunks[0].text == "@deco\n" * 7
        assert chunks[1].text.startswith("@deco\ndef f():")

    def test_chunk_starting_mid_line(self) -> None:
        """A chunk that starts in the middle of a decorator line (after
        a hard cut) never moves the cut before its start.
        """
        text = "@" + "d" * 80 + "\n@deco\ndef f():\n" + "b" * 80
        doc = make_doc(text, "a.py")
        Invariants.check(doc, PyChunker(70).chunk_document(doc), 70)

    def test_non_decorator_line_stops(self) -> None:
        """Only ``@`` lines are pulled: a comment or code above them
        stays in the previous chunk.
        """
        text = ("a" * 30 + "\nx = 1\n@deco\ndef f():\n" + "b" * 80)
        chunks = PyChunker(70).chunk_document(make_doc(text, "a.py"))
        assert chunks[0].text.endswith("x = 1\n")
        assert chunks[1].text.startswith("@deco\ndef f():")

    def test_without_definitions_like_fixed(self) -> None:
        """Without definitions it behaves like FixedSizeChunker."""
        text = "x = 1\n" * 20 + "\n\n" + "y = 2 " * 30
        doc = make_doc(text, "a.py")
        assert (PyChunker(50).chunk_document(doc)
                == FixedSizeChunker(50).chunk_document(doc))

    def test_early_definition_ignored(self) -> None:
        """A definition in the first half is not a cut point: the cut
        goes to a later separator instead of a tiny chunk.
        """
        text = "x\ndef f():\n    " + "a" * 30 + "\n" + "b" * 50
        chunks = PyChunker(60).chunk_document(make_doc(text, "a.py"))
        assert chunks[0].text.startswith("x\ndef f():")
        assert len(chunks[0].text) > 30

    @pytest.mark.parametrize("seed", range(20))
    @pytest.mark.parametrize("size", [1, 7, 40, 300])
    def test_invariants_random(self, seed: int, size: int) -> None:
        """Exact offsets, no gaps, never above the max, on random code
        full of definitions, decorators and blank lines.
        """
        rng = random.Random(seed)
        pieces = ["x = 1", " ", "\n", "\n\n", "\ndef f():", "\nclass A:",
                  "\n@deco", "\nasync def g():", "\n    def m(self):",
                  "\n    @property", "\n    async def h(self):", "pass"]
        text = "".join(rng.choice(pieces) for _ in range(rng.randint(0, 120)))
        doc = make_doc(text, "a.py")
        Invariants.check(doc, PyChunker(size).chunk_document(doc), size)


@needs_corpus
class TestPyChunkerRealCorpus:
    """PyChunker on every Python file of the vLLM corpus."""

    def test_invariants(self, docs: list[Document]) -> None:
        """Every chunk satisfies the invariants, at 2000 and 500."""
        for size in (2000, 500):
            chunker = PyChunker(size)
            for doc in docs:
                if doc.file_path.endswith(".py"):
                    Invariants.check(doc, chunker.chunk_document(doc), size)

    def test_more_chunks_start_at_definition(self,
                                             docs: list[Document]) -> None:
        """More chunks start at a definition than with FixedSizeChunker
        (measured: ~59 % against ~33 % at 2000).
        """
        def share(chunker: FixedSizeChunker) -> float:
            """Fraction of the ``.py`` chunks that start at a definition."""
            starts = total = 0
            for doc in docs:
                if not doc.file_path.endswith(".py"):
                    continue
                for chunk in chunker.chunk_document(doc)[1:]:
                    total += 1
                    starts += bool(DEFINITION.match(chunk.text))
            return starts / total

        py, fixed = share(PyChunker(2000)), share(FixedSizeChunker(2000))
        assert py > 0.5
        assert py > fixed + 0.15
