"""Tests for ``src.tokenizer_bm.Tokenizer`` and ``src.bm25_index.BM25Index``.

Scores are checked against ``Reference``, an independent, deliberately
naive BM25 written straight from the formula, so a mistake in the index
cannot hide behind the same mistake in the test.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter
from math import log
from pathlib import Path

import pytest

from src.bm25_index import BM25Index
from src.chunker import FixedSizeChunker
from src.exceptions import (DatasetError, IndexBuildError,
                            IndexNotFoundError, PathError, RagError)
from src.incremental_indexer import IncrementalIndexer
from src.models import Chunk, SearchHit
from src.reader import CorpusReader
from src.tokenizer_bm import Tokenizer
from conftest import RealData, needs_corpus, needs_unanswered

REAL_RAW = CorpusReader.PROJECT_ROOT / "data" / "raw"


def make_chunks(*texts: str) -> list[Chunk]:
    """One chunk per text, each in its own fake file."""
    return [Chunk(file_path=f"data/raw/{i}.md", first_character_index=0,
                  last_character_index=len(text), text=text)
            for i, text in enumerate(texts)]


class Reference:
    """Naive BM25 straight from the formula, for comparison.

    Like ``build``, a chunk's terms are those of its text plus those of
    its file path.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        """Tokenize every chunk (text and path) once."""
        tokenizer = Tokenizer()
        self.docs = [tokenizer.tokenize(c.text)
                     + tokenizer.tokenize(c.file_path) for c in chunks]
        self.n = len(self.docs)
        self.avg = sum(len(d) for d in self.docs) / self.n

    def idf(self, term: str) -> float:
        """``ln((N - df + 0.5) / (df + 0.5) + 1)``."""
        df = sum(1 for d in self.docs if term in d)
        return log((self.n - df + 0.5) / (df + 0.5) + 1)

    def score(self, query: str, doc_id: int) -> float:
        """Sum over the distinct query terms (stopwords dropped, like
        ``search``) present in the document.
        """
        doc = self.docs[doc_id]
        counts = Counter(doc)
        total = 0.0
        for term in set(Tokenizer().tokenize_query(query)):
            tf = counts[term]
            if tf == 0:
                continue
            norm = BM25Index.K1 * (1 - BM25Index.B
                                   + BM25Index.B * len(doc) / self.avg)
            total += self.idf(term) * tf * (BM25Index.K1 + 1) / (tf + norm)
        return total


class TestTokenizer:
    """Text → list of lowercase words."""

    @pytest.mark.parametrize("text, tokens", [
        ("How do I load a LoRA adapter?",
         ["how", "do", "i", "load", "a", "lora", "adapter"]),
        ("max_model_len", ["max", "model", "len"]),
        ("vllm.entrypoints.openai", ["vllm", "entrypoints", "openai"]),
        ("see lora.md", ["see", "lora", "md"]),
        ("--max-num-seqs=256", ["max", "num", "seqs", "256"]),
        ("fp8 and int4", ["fp8", "and", "int4"]),
        ("lora LoRA LORA", ["lora", "lora", "lora"]),
        ("", []),
        ("¿? --- ``` !!", []),
        ("   \n\t ", []),
    ])
    def test_tokenize(self, text: str, tokens: list[str]) -> None:
        """Lowercase, split on anything that is not a letter or digit."""
        assert Tokenizer().tokenize(text) == tokens

    def test_keeps_order_and_repeats(self) -> None:
        """Order and repetitions are kept (counts matter for BM25)."""
        assert Tokenizer().tokenize("b a b") == ["b", "a", "b"]

    @pytest.mark.parametrize("text, tokens", [
        ("café", ["cafe"]),
        ("Días de vacación", ["dias", "de", "vacacion"]),
        ("año", ["ano"]),
        ("cigüeña", ["ciguena"]),
    ])
    def test_accents_removed(self, text: str, tokens: list[str]) -> None:
        """Accents are removed, so ``días`` and ``dias`` are one term."""
        assert Tokenizer().tokenize(text) == tokens

    def test_splits_names(self) -> None:
        """A dotted snake_case name gives only its parts, in order."""
        assert Tokenizer().tokenize("self.max_len") == ["self", "max", "len"]

    def test_dots_split(self) -> None:
        """Dotted names are only split: no whole ``a.b`` term."""
        tokens = Tokenizer().tokenize("llm.generate 0.10.1 lora.md")
        assert all("." not in t for t in tokens)

    def test_pattern_compiled_once(self) -> None:
        """The pattern is a shared, precompiled class attribute."""
        assert Tokenizer.PATTERN is Tokenizer().PATTERN


class TestQueryStopwords:
    """``tokenize_query``: the query terms without the stopwords."""

    def test_drops_stopwords(self) -> None:
        """Question words and common words are removed."""
        assert Tokenizer().tokenize_query(
            "How do I load a LoRA adapter?") == ["load", "lora", "adapter"]

    def test_keeps_order_and_repeats(self) -> None:
        """The rest stays as ``tokenize`` gives it."""
        assert Tokenizer().tokenize_query("the lora and the lora") == [
            "lora", "lora"]

    def test_snake_names_split(self) -> None:
        """A snake_case name is filtered part by part."""
        tokens = Tokenizer().tokenize_query("What is is_driver_worker?")
        assert tokens == ["driver", "worker"]

    def test_spanish_stopwords(self) -> None:
        """Spanish question words go, with or without accents."""
        assert Tokenizer().tokenize_query("¿Cuántos días de permiso?") == [
            "dias", "permiso", "permisos", "licencias"]

    def test_negation_kept(self) -> None:
        """``no`` tells apart paid and unpaid leave."""
        assert "no" in Tokenizer().tokenize_query("permisos no retribuidos")

    @pytest.mark.parametrize("text", ["", "What is it?", "how does the"])
    def test_only_stopwords(self, text: str) -> None:
        """Nothing left when every word is a stopword."""
        assert Tokenizer().tokenize_query(text) == []

    def test_stopwords_are_lowercase_terms(self) -> None:
        """Every stopword is a term ``tokenize`` can produce, or it would
        never match.
        """
        for word in Tokenizer.STOPWORDS:
            assert Tokenizer().tokenize(word) == [word]

    def test_index_not_filtered(self) -> None:
        """Only the query is filtered: the chunks keep their stopwords."""
        index = BM25Index.build(make_chunks("how the server works"))
        assert "how" in index._postings
        assert "the" in index._postings

    def test_stopwords_only_query_finds_nothing(self) -> None:
        """A query made only of stopwords scores no chunk."""
        index = BM25Index.build(make_chunks("how the server works"))
        assert index.search("How is the?", 5) == []

    def test_question_words_do_not_decide(self) -> None:
        """A chunk that only shares the question wording does not beat
        the one about the subject.
        """
        index = BM25Index.build(make_chunks(
            "How and why: what is where, how and why.",
            "Load a LoRA adapter with the server."))
        hits = index.search("How do I load a LoRA adapter?", 2)
        assert hits[0].chunk.text == "Load a LoRA adapter with the server."
        assert len(hits) == 1


class TestSnakeTerms:
    """snake_case names are only split: no whole term is kept."""

    @pytest.mark.parametrize("text, parts", [
        ("max_model_len", ["max", "model", "len"]),
        ("MAX_MODEL_LEN", ["max", "model", "len"]),
        ("x_1 y_2_z", ["x", "1", "y", "2", "z"]),
        ("self.max_model_len", ["self", "max", "model", "len"]),
        ("api_server.py", ["api", "server", "py"]),
    ])
    def test_split_at_underscores(self, text: str, parts: list[str]) -> None:
        """Underscores separate terms, like any other symbol."""
        assert Tokenizer().tokenize(text) == parts

    @pytest.mark.parametrize("text", [
        "__init__", "_private", "foo_", "a__b", "no underscores", "",
    ])
    def test_no_snake_term(self, text: str) -> None:
        """No term ever contains an underscore."""
        assert all("_" not in t for t in Tokenizer().tokenize(text))

    def test_repetitions_kept(self) -> None:
        """A name written twice counts twice (BM25 tf)."""
        tokens = Tokenizer().tokenize("max_len and max_len")
        assert tokens.count("max") == 2

    def test_exact_name_ranks_first(self) -> None:
        """A question quoting ``max_model_len`` prefers the chunk that
        writes it, over a chunk with the words apart.
        """
        index = BM25Index.build(make_chunks(
            "The max length of the model and its len.",
            "Set max_model_len to limit the context."))
        hits = index.search("What is max_model_len?", 2)
        assert hits[0].chunk.text == "Set max_model_len to limit the context."


class TestBuild:
    """What ``build`` stores.

    Each chunk is indexed with the terms of its text plus those of its
    file path. ``make_chunks`` paths (``data/raw/0.md``) add 4 terms:
    ``data``, ``raw``, ``0`` and ``md``.
    """

    PATH_TERMS = 4

    def test_postings(self) -> None:
        """term → {chunk position: occurrences}."""
        index = BM25Index.build(make_chunks(
            "Load a LoRA adapter. LoRA is fast.", "Load the model."))
        assert index._postings["load"] == {0: 1, 1: 1}
        assert index._postings["lora"] == {0: 2}
        assert index._postings["model"] == {1: 1}
        assert "missing" not in index._postings

    def test_lengths_and_average(self) -> None:
        """Token counts per chunk (text + path) and their mean."""
        index = BM25Index.build(make_chunks("a b c d", "e f"))
        assert index._lengths == [4 + self.PATH_TERMS, 2 + self.PATH_TERMS]
        assert index._avg_length == 3.0 + self.PATH_TERMS
        assert index._n_chunks == 2

    def test_path_terms_indexed(self) -> None:
        """The file path is searchable even when the text never says it."""
        chunk = Chunk(file_path="data/raw/vllm/models/keye.py",
                      first_character_index=0, last_character_index=9,
                      text="class Foo")
        index = BM25Index.build([chunk])
        assert index._postings["keye"] == {0: 1}
        assert index._postings["py"] == {0: 1}
        assert index.search("keye", 1)[0].chunk == chunk

    def test_path_terms_not_in_text(self) -> None:
        """Only the indexed terms change: text and offsets stay."""
        chunks = make_chunks("a b")
        index = BM25Index.build(chunks)
        assert index._chunks[0].text == "a b"
        assert index._chunks == chunks

    def test_chunks_kept_in_order(self) -> None:
        """Chunk ids are positions in the given list."""
        chunks = make_chunks("x", "y", "z")
        index = BM25Index.build(chunks)
        assert index._chunks == chunks

    def test_chunk_without_tokens_allowed(self) -> None:
        """A symbols-only text is fine: its path still gives terms."""
        index = BM25Index.build(make_chunks("---", "real text"))
        assert index._lengths == [self.PATH_TERMS, 2 + self.PATH_TERMS]

    def test_no_chunks(self) -> None:
        """Nothing to index: clear error instead of dividing by zero."""
        with pytest.raises(IndexBuildError, match="Nothing to index"):
            BM25Index.build([])

    def test_only_symbols(self) -> None:
        """Chunks exist but neither text nor path has a term: same
        error.
        """
        chunks = [Chunk(file_path="/-/", first_character_index=0,
                        last_character_index=len(text), text=text)
                  for text in ("---", "``` ***")]
        with pytest.raises(IndexBuildError, match="2 chunks, 0 tokens"):
            BM25Index.build(chunks)

    def test_error_is_rag_error(self) -> None:
        """``main`` handles it as a project error."""
        with pytest.raises(RagError):
            BM25Index.build([])


def chunk_at(path: str, start: int, text: str) -> Chunk:
    """A chunk of ``path`` starting at ``start``."""
    return Chunk(file_path=path, first_character_index=start,
                 last_character_index=start + len(text), text=text)


class TestUpdate:
    """``update`` (bonus 3): the same index as ``build`` over the plan's
    chunks, tokenizing only the fresh ones."""

    OLD = [chunk_at("a.md", 0, "load the index"),
           chunk_at("a.md", 50, "search the index"),
           chunk_at("b.md", 0, "old words here"),
           chunk_at("c.py", 0, "def load(): pass")]

    @staticmethod
    def assert_same(updated: BM25Index, full: BM25Index) -> None:
        """Same stored data, so the same scores for any query."""
        assert updated._chunks == full._chunks
        assert updated._lengths == full._lengths
        assert updated._postings == full._postings
        assert updated._avg_length == full._avg_length
        assert updated.search("load index", 10) == full.search(
            "load index", 10)

    def check(self, to_remove: list[str], new: list[Chunk]) -> BM25Index:
        """Update the index of ``OLD`` and compare with a full build."""
        plan = IncrementalIndexer._plan(self.OLD, to_remove, new)
        updated = BM25Index.build(self.OLD).update(plan)
        self.assert_same(updated, BM25Index.build(plan.chunks))
        return updated

    def test_nothing_changed(self) -> None:
        """An empty plan gives the same index."""
        self.check([], [])

    def test_new_file_shifts_ids(self) -> None:
        """A new file in the middle: later chunk ids move up in the
        postings and lengths."""
        self.check([], [chunk_at("aa.md", 0, "build index fast")])

    def test_modified_file(self) -> None:
        """A modified file: its old terms leave, its new ones enter."""
        updated = self.check(["b.md"],
                             [chunk_at("b.md", 0, "new search text")])
        assert "old" not in updated._postings
        assert "new" in updated._postings

    def test_deleted_file(self) -> None:
        """A deleted file: a term only it had leaves the postings (its
        IDF would count a chunk that is gone)."""
        updated = self.check(["b.md"], [])
        assert "words" not in updated._postings

    def test_all_at_once(self) -> None:
        """New, modified and deleted together."""
        self.check(["a.md", "c.py"],
                   [chunk_at("a.md", 0, "load a model"),
                    chunk_at("a.md", 9, "and more"),
                    chunk_at("aa.md", 0, "build index fast")])

    def test_only_fresh_tokenized(self, monkeypatch: pytest.MonkeyPatch
                                  ) -> None:
        """Only the fresh chunks are tokenized: that is the speed-up."""
        index = BM25Index.build(self.OLD)
        plan = IncrementalIndexer._plan(
            self.OLD, ["b.md"], [chunk_at("aa.md", 0, "build index"),
                                 chunk_at("b.md", 0, "new text")])
        seen: list[str] = []
        original = BM25Index._chunk_tokens

        def spy(tokenizer: Tokenizer, chunk: Chunk) -> list[str]:
            seen.append(chunk.text)
            return original(tokenizer, chunk)

        monkeypatch.setattr(BM25Index, "_chunk_tokens", staticmethod(spy))
        index.update(plan)
        assert sorted(seen) == ["build index", "new text"]

    def test_self_unchanged(self) -> None:
        """The old index is not changed: ``update`` returns a new one."""
        index = BM25Index.build(self.OLD)
        postings = {term: dict(ids) for term, ids in index._postings.items()}
        lengths = list(index._lengths)
        plan = IncrementalIndexer._plan(self.OLD, ["b.md"], [])
        assert index.update(plan) is not index
        assert index._postings == postings
        assert index._lengths == lengths
        assert index._chunks == self.OLD

    def test_everything_removed(self) -> None:
        """No chunk left: the same error as ``build`` with nothing."""
        plan = IncrementalIndexer._plan(self.OLD, ["a.md", "b.md", "c.py"],
                                        [])
        with pytest.raises(IndexBuildError):
            BM25Index.build(self.OLD).update(plan)


class TestIdf:
    """Rarity of a term."""

    TEXTS = ["common rare", "common", "common", "common"]

    def test_rare_beats_common(self) -> None:
        """A term in fewer chunks has a higher idf."""
        index = BM25Index.build(make_chunks(*self.TEXTS))
        assert index._idf("rare") > index._idf("common")

    def test_matches_formula(self) -> None:
        """``ln((N - df + 0.5) / (df + 0.5) + 1)``."""
        index = BM25Index.build(make_chunks(*self.TEXTS))
        assert index._idf("rare") == pytest.approx(log(3.5 / 1.5 + 1))
        assert index._idf("common") == pytest.approx(log(0.5 / 4.5 + 1))

    def test_never_negative(self) -> None:
        """A term in every chunk still has idf ≥ 0."""
        index = BM25Index.build(make_chunks(*self.TEXTS))
        assert index._idf("common") >= 0


class TestTermScore:
    """Contribution of one term to one chunk."""

    @pytest.fixture
    def index(self) -> BM25Index:
        """Chunks of lengths 2, 4 and 12, plus 4 path terms each: 6, 8
        and 16 (average 10).
        """
        return BM25Index.build(make_chunks(
            "a b", "a b c d", " ".join(["w"] * 12)))

    def test_matches_formula(self, index: BM25Index) -> None:
        """Same value as the written formula."""
        k1, b = BM25Index.K1, BM25Index.B
        expected = 2.0 * 3 * (k1 + 1) / (3 + k1 * (1 - b + b * 8 / 10))
        assert index._term_score(2.0, 3, 1) == pytest.approx(expected)

    def test_grows_with_tf(self, index: BM25Index) -> None:
        """More occurrences, higher score."""
        scores = [index._term_score(1.0, tf, 1) for tf in (1, 2, 5, 20)]
        assert scores == sorted(scores)

    def test_saturates(self, index: BM25Index) -> None:
        """Never above ``idf × (K1 + 1)`` however many repetitions."""
        assert index._term_score(1.0, 10**6, 1) < BM25Index.K1 + 1

    def test_longer_chunk_scores_less(self, index: BM25Index) -> None:
        """Same tf in a longer chunk counts less."""
        short = index._term_score(1.0, 2, 0)
        long = index._term_score(1.0, 2, 2)
        assert short > long

    def test_proportional_to_idf(self, index: BM25Index) -> None:
        """Doubling idf doubles the contribution."""
        assert index._term_score(2.0, 3, 1) == pytest.approx(
            2 * index._term_score(1.0, 3, 1))


class TestSearch:
    """Ranking of chunks for a query."""

    TEXTS = [
        "Serve the model with the OpenAI compatible server.",
        "Configure the server port and host for the server.",
        "Load a LoRA adapter at runtime.",
        "Unrelated text about tokenizers and chat templates.",
        "The server, the server, the server, the server.",
    ]

    @pytest.fixture
    def index(self) -> BM25Index:
        """Index over ``TEXTS``."""
        return BM25Index.build(make_chunks(*self.TEXTS))

    def test_returns_search_hits(self, index: BM25Index) -> None:
        """Each result is a ``SearchHit`` holding the real chunk."""
        hits = index.search("lora adapter", 3)
        assert all(isinstance(h, SearchHit) for h in hits)
        assert hits[0].chunk.text == self.TEXTS[2]

    def test_scores_match_reference(self, index: BM25Index) -> None:
        """Every score equals the naive reference implementation."""
        reference = Reference(make_chunks(*self.TEXTS))
        query = "configure the openai server"
        for hit in index.search(query, 5):
            doc_id = self.TEXTS.index(hit.chunk.text)
            assert hit.score == pytest.approx(reference.score(query, doc_id))

    def test_sorted_by_score(self, index: BM25Index) -> None:
        """Best first."""
        scores = [h.score for h in index.search("the server", 5)]
        assert scores == sorted(scores, reverse=True)

    def test_rare_term_wins(self, index: BM25Index) -> None:
        """"openai" (1 chunk) outweighs repeated "server" (3 chunks)."""
        assert index.search("openai server", 1)[0].chunk.text == \
            self.TEXTS[0]

    def test_saturation_limits_repetition(self, index: BM25Index) -> None:
        """Repeating "server" 4 times does not beat a chunk that also
        matches "configure".
        """
        top = index.search("configure server", 1)[0]
        assert top.chunk.text == self.TEXTS[1]

    def test_only_matching_chunks(self, index: BM25Index) -> None:
        """Chunks sharing no term are not returned, even with a big k."""
        hits = index.search("lora", 10)
        assert [h.chunk.text for h in hits] == [self.TEXTS[2]]

    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_k_limits_results(self, index: BM25Index, k: int) -> None:
        """At most k results."""
        assert len(index.search("the server model", k)) == k

    def test_k_zero(self, index: BM25Index) -> None:
        """k = 0 gives nothing."""
        assert index.search("server", 0) == []

    @pytest.mark.parametrize("query", ["", "   ", "¿?", "---",
                                       "zzzqqq xyzzy"])
    def test_no_results(self, index: BM25Index, query: str) -> None:
        """No tokens or no known term: empty list, no error."""
        assert index.search(query, 5) == []

    def test_case_insensitive(self, index: BM25Index) -> None:
        """The query goes through the same tokenizer."""
        assert index.search("LoRA ADAPTER", 3) == index.search(
            "lora adapter", 3)

    def test_repeated_query_terms_count_once(self, index: BM25Index) -> None:
        """"lora lora lora" scores like "lora"."""
        assert index.search("lora lora lora", 3) == index.search("lora", 3)

    def test_ties_broken_by_position(self) -> None:
        """Identical chunks: the earlier one first, every time."""
        index = BM25Index.build(make_chunks("alpha beta", "gamma",
                                            "alpha beta", "alpha beta"))
        for _ in range(5):
            hits = index.search("beta alpha", 3)
            assert [h.chunk.file_path for h in hits] == [
                "data/raw/0.md", "data/raw/2.md", "data/raw/3.md"]

    def test_ties_stable_across_runs(self) -> None:
        """Tied chunks come in the same order in every Python process.

        The query terms go through a ``set``, whose order changes between
        runs (hash randomization, ``PYTHONHASHSEED``). Chunk 0 only has
        "beta" and chunk 1 only "alpha", with identical scores; without
        the tie-break by position, some runs would return ``[1, 0]``.
        """
        script = (
            "from src.bm25_index import BM25Index\n"
            "from src.models import Chunk\n"
            "chunks = [Chunk(file_path=f'{i}', first_character_index=0,"
            " last_character_index=1, text=t)"
            " for i, t in enumerate(['beta x', 'alpha x'])]\n"
            "index = BM25Index.build(chunks)\n"
            "print([h.chunk.file_path for h in index.search('alpha beta', 2)])"
        )
        orders = set()
        for seed in range(12):
            env = {**os.environ, "PYTHONHASHSEED": str(seed)}
            result = subprocess.run(
                [sys.executable, "-c", script], env=env, check=True,
                capture_output=True, text=True,
                cwd=CorpusReader.PROJECT_ROOT)
            orders.add(result.stdout.strip())
        assert orders == {"['0', '1']"}

    def test_deterministic(self, index: BM25Index) -> None:
        """Same query, same answer."""
        first = index.search("the server model openai", 5)
        assert all(index.search("the server model openai", 5) == first
                   for _ in range(5))


class TestSaveLoad:
    """Persisting the index between ``index`` and the search commands."""

    TEXTS = ["Load a LoRA adapter at runtime.",
             "Configure the OpenAI compatible server.",
             "The server, the server, the server.",
             "---"]

    @pytest.fixture
    def index(self) -> BM25Index:
        """A small index to save."""
        return BM25Index.build(make_chunks(*self.TEXTS))

    def test_file_created(self, index: BM25Index, tmp_path: Path) -> None:
        """``save`` writes ``FILE_NAME`` inside the folder."""
        index.save(tmp_path)
        assert (tmp_path / BM25Index.FILE_NAME).is_file()

    def test_creates_missing_folders(self, index: BM25Index,
                                     tmp_path: Path) -> None:
        """Nested folders that do not exist yet are created."""
        target = tmp_path / "a" / "b" / "c"
        index.save(target)
        assert (target / BM25Index.FILE_NAME).is_file()

    def test_valid_json(self, index: BM25Index, tmp_path: Path) -> None:
        """The file is JSON with the three ``IndexData`` fields."""
        index.save(tmp_path)
        data = json.loads((tmp_path / BM25Index.FILE_NAME).read_text(
            encoding="utf-8"))
        assert set(data) == {"chunks", "postings", "lengths"}

    def test_round_trip_data(self, index: BM25Index,
                             tmp_path: Path) -> None:
        """Loading gives back exactly what was saved (JSON turns the chunk
        ids into text; loading must turn them back into ``int``).
        """
        index.save(tmp_path)
        loaded = BM25Index.load(tmp_path)
        assert loaded._chunks == index._chunks
        assert loaded._postings == index._postings
        assert loaded._lengths == index._lengths
        assert loaded._avg_length == index._avg_length
        assert all(isinstance(chunk_id, int)
                   for ids in loaded._postings.values() for chunk_id in ids)

    @pytest.mark.parametrize("query", ["lora adapter", "the server",
                                       "configure openai server", "nothing"])
    def test_round_trip_search(self, index: BM25Index, tmp_path: Path,
                               query: str) -> None:
        """A loaded index answers exactly like the original."""
        index.save(tmp_path)
        assert BM25Index.load(tmp_path).search(query, 4) == \
            index.search(query, 4)

    def test_overwrite(self, tmp_path: Path) -> None:
        """Saving again replaces the previous index."""
        BM25Index.build(make_chunks("old text")).save(tmp_path)
        BM25Index.build(make_chunks("new words")).save(tmp_path)
        loaded = BM25Index.load(tmp_path)
        assert loaded.search("new", 1)[0].chunk.text == "new words"
        assert loaded.search("old", 1) == []

    def test_save_into_a_file(self, index: BM25Index,
                              tmp_path: Path) -> None:
        """A file where the folder should be: ``PathError``."""
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("x", encoding="utf-8")
        with pytest.raises(PathError, match="Cannot use") as info:
            index.save(not_a_dir)
        assert isinstance(info.value.__cause__, OSError)

    @pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0,
                        reason="root ignores permissions")
    def test_save_without_permission(self, index: BM25Index,
                                     tmp_path: Path) -> None:
        """A read-only folder: ``PathError`` saying why."""
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o500)
        try:
            with pytest.raises(PathError, match="Permission denied"):
                index.save(locked / "index")
        finally:
            locked.chmod(0o700)

    def test_load_missing_folder(self, tmp_path: Path) -> None:
        """No index yet: tell the user to run ``index`` first."""
        with pytest.raises(IndexNotFoundError) as info:
            BM25Index.load(tmp_path / "nothing_here")
        assert str(info.value) == (
            f"No index found in '{tmp_path / 'nothing_here'}'. "
            "Run 'index' first.")
        assert isinstance(info.value.__cause__, FileNotFoundError)

    def test_load_folder_without_index(self, tmp_path: Path) -> None:
        """The folder exists but has no index file: same message."""
        with pytest.raises(IndexNotFoundError, match="Run 'index' first"):
            BM25Index.load(tmp_path)

    def test_load_from_a_file(self, tmp_path: Path) -> None:
        """``index_dir`` is a file: ``PathError`` (not "not found")."""
        not_a_dir = tmp_path / "file.txt"
        not_a_dir.write_text("x", encoding="utf-8")
        with pytest.raises(PathError, match="Cannot use"):
            BM25Index.load(not_a_dir)

    @pytest.mark.parametrize("content", [
        "",
        "not json at all",
        '{"chunks": [,]}',
        "[1, 2, 3]",
        '{"chunks": [], "postings": {}}',
        '{"chunks": [], "postings": {"a": {"x": 1}}, "lengths": []}',
    ])
    def test_load_corrupt(self, tmp_path: Path, content: str) -> None:
        """Broken or foreign content: ``DatasetError`` naming the file."""
        (tmp_path / BM25Index.FILE_NAME).write_text(content,
                                                    encoding="utf-8")
        with pytest.raises(DatasetError) as info:
            BM25Index.load(tmp_path)
        assert str(tmp_path / BM25Index.FILE_NAME) in str(info.value)

    def test_errors_are_rag_errors(self, tmp_path: Path) -> None:
        """``main`` shows all of them without a traceback."""
        for error in (IndexNotFoundError, PathError, DatasetError):
            assert issubclass(error, RagError)


@pytest.fixture(scope="module")
def real_index() -> BM25Index:
    """Index over the whole vLLM corpus, built once."""
    docs = CorpusReader(REAL_RAW).read().documents
    return BM25Index.build(FixedSizeChunker(2000).chunk_documents(docs))


@needs_corpus
class TestRealCorpus:
    """Search over the real corpus."""

    def test_known_question(self, real_index: BM25Index) -> None:
        """A dataset question: the top hit overlaps its ground truth
        (``lora.md`` 4695-6098) with IoU ≥ 0.05, as the moulinette asks.
        """
        hit = real_index.search("What HTTP endpoint is used to dynamically "
                                "load a LoRA adapter in vLLM?", 1)[0]
        assert hit.chunk.file_path.endswith("docs/features/lora.md")
        start = hit.chunk.first_character_index
        end = hit.chunk.last_character_index
        overlap = max(0, min(end, 6098) - max(start, 4695))
        union = max(end, 6098) - min(start, 4695)
        assert overlap / union >= 0.05

    def test_update_equals_build(self, real_index: BM25Index) -> None:
        """Removing some files and moving every other chunk id gives the
        same index as building it again (bonus 3)."""
        chunks = real_index._chunks
        removed = sorted({chunk.file_path for chunk in chunks})[::50]
        plan = IncrementalIndexer._plan(chunks, removed, [])
        updated = real_index.update(plan)
        full = BM25Index.build(plan.chunks)
        assert updated._lengths == full._lengths
        assert updated._postings == full._postings
        query = "How do I load a LoRA adapter?"
        assert updated.search(query, 10) == full.search(query, 10)

    @needs_unanswered
    def test_speed_all_public_questions(self,
                                        real_index: BM25Index) -> None:
        """Every dataset question searched well under the subject's limit
        (90 s for 200 questions).
        """
        questions = []
        for path in RealData.unanswered_files():
            data = json.loads(path.read_text(encoding="utf-8"))
            questions += [q["question"] for q in data["rag_questions"]]
        start = time.perf_counter()
        for question in questions:
            assert len(real_index.search(question, 10)) == 10
        assert time.perf_counter() - start < 30
