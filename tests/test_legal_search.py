"""Article chunks, query synonyms and the search by agreement."""
from src.base_retriever import Retriever
from src.chunker import ArticleChunker, MdChunker
from src.convenio_search import ConvenioSearch
from src.models import Chunk, Document, SearchHit
from src.tokenizer_bm import Tokenizer


def doc(text: str) -> Document:
    """A document with a fixed path."""
    return Document(file_path="data/raw/convenio_x.txt", text=text)


class TestArticleChunker:
    """Every article starts a chunk and gives it its title."""

    TEXT = ("Preámbulo.\n\nArtículo 1. Vacaciones.\n\nTreinta días.\n\n"
            "Artículo 2.\n\nPermisos\n\nVer el Artículo 37.3 del texto.\n")

    def test_ranges_point_to_the_text(self) -> None:
        """The ranges are those of the whole document."""
        for chunk in ArticleChunker(2000).chunk_document(doc(self.TEXT)):
            assert (self.TEXT[chunk.first_character_index:
                              chunk.last_character_index] == chunk.text)

    def test_titles(self) -> None:
        """A heading on two lines is joined; a preamble has no title."""
        chunks = ArticleChunker(2000).chunk_document(doc(self.TEXT))
        assert [c.title for c in chunks] == [
            "", "Artículo 1. Vacaciones.", "Artículo 2. Permisos"]

    def test_long_article_keeps_title(self) -> None:
        """Every piece of a long article has its title."""
        text = "Artículo 9. Largo.\n\n" + "palabra " * 300
        chunks = ArticleChunker(200).chunk_document(doc(text))
        assert len(chunks) > 1
        assert {c.title for c in chunks} == {"Artículo 9. Largo."}

    def test_without_articles_as_md(self) -> None:
        """Any other text gets the chunks of MdChunker."""
        text = "# Title\n\nSome text.\n\n" * 200
        assert (ArticleChunker(300).chunk_document(doc(text))
                == MdChunker(300).chunk_document(doc(text)))


class TestSynonyms:
    """The query also searches how the legal texts say it."""

    def test_word(self) -> None:
        """Accents do not matter."""
        assert "nocturnidad" in Tokenizer().tokenize_query("de NOCHE")

    def test_phrase_needs_its_order(self) -> None:
        """A key of several words is matched as a phrase."""
        tokenizer = Tokenizer()
        assert "matrimonio" in tokenizer.tokenize_query("si me caso")
        assert "matrimonio" not in tokenizer.tokenize_query("en caso de me")

    def test_index_terms_unchanged(self) -> None:
        """``tokenize`` never adds synonyms."""
        assert Tokenizer().tokenize("noche") == ["noche"]


class FakeRetriever(Retriever):
    """Returns ``counts[source]`` hits from each searched source."""

    def __init__(self, counts: dict[str, int]) -> None:
        """Store how many hits each source has."""
        self.counts = counts

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Fake hits of the first source."""
        name = sorted(sources or {"all"})[0]
        chunk = Chunk(file_path=f"{name}.txt", first_character_index=0,
                      last_character_index=1, text="x")
        return [SearchHit(chunk=chunk, score=1.0, chunk_id=0)
                for _ in range(min(k, self.counts.get(name, 0)))]


def names(retriever: Retriever, k: int, convenio: str | None) -> list[str]:
    """The file of each hit."""
    return [hit.chunk.file_path.removesuffix(".txt") for hit in
            ConvenioSearch.search(retriever, "q", k, convenio)]


class TestConvenioSearch:
    """The agreement and the Statute share the results."""

    def test_shares(self) -> None:
        """7 places for the agreement and 3 for the Statute."""
        fake = FakeRetriever({"convenio_tic": 20,
                              "estatuto_trabajadores": 20})
        assert names(fake, 10, "tic") == (["convenio_tic"] * 7
                                          + ["estatuto_trabajadores"] * 3)

    def test_other_side_fills(self) -> None:
        """Free places go to the other text."""
        fake = FakeRetriever({"convenio_tic": 2,
                              "estatuto_trabajadores": 20})
        assert names(fake, 10, "tic") == (["convenio_tic"] * 2
                                          + ["estatuto_trabajadores"] * 8)

    def test_statute_only(self) -> None:
        """``estatuto`` searches only the Statute."""
        fake = FakeRetriever({"convenio_tic": 20,
                              "estatuto_trabajadores": 20})
        assert names(fake, 4, "estatuto") == ["estatuto_trabajadores"] * 4

    def test_no_filter(self) -> None:
        """Without an agreement, the whole corpus."""
        assert names(FakeRetriever({"all": 3}), 3, None) == ["all"] * 3
