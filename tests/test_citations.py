"""The catalog of legal texts and the ``/convenios``, ``/citations`` API."""
from pathlib import Path

from fastapi.testclient import TestClient

from src.api import RagApi
from src.base_retriever import Retriever
from src.models import Chunk, SearchHit
from src.norm_catalog import NormCatalog


def hit(name: str, title: str = "Artículo 1. Vacaciones.") -> SearchHit:
    """A hit of the file ``name``."""
    chunk = Chunk(file_path=f"data/laboral/raw/{name}.txt",
                  first_character_index=3, last_character_index=9,
                  text="texto", title=title)
    return SearchHit(chunk=chunk, score=0.5, chunk_id=0)


class Fake(Retriever):
    """One hit from each file searched."""

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return a hit for every source."""
        return [hit(name) for name in sorted(sources or [])][:k]


def client(index_dir: str | None = None) -> TestClient:
    """A client of an API over the fake retriever."""
    return TestClient(RagApi(Fake(), index_dir=index_dir).app)


class TestCatalog:
    """Each legal text can be checked by the reader."""

    def test_every_norm_is_linked(self) -> None:
        """Each text links to its own document in the BOE."""
        for norm in NormCatalog.NORMS:
            assert norm.url.startswith("https://www.boe.es/")
            assert norm.url.endswith(norm.boe_id)

    def test_citation(self) -> None:
        """A known file gets its name, article and link."""
        citation = NormCatalog.citation(hit("convenio_tic"))
        assert citation.kind == "convenio"
        assert citation.article == "Artículo 1. Vacaciones."
        assert citation.url.endswith("BOE-A-2025-7766")

    def test_statute_kind(self) -> None:
        """The Statute is told apart from the agreements."""
        citation = NormCatalog.citation(hit("estatuto_trabajadores"))
        assert citation.kind == "estatuto"
        assert citation.norm == "Estatuto de los Trabajadores"

    def test_unknown_file(self) -> None:
        """A file out of the catalog has no link."""
        citation = NormCatalog.citation(hit("other", title=""))
        assert (citation.norm, citation.url) == ("other", "")


class TestEndpoints:
    """The two endpoints of the web page."""

    def test_convenios_all(self) -> None:
        """Without an index, the whole catalog."""
        response = client().get("/convenios")
        assert response.status_code == 200
        assert len(response.json()) == len(NormCatalog.NORMS)

    def test_convenios_of_the_index(self, tmp_path: Path) -> None:
        """Only the texts of the index can be chosen."""
        (tmp_path / "manifest.json").write_text(
            '{"max_chunk_size": 2000, "raw_dir": "r", "files": '
            '{"r/convenio_tic.txt": {"mtime_ns": 1, "size": 1}}}')
        ids = [n["id"] for n in client(str(tmp_path)).get(
            "/convenios").json()]
        assert ids == ["tic"]

    def test_citations_agreement_first(self) -> None:
        """The agreement's citations come before the Statute's."""
        body = client().get("/citations", params={
            "query": "q", "convenio": "tic", "k": 10}).json()
        assert body["convenio"] == "tic"
        assert [c["kind"] for c in body["citations"]] == ["convenio",
                                                          "estatuto"]

    def test_citations_need_a_convenio(self) -> None:
        """Choosing a text is mandatory."""
        assert client().get("/citations",
                            params={"query": "q"}).status_code == 422

    def test_unknown_convenio(self, tmp_path: Path) -> None:
        """A text that is not in the index is a 400."""
        (tmp_path / "manifest.json").write_text(
            '{"max_chunk_size": 2000, "raw_dir": "r", "files": '
            '{"r/convenio_tic.txt": {"mtime_ns": 1, "size": 1}}}')
        response = client(str(tmp_path)).get("/citations", params={
            "query": "q", "convenio": "patata"})
        assert response.status_code == 400
