"""The catalog of legal texts and the ``/convenios``, ``/citations`` API."""
from collections.abc import Iterator
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from src.api import RagApi
from src.base_retriever import Retriever
from src.models import Chunk, SearchHit
from src.norm_catalog import NormCatalog


def hit(name: str, title: str = "Artículo 1. Vacaciones.") -> SearchHit:
    """A hit of the file ``name``."""
    chunk = Chunk(file_path=f"data/labour/raw/{name}.txt",
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

    def test_consult(self) -> None:
        """The answer comes with its citations."""
        api = RagApi(Fake())
        calls: list[str] = []

        class Generator:
            def generate(self, query: str, chunks: list[Chunk]) -> str:
                calls.append(query)
                return "Quince días"

        api._generator = Generator()  # type: ignore[assignment]
        body = TestClient(api.app).get("/consult", params={
            "query": "q", "convenio": "tic"}).json()
        assert body["answer"] == "Quince días"
        assert len(body["citations"]) == 2
        assert calls == ["q"]

    def test_sources_mode_generates_nothing(self) -> None:
        """Only the sources: the model is never called."""
        api = RagApi(Fake())
        api._generator = None
        body = TestClient(api.app).get("/consult", params={
            "query": "q", "convenio": "tic", "mode": "sources"}).json()
        assert body["answer"] == ""
        assert len(body["citations"]) == 2

    def test_local_mode_can_be_switched_off(
            self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With the local model off, the server says so."""
        monkeypatch.setattr(RagApi, "LOCAL_READY", False)
        assert client().get("/consult", params={
            "query": "q", "convenio": "tic",
            "mode": "local"}).status_code == 501

    def test_local_mode_uses_the_local_model(self) -> None:
        """The local answer comes from the local model, not Gemini."""
        api = RagApi(Fake())
        calls: list[str] = []

        class Local:
            def iter_generate(self, query: str, chunks: list[Chunk],
                              *rest: str | None
                              ) -> Iterator[tuple[str, ...]]:
                calls.append(query)
                yield ("answer", "Local")

        api._local = Local()  # type: ignore[assignment]
        body = TestClient(api.app).get("/consult", params={
            "query": "q", "convenio": "tic", "mode": "local"}).json()
        assert (body["answer"], calls) == ("Local", ["q"])

    def test_key_status_and_entry(self,
                                  monkeypatch: pytest.MonkeyPatch) -> None:
        """The key typed in the page is used, and only kept in memory."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        api = client()
        assert api.get("/status").json()["gemini_key"] is False
        assert api.post("/key", json={"key": "corta"}).status_code == 422
        assert api.post("/key", json={"key": "AIza-una-clave-larga"}
                        ).json()["gemini_key"] is True

    def test_consult_needs_a_convenio(self) -> None:
        """Choosing a text is mandatory."""
        assert client().get("/consult",
                            params={"query": "q"}).status_code == 422

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
