"""Later BOE publications: read, matched to articles, shown and sent."""
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from src.amendments import AmendmentBook
from src.api import RagApi
from src.base_retriever import Retriever
from src.gemini_generator import GeminiGenerator
from src.models import Chunk, SearchHit

REAL = Path("data/labour/amendments.json")


def book(tmp_path: Path) -> AmendmentBook:
    """A small list of publications."""
    path = tmp_path / "m.json"
    path.write_text(
        '{"reviewed": "2026-11-10", "agreements": {"contact_center": ['
        '{"boe_id": "B-1", "date": "2024-02-22", "kind": "Sentencia",'
        ' "summary": "Anula", "articles": [30]},'
        '{"boe_id": "B-2", "date": "2026-03-06", "kind": "Tablas",'
        ' "summary": "Tablas", "others": ["ANEXO II"]}]}}', encoding="utf-8")
    return AmendmentBook(path)


class TestBook:
    """Which texts each publication affects."""

    def test_article(self, tmp_path: Path) -> None:
        """An article number matches only that article."""
        found = book(tmp_path).for_title("contact_center",
                                         "Artículo 30. Permisos.")
        assert [a.boe_id for a in found] == ["B-1"]
        assert book(tmp_path).for_title("contact_center",
                                        "Artículo 3. Otro.") == []

    def test_annex(self, tmp_path: Path) -> None:
        """An annex matches by its name, not by a prefix of it."""
        notes = book(tmp_path)
        assert notes.for_title("contact_center", "ANEXO II Tablas")
        assert not notes.for_title("contact_center", "ANEXO III Otras")

    def test_newest_first_with_link(self, tmp_path: Path) -> None:
        """The list of an agreement links to the BOE."""
        items = book(tmp_path).for_norm("contact_center")
        assert [a.boe_id for a in items] == ["B-2", "B-1"]
        assert items[0].url.endswith("B-2")

    def test_missing_file_means_no_warnings(self, tmp_path: Path) -> None:
        """Nothing is invented without the file."""
        assert AmendmentBook(tmp_path / "no.json").for_norm("tic") == []

    def test_real_file_reads(self) -> None:
        """The shipped list loads and covers the ruling on art. 30."""
        notes = AmendmentBook(REAL)
        assert notes.reviewed
        assert len(notes.for_title("contact_center",
                                   "Artículo 30. Permisos retribuidos.")) == 2


class Fake(Retriever):
    """Always the art. 30 of contact center."""

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """One hit."""
        chunk = Chunk(file_path="data/labour/raw/convenio_contact_center.txt",
                      first_character_index=0, last_character_index=1,
                      text="x", title="Artículo 30. Permisos retribuidos.")
        return [SearchHit(chunk=chunk, score=1.0, chunk_id=0)]


class TestApi:
    """The warnings reach the page and the model."""

    def api(self, tmp_path: Path) -> tuple[TestClient, list[Any]]:
        """An API whose generator records what it receives."""
        index = tmp_path / "processed"
        index.mkdir()
        (tmp_path / AmendmentBook.FILE_NAME).write_text(
            REAL.read_text(encoding="utf-8"), encoding="utf-8")
        rag = RagApi(Fake(), index_dir=str(index), generator="gemini")
        calls: list[Any] = []

        class Recorder:
            def generate(self, *args: Any) -> str:
                calls.append(args)
                return "Respuesta"

        rag._generator = Recorder()  # type: ignore[assignment]
        return TestClient(rag.app), calls

    def test_citation_carries_the_ruling(self, tmp_path: Path) -> None:
        """The article card can show it was changed."""
        client, _ = self.api(tmp_path)
        body = client.get("/citations", params={
            "query": "q", "convenio": "contact_center"}).json()
        assert len(body["citations"][0]["amendments"]) == 2

    def test_model_is_told(self, tmp_path: Path) -> None:
        """The model receives the publications of what it cites."""
        client, calls = self.api(tmp_path)
        client.get("/consult", params={"query": "q",
                                       "convenio": "contact_center"})
        assert "BOE-A-2024-3432" in calls[0][4]

    def test_norm_list(self, tmp_path: Path) -> None:
        """The selector knows how many changes each agreement has."""
        client, _ = self.api(tmp_path)
        norms = {n["id"]: n for n in client.get("/convenios").json()}
        assert len(norms["construccion"]["amendments"]) == 3
        assert norms["estatuto"]["amendments"] == []


def test_prompt_section(tmp_path: Path) -> None:
    """The notes go in their own section of the request."""
    bodies: list[dict[str, Any]] = []

    def send(body: dict[str, Any]) -> dict[str, Any]:
        bodies.append(body)
        return {"candidates": [{"finishReason": "STOP", "content": {
            "parts": [{"text": "Ok"}]}}]}

    chunk = Chunk(file_path="a.txt", first_character_index=0,
                  last_character_index=1, text="x")
    GeminiGenerator(tmp_path, send).generate("q", [chunk], notes="- art. 30")
    assert "MODIFICACIONES POSTERIORES:" in (
        bodies[0]["contents"][0]["parts"][0]["text"])
