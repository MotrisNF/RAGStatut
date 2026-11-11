"""The local answer sent part by part: API events and the page."""
import json
from pathlib import Path
import sys
from typing import Any, Iterator

from fastapi.testclient import TestClient
import pytest

from src.api import RagApi
from src.base_retriever import Retriever
from src.exceptions import RagError
from src.models import Chunk, SearchHit


class Fake(Retriever):
    """One hit of the agreement and one of the Statute."""

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Two hits."""
        hits = []
        for name in sorted(sources or []):
            chunk = Chunk(file_path=f"data/labour/raw/{name}.txt",
                          first_character_index=0, last_character_index=1,
                          text="x", title="Artículo 1.")
            hits.append(SearchHit(chunk=chunk, score=1.0, chunk_id=0))
        return hits


class Steps:
    """A local model that reports its steps like the real one."""

    def __init__(self, fail: bool = False) -> None:
        """Choose whether it fails half way."""
        self.fail = fail
        self.calls = 0

    def iter_generate(self, *args: Any) -> Iterator[tuple[str, ...]]:
        """Two parts, then the answer."""
        self.calls += 1
        yield ("stage", "convenio")
        yield ("part", "Tu convenio", "Cuatro días (art. 30).")
        if self.fail:
            raise RagError("sin memoria")
        yield ("stage", "respuesta")
        yield ("part", "Respuesta", "Cuatro días.")
        yield ("answer", "Respuesta: Cuatro días.\n\nTu convenio: Cuatro.")


def stream(local: Steps, tmp_path: Path | None = None) -> list[dict[str, Any]]:
    """The events of one streamed question."""
    api = RagApi(Fake(), index_dir=str(tmp_path) if tmp_path else None)
    api._local = local  # type: ignore[assignment]
    response = TestClient(api.app).get("/consult/stream", params={
        "query": "q", "convenio": "tic"})
    assert response.status_code == 200
    return [json.loads(line) for line in response.text.splitlines() if line]


class TestApi:
    """The order and content of the events."""

    def test_steps_then_result(self) -> None:
        """Each step as it happens, the whole result last."""
        events = stream(Steps())
        assert [e["event"] for e in events] == [
            "stage", "part", "stage", "part", "done"]
        assert events[1]["section"] == "Tu convenio"
        assert events[-1]["result"]["answer"].startswith("Respuesta:")
        assert len(events[-1]["result"]["citations"]) == 2

    def test_error_is_an_event(self) -> None:
        """A failure half way ends the stream with its message."""
        events = stream(Steps(fail=True))
        assert events[-1] == {"event": "error", "detail": "sin memoria"}

    def test_unknown_agreement_before_streaming(self,
                                                tmp_path: Path) -> None:
        """Bad input is still a normal error, not a stream."""
        (tmp_path / "manifest.json").write_text(
            '{"max_chunk_size": 2000, "raw_dir": "r", "files": '
            '{"r/convenio_tic.txt": {"mtime_ns": 1, "size": 1}}}')
        api = RagApi(Fake(), index_dir=str(tmp_path))
        response = TestClient(api.app).get("/consult/stream", params={
            "query": "q", "convenio": "patata"})
        assert response.status_code == 400


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))
import client  # noqa: E402


class TestPage:
    """The page shows the parts while they arrive."""

    def test_parts_shown_then_result(self,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """Every part reaches the page, and the final result is used."""
        import streamlit as st
        from streamlit.testing.v1 import AppTest

        # The page caches the agreements: start and end without them, so
        # the fake list does not leak into other tests.
        st.cache_data.clear()

        events = [
            {"event": "stage", "stage": "convenio"},
            {"event": "part", "section": "Tu convenio", "text": "Cuatro."},
            {"event": "done", "result": {
                "query": "q", "convenio": "tic", "answer": "Respuesta: Ok.",
                "citations": [], "checks": []}}]
        monkeypatch.setattr(client, "consult_stream",
                            lambda *args: iter(events))
        monkeypatch.setattr(client, "citations", lambda q, c: {
            "citations": [{"kind": "convenio", "norm": "C", "url": "",
                           "article": "Artículo 1.", "text": "x"}]})
        monkeypatch.setattr(client, "norms", lambda: [
            {"id": "tic", "name": "Convenio TIC", "validity": "2027",
             "url": "", "boe_id": "", "source": "t"}])
        monkeypatch.setattr(client, "contracts", lambda: [])
        monkeypatch.setattr(client, "status", lambda: {
            "gemini_key": False, "local_ready": True})
        app = AppTest.from_file(str(Path(client.__file__).with_name(
            "app.py")), default_timeout=20)
        app.session_state["answer_mode"] = "local"
        app.run()
        app.selectbox[0].set_value("tic").run()
        app.text_area[0].set_value("¿Vacaciones?").run()
        next(b for b in app.button if b.label == "Preguntar").click().run()
        assert not app.exception
        st.cache_data.clear()
        assert app.session_state["results"][0]["answer"] == "Respuesta: Ok."
