"""GeminiGenerator with a fake transport: never the API."""
from pathlib import Path
from typing import Any

import pytest

from src.answer_cache import AnswerCache
from src.exceptions import RagError
from src.gemini_generator import GeminiGenerator
from src.models import NO_SOURCES, Chunk


def chunk(name: str = "convenio_tic", text: str = "Quince días.",
          title: str = "Artículo 22. Permisos.") -> Chunk:
    """A chunk of a legal text."""
    return Chunk(file_path=f"data/laboral/raw/{name}.txt",
                 first_character_index=0, last_character_index=len(text),
                 text=text, title=title)


def reply(text: str = "Respuesta", reason: str = "STOP",
          **part: Any) -> dict[str, Any]:
    """A response of the API."""
    return {"candidates": [{"finishReason": reason, "content": {
        "parts": [{"text": text, **part}]}}]}


class Sender:
    """Records the bodies and gives a fixed response."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        """Choose the response."""
        self.bodies: list[dict[str, Any]] = []
        self.response = reply() if response is None else response

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        """Record and answer."""
        self.bodies.append(body)
        return self.response


@pytest.fixture(autouse=True)
def no_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test starts with the default model."""
    monkeypatch.delenv(GeminiGenerator.MODEL_ENV, raising=False)


def lines(folder: Path, model: str = GeminiGenerator.DEFAULT_MODEL) -> int:
    """Calls written in the ledger of a model."""
    path = folder / f"ledger_{model}.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


class TestPrompt:
    """What the model receives."""

    def test_labels(self, tmp_path: Path) -> None:
        """Each text says which law and article it comes from."""
        send = Sender()
        GeminiGenerator(tmp_path, send).generate("¿Boda?", [chunk()])
        text = send.bodies[0]["contents"][0]["parts"][0]["text"]
        assert "Artículo 22. Permisos." in text
        assert "XIX Convenio" in text
        assert text.endswith("PREGUNTA: ¿Boda?")

    def test_context_limit(self, tmp_path: Path) -> None:
        """The best chunks are kept when not all of them fit."""
        send = Sender()
        big = chunk(text="x" * GeminiGenerator.MAX_CONTEXT_CHARS)
        big.title = ""
        GeminiGenerator(tmp_path, send).generate("q", [chunk(), big])
        text = send.bodies[0]["contents"][0]["parts"][0]["text"]
        assert "Quince días." in text
        assert "xxx" not in text

    def test_statute_keeps_its_share(self, tmp_path: Path) -> None:
        """A long agreement cannot push the Statute out."""
        send = Sender()
        room = int(GeminiGenerator.MAX_CONTEXT_CHARS
                   * GeminiGenerator.AGREEMENT_SHARE)
        long = chunk(text="x" * (room - 200), title="")
        law = chunk("estatuto_trabajadores", "Ley.", "Artículo 37. Permisos.")
        GeminiGenerator(tmp_path, send).generate("q", [long, long, law])
        assert "Ley." in send.bodies[0]["contents"][0]["parts"][0]["text"]

    def test_no_sampling_parameters(self, tmp_path: Path) -> None:
        """Recent models ignore them, so they are not sent."""
        send = Sender()
        GeminiGenerator(tmp_path, send).generate("q", [chunk()])
        assert "temperature" not in send.bodies[0]["generationConfig"]


class TestWholeArticles:
    """A retrieved piece brings the rest of its article."""

    TEXT = ("Artículo 9. Contratos.\n\nContrato A: un mes de prueba.\n\n"
            "Contrato B: sin prueba.\n\nArtículo 10. Otro.\n\nNada.\n")

    @pytest.fixture
    def path(self, tmp_path: Path,
             monkeypatch: pytest.MonkeyPatch) -> str:
        """A corpus file inside a fake project."""
        monkeypatch.setattr("src.reader.CorpusReader.PROJECT_ROOT", tmp_path)
        (tmp_path / "a.txt").write_text(self.TEXT, encoding="utf-8")
        return "a.txt"

    def piece(self, path: str, text: str, title: str) -> Chunk:
        """A chunk with its real position in the file."""
        start = self.TEXT.index(text)
        return Chunk(file_path=path, first_character_index=start,
                     last_character_index=start + len(text), text=text,
                     title=title)

    def context(self, tmp_path: Path, chunks: list[Chunk]) -> str:
        """The text the model receives."""
        send = Sender()
        GeminiGenerator(tmp_path, send).generate("q", chunks)
        text: str = send.bodies[0]["contents"][0]["parts"][0]["text"]
        return text

    def test_piece_brings_its_article(self, tmp_path: Path,
                                      path: str) -> None:
        """The part the piece refers to reaches the model."""
        text = self.context(tmp_path, [self.piece(
            path, "Contrato B: sin prueba.", "Artículo 9. Contratos.")])
        assert "Contrato A: un mes" in text
        assert "Artículo 10" not in text

    def test_article_sent_once(self, tmp_path: Path, path: str) -> None:
        """Two pieces of an article do not repeat it."""
        title = "Artículo 9. Contratos."
        text = self.context(tmp_path, [
            self.piece(path, "Contrato B: sin prueba.", title),
            self.piece(path, "Contrato A: un mes de prueba.", title)])
        assert text.count("Contrato B: sin prueba.") == 1

    def test_changed_file_keeps_the_chunk(self, tmp_path: Path,
                                          path: str) -> None:
        """If the file no longer has the text, the chunk is used."""
        stale = Chunk(file_path=path, first_character_index=0,
                      last_character_index=5, text="Viejo", title="T")
        assert "Viejo" in self.context(tmp_path, [stale])


class TestAnswer:
    """Only complete answers are shown."""

    def test_text(self, tmp_path: Path) -> None:
        """A finished answer is returned."""
        generator = GeminiGenerator(tmp_path, Sender(reply(" Hola ")))
        assert generator.generate("q", [chunk()]) == "Hola"

    def test_thoughts_left_out(self, tmp_path: Path) -> None:
        """The model's thoughts are not part of the answer."""
        response = reply("visible")
        response["candidates"][0]["content"]["parts"].insert(
            0, {"text": "pensando", "thought": True})
        generator = GeminiGenerator(tmp_path, Sender(response))
        assert generator.generate("q", [chunk()]) == "visible"

    @pytest.mark.parametrize("reason", ["MAX_TOKENS", "RECITATION",
                                        "SAFETY"])
    def test_incomplete_is_an_error(self, tmp_path: Path,
                                    reason: str) -> None:
        """A cut answer could mislead: it is refused."""
        generator = GeminiGenerator(tmp_path, Sender(reply(reason=reason)))
        with pytest.raises(RagError, match=reason):
            generator.generate("q", [chunk()])

    @pytest.mark.parametrize("response", [
        {}, {"candidates": []}, reply(text="  ")])
    def test_empty_is_an_error(self, tmp_path: Path,
                               response: dict[str, Any]) -> None:
        """No text is an error, not an empty answer."""
        generator = GeminiGenerator(tmp_path, Sender(response))
        with pytest.raises(RagError):
            generator.generate("q", [chunk()])


class TestQuota:
    """No call goes over the limits."""

    def test_no_chunks_no_call(self, tmp_path: Path) -> None:
        """Without sources nothing is spent."""
        send = Sender()
        answer = GeminiGenerator(tmp_path, send).generate("q", [])
        assert answer == NO_SOURCES
        assert send.bodies == []
        assert lines(tmp_path) == 0

    def test_written_before_sending(self, tmp_path: Path) -> None:
        """A failed call also counts."""
        def failing(body: dict[str, Any]) -> dict[str, Any]:
            assert lines(tmp_path) == 1
            raise RagError("HTTP 500")

        with pytest.raises(RagError):
            GeminiGenerator(tmp_path, failing).generate("q", [chunk()])
        assert lines(tmp_path) == 1

    def test_minute_limit(self, tmp_path: Path) -> None:
        """Over the limit nothing is sent."""
        send = Sender()
        generator = GeminiGenerator(tmp_path, send)
        allowed = GeminiGenerator.LIMITS[generator.model][1]
        for _ in range(allowed):
            generator.generate("q", [chunk()])
        with pytest.raises(RagError, match="límite"):
            generator.generate("q", [chunk()])
        assert len(send.bodies) == allowed

    def test_each_model_has_its_ledger(self, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch
                                       ) -> None:
        """Google counts each model apart."""
        monkeypatch.setenv(GeminiGenerator.MODEL_ENV, "gemini-3.8-flash")
        GeminiGenerator(tmp_path, Sender()).generate("q", [chunk()])
        assert lines(tmp_path, "gemini-3.8-flash") == 1
        assert lines(tmp_path) == 0

    def test_unknown_model_refused(self, tmp_path: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
        """A model without known limits cannot be used."""
        monkeypatch.setenv(GeminiGenerator.MODEL_ENV, "gemini-4-argon")
        with pytest.raises(RagError, match="Unknown"):
            GeminiGenerator(tmp_path, Sender())

    def test_no_key_no_network(self, tmp_path: Path,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the key nothing is sent or written."""
        monkeypatch.delenv(GeminiGenerator.API_KEY_ENV, raising=False)

        def forbidden(*args: Any, **kwargs: Any) -> None:
            raise AssertionError("network used")

        monkeypatch.setattr("urllib.request.urlopen", forbidden)
        with pytest.raises(RagError):
            GeminiGenerator(tmp_path).generate("q", [chunk()])
        assert lines(tmp_path) == 0


class TestAnswerCache:
    """Answers of another model or prompt are never reused."""

    def test_model_changes_namespace(self,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
        """Another model is another namespace."""
        first = GeminiGenerator.namespace()
        monkeypatch.setenv(GeminiGenerator.MODEL_ENV, "gemini-3.8-flash")
        assert GeminiGenerator.namespace() != first

    def test_not_shared_with_qwen(self, tmp_path: Path) -> None:
        """Gemini's answers are in their own file."""
        assert (AnswerCache.file_path(tmp_path, "gemini")
                != AnswerCache.file_path(tmp_path))

    def test_custom_namespace(self, tmp_path: Path) -> None:
        """A saved answer is only read with its namespace."""
        path = tmp_path / "c.json"
        cache = AnswerCache(path, namespace="one")
        cache.put("k", "answer")
        cache.flush()
        assert AnswerCache(path, namespace="one").get("k") == "answer"
        assert AnswerCache(path, namespace="two").get("k") is None
