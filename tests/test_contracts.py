"""Fictional contracts: reading them and answering about them, offline."""
import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient
import pytest

from src.api import RagApi
from src.answer_cache import AnswerCache
from src.base_retriever import Retriever
from src.contracts import ContractStore
from src.gemini_generator import GeminiGenerator
from src.models import Chunk, GeneratorMode, SearchHit

TEXT = ("CONTRATO\nLa trabajadora: Ana Ruiz\nCUARTA. Vacaciones: 20 días.\n"
        "SEGUNDA. Puesto: Cajera. Clasificación profesional: X.\n")


@pytest.fixture
def folder(tmp_path: Path) -> Path:
    """Two contracts and the answer key that must never be read."""
    root = tmp_path / "contracts"
    (root / "tic").mkdir(parents=True)
    (root / "tic" / "tic_01_ana-ruiz.txt").write_text(TEXT, encoding="utf-8")
    (root / "tic" / "tic_02_eva-gil.txt").write_text(
        TEXT.replace("Ana Ruiz", "Eva Gil"), encoding="utf-8")
    (root / "_answer_key.json").write_text(
        json.dumps([{"discrepancias": "SECRETO"}]), encoding="utf-8")
    return root


class Fake(Retriever):
    """One hit, always."""

    def search(self, query: str, k: int,
               sources: frozenset[str] | None = None) -> list[SearchHit]:
        """Return a hit."""
        chunk = Chunk(file_path="data/x/convenio_tic.txt",
                      first_character_index=0, last_character_index=1,
                      text="x", title="")
        return [SearchHit(chunk=chunk, score=1.0, chunk_id=0)]


class Recorder:
    """A generator that records what it receives."""

    def __init__(self) -> None:
        """No call yet."""
        self.calls: list[tuple[str, str | None]] = []

    def generate(self, query: str, chunks: list[Chunk],
                 contract: str | None = None,
                 checks: str | None = None,
                 notes: str | None = None) -> str:
        """Record and answer."""
        self.calls.append((query, contract))
        return "Respuesta"


def api(folder: Path, generator: GeneratorMode = "gemini",
        cache: Path | None = None) -> tuple[TestClient, Recorder]:
    """An API over the fake retriever."""
    index = folder.parent / "processed"
    index.mkdir(exist_ok=True)
    answers = AnswerCache(cache, namespace="n") if cache else None
    rag = RagApi(Fake(), index_dir=str(index), generator=generator,
                 answers=answers)
    record = Recorder()
    rag._generator = record  # type: ignore[assignment]
    return TestClient(rag.app), record


class TestStore:
    """What is read from the folder."""

    def test_fields(self, folder: Path) -> None:
        """Name, job and agreement come from the file."""
        found = ContractStore(folder).get("tic_01")
        assert found is not None
        assert (found.name, found.job, found.convenio) == (
            "Ana Ruiz", "Cajera", "tic")

    def test_answer_key_never_read(self, folder: Path) -> None:
        """The planted answers are invisible to the store."""
        assert all("SECRETO" not in c.text
                   for c in ContractStore(folder).all())

    def test_only_active_are_offered(self, folder: Path) -> None:
        """The others stay on disk but are not offered."""
        (folder / "_active.json").write_text('["tic_02"]', encoding="utf-8")
        assert [c.id for c in ContractStore(folder).all()] == ["tic_02"]

    def test_real_contracts_load(self) -> None:
        """The offered files parse, with no answer key inside."""
        root = Path("data/labour/contracts")
        if not root.exists():
            pytest.skip("contracts not present")
        found = ContractStore(root).all()
        assert len(found) == 15
        assert all("discrepancias" not in c.text for c in found)


class TestPrompt:
    """What Gemini receives."""

    def send(self, tmp_path: Path, contract: str | None) -> dict[str, Any]:
        """The request body built for a question."""
        bodies: list[dict[str, Any]] = []

        def transport(body: dict[str, Any]) -> dict[str, Any]:
            bodies.append(body)
            return {"candidates": [{"finishReason": "STOP", "content": {
                "parts": [{"text": "Ok"}]}}]}

        chunk = Chunk(file_path="data/x/a.txt", first_character_index=0,
                      last_character_index=1, text="x")
        GeminiGenerator(tmp_path, transport).generate("q", [chunk], contract)
        return bodies[0]

    def test_contract_sent_with_its_rules(self, tmp_path: Path) -> None:
        """The contract and its rules go in."""
        body = self.send(tmp_path, TEXT)
        assert "CONTRATO:" in body["contents"][0]["parts"][0]["text"]
        assert "Posible discrepancia" in (
            body["systemInstruction"]["parts"][0]["text"])

    def test_no_contract_no_rules(self, tmp_path: Path) -> None:
        """Without a contract the prompt is the usual one."""
        body = self.send(tmp_path, None)
        assert "CONTRATO:" not in body["contents"][0]["parts"][0]["text"]
        assert "Posible discrepancia" not in (
            body["systemInstruction"]["parts"][0]["text"])

    def test_namespace_ignores_other_code(self) -> None:
        """Only the model and the prompts identify the answers."""
        assert len(GeminiGenerator.namespace()) == 64


class TestApi:
    """The contracts through the API."""

    def test_list(self, folder: Path) -> None:
        """The list has the contracts, without the answer key."""
        client, _ = api(folder)
        body = client.get("/contracts")
        assert [c["id"] for c in body.json()] == ["tic_01", "tic_02"]
        assert "SECRETO" not in body.text

    def test_contract_reaches_the_generator(self, folder: Path) -> None:
        """The text of the chosen contract is used."""
        client, record = api(folder)
        client.get("/consult", params={"query": "q", "convenio": "tic",
                                       "contract": "tic_01"})
        assert record.calls[0][1] is not None
        assert "Ana Ruiz" in record.calls[0][1]

    def test_two_people_do_not_share_an_answer(self, folder: Path) -> None:
        """Same question, another contract: another answer."""
        client, record = api(folder, cache=folder.parent / "c.json")
        for contract in ("tic_01", "tic_02", "tic_01"):
            client.get("/consult", params={"query": "q", "convenio": "tic",
                                           "contract": contract})
        assert len(record.calls) == 2

    def test_unknown_contract(self, folder: Path) -> None:
        """A contract that does not exist is a 404, not an answer."""
        client, record = api(folder)
        response = client.get("/consult", params={
            "query": "q", "convenio": "tic", "contract": "nope"})
        assert response.status_code == 404
        assert record.calls == []

    def test_local_model_refused(self, folder: Path) -> None:
        """Contracts need Gemini."""
        client, _ = api(folder, generator="qwen")
        assert client.get("/consult", params={
            "query": "q", "convenio": "tic",
            "contract": "tic_01"}).status_code == 400
