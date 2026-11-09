"""Tests for the pydantic models in ``src.models``."""
import json
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.models import (
    AnsweredQuestion,
    CorpusReadResult,
    Document,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    UnansweredQuestion,
)
from conftest import RealData, needs_answered, needs_unanswered

SOURCE = {
    "file_path": "data/raw/vllm-0.10.1/docs/features/lora.md",
    "first_character_index": 4695,
    "last_character_index": 6098,
}


class TestMinimalSource:
    """One source location."""

    def test_valid(self) -> None:
        """The three fields are stored as given."""
        source = MinimalSource.model_validate(SOURCE)
        assert source.file_path == SOURCE["file_path"]
        assert source.first_character_index == 4695
        assert source.last_character_index == 6098

    @pytest.mark.parametrize("missing", list(SOURCE))
    def test_every_field_required(self, missing: str) -> None:
        """Removing any field fails."""
        data = {k: v for k, v in SOURCE.items() if k != missing}
        with pytest.raises(ValidationError):
            MinimalSource.model_validate(data)

    def test_index_must_be_int(self) -> None:
        """Non-numeric indexes are rejected."""
        with pytest.raises(ValidationError):
            MinimalSource.model_validate(
                {**SOURCE, "first_character_index": "start"})


class TestQuestions:
    """Unanswered and answered questions."""

    def test_id_generated(self) -> None:
        """Without ``question_id`` a valid UUID is created."""
        question = UnansweredQuestion(question="What is LoRA?")
        uuid.UUID(question.question_id)

    def test_ids_are_unique(self) -> None:
        """Each question gets its own generated id."""
        ids = {UnansweredQuestion(question="q").question_id
               for _ in range(50)}
        assert len(ids) == 50

    def test_given_id_kept(self) -> None:
        """An explicit id is not replaced."""
        assert UnansweredQuestion(question_id="q1",
                                  question="q").question_id == "q1"

    def test_question_required(self) -> None:
        """``question`` has no default."""
        with pytest.raises(ValidationError):
            UnansweredQuestion.model_validate({"question_id": "q1"})

    def test_answered_is_unanswered(self) -> None:
        """An answered question is also an unanswered question."""
        answered = AnsweredQuestion(
            question="q", sources=[MinimalSource.model_validate(SOURCE)],
            answer="a")
        assert isinstance(answered, UnansweredQuestion)

    def test_answered_needs_sources_and_answer(self) -> None:
        """``sources`` and ``answer`` are required."""
        with pytest.raises(ValidationError):
            AnsweredQuestion.model_validate({"question": "q"})

    def test_extra_fields_ignored(self) -> None:
        """Dataset fields not in the model do not break validation."""
        question = AnsweredQuestion.model_validate({
            "question": "q", "sources": [SOURCE], "answer": "a",
            "difficulty": "synthetic", "is_valid": True,
        })
        assert "difficulty" not in question.model_dump()


class TestRagDataset:
    """The question files."""

    def test_union_picks_right_model(self) -> None:
        """Answered entries become AnsweredQuestion, others Unanswered."""
        dataset = RagDataset.model_validate({"rag_questions": [
            {"question": "a", "sources": [SOURCE], "answer": "x"},
            {"question": "b"},
        ]})
        kinds = [type(q) for q in dataset.rag_questions]
        assert kinds == [AnsweredQuestion, UnansweredQuestion]

    def test_empty_list_valid(self) -> None:
        """A dataset may have no questions."""
        assert RagDataset(rag_questions=[]).rag_questions == []

    def test_wrong_root_key(self) -> None:
        """The root key must be ``rag_questions``."""
        with pytest.raises(ValidationError):
            RagDataset.model_validate({"questions": []})

    def test_broken_json(self) -> None:
        """Invalid JSON text raises ``ValidationError``."""
        with pytest.raises(ValidationError):
            RagDataset.model_validate_json('{"rag_questions": [,]}')


@needs_unanswered
class TestRealDatasets:
    """The question datasets in ``data/datasets``, whatever their names
    (the public ones, or the ones brought to the evaluation)."""

    @pytest.mark.parametrize("path", RealData.unanswered_files(),
                             ids=lambda path: path.name)
    def test_unanswered_valid(self, path: Path) -> None:
        """Every unanswered dataset validates."""
        dataset = RagDataset.model_validate_json(
            path.read_text(encoding="utf-8"))
        assert len(dataset.rag_questions) > 0

    @needs_answered
    @pytest.mark.parametrize("path", RealData.answered_files(),
                             ids=lambda path: path.name)
    def test_answered_valid(self, path: Path) -> None:
        """Every answered entry is parsed as AnsweredQuestion."""
        dataset = RagDataset.model_validate_json(
            path.read_text(encoding="utf-8"))
        assert all(isinstance(q, AnsweredQuestion)
                   for q in dataset.rag_questions)

    @pytest.mark.parametrize("name", RealData.dataset_names())
    def test_ids_match(self, name: str) -> None:
        """A dataset in both folders lists the same questions in both."""
        def ids(folder: Path) -> list[str]:
            """Question ids of the dataset in ``folder``."""
            path = folder / name
            data = RagDataset.model_validate_json(
                path.read_text(encoding="utf-8"))
            return [q.question_id for q in data.rag_questions]
        assert ids(RealData.UNANSWERED) == ids(RealData.ANSWERED)


class TestSearchResults:
    """Output models of search and answer generation."""

    def result(self) -> MinimalSearchResults:
        """A search result with one source."""
        return MinimalSearchResults(
            question_id="q1", question="How to configure OpenAI server?",
            retrieved_sources=[MinimalSource.model_validate(SOURCE)])

    def test_answer_is_search_result(self) -> None:
        """MinimalAnswer extends MinimalSearchResults."""
        answer = MinimalAnswer(**self.result().model_dump(), answer="a")
        assert isinstance(answer, MinimalSearchResults)

    def test_json_keys_match_subject(self) -> None:
        """The JSON output uses exactly the subject's keys."""
        output = StudentSearchResults(search_results=[self.result()], k=10)
        data = json.loads(output.model_dump_json())
        assert set(data) == {"search_results", "k"}
        entry = data["search_results"][0]
        assert set(entry) == {"question_id", "question", "retrieved_sources"}
        assert set(entry["retrieved_sources"][0]) == set(SOURCE)

    def test_answer_json_keys(self) -> None:
        """Answers add only the ``answer`` key."""
        answer = MinimalAnswer(**self.result().model_dump(), answer="a")
        output = StudentSearchResultsAndAnswer(search_results=[answer], k=5)
        entry = json.loads(output.model_dump_json())["search_results"][0]
        assert set(entry) == {"question_id", "question",
                              "retrieved_sources", "answer"}

    def test_round_trip(self) -> None:
        """Writing to JSON and reading back gives the same object."""
        output = StudentSearchResults(search_results=[self.result()], k=10)
        again = StudentSearchResults.model_validate_json(
            output.model_dump_json())
        assert again == output

    def test_answer_file_needs_answers(self) -> None:
        """A search result without answer is not a MinimalAnswer."""
        with pytest.raises(ValidationError):
            StudentSearchResultsAndAnswer.model_validate({
                "search_results": [self.result().model_dump()], "k": 10})

    def test_k_required(self) -> None:
        """``k`` is part of the output."""
        with pytest.raises(ValidationError):
            StudentSearchResults.model_validate({"search_results": []})


class TestCorpusModels:
    """Models produced by the corpus reader."""

    def test_document(self) -> None:
        """A document keeps its path and text."""
        doc = Document(file_path="data/raw/a.md", text="hello")
        assert (doc.file_path, doc.text) == ("data/raw/a.md", "hello")

    def test_document_positional_rejected(self) -> None:
        """pydantic models only take keyword arguments."""
        with pytest.raises(TypeError):
            Document("data/raw/a.md", "hello")  # type: ignore[call-arg]

    def test_found(self) -> None:
        """``found`` adds read and skipped files."""
        result = CorpusReadResult(
            documents=[Document(file_path="a", text=""),
                       Document(file_path="b", text="")],
            skipped=["c"])
        assert result.found == 3

    def test_found_not_serialized(self) -> None:
        """``found`` is computed, not stored."""
        result = CorpusReadResult(documents=[], skipped=[])
        assert result.found == 0
        assert "found" not in result.model_dump()
