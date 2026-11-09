"""Checks on the mini corpus and datasets in ``tests/data``.

They guard the test data itself: if a dataset is edited by hand (new
question, moved range...), these tests say what is wrong before any
recall is computed with it. ``TestPipeline`` runs index → search_dataset
→ evaluate on the mini corpus, like ``make test_rag``.
"""
import json
from pathlib import Path

import fire
import pytest

from src.bm25_index import BM25Index
from src.commands import RagCLI
from src.evaluator import Evaluator
from src.models import AnsweredQuestion, RagDataset, StudentSearchResults
from src.reader import CorpusReader

DATA = Path(__file__).parent / "data"
ROOT = CorpusReader.PROJECT_ROOT
NAMES = ["dataset_mini.json", "dataset_mini_hard.json"]
DIFFICULTIES = {"paraphrase", "identifier", "small-answer", "distractor"}


def load(folder: str, name: str) -> RagDataset:
    """Read a mini dataset."""
    path = DATA / "datasets" / folder / name
    return RagDataset.model_validate_json(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", NAMES)
class TestDatasets:
    """Each dataset is consistent with the corpus and with its pair."""

    def test_same_questions(self, name: str) -> None:
        """Unanswered and answered list the same ids and texts."""
        plain = load("UnansweredQuestions", name).rag_questions
        answered = load("AnsweredQuestions", name).rag_questions
        assert [(q.question_id, q.question) for q in plain] == [
            (q.question_id, q.question) for q in answered]

    def test_unique_ids(self, name: str) -> None:
        """No repeated question id."""
        ids = [q.question_id for q in load("AnsweredQuestions",
                                           name).rag_questions]
        assert len(ids) == len(set(ids))

    def test_answered(self, name: str) -> None:
        """Every answered question has an answer and at least one source."""
        for question in load("AnsweredQuestions", name).rag_questions:
            assert isinstance(question, AnsweredQuestion)
            assert question.answer and question.sources

    def test_sources_point_into_the_corpus(self, name: str) -> None:
        """Each source is a real file of the mini corpus, with a valid,
        non-blank range of at most 2000 characters.
        """
        for question in load("AnsweredQuestions", name).rag_questions:
            assert isinstance(question, AnsweredQuestion)
            for source in question.sources:
                assert source.file_path.startswith("tests/data/raw/")
                text = (ROOT / source.file_path).read_text(encoding="utf-8")
                start = source.first_character_index
                end = source.last_character_index
                assert 0 <= start < end <= len(text), question.question_id
                assert end - start <= 2000
                assert text[start:end].strip(), question.question_id


def test_hard_questions_say_what_they_test() -> None:
    """Every hard question carries a known ``difficulty`` label."""
    path = DATA / "datasets" / "AnsweredQuestions" / "dataset_mini_hard.json"
    questions = json.loads(path.read_text(encoding="utf-8"))["rag_questions"]
    assert {q["difficulty"] for q in questions} == DIFFICULTIES


class TestPipeline:
    """index → search_dataset → evaluate on the mini corpus."""

    def run(self, tmp_path: Path, name: str, chunk: int) -> dict[int, float]:
        """Run the three steps and return recall@cutoff."""
        index_dir = tmp_path / "processed"
        output = tmp_path / "output"
        fire.Fire(RagCLI, command=[
            "index", "--raw_dir", str(DATA / "raw"), "--index_dir",
            str(index_dir), "--max_chunk_size", str(chunk)])
        fire.Fire(RagCLI, command=[
            "search_dataset", "--dataset_path",
            str(DATA / "datasets" / "UnansweredQuestions" / name),
            "--k", "5", "--save_directory", str(output),
            "--index_dir", str(index_dir)])
        results = StudentSearchResults.model_validate_json(
            (output / name).read_text(encoding="utf-8"))
        report = Evaluator().evaluate(results, load("AnsweredQuestions",
                                                    name), 5)
        return report.recall

    def test_easy_is_perfect(self, tmp_path: Path) -> None:
        """Easy questions are all found first with 500-char chunks."""
        recall = self.run(tmp_path, "dataset_mini.json", 500)
        assert recall == {1: 1.0, 3: 1.0, 5: 1.0}

    def test_hard_leaves_room(self, tmp_path: Path) -> None:
        """Hard questions are harder: some are missed, not all."""
        recall = self.run(tmp_path, "dataset_mini_hard.json", 500)
        assert 0.0 < recall[5] < 1.0

    def test_index_is_loadable(self, tmp_path: Path) -> None:
        """The mini index can be loaded and searched."""
        self.run(tmp_path, "dataset_mini.json", 500)
        hits = BM25Index.load(tmp_path / "processed").search("lora", 1)
        assert hits[0].chunk.file_path.startswith("tests/data/raw/")
