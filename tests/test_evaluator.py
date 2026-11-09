"""Tests for ``src.evaluator.Evaluator``.

The evaluator receives already loaded models, so most tests build tiny
datasets and results by hand, with ranges chosen so the expected recall
is known in advance. ``TestMatchesMoulinette`` checks real search results
(stored in ``tests/data``) against the numbers the moulinette reported.
"""
from pathlib import Path

import pytest

from src.evaluator import Evaluator
from src.exceptions import (DatasetEvaluationError, EvaluationError,
                            RagError, ResultsEvaluationError)
from src.models import (AnsweredQuestion, EvaluationInform,
                        MinimalSearchResults, MinimalSource, RagDataset,
                        StudentSearchResults, UnansweredQuestion)

DATA = Path(__file__).parent / "data"
ANSWERED = Path(__file__).parent.parent / "data" / "datasets" / \
    "AnsweredQuestions"


def src(start: int, end: int, path: str = "data/raw/a.md") -> MinimalSource:
    """A source location."""
    return MinimalSource(file_path=path, first_character_index=start,
                         last_character_index=end)


def answered(question_id: str, *sources: MinimalSource) -> AnsweredQuestion:
    """A ground-truth question with its correct sources."""
    return AnsweredQuestion(question_id=question_id, question="q",
                            sources=list(sources), answer="a")


def dataset(*questions: UnansweredQuestion) -> RagDataset:
    """A dataset with the given questions."""
    return RagDataset(rag_questions=list(questions))


def results(k: int, **retrieved: list[MinimalSource]) -> StudentSearchResults:
    """Search results: ``question_id=[sources...]``, in that order."""
    return StudentSearchResults(
        search_results=[MinimalSearchResults(question_id=qid, question="q",
                                             retrieved_sources=sources)
                        for qid, sources in retrieved.items()],
        k=k)


class TestIou:
    """Intersection over union of two character ranges."""

    @pytest.mark.parametrize("a, b, expected", [
        ((100, 300), (100, 300), 1.0),
        ((100, 300), (200, 400), 100 / 300),
        ((0, 100), (500, 600), 0.0),
        ((0, 100), (100, 200), 0.0),
        ((3835, 5716), (4695, 6098), 1021 / 2263),
        ((0, 2000), (900, 940), 40 / 2000),
        ((10, 20), (0, 100), 10 / 100),
    ])
    def test_values(self, a: tuple[int, int], b: tuple[int, int],
                    expected: float) -> None:
        """Known values, including touching and nested ranges."""
        assert Evaluator()._iou(src(*a), src(*b)) == pytest.approx(expected)

    def test_symmetric(self) -> None:
        """``iou(a, b) == iou(b, a)``."""
        a, b = src(0, 150), src(100, 400)
        assert Evaluator()._iou(a, b) == Evaluator()._iou(b, a)

    def test_between_zero_and_one(self) -> None:
        """Always in ``[0, 1]``."""
        for a, b in [((0, 5), (3, 9)), ((0, 1), (0, 1000)), ((7, 8), (1, 2))]:
            assert 0.0 <= Evaluator()._iou(src(*a), src(*b)) <= 1.0


class TestIsFound:
    """A correct source is found when one result shares its file and
    overlaps it with IoU ≥ 0.05."""

    def test_same_file_overlap(self) -> None:
        """Overlapping range in the same file: found."""
        assert Evaluator()._is_found(src(100, 300), [src(200, 400)])

    def test_other_file_never(self) -> None:
        """Identical range in another file: not found."""
        assert not Evaluator()._is_found(src(0, 100),
                                         [src(0, 100, "data/raw/b.md")])

    def test_threshold_included(self) -> None:
        """IoU exactly 0.05 counts."""
        assert Evaluator()._is_found(src(0, 5), [src(0, 100)])

    def test_below_threshold(self) -> None:
        """IoU 0.04 does not count, even if the answer is inside."""
        assert not Evaluator()._is_found(src(0, 4), [src(0, 100)])

    def test_any_of_many(self) -> None:
        """One good result among bad ones is enough."""
        retrieved = [src(0, 10, "data/raw/x.md"), src(900, 1000),
                     src(100, 300)]
        assert Evaluator()._is_found(src(150, 250), retrieved)

    def test_no_results(self) -> None:
        """Nothing retrieved: not found."""
        assert not Evaluator()._is_found(src(0, 100), [])


class TestQuestionRecall:
    """Share of the correct sources found for one question."""

    def test_all_found(self) -> None:
        """Every correct source found: 1."""
        assert Evaluator()._question_recall([src(0, 100)],
                                            [src(0, 100)]) == 1.0

    def test_none_found(self) -> None:
        """No correct source found: 0."""
        assert Evaluator()._question_recall([src(0, 100)],
                                            [src(500, 600)]) == 0.0

    def test_partial(self) -> None:
        """One of two correct sources found: 0.5."""
        correct = [src(0, 100), src(0, 100, "data/raw/b.md")]
        assert Evaluator()._question_recall(correct, [src(0, 100)]) == 0.5

    def test_one_result_finds_two(self) -> None:
        """A wide result can cover two close correct sources."""
        correct = [src(0, 50), src(50, 100)]
        assert Evaluator()._question_recall(correct, [src(0, 100)]) == 1.0


class TestValidateDataset:
    """Only datasets with answers can be evaluated."""

    def test_mapping(self) -> None:
        """question_id → correct sources, in dataset order."""
        expected = Evaluator()._validate_dataset(dataset(
            answered("q2", src(0, 10)), answered("q1", src(5, 9))))
        assert list(expected) == ["q2", "q1"]
        assert expected["q1"] == [src(5, 9)]

    def test_empty(self) -> None:
        """No questions: error (recall would divide by 0)."""
        with pytest.raises(DatasetEvaluationError, match="no questions"):
            Evaluator()._validate_dataset(dataset())

    def test_unanswered_question(self) -> None:
        """An UnansweredQuestion has no sources: error naming it."""
        with pytest.raises(DatasetEvaluationError, match="'q2'"):
            Evaluator()._validate_dataset(dataset(
                answered("q1", src(0, 10)),
                UnansweredQuestion(question_id="q2", question="q")))

    def test_answered_without_sources(self) -> None:
        """An answered question with ``sources = []``: error."""
        with pytest.raises(DatasetEvaluationError, match="'q1'"):
            Evaluator()._validate_dataset(dataset(answered("q1")))


class TestValidateResults:
    """Results must be valid for the moulinette."""

    def test_mapping(self) -> None:
        """question_id → retrieved sources, keeping their order."""
        retrieved = Evaluator()._validate_results(
            results(2, q1=[src(5, 9), src(0, 4)]))
        assert retrieved == {"q1": [src(5, 9), src(0, 4)]}

    def test_empty_allowed(self) -> None:
        """No results at all: valid (every question will be missing)."""
        assert Evaluator()._validate_results(results(10)) == {}

    def test_question_without_sources_kept(self) -> None:
        """A question with nothing retrieved is present, not missing."""
        assert Evaluator()._validate_results(results(5, q1=[])) == {"q1": []}

    def test_repeated_question(self) -> None:
        """The same id twice: error."""
        repeated = StudentSearchResults(search_results=[
            MinimalSearchResults(question_id="q1", question="q",
                                 retrieved_sources=[]),
            MinimalSearchResults(question_id="q1", question="q",
                                 retrieved_sources=[])], k=5)
        with pytest.raises(ResultsEvaluationError,
                           match="'q1' appears more than once"):
            Evaluator()._validate_results(repeated)

    def test_more_sources_than_k(self) -> None:
        """More sources than k: error."""
        with pytest.raises(ResultsEvaluationError,
                           match="has 3 sources, but k is 2"):
            Evaluator()._validate_results(
                results(2, q1=[src(0, 5), src(5, 9), src(9, 12)]))

    @pytest.mark.parametrize("start, end", [(-1, 5), (9, 5), (7, 7)])
    def test_impossible_range(self, start: int, end: int) -> None:
        """Negative start, end before start, or empty range: error."""
        with pytest.raises(ResultsEvaluationError,
                           match="impossible character range"):
            Evaluator()._validate_results(results(1, q1=[src(start, end)]))

    def test_too_long(self) -> None:
        """Longer than 2000 characters: error with the length."""
        with pytest.raises(ResultsEvaluationError,
                           match="has 2001 characters"):
            Evaluator()._validate_results(results(1, q1=[src(0, 2001)]))

    def test_exactly_max_allowed(self) -> None:
        """Exactly 2000 characters is valid."""
        Evaluator()._validate_results(results(1, q1=[src(500, 2500)]))


class TestEvaluate:
    """Recall@k over a whole dataset."""

    def test_returns_report(self) -> None:
        """The result is an ``EvaluationInform``."""
        report = Evaluator().evaluate(results(1, q1=[src(0, 10)]),
                                      dataset(answered("q1", src(0, 10))), 1)
        assert isinstance(report, EvaluationInform)

    def test_perfect(self) -> None:
        """Every answer in first position: recall 1 at every cutoff."""
        report = Evaluator().evaluate(
            results(10, q1=[src(0, 10)], q2=[src(50, 60)]),
            dataset(answered("q1", src(0, 10)), answered("q2", src(50, 60))),
            10)
        assert report.recall == {1: 1.0, 3: 1.0, 5: 1.0, 10: 1.0}
        assert report.missing == []
        assert report.unknown == []

    def test_position_matters(self) -> None:
        """The answer in 4th position counts for @5 and @10 only."""
        wrong = [src(1000 + 100 * i, 1050 + 100 * i) for i in range(3)]
        report = Evaluator().evaluate(
            results(10, q1=[*wrong, src(0, 10)]),
            dataset(answered("q1", src(0, 10))), 10)
        assert report.recall == {1: 0.0, 3: 0.0, 5: 1.0, 10: 1.0}

    def test_average_over_questions(self) -> None:
        """Mean of the per-question recalls."""
        report = Evaluator().evaluate(
            results(5, q1=[src(0, 10)], q2=[src(500, 600)],
                    q3=[src(0, 10, "data/raw/c.md")], q4=[src(70, 80)]),
            dataset(answered("q1", src(0, 10)), answered("q2", src(0, 10)),
                    answered("q3", src(0, 10, "data/raw/c.md")),
                    answered("q4", src(70, 80))),
            5)
        assert report.recall[1] == pytest.approx(0.75)

    def test_missing_count_as_zero(self) -> None:
        """Questions without results lower the mean and are listed."""
        report = Evaluator().evaluate(
            results(5, q1=[src(0, 10)]),
            dataset(answered("q1", src(0, 10)), answered("q2", src(0, 10))),
            5)
        assert report.recall[1] == 0.5
        assert report.missing == ["q2"]

    def test_unknown_ignored(self) -> None:
        """Results for questions outside the dataset do not count."""
        report = Evaluator().evaluate(
            results(5, q1=[src(0, 10)], zz=[src(0, 10)]),
            dataset(answered("q1", src(0, 10))), 5)
        assert report.recall[1] == 1.0
        assert report.unknown == ["zz"]

    def test_empty_results(self) -> None:
        """Nothing retrieved at all: recall 0 everywhere, all missing."""
        report = Evaluator().evaluate(
            results(10), dataset(answered("q1", src(0, 10))), 10)
        assert report.recall == {1: 0.0, 3: 0.0, 5: 0.0, 10: 0.0}
        assert report.missing == ["q1"]

    @pytest.mark.parametrize("asked_k, results_k, cutoffs", [
        (10, 10, [1, 3, 5, 10]),
        (5, 10, [1, 3, 5]),
        (10, 3, [1, 3]),
        (4, 7, [1, 3]),
        (1, 1, [1]),
    ])
    def test_cutoffs(self, asked_k: int, results_k: int,
                     cutoffs: list[int]) -> None:
        """Only cutoffs ≤ the asked k and ≤ the k of the results."""
        report = Evaluator().evaluate(
            results(results_k, q1=[src(0, 10)]),
            dataset(answered("q1", src(0, 10))), asked_k)
        assert list(report.recall) == cutoffs

    def test_multi_source_question(self) -> None:
        """A question with two correct sources, one found: 0.5."""
        report = Evaluator().evaluate(
            results(5, q1=[src(0, 10)]),
            dataset(answered("q1", src(0, 10), src(0, 10, "data/raw/b.md"))),
            5)
        assert report.recall[5] == 0.5

    def test_failed_lists_not_found(self) -> None:
        """Questions not found within k (missing included), in dataset
        order; found ones are not listed.
        """
        report = Evaluator().evaluate(
            results(5, q1=[src(0, 10)], q3=[src(500, 600)]),
            dataset(answered("q1", src(0, 10)), answered("q2", src(0, 10)),
                    answered("q3", src(0, 10))),
            5)
        assert report.failed == ["q2", "q3"]

    @pytest.mark.parametrize("asked_k, failed", [(3, ["q1"]), (5, [])])
    def test_failed_depends_on_k(self, asked_k: int,
                                 failed: list[str]) -> None:
        """An answer in 4th position fails for k = 3, not for k = 5."""
        wrong = [src(1000 + 100 * i, 1050 + 100 * i) for i in range(3)]
        report = Evaluator().evaluate(
            results(10, q1=[*wrong, src(0, 10)]),
            dataset(answered("q1", src(0, 10))), asked_k)
        assert report.failed == failed

    def test_failed_needs_all_sources(self) -> None:
        """Finding one of two correct sources is still a failure."""
        report = Evaluator().evaluate(
            results(5, q1=[src(0, 10)]),
            dataset(answered("q1", src(0, 10), src(0, 10, "data/raw/b.md"))),
            5)
        assert report.failed == ["q1"]

    def test_dataset_error_propagates(self) -> None:
        """Dataset problems surface as ``DatasetEvaluationError``."""
        with pytest.raises(DatasetEvaluationError):
            Evaluator().evaluate(results(5), dataset(), 5)

    def test_results_error_propagates(self) -> None:
        """Results problems surface as ``ResultsEvaluationError``."""
        with pytest.raises(ResultsEvaluationError):
            Evaluator().evaluate(results(1, q1=[src(0, 3000)]),
                                 dataset(answered("q1", src(0, 10))), 1)


class TestErrors:
    """The evaluation error hierarchy."""

    def test_hierarchy(self) -> None:
        """Both are ``EvaluationError`` and therefore ``RagError``."""
        for error in (DatasetEvaluationError, ResultsEvaluationError):
            assert issubclass(error, EvaluationError)
            assert issubclass(error, RagError)

    def test_problem_without_prefix(self) -> None:
        """``problem`` keeps the bare message for re-wrapping."""
        error = ResultsEvaluationError("bad range")
        assert error.problem == "bad range"
        assert str(error) == "Evaluation error: bad range"


@pytest.mark.skipif(not ANSWERED.is_dir(), reason="datasets not present")
class TestMatchesMoulinette:
    """Real search results give the moulinette's exact numbers.

    ``tests/data/search_results_{docs,code}.json`` are search results
    saved once and scored by the moulinette against the public datasets,
    so the expected numbers only hold for those two datasets: each case
    is skipped when its public dataset is not in ``data/datasets``.
    """

    @pytest.mark.parametrize("name, expected", [
        ("docs", {1: 0.61, 3: 0.78, 5: 0.85, 10: 0.88}),
        ("code", {1: 31 / 99, 3: 53 / 99, 5: 59 / 99, 10: 67 / 99}),
    ])
    def test_same_recall(self, name: str,
                         expected: dict[int, float]) -> None:
        """Same recall@1/3/5/10 as ``moulinette
        evaluate_student_search_results``.
        """
        dataset = ANSWERED / f"dataset_{name}_public.json"
        if not dataset.is_file():
            pytest.skip(f"public dataset not present ({dataset.name})")
        found = StudentSearchResults.model_validate_json(
            (DATA / f"search_results_{name}.json").read_text(
                encoding="utf-8"))
        truth = RagDataset.model_validate_json(
            dataset.read_text(encoding="utf-8"))
        report = Evaluator().evaluate(found, truth, 10)
        assert report.recall == pytest.approx(expected)
        assert report.missing == []
        assert report.unknown == []
