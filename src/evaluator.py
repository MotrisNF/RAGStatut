"""Recall@k of search results against a ground-truth dataset.

The same metric as the moulinette: a correct source is found when one of
the first k results is in the same file and overlaps it with IoU ≥ 0.05.
"""
from tqdm import tqdm

from .exceptions import DatasetEvaluationError, ResultsEvaluationError
from .models import (AnsweredQuestion, MinimalSource, StudentSearchResults,
                     RagDataset, EvaluationInform)


class Evaluator:
    """Compute recall@k from already loaded results and dataset.

    It never reads files: the ``evaluate`` command loads both JSON files
    and adds their paths to the error messages.

    Attributes:
        MAX_CNTX_LEN: Longest valid source, as the moulinette's
            ``max_context_length``.
        MIN_IOU: Smallest overlap that counts a correct source as found.
        CUTOFFS: The k values reported (recall@1, @3, @5, @10).
    """

    MAX_CNTX_LEN: int = 2000
    MIN_IOU: float = 0.05
    CUTOFFS: tuple[int, ...] = (1, 3, 5, 10)

    def _iou(self,
             a: MinimalSource,
             b: MinimalSource
             ) -> float:
        """Intersection over union of two character ranges.

        1.0 for identical ranges, 0.0 when they do not overlap. The file
        is not compared here (see ``_is_found``).

        Args:
            a: A source.
            b: Another source.

        Returns:
            The overlap, between 0 and 1.
        """
        intersection = max(0, min(a.last_character_index,
                           b.last_character_index)
                           - max(a.first_character_index,
                                 b.first_character_index))
        union = ((a.last_character_index - a.first_character_index)
                 + (b.last_character_index - b.first_character_index)
                 - intersection)
        return intersection / union

    def _is_found(self,
                  correct: MinimalSource,
                  results: list[MinimalSource]
                  ) -> bool:
        """Whether one of the results finds a correct source.

        Args:
            correct: A ground-truth source.
            results: The retrieved sources to look into.

        Returns:
            True if a result is in the same file and has IoU ≥ ``MIN_IOU``.
        """
        return any(
            source.file_path == correct.file_path
            and self._iou(source, correct) >= self.MIN_IOU
            for source in results
        )

    def _question_recall(self,
                         correct: list[MinimalSource],
                         result: list[MinimalSource]
                         ) -> float:
        """Share of the correct sources found among the results.

        Args:
            correct: Ground-truth sources of one question (never empty,
                ``_validate_dataset`` guarantees it).
            result: Retrieved sources of that question, already cut to k.

        Returns:
            Found sources / correct sources, between 0 and 1.
        """
        found = sum(
            1
            for source in correct
            if self._is_found(source, result)
        )

        return found / len(correct)

    def _validate_dataset(self,
                          dataset: RagDataset
                          ) -> dict[str, list[MinimalSource]]:
        """Check that every question can be evaluated.

        Args:
            dataset: The ground-truth dataset.

        Returns:
            ``question_id → correct sources``, in dataset order.

        Raises:
            DatasetEvaluationError: The dataset is empty, or a question is
                not an ``AnsweredQuestion`` or has no sources.
        """
        if not dataset.rag_questions:
            raise DatasetEvaluationError("the dataset has no questions")

        expected: dict[str, list[MinimalSource]] = {}

        for q in dataset.rag_questions:
            if (not isinstance(q, AnsweredQuestion)
                    or not q.sources):
                raise DatasetEvaluationError(
                    f"question '{q.question_id}' has no sources: use a "
                    "valid AnsweredQuestions dataset"
                )
            expected[q.question_id] = q.sources

        return expected

    def _validate_results(self,
                          results: StudentSearchResults
                          ) -> dict[str, list[MinimalSource]]:
        """Check that the results would be valid for the moulinette.

        Empty results are allowed (every question is then missing).

        Args:
            results: The search results to evaluate.

        Returns:
            ``question_id → retrieved sources``, keeping their order.

        Raises:
            ResultsEvaluationError: A question appears twice, has more
                sources than ``k``, or a source has an impossible range or
                is longer than ``MAX_CNTX_LEN``.
        """
        retrieved: dict[str, list[MinimalSource]] = {}

        for entry in results.search_results:
            question = entry.question_id
            if question in retrieved:
                raise ResultsEvaluationError(
                    f"question '{question}' appears more than once"
                )
            if len(entry.retrieved_sources) > results.k:
                raise ResultsEvaluationError(
                    f"question '{question}' has "
                    f"{len(entry.retrieved_sources)} sources, but k is "
                    f"{results.k}"
                )
            for source in entry.retrieved_sources:
                start = source.first_character_index
                end = source.last_character_index
                where = f"{source.file_path} [{start}:{end}]"
                if start < 0 or end <= start:
                    raise ResultsEvaluationError(
                        f"question '{question}': source {where} has an "
                        "impossible character range"
                    )
                if end - start > self.MAX_CNTX_LEN:
                    raise ResultsEvaluationError(
                        f"question '{question}': source {where} has "
                        f"{end - start} characters (max {self.MAX_CNTX_LEN})"
                    )
            retrieved[question] = entry.retrieved_sources

        return retrieved

    def evaluate(self,
                 results: StudentSearchResults,
                 dataset: RagDataset,
                 k: int = 10
                 ) -> EvaluationInform:
        """Recall@k of the results, averaged over the dataset questions.

        Questions missing from the results count as 0; results for
        questions outside the dataset are ignored. Only the cutoffs that
        are ≤ ``k`` and ≤ ``results.k`` are reported.

        Args:
            results: The search results.
            dataset: The ground truth (AnsweredQuestions).
            k: Largest cutoff to report.

        Returns:
            Recall per cutoff, the missing and unknown question ids, and
            the questions not fully found within the first k results.

        Raises:
            DatasetEvaluationError: The dataset cannot be used.
            ResultsEvaluationError: The results are not valid.
        """
        expected = self._validate_dataset(dataset)
        retrieved = self._validate_results(results)

        missing: list[str] = [
            question_id
            for question_id in expected
            if question_id not in retrieved
            ]
        unknown: list[str] = [
            question_id
            for question_id in retrieved
            if question_id not in expected
            ]
        cutoffs = [cutoff for cutoff in self.CUTOFFS
                   if cutoff <= k and cutoff <= results.k]
        totals: dict[int, float] = {cutoff: 0.0 for cutoff in cutoffs}

        for question_id in tqdm(expected, desc="Evaluating",
                                unit="question"):
            if question_id not in retrieved:
                continue
            for cutoff in cutoffs:
                totals[cutoff] += self._question_recall(
                    expected[question_id], retrieved[question_id][:cutoff]
                )

        recall = {cutoff: total / len(expected)
                  for cutoff, total in totals.items()}

        limit = min(k, results.k)
        failed = [
            question_id
            for question_id in expected
            if question_id not in retrieved
            or self._question_recall(expected[question_id],
                                     retrieved[question_id][:limit]) < 1.0
        ]

        return EvaluationInform(recall, missing, unknown, failed)
