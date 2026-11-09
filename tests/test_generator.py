"""Tests for ``src.generator`` that do not load the model."""
import pytest

from src.generator import Generator


@pytest.fixture
def generator() -> Generator:
    """A ``Generator`` without its model: only pure methods are used."""
    return Generator.__new__(Generator)


class TestCleanAnswer:
    """The refusal sentence never contradicts a real answer."""

    def test_refusal_tail_removed(self, generator: Generator) -> None:
        """A real answer followed by the refusal keeps only the answer."""
        answer = ("You should checkout commit `e5be006`. "
                  + Generator.REFUSAL)
        assert (generator._clean_answer(answer)
                == "You should checkout commit `e5be006`.")

    def test_refusal_in_the_middle_removed(self,
                                           generator: Generator) -> None:
        """The refusal is removed wherever it appears."""
        answer = f"Use `values.yaml`. {Generator.REFUSAL} It has a table."
        assert "does not contain" not in generator._clean_answer(answer)

    @pytest.mark.parametrize("answer", [
        Generator.REFUSAL,
        f"  {Generator.REFUSAL}\n",
        f"\n{Generator.REFUSAL}",
    ])
    def test_only_refusal_kept(self, generator: Generator,
                               answer: str) -> None:
        """An answer that is only the refusal stays the refusal."""
        assert generator._clean_answer(answer) == Generator.REFUSAL

    def test_normal_answer_unchanged(self, generator: Generator) -> None:
        """Answers without the refusal are only stripped."""
        assert generator._clean_answer("  The default is `False`.\n") == \
            "The default is `False`."


class TestExamples:
    """The few-shot example must not teach copyable decorations."""

    def test_example_answer_has_no_parenthetical(self) -> None:
        """The model copied a trailing "(ServerConfig)" into answers."""
        assert "(" not in Generator.EXAMPLE_ANSWER
