"""The cleaning of local answers; the model itself is never loaded."""
import pytest

pytest.importorskip("torch")
from src.local_generator import LocalGenerator  # noqa: E402


@pytest.mark.parametrize("answer", [
    "Respuesta:",
    "Respuesta: (art. 37. Descanso semanal, fiestas y permisos).\n"
    "Tu convenio:\nEstatuto de los Trabajadores: (art. 37. Descanso).",
    "   ",
])
def test_empty_template_is_replaced(answer: str) -> None:
    """Labels with nothing after them are not shown as an answer."""
    assert LocalGenerator.check_answer(answer) == LocalGenerator.EMPTY


def test_empty_sections_are_dropped() -> None:
    """Only the sections with content stay."""
    answer = ("Tu convenio: Cuatro días naturales (art. 30).\n"
              "Estatuto de los Trabajadores:\nRespuesta: Cuatro días.")
    assert LocalGenerator.check_answer(answer) == (
        "Tu convenio: Cuatro días naturales (art. 30).\n\n"
        "Respuesta: Cuatro días.")


def test_plain_text_is_kept() -> None:
    """A refusal without labels stays as it is."""
    text = "No aparece en los textos consultados."
    assert LocalGenerator.check_answer(text) == text


def test_context_in_two_blocks() -> None:
    """The agreement and the Statute go apart, each under its heading."""
    from src.models import Chunk

    class Tokens:
        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            return [0] * len(text.split())

    model = LocalGenerator.__new__(LocalGenerator)
    model._tokenizer = Tokens()
    law = Chunk(file_path="data/labour/raw/estatuto_trabajadores.txt",
                first_character_index=0, last_character_index=1,
                text="Dos días.", title="Artículo 37. Permisos.")
    own = Chunk(file_path="data/labour/raw/convenio_tic.txt",
                first_character_index=0, last_character_index=1,
                text="Cuatro días.", title="Artículo 30. Permisos.")
    context = model._build_context([law, own])
    first = context.index(LocalGenerator.AGREEMENT_HEADING)
    second = context.index(LocalGenerator.STATUTE_HEADING)
    assert first < context.index("Cuatro días.") < second
    assert second < context.index("Dos días.")


def test_statute_keeps_its_share() -> None:
    """Many long agreement texts do not push the Statute out."""
    from src.models import Chunk

    class Tokens:
        def encode(self, text: str, add_special_tokens: bool) -> list[int]:
            return [0] * len(text.split())

    model = LocalGenerator.__new__(LocalGenerator)
    model._tokenizer = Tokens()
    words = " ".join(["palabra"] * 450)
    own = [Chunk(file_path="data/labour/raw/convenio_tic.txt",
                 first_character_index=i, last_character_index=i + 1,
                 text=words, title="Artículo 30.") for i in range(7)]
    law = Chunk(file_path="data/labour/raw/estatuto_trabajadores.txt",
                first_character_index=0, last_character_index=1,
                text="Dos días " + words, title="Artículo 37.")
    assert "Dos días" in model._build_context(own + [law])


class Fake(LocalGenerator):
    """Records each call instead of running the model."""

    def __init__(self, answers: dict | None = None) -> None:  # type: ignore
        """No model: only the calls are recorded."""
        self.calls: list[tuple[str | None, str]] = []
        self.answers = answers or {
            None: "Cuatro días.", "convenio": "Cuatro días (art. 30).",
            "estatuto": "Dos días (art. 37)."}

    def _call(self, question, chunks, prompt, only, extra,  # type: ignore
              tokens, given=None):
        self.calls.append((only, prompt))
        self.given = given
        return self.answers[only]


def chunks_of(*names: str) -> list:  # type: ignore[type-arg]
    """One chunk of each file."""
    from src.models import Chunk
    return [Chunk(file_path=f"data/labour/raw/{n}.txt",
                  first_character_index=0, last_character_index=1, text="x",
                  title="Artículo 1.") for n in names]


@pytest.fixture
def split(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tests of the three calls run with them switched on."""
    monkeypatch.setattr(LocalGenerator, "SPLIT_CALLS", True)


@pytest.mark.usefixtures("split")
def test_three_calls_one_per_law() -> None:
    """Each law is written alone; the answer last, from those two."""
    model = Fake()
    answer = model.generate("q", chunks_of("convenio_tic",
                                           "estatuto_trabajadores"))
    assert [only for only, _ in model.calls] == ["convenio", "estatuto",
                                                 None]
    assert "Dos días (art. 37)." in model.given
    assert answer == ("Respuesta: Cuatro días.\n\n"
                      "Tu convenio: Cuatro días (art. 30).\n\n"
                      "Estatuto de los Trabajadores: Dos días (art. 37).")


@pytest.mark.usefixtures("split")
def test_law_without_texts_is_not_asked() -> None:
    """No Statute text: no call, and no Statute section."""
    model = Fake()
    answer = model.generate("q", chunks_of("convenio_tic"))
    assert [only for only, _ in model.calls] == ["convenio"]
    assert "Estatuto" not in answer


def test_switch_back_to_one_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """Switching SPLIT_CALLS never reuses the answers of the other way."""
    first = LocalGenerator.namespace()
    monkeypatch.setattr(LocalGenerator, "SPLIT_CALLS",
                        not LocalGenerator.SPLIT_CALLS)
    assert LocalGenerator.namespace() != first


def test_context_change_is_a_new_answer(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """A different share of the context never reuses an old answer."""
    first = LocalGenerator.namespace()
    monkeypatch.setattr(LocalGenerator, "AGREEMENT_SHARE", 0.5)
    assert LocalGenerator.namespace() != first


def test_precision_defaults_to_bfloat16(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """bfloat16 unless float32 is asked for."""
    monkeypatch.delenv(LocalGenerator.DTYPE_ENV, raising=False)
    assert LocalGenerator.dtype() == "bfloat16"
    monkeypatch.setenv(LocalGenerator.DTYPE_ENV, "float32")
    assert LocalGenerator.dtype() == "float32"


@pytest.mark.usefixtures("split")
def test_contract_question_tells_parts_what_to_explain() -> None:
    """With a contract, the parts explain the topics, not the contract."""
    model = Fake()
    model.generate("Revisa este contrato", chunks_of(
        "convenio_tic", "estatuto_trabajadores"), contract="CONTRATO x")
    parts = [prompt for only, prompt in model.calls if only]
    assert all(LocalGenerator.CONTRACT_PART_RULE in p for p in parts)
    plain = Fake()
    plain.generate("q", chunks_of("convenio_tic"))
    assert LocalGenerator.CONTRACT_PART_RULE not in plain.calls[0][1]


@pytest.mark.usefixtures("split")
def test_wrong_nothing_is_replaced() -> None:
    """"Not found" while the Statute said something is never shown."""
    model = Fake({None: "No aparece en los textos consultados.",
                  "convenio": "No lo menciona.",
                  "estatuto": "Dos días (art. 37)."})
    answer = model.generate("q", chunks_of("convenio_tic",
                                           "estatuto_trabajadores"),
                            contract="CONTRATO x")
    assert answer.startswith("Respuesta: Dos días (art. 37).")


@pytest.mark.usefixtures("split")
def test_nothing_found_needs_no_answer_call() -> None:
    """When no law says anything, the code answers alone."""
    model = Fake({"convenio": "No lo menciona.",
                  "estatuto": "No lo menciona."})
    answer = model.generate("q", chunks_of("convenio_tic",
                                           "estatuto_trabajadores"))
    assert [only for only, _ in model.calls] == ["convenio", "estatuto"]
    assert answer.startswith("Respuesta: No aparece")


@pytest.mark.usefixtures("split")
def test_one_law_is_the_whole_answer() -> None:
    """One law with content: no summary call and no repeated detail."""
    model = Fake()
    answer = model.generate("q", chunks_of("estatuto_trabajadores"))
    assert [only for only, _ in model.calls] == ["estatuto"]
    assert answer == "Respuesta: Dos días (art. 37)."
