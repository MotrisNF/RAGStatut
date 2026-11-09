"""The MiConvenio page, with the API replaced: nothing leaves the test."""
from pathlib import Path
import sys
from typing import Any

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent / "web"
sys.path.insert(0, str(ROOT))
import app_helpers as helpers  # noqa: E402
import client  # noqa: E402

NORMS = [
    {"id": "estatuto", "source": "e", "name": "Estatuto", "boe_id": "E",
     "url": "https://www.boe.es/e", "validity": "Consolidado"},
    {"id": "tic", "source": "t", "name": "Convenio TIC", "boe_id": "T",
     "url": "https://www.boe.es/t", "validity": "Hasta 2027"},
    {"id": "obra", "source": "o", "name": "Convenio Obra", "boe_id": "O",
     "url": "https://www.boe.es/o", "validity": "Hasta 2026"},
]


def citation(article: str, text: str = "Texto") -> dict[str, Any]:
    """A citation of an agreement."""
    return {"kind": "convenio", "norm": "Convenio TIC", "article": article,
            "text": text, "url": "https://www.boe.es/t"}


class Fake:
    """Stands for the API and counts the answers asked for."""

    def __init__(self, error: str = "") -> None:
        """Choose whether the answers fail."""
        self.error = error
        self.calls: list[str] = []

    def contracts(self) -> list[dict[str, Any]]:
        """The people."""
        return [{"id": "tic_01", "name": "Ana Ruiz", "job": "Cajera",
                 "convenio": "tic", "text": "CONTRATO de Ana"}]

    def citations(self, query: str, convenio: str) -> dict[str, Any]:
        """Search."""
        return {"citations": [citation("Artículo 22.")]}

    def consult(self, query: str, convenio: str,
                contract: str | None = None) -> dict[str, Any]:
        """Answer, or fail."""
        self.contract = contract
        if self.error:
            raise client.ApiError(self.error)
        self.calls.append(convenio)
        return {"query": query, "convenio": convenio,
                "answer": f"Respuesta de {convenio}",
                "citations": [citation("Artículo 22.", "Uno"),
                              citation("Artículo 22.", "Dos")]}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Fake:
    """The API replaced by a fake."""
    api = Fake()
    monkeypatch.setattr(client, "norms", lambda: NORMS)
    monkeypatch.setattr(client, "citations", api.citations)
    monkeypatch.setattr(client, "contracts", api.contracts)
    monkeypatch.setattr(client, "consult", api.consult)
    return api


def page() -> AppTest:
    """The page, freshly opened."""
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=20)
    app.run()
    assert not app.exception
    return app


def button(app: AppTest, label: str) -> Any:
    """The button with that label."""
    return next(b for b in app.button if b.label == label)


def asked(app: AppTest) -> AppTest:
    """Ask a question about the TIC agreement."""
    app.selectbox[0].set_value("tic").run()
    app.text_area[0].set_value("¿Vacaciones?").run()
    button(app, "Preguntar").click().run()
    assert not app.exception
    return app


def texts(app: AppTest) -> str:
    """All the text on the screen."""
    return " ".join(m.value for m in app.markdown)


class TestFormat:
    """The model decorates the titles differently each time."""

    @pytest.mark.parametrize("title", [
        "**Tu convenio**", "### Tu convenio", "«Tu convenio»",
        "Tu convenio:", "Tu convenio"])
    def test_section_titles(self, title: str) -> None:
        """Every form ends up as the same section."""
        assert helpers.split_sections(f"{title}\nTexto.") == [
            ("Tu convenio", "Texto.")]

    def test_title_in_line(self) -> None:
        """A title followed by the text is split."""
        assert helpers.split_sections("Tu convenio: Cuatro días.") == [
            ("Tu convenio", "Cuatro días.")]

    def test_no_section(self) -> None:
        """A sentence that names the agreement is not a title."""
        text = "Según tu convenio, cuatro días."
        assert helpers.split_sections(text) == [("", text)]

    def test_direct_answer_section(self) -> None:
        """The direct answer is recognised as a section."""
        titles = [t for t, _ in helpers.split_sections(
            "Respuesta: Sí, 23 días.\n\nTu convenio: Art.")]
        assert titles == ["Respuesta", "Tu convenio"]

    def test_contract_sections(self) -> None:
        """The sections of a person's answer are recognised."""
        titles = [t for t, _ in helpers.split_sections(
            "Tu contrato: Veinte.\nPosible discrepancia: Menos que 23.")]
        assert titles == ["Tu contrato", "Posible discrepancia"]

    @pytest.mark.parametrize("reference, title, expected", [
        ("art. 22", "Artículo 22. Permisos.", True),
        ("art. 220", "Artículo 22. Permisos.", False),
        ("art. 37.3.b bis", "Artículo 37. Descanso.", True),
        ("disposición transitoria undécima.2",
         "Disposición transitoria undécima. Despido.", True),
        ("anexo II", "ANEXO II Tablas", True),
        ("anexo I", "ANEXO II Tablas", False),
    ])
    def test_refers_to(self, reference: str, title: str,
                       expected: bool) -> None:
        """A reference only points to its own article."""
        assert helpers.refers_to(reference, title) is expected


class TestPills:
    """References become links, and the content stays harmless."""

    GROUPS = [{"kind": "convenio", "norm": "C", "article": "Artículo 22. P.",
               "url": "https://www.boe.es/t", "pieces": ["Uno"]},
              {"kind": "estatuto", "norm": "E", "article": "Artículo 22. E.",
               "url": "https://www.boe.es/e", "pieces": ["Dos"]}]

    def test_pill_replaces_parentheses(self) -> None:
        """``(art. 22)`` becomes a pill without brackets."""
        out = helpers.link_references("Días (art. 22).", self.GROUPS,
                                      "convenio", 0)
        assert out.startswith('Días <a class="cite"')
        assert "(" not in out

    def test_prefers_the_section_kind(self) -> None:
        """In the Statute section, the Statute's article is linked."""
        out = helpers.link_references("(art. 22)", self.GROUPS,
                                      "estatuto", 0)
        assert 'title="Artículo 22. E."' in out

    def test_not_retrieved_has_no_link(self) -> None:
        """A reference with no article behind it is not a link."""
        out = helpers.link_references("(art. 99)", self.GROUPS,
                                      "convenio", 0)
        assert "<a " not in out
        assert '<span class="cite">' in out

    def test_answer_is_escaped(self) -> None:
        """The model cannot inject HTML in the page."""
        out = helpers.link_references("<img src=x onerror=alert(1)>",
                                      self.GROUPS, "convenio", 0)
        assert "<img" not in out

    def test_source_is_escaped(self) -> None:
        """Neither can the text of a document."""
        group = {**self.GROUPS[0], "pieces": ["<script>x</script>"]}
        assert "<script" not in helpers.source_card(group, 0)


class TestPage:
    """What a person sees."""

    def test_needs_convenio_and_question(self, fake: Fake) -> None:
        """Nothing can be asked until both are chosen."""
        app = page()
        assert button(app, "Preguntar").disabled
        app.selectbox[0].set_value("tic").run()
        assert button(app, "Preguntar").disabled

    def test_example_fills_the_question(self, fake: Fake) -> None:
        """An example does not spend anything by itself."""
        app = page()
        next(b for b in app.button if "vacaciones" in b.label).click().run()
        assert app.text_area[0].value
        assert fake.calls == []

    def test_answer_and_sources(self, fake: Fake) -> None:
        """The answer shows with its article, once per article."""
        app = asked(page())
        assert "Respuesta de tic" in texts(app)
        assert texts(app).count('<div class="src"') == 1
        assert "Artículo 22." in texts(app)

    def test_compare_puts_two_side_by_side(self, fake: Fake) -> None:
        """A second agreement spends one more answer, not two."""
        app = asked(page())
        app.selectbox[1].set_value("obra").run()
        button(app, "Comparar").click().run()
        assert not app.exception
        assert fake.calls == ["tic", "obra"]
        assert "Respuesta de tic" in texts(app)
        assert "Respuesta de obra" in texts(app)

    def test_person_uses_their_agreement(self, fake: Fake) -> None:
        """Choosing a person chooses the agreement and sends the contract."""
        app = page()
        app.radio[0].set_value("Por persona (ficticia)").run()
        app.selectbox[0].select_index(0).run()
        app.text_area[0].set_value("¿Vacaciones?").run()
        button(app, "Preguntar").click().run()
        assert not app.exception
        assert fake.calls == ["tic"]
        assert fake.contract == "tic_01"
        assert "Ana Ruiz" in texts(app)

    def test_error_is_readable(self, fake: Fake,
                               monkeypatch: pytest.MonkeyPatch) -> None:
        """A failure shows its message, not a traceback."""
        failing = Fake("Se ha alcanzado el límite")
        monkeypatch.setattr(client, "consult", failing.consult)
        app = asked(page())
        assert "límite" in app.error[0].value

    def test_api_down(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the API the page explains what to do."""
        def down() -> list[dict[str, Any]]:
            raise client.ApiError("¿Has arrancado «make api»?")

        monkeypatch.setattr(client, "norms", down)
        import streamlit as st
        st.cache_data.clear()
        assert "make api" in page().error[0].value
