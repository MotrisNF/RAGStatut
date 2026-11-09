"""MiConvenio: the page where a worker asks about their rights.

Run with ``make web`` (and ``make api`` in another terminal). It is a
normal Python script: Streamlit runs it from top to bottom again every time
the user touches something, and ``st.session_state`` is what survives
between those runs. The look lives in ``style.css``.
"""
import html
from pathlib import Path
from typing import Any

import streamlit as st

from app_helpers import (KINDS, group_citations, link_references,
                         sources_panel, split_sections)
import client

EXAMPLES = [
    "¿Cuántos días de vacaciones me corresponden al año?",
    "¿Cuántos días libres me dan si me caso?",
    "Se ha muerto mi padre, ¿cuántos días de permiso tengo?",
    "¿Cuánto dura el periodo de prueba?",
]
REVIEW = ("Revisa este contrato: ¿alguna condición es peor que lo que dice "
          "el convenio sobre vacaciones, jornada anual o periodo de prueba?")
PERSON_EXAMPLES = [
    REVIEW,
    "¿Cuántos días de vacaciones le corresponden?",
    "¿Cuánto puede durar su periodo de prueba?",
    "¿Cuántas horas al año tiene que trabajar?",
]
BY_NORM = "Por convenio"
BY_PERSON = "Por persona (ficticia)"
STATUTE = "estatuto"
VISIBLE = ("", "Respuesta", "Posible discrepancia")
DISCLAIMER = (
    "MiConvenio es orientativo y no sustituye el asesoramiento legal. "
    "Consulta con tu representación sindical o un profesional antes de "
    "tomar una decisión. Las respuestas se basan en la versión publicada "
    "de cada norma: puede haber modificaciones posteriores (tablas "
    "salariales, acuerdos o sentencias). No escribas datos personales: "
    "las preguntas se envían a un servicio externo."
)
STYLE = Path(__file__).with_name("style.css")


def start_state() -> None:
    """Create what must survive between runs of the script."""
    defaults: dict[str, Any] = {"question": "", "results": [], "error": "",
                                "asked": "", "person": None}
    for name, value in defaults.items():
        st.session_state.setdefault(name, value)


def html_block(markup: str) -> None:
    """Show HTML built by this page; its contents are already escaped."""
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def load_norms() -> list[dict[str, Any]]:
    """The legal texts, remembered for a few minutes."""
    return client.norms()


@st.cache_data(ttl=300, show_spinner=False)
def load_contracts() -> list[dict[str, Any]]:
    """The fictional contracts, remembered for a few minutes."""
    return client.contracts()


def ask(question: str, convenio: str,
        contract: str | None = None) -> dict[str, Any] | None:
    """Ask the API, telling the user what is happening at each step.

    Args:
        question: The question.
        convenio: The chosen legal text.
        contract: Identifier of a fictional contract to apply it to.

    Returns:
        The answer, or ``None`` if something failed (the error is kept
        in ``st.session_state.error``).
    """
    with st.status("Buscando en los textos legales…", expanded=True) as box:
        try:
            found = client.citations(question, convenio)
            total = len({c["article"] for c in found["citations"]})
            st.write(f"He encontrado {total} artículos relacionados.")
            box.update(label="Redactando la respuesta…")
            result = client.consult(question, convenio, contract)
        except client.ApiError as error:
            box.update(label="No se ha podido responder", state="error")
            st.session_state.error = str(error)
            return None
        box.update(label="Respuesta lista", state="complete", expanded=False)
    return result


def name_of(norms: list[dict[str, Any]], identifier: str) -> str:
    """The name shown for a legal text."""
    if identifier == STATUTE:
        return "Solo el Estatuto de los Trabajadores"
    return str(next(n["name"] for n in norms if n["id"] == identifier))


def use_example(text: str) -> None:
    """Put an example in the question box."""
    st.session_state.question = text


def run_question(question: str, convenio: str,
                 person: dict[str, Any] | None) -> None:
    """Ask a new question; it replaces whatever was on the screen."""
    st.session_state.error = ""
    st.session_state.results = []
    st.session_state.asked = question
    st.session_state.person = person
    result = ask(question, convenio, person["id"] if person else None)
    if result is not None:
        st.session_state.results = [result]


def run_comparison(other: str) -> None:
    """Put a second legal text next to the first answer."""
    st.session_state.error = ""
    person = st.session_state.person
    result = ask(st.session_state.asked, other,
                 person["id"] if person else None)
    if result is not None:
        st.session_state.results = [st.session_state.results[0], result]


def show_checks(index: int, checks: list[dict[str, Any]]) -> None:
    """Show the comparison made by the code, not by the model."""
    if not checks:
        return
    marks = {"correcto": ("ok", "Correcto"), "peor": ("bad", "Peor"),
             "depende": ("doubt", "Depende")}
    rows = []
    for check in checks:
        css, label = marks[check["status"]]
        cells = [html.escape(str(check[name]), quote=False)
                 for name in ("field", "agreed", "limit", "article")]
        rows.append(f'<div class="check"><span class="state {css}">{label}'
                    f'</span><span><strong>{cells[0]}:</strong> {cells[1]}'
                    f' · convenio: {cells[2]} ({cells[3]})</span></div>')
    with st.container(key=f"answer_{index}_checks"):
        html_block('<span class="card-label contrato">Comprobación '
                   'automática</span>' + "".join(rows))


def show_section(index: int, number: int, title: str, body: str,
                 groups: list[dict[str, Any]]) -> None:
    """Show one section of an answer as a card."""
    kind = KINDS.get(title, "none")
    prefix = "headline" if title in ("Respuesta", "") else "answer"
    with st.container(key=f"{prefix}_{index}_{number}"):
        html_block(f'<span class="card-label {kind}">{title}</span>')
        html_block(link_references(body, groups, kind, index))


def show_answer(index: int, result: dict[str, Any],
                groups: list[dict[str, Any]]) -> None:
    """Show the sections of an answer as separate cards."""
    if not result["citations"]:
        with st.container(key=f"answer_{index}_empty"):
            st.write("No he encontrado artículos sobre esto en los textos "
                     "consultados.")
        return
    show_checks(index, result.get("checks", []))
    sections = list(enumerate(split_sections(result["answer"])))
    # The model writes its conclusion last, after comparing; it is shown
    # first, because it is what the person came for.
    always = sorted((s for s in sections if s[1][0] in VISIBLE),
                    key=lambda s: VISIBLE.index(s[1][0]))
    detail = [s for s in sections if s[1][0] not in VISIBLE]
    for number, (title, body) in always:
        show_section(index, number, title, body, groups)
    if detail:
        with st.expander("Ver el detalle norma por norma"):
            for number, (title, body) in detail:
                show_section(index, number, title, body, groups)


def show_result(index: int, result: dict[str, Any], norm: dict[str, Any],
                alone: bool) -> None:
    """Show an answer with its legal text and its articles.

    Args:
        index: Position of the answer on the page.
        result: What the API answered.
        norm: The legal text it comes from.
        alone: Whether there is room to put the articles at its side.
    """
    name = html.escape(norm["name"], quote=False)
    validity = html.escape(norm["validity"], quote=False)
    html_block(f'<div class="norm-line"><strong>{name}</strong>'
               f'<span class="chip">{validity}</span></div>')
    groups = group_citations(result["citations"])
    panel = sources_panel(groups, result["answer"], index)
    if alone and panel:
        left, right = st.columns([3, 2], gap="large")
        with left:
            show_answer(index, result, groups)
        with right:
            html_block(panel)
        return
    show_answer(index, result, groups)
    if panel:
        html_block(panel)


def show_comparison_picker(norms: list[dict[str, Any]]) -> None:
    """Offer another agreement to compare the same question with."""
    current = st.session_state.results[0]["convenio"]
    others = [n["id"] for n in norms if n["id"] not in (current, STATUTE)]
    if not others:
        return
    html_block('<div class="compare-title">¿Y en otro sector? '
               '<span>Compara la misma pregunta con otro convenio.</span>'
               '</div>')
    with st.container(key="compare_card"):
        pick, go = st.columns([4, 1], vertical_alignment="center")
        other = pick.selectbox("Otro convenio", others, index=None,
                               placeholder="Elige otro convenio",
                               format_func=lambda i: name_of(norms, i),
                               label_visibility="collapsed")
        if go.button("Comparar", disabled=other is None,
                     width="stretch") and other is not None:
            run_comparison(other)
            st.rerun()


def person_label(person: dict[str, Any], norms: list[dict[str, Any]]) -> str:
    """How a person is shown in the selector."""
    sector = person["convenio"].replace("_", " ")
    return f"{person['name']} · {person['job']} · {sector}"


def show_ask_card(norms: list[dict[str, Any]],
                  contracts: list[dict[str, Any]]) -> None:
    """The card where the question is written."""
    choices = [n["id"] for n in norms if n["id"] != STATUTE] + [STATUTE]
    with st.container(key="ask_card"):
        by_person = (st.radio("Modo", [BY_NORM, BY_PERSON], horizontal=True,
                              key="mode", label_visibility="collapsed")
                     == BY_PERSON)
        st.text_area("¿Qué quieres saber?", key="question", height=90,
                     placeholder="Escribe tu duda, por ejemplo: ¿cuántos "
                                 "días libres me dan si me caso?",
                     label_visibility="collapsed")
        pick, go = st.columns([4, 1], vertical_alignment="center")
        person: dict[str, Any] | None = None
        if by_person:
            person = pick.selectbox(
                "Persona", contracts, index=None,
                placeholder=f"Elige una de las {len(contracts)} personas",
                format_func=lambda p: person_label(p, norms),
                label_visibility="collapsed")
            convenio = person["convenio"] if person else None
        else:
            convenio = pick.selectbox(
                "¿Cuál es tu convenio?", choices, index=None,
                placeholder="Elige tu convenio",
                format_func=lambda i: name_of(norms, i),
                label_visibility="collapsed")
        question = st.session_state.question.strip()
        if go.button("Preguntar", type="primary", width="stretch",
                     disabled=not (question and convenio)) and convenio:
            run_question(question, convenio, person)
            st.rerun()
        if person:
            with st.expander(f"Ver el contrato de {person['name']}"):
                st.text(person["text"])


def show_examples() -> None:
    """Questions that can be tried with one click."""
    html_block('<div class="examples-label">O empieza con una de estas'
               '</div>')
    texts = (PERSON_EXAMPLES if st.session_state.get("mode") == BY_PERSON
             else EXAMPLES)
    for number, (column, text) in enumerate(zip(st.columns(2) * 2, texts)):
        column.button(text, key=f"ex_{number}", on_click=use_example,
                      args=(text,), width="stretch")


def main() -> None:
    """Draw the page."""
    start_state()
    results = st.session_state.results
    st.set_page_config(page_title="MiConvenio", page_icon="⚖️",
                       layout="wide" if results else "centered")
    html_block(f"<style>{STYLE.read_text(encoding='utf-8')}</style>")
    html_block('<div class="brand">MiConvenio</div>')
    if not results:
        html_block('<div class="hero"><h1>¿Qué te toca según '
                   '<em>tu convenio</em>?</h1><p>Pregunta como hablas. Te '
                   'respondo con lo que dicen tu convenio y el Estatuto de '
                   'los Trabajadores, citando cada artículo.</p></div>')
    try:
        norms = load_norms()
    except client.ApiError as error:
        st.error(str(error))
        st.stop()

    try:
        contracts = load_contracts()
    except client.ApiError:
        contracts = []
    show_ask_card(norms, contracts)
    if st.session_state.error:
        st.error(st.session_state.error)
    if not results:
        show_examples()
    else:
        asked = html.escape(st.session_state.asked, quote=False)
        html_block(f'<div class="asked">{asked}</div>')
        person = st.session_state.person
        if person:
            who = html.escape(f"{person['name']}, {person['job']}",
                              quote=False)
            html_block(f'<div class="norm-line"><strong>{who}</strong>'
                       '<span class="chip">Contrato ficticio</span></div>')
        by_id = {n["id"]: n for n in norms}
        alone = len(results) == 1
        for index, (column, result) in enumerate(
                zip(st.columns(len(results), gap="large"), results)):
            with column:
                show_result(index, result, by_id[result["convenio"]], alone)
        show_comparison_picker(norms)
    html_block(f'<div class="foot">{DISCLAIMER}</div>')


main()
