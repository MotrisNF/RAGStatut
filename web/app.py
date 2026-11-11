"""MiConvenio: the page where a worker asks about their rights.

Run with ``make web`` (and ``make api`` in another terminal). It is a
normal Python script: Streamlit runs it from top to bottom again every time
the user touches something, and ``st.session_state`` is what survives
between those runs. The look lives in ``style.css``.
"""
import html
from pathlib import Path
import queue
import threading
import time
from typing import Any

import streamlit as st

from app_helpers import (KINDS, amendment_notes, estimated_progress,
                         group_citations, link_references, source_card,
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
KEY_PAGE = "https://aistudio.google.com/apikey"
SOURCES, LOCAL = "sources", "local"
WAIT_GEMINI = ("Suele tardar unos segundos; mientras, puedes leer los "
               "artículos.")
WAIT_LOCAL = ("Se está redactando en este ordenador: puede tardar un par "
              "de minutos, y la primera vez algo más porque carga el modelo. "
              "Ten paciencia; mientras, puedes leer los artículos.")
LOCAL_RISKS = (
    "- Puede **inventar cifras o plazos** que no dicen los textos.\n"
    "- Puede **citar el artículo correcto con el apartado equivocado**, "
    "un error difícil de detectar.\n"
    "- Usa un modelo abierto (Qwen3-4B) y tarda **varios minutos** por "
    "respuesta.\n"
    "- Necesita unos **8 GB de memoria libre**; con menos, el ordenador "
    "puede ir muy lento o cerrar el servidor. La primera vez descarga el "
    "modelo (unos 8 GB).\n"
    "- Comprueba siempre la respuesta con los artículos que se muestran.")


def start_state() -> None:
    """Create what must survive between runs of the script."""
    defaults: dict[str, Any] = {"question": "", "results": [], "error": "",
                                "asked": "", "person": None,
                                "answer_mode": None, "more_options": False}
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


def answer_mode(has_key: bool) -> str | None:
    """How answers are written now: Gemini with a key, else the choice."""
    return "gemini" if has_key else st.session_state.answer_mode


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
    mode = st.session_state.get("current_mode") or "gemini"
    try:
        with st.spinner("Buscando en los textos legales…"):
            found = client.citations(question, convenio)
        # A plain line, not a box: the articles themselves are the cards
        # shown below while the answer is written.
        total = len(group_citations(found["citations"]))
        html_block(f'<div class="found-line">✓ He encontrado {total} '
                   'artículos relacionados</div>')
        if mode != SOURCES and found["citations"]:
            # The search ends long before the writing: a card with the
            # progress, and the articles found to read while waiting.
            return ask_with_progress(question, convenio, contract,
                                     found["citations"], mode)
        return dict(client.consult(question, convenio, contract, mode))
    except client.ApiError as error:
        st.session_state.error = str(error)
        return None


STAGES = {
    "loading": "Cargando el modelo en este ordenador (solo la primera vez)…",
    "convenio": "Redactando lo que dice tu convenio (1/3)…",
    "estatuto": "Redactando lo que dice el Estatuto (2/3)…",
    "respuesta": "Resumiendo la respuesta (3/3)…",
    "todo": "Redactando la respuesta…",
    "gemini": "Redactando la respuesta con Gemini…",
}


# Expected seconds of each stage with Qwen3-4B. The bar only estimates:
# it never fills before the stage ends.
STAGE_SECONDS = {"loading": 20, "convenio": 30, "estatuto": 30,
                 "respuesta": 30, "todo": 90, "gemini": 6}


def expected_seconds() -> dict[str, float]:
    """How long each stage usually takes: what the server learned from
    the answers already generated here, or ``STAGE_SECONDS``."""
    try:
        learned = client.status().get("stage_seconds") or {}
    except client.ApiError:
        learned = {}
    return {**STAGE_SECONDS, **{stage: float(seconds)
                                for stage, seconds in learned.items()}}


# How often the bar moves while waiting for the server.
TICK = 0.25


def read_stream(events: "queue.Queue[tuple[str, Any]]", question: str,
                convenio: str, contract: str | None) -> None:
    """Read the local answer in a thread, leaving each event in a queue.

    The page keeps redrawing the bar meanwhile. Nothing here touches
    Streamlit: only the page does.
    """
    try:
        for event in client.consult_stream(question, convenio, contract):
            events.put(("event", event))
    except client.ApiError as error:
        events.put(("error", error))
    events.put(("end", None))


def read_consult(events: "queue.Queue[tuple[str, Any]]", question: str,
                 convenio: str, contract: str | None, mode: str) -> None:
    """Ask Gemini in a thread, as one stage followed by the result."""
    events.put(("event", {"event": "stage", "stage": mode}))
    try:
        result = client.consult(question, convenio, contract, mode)
        events.put(("event", {"event": "done", "result": result}))
    except client.ApiError as error:
        events.put(("error", error))
    events.put(("end", None))


def ask_with_progress(question: str, convenio: str, contract: str | None,
                      citations: list[dict[str, Any]],
                      mode: str) -> dict[str, Any]:
    """Ask the model, with a bar that estimates the progress of each
    stage; a local answer also shows each part as soon as it is written.

    Raises:
        client.ApiError: The API failed; ``ask`` shows it.
    """
    groups = group_citations(citations)
    status = st.container()
    show_found_cards(groups)
    events: "queue.Queue[tuple[str, Any]]" = queue.Queue()
    if mode == LOCAL:
        first, wait = "convenio", WAIT_LOCAL
        threading.Thread(target=read_stream,
                         args=(events, question, convenio, contract),
                         daemon=True).start()
    else:
        first, wait = mode, WAIT_GEMINI
        threading.Thread(target=read_consult,
                         args=(events, question, convenio, contract, mode),
                         daemon=True).start()
    # A plain card, not a status box: a box folds itself on every update,
    # and the progress must stay in sight.
    with status, st.container(key="headline_local"):
        title = st.empty()
        bar = st.progress(0.0)
        st.caption(wait)
        written = st.container()
        result: dict[str, Any] | None = None
        stage, started = first, time.monotonic()
        label = STAGES[stage]
        shown = 0.0
        expected = expected_seconds()
        while True:
            try:
                kind, item = events.get(timeout=TICK)
            except queue.Empty:
                elapsed = time.monotonic() - started
                shown = estimated_progress(elapsed, expected.get(stage, 30))
                bar.progress(shown)
                show_stage(title, label, int(elapsed))
                continue
            if kind == "error":
                raise item
            if kind == "end":
                break
            event = item
            if event["event"] == "stage":
                stage, started = event["stage"], time.monotonic()
                label = STAGES.get(stage, "Redactando la respuesta…")
                shown = 0.0
                bar.progress(shown)
                show_stage(title, label, 0)
            elif event["event"] == "part":
                # The stage is over: the bar fills first, then its text.
                shown = fill_bar(bar, shown)
                kind = KINDS.get(event["section"], "none")
                with written:
                    html_block(f'<span class="card-label {kind}">'
                               f'{html.escape(event["section"])}</span>')
                    html_block(link_references(event["text"], groups, kind,
                                               9))
            elif event["event"] == "done":
                shown = fill_bar(bar, shown)
                result = dict(event["result"])
        if result is None:
            raise client.ApiError("La respuesta local se ha cortado.")
    return result


# The bar fills in this many steps, this many seconds apart, when a
# stage ends: quick, but visible.
FILL_STEPS = 10
FILL_PAUSE = 0.04


def fill_bar(bar: Any, shown: float) -> float:
    """Fill the bar to 100 % quickly, from where it was.

    Args:
        bar: The progress bar.
        shown: Its current value.

    Returns:
        The new value, 1.0.
    """
    for step in range(1, FILL_STEPS + 1):
        bar.progress(shown + (1.0 - shown) * step / FILL_STEPS)
        time.sleep(FILL_PAUSE)
    return 1.0


def show_stage(place: Any, label: str, seconds: int) -> None:
    """The title of the stage being written, with its elapsed time."""
    place.markdown(f'<div class="stage-line"><span class="card-label '
                   f'respuesta">{html.escape(label)}</span>'
                   f'<span class="stage-time">{seconds} s</span></div>',
                   unsafe_allow_html=True)


#: Position given to the cards shown while waiting, apart from the answers.
WAITING = 9


def show_found_cards(groups: list[dict[str, Any]]) -> None:
    """The articles found, ready to be read while the answer is written."""
    html_block('<h5 class="src-h">Mientras tanto, puedes leer los artículos '
               'encontrados</h5>'
               + "".join(source_card(group, WAITING) for group in groups))


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
    st.session_state.pending = convenio
    # The box is emptied on the next run: a widget cannot be changed once
    # drawn in this one. The agreement or person chosen stays.
    st.session_state.clear_question = True


def run_comparison(other: str) -> None:
    """Put a second legal text next to the first answer."""
    st.session_state.error = ""
    st.session_state.results = st.session_state.results[:1]
    st.session_state.pending = other


def busy() -> bool:
    """Whether a question is being answered.

    Every control that could send another question is disabled until it
    ends, so one person never runs two answers at once.
    """
    return bool(st.session_state.get("pending"))


def run_pending() -> None:
    """Answer the question waiting to be asked, below the question card.

    The buttons only leave the request here and rerun the page, so the
    wait is drawn in the results area, not inside the card, with the
    controls disabled. The request is cleared when it ends, even on an
    error, so the page never stays locked.
    """
    convenio = st.session_state.get("pending")
    if not convenio:
        return
    try:
        asked = html.escape(st.session_state.asked, quote=False)
        html_block(f'<div class="asked">{asked}</div>')
        person = st.session_state.person
        result = ask(st.session_state.asked, convenio,
                     person["id"] if person else None)
        if result is not None:
            st.session_state.results = st.session_state.results + [result]
    finally:
        st.session_state.pending = None
    st.rerun()


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
    if not result["answer"]:
        with st.container(key=f"headline_{index}_sources"):
            html_block('<span class="card-label respuesta">Solo '
                       'referencias</span>')
            st.write("Estos son los artículos que responden a tu pregunta. "
                     "Ábrelos para leer lo que dicen; los avisos y la "
                     "comprobación del contrato son exactos.")
        return
    sections = list(enumerate(split_sections(result["answer"])))
    # The model writes its conclusion last, after comparing; it is shown
    # first, because it is what the person came for.
    always = sorted((s for s in sections if s[1][0] in VISIBLE),
                    key=lambda s: VISIBLE.index(s[1][0]))
    detail = [s for s in sections if s[1][0] not in VISIBLE]
    if not always:
        # An answer without the "Respuesta" section (an older one): show
        # it all rather than an empty column.
        always, detail = sections, []
    for number, (title, body) in always:
        show_section(index, number, title, body, groups)
    if detail:
        with st.expander("Ver el detalle norma por norma"):
            for number, (title, body) in detail:
                show_section(index, number, title, body, groups)


def show_result(index: int, result: dict[str, Any], norm: dict[str, Any],
                alone: bool) -> Any:
    """Show an answer with its legal text and its articles.

    Args:
        index: Position of the answer on the page.
        result: What the API answered.
        norm: The legal text it comes from.
        alone: Whether there is room to put the articles at its side.

    Returns:
        The column of the answer, where more can be added below it.
    """
    name = html.escape(norm["name"], quote=False)
    validity = html.escape(norm["validity"], quote=False)
    changed = norm.get("amendments") or []
    warning = (f'<span class="chip warn">⚠️ {len(changed)} modificaciones '
               'posteriores</span>' if changed else "")
    html_block(f'<div class="norm-line"><strong>{name}</strong>'
               f'<span class="chip">{validity}</span>{warning}</div>')
    if changed:
        reviewed = "/".join(reversed(norm.get("reviewed", "").split("-")))
        with st.expander("Ver las modificaciones publicadas en el BOE"):
            html_block(amendment_notes(changed))
            st.caption(f"Revisado el {reviewed}. El texto que consulta "
                       "MiConvenio es el original; estas publicaciones no "
                       "están incluidas en la búsqueda.")
    groups = group_citations(result["citations"])
    panel = sources_panel(groups, result["answer"], index)
    if alone and panel:
        left, right = st.columns([3, 2], gap="large")
        with left:
            show_answer(index, result, groups)
        with right:
            html_block(panel)
        return left
    show_answer(index, result, groups)
    if panel:
        html_block(panel)
    return st.container()


def show_comparison_picker(norms: list[dict[str, Any]]) -> None:
    """Offer another agreement to compare the same question with."""
    current = st.session_state.results[0]["convenio"]
    others = [n["id"] for n in norms if n["id"] not in (current, STATUTE)]
    if not others:
        return
    with st.expander("¿Y en otro sector? Compara con otro convenio"), \
            st.container(key="compare_card"):
        pick, go = st.columns([4, 1], vertical_alignment="center")
        other = pick.selectbox("Otro convenio", others, index=None,
                               placeholder="Elige otro convenio",
                               format_func=lambda i: name_of(norms, i),
                               label_visibility="collapsed",
                               disabled=busy())
        if go.button("Comparar", disabled=other is None or busy(),
                     width="stretch") and other is not None:
            run_comparison(other)
            st.rerun()


def person_label(person: dict[str, Any], norms: list[dict[str, Any]]) -> str:
    """How a person is shown in the selector."""
    sector = person["convenio"].replace("_", " ")
    return f"{person['name']} · {person['job']} · {sector}"


def save_key() -> None:
    """Send the typed key to the server, then forget it here."""
    key = st.session_state.get("typed_key", "").strip()
    if not key:
        return
    try:
        client.set_key(key)
        st.session_state.key_error = ""
    except client.ApiError as error:
        st.session_state.key_error = str(error)
    st.session_state.typed_key = ""


def choose(mode: str | None) -> None:
    """Remember how to answer without a key."""
    st.session_state.answer_mode = mode
    st.session_state.more_options = False


def show_key_card() -> None:
    """Without a key: how to get one, where to paste it, and the rest."""
    with st.container(key="key_card"):
        html_block('<span class="card-label respuesta">Falta la clave de '
                   'Gemini</span>')
        st.markdown("MiConvenio redacta las respuestas con Gemini, de "
                    "Google. La clave es **gratuita** y se consigue en un "
                    f"minuto con una cuenta de Google: [créala aquí]"
                    f"({KEY_PAGE}).")
        paste, save = st.columns([4, 1], vertical_alignment="bottom")
        paste.text_input("Pega tu clave", type="password", key="typed_key",
                         placeholder="Pega aquí tu clave de Gemini",
                         label_visibility="collapsed", disabled=busy())
        save.button("Guardar clave", on_click=save_key, width="stretch",
                    disabled=busy())
        if st.session_state.get("key_error"):
            st.error(st.session_state.key_error)
        st.caption("La clave solo se guarda en la memoria del servidor de "
                   "este ordenador mientras esté en marcha.")
        if st.button("Probar otras opciones", key="more", disabled=busy()):
            st.session_state.more_options = True
    if st.session_state.more_options:
        show_other_options()


def show_other_options() -> None:
    """The two ways to go on without a key."""
    sources, local = st.columns(2, gap="medium")
    with sources.container(key="option_sources"):
        html_block('<span class="card-label contrato">Recomendado</span>')
        st.markdown("**Solo referencias**")
        st.write("Te muestro los artículos que responden a tu pregunta, con "
                 "sus avisos y la comprobación del contrato, sin redactar "
                 "nada. No puede equivocarse al resumir.")
        st.button("Usar solo referencias", on_click=choose, args=(SOURCES,),
                  width="stretch", type="primary", disabled=busy())
    with local.container(key="option_local"):
        html_block('<span class="card-label discrepancia">Experimental'
                   '</span>')
        st.markdown("**Generar en este ordenador**")
        st.markdown(LOCAL_RISKS)
        st.button("Generar en local", on_click=choose, args=(LOCAL,),
                  width="stretch", disabled=busy())


def show_mode_line(mode: str) -> None:
    """Which mode is on, and a way to change it."""
    names = {SOURCES: "solo referencias", LOCAL: "generación en local "
             "(experimental)"}
    text, change = st.columns([5, 1], vertical_alignment="center")
    text.caption(f"Modo sin clave: **{names[mode]}**.")
    change.button("Cambiar", key="change_mode", on_click=choose,
                  args=(None,), disabled=busy())


def show_ask_card(norms: list[dict[str, Any]],
                  contracts: list[dict[str, Any]]) -> None:
    """The card where the question is written."""
    choices = [n["id"] for n in norms if n["id"] != STATUTE] + [STATUTE]
    if st.session_state.pop("clear_question", False):
        st.session_state.question = ""
    with st.container(key="ask_card"):
        by_person = (st.radio("Modo", [BY_NORM, BY_PERSON], horizontal=True,
                              key="mode", label_visibility="collapsed",
                              disabled=busy())
                     == BY_PERSON)
        st.text_area("¿Qué quieres saber?", key="question", height=90,
                     placeholder="Escribe tu duda, por ejemplo: ¿cuántos "
                                 "días libres me dan si me caso?",
                     label_visibility="collapsed", disabled=busy())
        pick, go = st.columns([4, 1], vertical_alignment="center")
        person: dict[str, Any] | None = None
        if by_person:
            person = pick.selectbox(
                "Persona", contracts, index=None, key="pick_person",
                placeholder=f"Elige una de las {len(contracts)} personas",
                format_func=lambda p: person_label(p, norms),
                label_visibility="collapsed", disabled=busy())
            convenio = person["convenio"] if person else None
        else:
            convenio = pick.selectbox(
                "¿Cuál es tu convenio?", choices, index=None,
                key="pick_convenio",
                placeholder="Elige tu convenio",
                format_func=lambda i: name_of(norms, i),
                label_visibility="collapsed", disabled=busy())
        question = st.session_state.question.strip()
        ready = st.session_state.get("current_mode") is not None
        label = "Espera…" if busy() else "Preguntar"
        if go.button(label, type="primary", width="stretch",
                     disabled=busy() or not (question and convenio and ready)
                     ) and convenio:
            run_question(question, convenio, person)
            st.rerun()
        if person:
            with st.expander(f"Ver el contrato de {person['name']}"):
                st.text(person["text"])


def show_question_area(mode: str | None, norms: list[dict[str, Any]],
                       contracts: list[dict[str, Any]]) -> None:
    """The key or mode line, then the card where the question is written."""
    if mode is None:
        show_key_card()
    elif mode != "gemini":
        show_mode_line(mode)
    show_ask_card(norms, contracts)


def show_examples() -> None:
    """Questions that can be tried with one click."""
    html_block('<div class="examples-label">O empieza con una de estas'
               '</div>')
    texts = (PERSON_EXAMPLES if st.session_state.get("mode") == BY_PERSON
             else EXAMPLES)
    for number, (column, text) in enumerate(zip(st.columns(2) * 2, texts)):
        column.button(text, key=f"ex_{number}", on_click=use_example,
                      args=(text,), width="stretch", disabled=busy())


def main() -> None:
    """Draw the page."""
    start_state()
    results = st.session_state.results
    # Once a question is asked, the answer goes first and the card for the
    # next question moves to the end of the page.
    answering = bool(results) or busy()
    st.set_page_config(page_title="MiConvenio", page_icon="⚖️",
                       layout="wide" if results else "centered")
    html_block(f"<style>{STYLE.read_text(encoding='utf-8')}</style>")
    html_block('<div class="brand">MiConvenio</div>')
    if not answering:
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
    try:
        has_key = bool(client.status().get("gemini_key"))
    except client.ApiError:
        has_key = False
    mode = answer_mode(has_key)
    st.session_state.current_mode = mode
    if not answering:
        show_question_area(mode, norms, contracts)
        if st.session_state.error:
            st.error(st.session_state.error)
        show_examples()
    else:
        run_pending()
        if st.session_state.error:
            st.error(st.session_state.error)
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
        answer_column = None
        for index, (column, result) in enumerate(
                zip(st.columns(len(results), gap="large"), results)):
            with column:
                answer_column = show_result(index, result,
                                            by_id[result["convenio"]], alone)
        # Folded under the answer, like its detail: the page stays clean.
        # With two answers side by side, under both of them.
        place = answer_column if alone and answer_column else st.container()
        with place:
            if alone:
                show_comparison_picker(norms)
            with st.expander("¿Tienes otra pregunta?"):
                show_question_area(mode, norms, contracts)
    html_block(f'<div class="foot">{DISCLAIMER}</div>')


main()
