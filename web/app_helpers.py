"""Text and HTML helpers of the page, kept apart to test them alone.

Everything that comes from the model or the corpus is escaped before it is
turned into HTML, so it can only ever be shown as text.
"""
import html
import math
import re
from typing import Any

# A line that is only a section title, however the model decorated it:
# "**Tu convenio**", "### Tu convenio", "«Tu convenio»", "Tu convenio:".
SECTION = re.compile(
    r"^[\s#*«»\"]*(Respuesta|Tu contrato|Tu convenio"
    r"|Estatuto de los Trabajadores|Posible discrepancia)[\s*«»\":]*$",
    re.MULTILINE)
# The same title with the text following it on the same line.
SECTION_INLINE = re.compile(
    r"^[\s#*«»\"]*(Respuesta|Tu contrato|Tu convenio"
    r"|Estatuto de los Trabajadores|Posible discrepancia)[*«»\"]*:\s*",
    re.MULTILINE)
REFERENCE = re.compile(
    r"\bart\.\s*\d+(?:\.\d+)*(?:\.?[a-z](?![a-záéíóúñ]))?(?:\s+bis\b)?"
    r"|\bdisposici[oó]n\s+(?:adicional|transitoria|derogatoria|final)"
    r"(?:\s+[a-záéíóúñ]+)?(?:\.\d+)?"
    r"|\banexo\s+[IVXLC\d]+\b", re.IGNORECASE)
KINDS = {"Respuesta": "respuesta", "Tu contrato": "contrato",
         "Tu convenio": "convenio",
         "Estatuto de los Trabajadores": "estatuto",
         "Posible discrepancia": "discrepancia"}


def format_answer(text: str) -> str:
    """Give the sections of an answer the same look every time.

    Args:
        text: The answer, as the model wrote it.

    Returns:
        The answer as Markdown, with each section title on its own line.
    """
    text = SECTION.sub(lambda m: f"\n#### {m[1]}\n", text)
    text = SECTION_INLINE.sub(lambda m: f"\n#### {m[1]}\n\n", text)
    return text.strip()


def split_sections(text: str) -> list[tuple[str, str]]:
    """Cut an answer into its sections.

    Args:
        text: The answer, as the model wrote it.

    Returns:
        ``(title, body)`` for each section, in order. The text with no
        section, like "No aparece...", has an empty title.
    """
    pieces = re.split(r"^#### (.+)$", format_answer(text), flags=re.MULTILINE)
    sections = [("", pieces[0].strip())] if pieces[0].strip() else []
    for title, body in zip(pieces[1::2], pieces[2::2]):
        sections.append((title, body.strip()))
    return sections


def group_citations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join the pieces of each article, keeping their order.

    Args:
        items: The citations of an answer.

    Returns:
        One entry per article, with all its text together.
    """
    groups: dict[tuple[str, str], dict[str, Any]] = {}
    for item in items:
        key = (item["norm"], item["article"])
        group = groups.setdefault(key, {**item, "pieces": []})
        if item["text"] not in group["pieces"]:
            group["pieces"].append(item["text"])
    return list(groups.values())


def refers_to(reference: str, title: str) -> bool:
    """Whether a reference of the answer points to an article's title.

    Args:
        reference: As the answer wrote it ("art. 37.3.b bis").
        title: Title of the article ("Artículo 37. Descanso...").

    Returns:
        True if it is the same article, disposition or annex.
    """
    number = re.match(r"art\.\s*(\d+)", reference, re.IGNORECASE)
    if number:
        return re.match(rf"Art[ií]culo {number[1]}\b", title) is not None
    base = re.sub(r"\.\d+$", "", reference.lower().strip())
    base = re.sub(r"\s+", " ", base)
    return title.lower().startswith(base) and (
        len(title) == len(base) or not title[len(base)].isalnum())


def is_cited(group: dict[str, Any], answer: str) -> bool:
    """Whether the answer refers to this article, disposition or annex."""
    return any(refers_to(found[0], group["article"])
               for found in REFERENCE.finditer(answer))


def order_groups(groups: list[dict[str, Any]], answer: str
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate the articles the answer cites from the rest."""
    cited = [group for group in groups if is_cited(group, answer)]
    others = [group for group in groups if not is_cited(group, answer)]
    return cited, others


def source_id(result_index: int, group: dict[str, Any]) -> str:
    """Identifier of an article's card on the page."""
    digest = abs(hash((group["norm"], group["article"]))) % 10**8
    return f"src-{result_index}-{digest}"


def link_references(body: str, groups: list[dict[str, Any]], kind: str,
                    result_index: int) -> str:
    """Escape a section and turn its references into clickable pills.

    Args:
        body: Markdown of the section.
        groups: The articles of the answer.
        kind: ``convenio`` or ``estatuto``: its articles are tried first.
        result_index: Which answer of the page this is.

    Returns:
        Markdown with HTML for ``st.markdown``. A reference to something
        that was not retrieved stays as a grey pill without a link.
    """
    ordered = sorted(groups, key=lambda g: g["kind"] != kind)

    def pill(found: re.Match[str]) -> str:
        label = found[0]
        target = next((g for g in ordered if refers_to(label, g["article"])),
                      None)
        if target is None:
            return f'<span class="cite">{label}</span>'
        title = html.escape(target["article"], quote=True)
        return (f'<a class="cite" href="#{source_id(result_index, target)}" '
                f'title="{title}">{label}</a>')

    escaped = html.escape(body, quote=False).replace("$", "\\$")
    linked = REFERENCE.sub(pill, escaped)
    return re.sub(r"\(((?:<a class=\"cite\"|<span class=\"cite\")[^()]*?"
                  r"</(?:a|span)>)\)", r"\1", linked)


def source_card(group: dict[str, Any], result_index: int) -> str:
    """HTML of one article, closed until the user opens it.

    It opens with the CSS rule ``:target``: following a link to its id,
    from its own title or from a pill of the answer, shows its text.
    """
    origin = "Estatuto" if group["kind"] == "estatuto" else "Convenio"
    title = html.escape(group["article"] or group["norm"], quote=False)
    paragraphs = "<br><br>".join(
        html.escape(piece.strip(), quote=False).replace("\n\n", "<br><br>")
        .replace("\n", " ") for piece in group["pieces"])
    link = ""
    if group["url"]:
        url = html.escape(group["url"], quote=True)
        link = (f'<a class="boe" href="{url}" target="_blank" '
                'rel="noopener">Ver en el BOE ↗</a>')
    card = source_id(result_index, group)
    changed = group.get("amendments") or []
    mark = '<span class="tag warn">Modificado</span>' if changed else ""
    return (f'<div class="src" id="{card}">'
            f'<a class="src-head" href="#{card}">'
            f'<span class="tag {group["kind"]}">{origin}</span>'
            f'<span>{title}</span>{mark}</a>'
            f'<div class="src-open">{amendment_notes(changed)}'
            f'<div class="src-body">{paragraphs.replace("$", "&#36;")}</div>'
            f'{link}<a class="src-close" href="#cerrar">Cerrar</a>'
            f'</div></div>')


def amendment_notes(items: list[dict[str, Any]]) -> str:
    """HTML of the later publications that changed a text."""
    rows = []
    for item in items:
        date = "/".join(reversed(item["date"].split("-")))
        url = html.escape(item["url"], quote=True)
        summary = html.escape(item["summary"], quote=False)
        rows.append(f'<div class="amend">⚠️ <strong>{date}</strong> · '
                    f'{summary} <a href="{url}" target="_blank" '
                    f'rel="noopener">BOE ↗</a></div>')
    return "".join(rows)


def sources_panel(groups: list[dict[str, Any]], answer: str,
                  result_index: int) -> str:
    """HTML of every article: first the ones the answer cites."""
    cited, others = order_groups(groups, answer)
    parts: list[str] = []
    if cited:
        parts.append('<h5 class="src-h">Artículos citados</h5>')
        parts += [source_card(group, result_index) for group in cited]
    if others:
        parts.append('<h5 class="src-h more">También consultados</h5>')
        parts += [source_card(group, result_index) for group in others]
    return "".join(parts)


def estimated_progress(elapsed: float, expected: float) -> float:
    """How full the bar of a stage is, from its elapsed time.

    It reaches 90 % at the expected time and then keeps creeping towards
    99 %, so a slow stage never looks stuck at 100 %.

    Args:
        elapsed: Seconds since the stage started.
        expected: Seconds the stage usually takes.

    Returns:
        A value between 0 and 0.99. With no usable expected time, 30 s.
    """
    if expected <= 0:
        expected = 30.0
    if elapsed <= expected:
        return 0.9 * elapsed / expected
    return 0.9 + 0.09 * (1 - math.exp(-(elapsed - expected) / expected))
