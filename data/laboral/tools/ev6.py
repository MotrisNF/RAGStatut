"""Compara Gemini solo, la busqueda local y su union. Sin llamadas a la API.

Solo usa preguntas con vector ya guardado. Reparto 7/3 convenio/Estatuto.
    python data/laboral/tools/ev6.py
"""
import os
import re
import sys
from pathlib import Path, PurePosixPath

sys.path[:0] = [os.getcwd(), os.path.dirname(__file__)]
from bank import E, Q as DEV  # noqa: E402
from bank2 import Q as NEW  # noqa: E402
from src.commands import RagCLI  # noqa: E402
from src.gemini_retriever import GeminiRetriever  # noqa: E402

INDEX = "data/laboral/processed"
ART = re.compile(r"^Art[ií]culo (\d+)(?: bis)?\.", re.M)
spans = {}
for path in Path("data/laboral/raw").glob("*.txt"):
    text = path.read_text(encoding="utf-8")
    found = list(ART.finditer(text))
    arts = {}
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        n = int(m.group(1))
        if n not in arts or end - m.start() > arts[n][1] - arts[n][0]:
            arts[n] = (m.start(), end)
    spans[path.stem] = arts

gemini = RagCLI._build_retriever(INDEX, "gemini")
assert isinstance(gemini, GeminiRetriever)
hybrid = gemini._fallback
bm25 = hybrid._bm25
COVERED = sorted(s for s in gemini._covered if s.startswith("convenio_"))
saved = gemini._saved


def has_vector(q):
    return GeminiRetriever.key(GeminiRetriever.query_text(q)) in saved


def hit(chunk, name, articles):
    if PurePosixPath(chunk.file_path).stem != name:
        return False
    first, last = chunk.first_character_index, chunk.last_character_index
    for a in articles:
        s, e = spans[name][a]
        over = min(e, last) - max(s, first)
        if over > 0 and over >= 0.3 * min(last - first, e - s):
            return True
    return False


def rrf(*lists):
    """Fusion por posicion: suma 1/(60 + puesto) de cada lista."""
    score, chunk = {}, {}
    for hits in lists:
        for place, h in enumerate(hits):
            score[h.chunk_id] = score.get(h.chunk_id, 0) + 1 / (60 + place)
            chunk[h.chunk_id] = h.chunk
    return [chunk[c] for c in sorted(score, key=lambda c: -score[c])]


STRATEGIES = {
    "Local (BM25+potion)": lambda q, k, s: [h.chunk for h in hybrid.search(q, k, s)],
    "Gemini solo": lambda q, k, s: [h.chunk for h in gemini.search(q, k, s)],
    "Union BM25 + Gemini": lambda q, k, s: rrf(bm25.search(q, 30, s),
                                               gemini.search(q, 30, s))[:k],
    "Union local + Gemini": lambda q, k, s: rrf(hybrid.search(q, 30, s),
                                                gemini.search(q, 30, s))[:k],
}


def split(fn, q, k, name):
    own = fn(q, k, frozenset({name}))
    law = fn(q, k, frozenset({E}))
    own_n = min(-(-k * 7 // 10), len(own))
    law_n = min(k - own_n, len(law))
    own_n = min(k - law_n, len(own))
    return own[:own_n] + law[:law_n]


print("Convenios con vectores:", ", ".join(c[9:] for c in COVERED))
for label, bank in (("BANCO ORIGINAL", DEV), ("PREGUNTAS NUEVAS", NEW)):
    usable = {q: g for q, g in bank.items() if has_vector(q)}
    print(f"\n== {label} ({len(usable)} de {len(bank)} preguntas con vector)")
    for tag, fn in STRATEGIES.items():
        line = f"{tag:22s}"
        for k in (5, 10):
            own = own_ok = law = law_ok = 0
            for q, gold in usable.items():
                for name in COVERED:
                    got = split(fn, q, k, name)
                    if name in gold:
                        own += 1
                        own_ok += any(hit(c, name, gold[name]) for c in got)
                    if E in gold:
                        law += 1
                        law_ok += any(hit(c, E, gold[E]) for c in got)
            line += (f" | k={k}: convenio {own_ok}/{own}"
                     f" Estatuto {law_ok}/{law}")
        print(line, flush=True)
