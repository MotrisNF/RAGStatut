"""Mide el reparto convenio/Estatuto con el indice real. No genera nada.

Desde la raiz del proyecto: python data/laboral/tools/ev3.py
"""
import os
import re
import sys
from pathlib import Path, PurePosixPath

sys.path[:0] = [os.getcwd(), os.path.dirname(__file__)]
from bank import E, Q  # noqa: E402
from src.bm25_index import BM25Index  # noqa: E402
from src.convenio_search import ConvenioSearch  # noqa: E402
from src.hybrid_retriever import HybridRetriever  # noqa: E402
from src.vector_index import VectorIndex  # noqa: E402

RAW = Path("data/laboral/raw")
INDEX = Path("data/laboral/processed")
ART = re.compile(r"^Art[ií]culo (\d+)(?: bis)?\.", re.M)
spans: dict[str, dict[int, tuple[int, int]]] = {}
for path in RAW.glob("*.txt"):
    text = path.read_text(encoding="utf-8")
    found = list(ART.finditer(text))
    arts: dict[int, tuple[int, int]] = {}
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        n = int(m.group(1))
        if n not in arts or end - m.start() > arts[n][1] - arts[n][0]:
            arts[n] = (m.start(), end)
    spans[path.stem] = arts


def hit(chunk, name, articles):
    """Si el chunk pertenece a alguno de los articulos correctos."""
    if PurePosixPath(chunk.file_path).stem != name:
        return False
    first, last = chunk.first_character_index, chunk.last_character_index
    for a in articles:
        s, e = spans[name][a]
        over = min(e, last) - max(s, first)
        if over > 0 and over >= 0.3 * min(last - first, e - s):
            return True
    return False


bm25 = BM25Index.load(INDEX)
retriever = HybridRetriever(bm25, VectorIndex.load(INDEX))
CONVENIOS = [n for n in spans if n.startswith("convenio_")]


def joint(query, k, name):
    """Una sola busqueda conjunta en convenio y Estatuto."""
    return retriever.search(query, k, frozenset({name, E}))


def split(share):
    def run(query, k, name):
        ConvenioSearch.CONVENIO_SHARE = share
        short = name.removeprefix("convenio_")
        return ConvenioSearch.search(retriever, query, k, short)
    return run


def measure(tag, search):
    line = f"{tag:14s}"
    for k in (5, 10):
        own = own_ok = law = law_ok = both = both_ok = 0
        for question, gold in Q.items():
            for name in CONVENIOS:
                hits = search(question, k, name)
                got_own = (any(hit(c.chunk, name, gold[name]) for c in hits)
                           if name in gold else None)
                got_law = (any(hit(c.chunk, E, gold[E]) for c in hits)
                           if E in gold else None)
                if got_own is not None:
                    own += 1
                    own_ok += got_own
                if got_law is not None:
                    law += 1
                    law_ok += got_law
                if got_own is not None and got_law is not None:
                    both += 1
                    both_ok += got_own and got_law
        line += (f" | k={k}: convenio {own_ok / own:.2f}"
                 f" Estatuto {law_ok / law:.2f}"
                 f" ambos {both_ok / both:.2f}")
    print(line + f" (casos: {own}/{law}/{both})", flush=True)


measure("conjunta", joint)
for s in (0.5, 0.6, 0.7, 0.8, 0.9):
    measure(f"reparto {int(s * 100)}/{100 - int(s * 100)}", split(s))
