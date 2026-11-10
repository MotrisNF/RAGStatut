"""Mide el reparto convenio/Estatuto con el indice real. No genera nada.

Desde la raiz del proyecto: python data/labour/tools/measure_convenio_split.py
"""
import os
import re
import sys
from pathlib import Path, PurePosixPath

sys.path[:0] = [os.getcwd(), os.path.dirname(__file__)]
from question_bank import E, Q as DEV  # noqa: E402
from question_bank_new import Q as NEW  # noqa: E402
from src.tokenizer_bm import Tokenizer  # noqa: E402
from src.bm25_index import BM25Index  # noqa: E402
from src.convenio_search import ConvenioSearch  # noqa: E402
from src.hybrid_retriever import HybridRetriever  # noqa: E402
from src.vector_index import VectorIndex  # noqa: E402

RAW = Path("data/labour/raw")
OLD_INDEX, NEW_INDEX = Path(sys.argv[1]), Path(sys.argv[2])
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


Q = DEV
retriever = None
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


def measure(tag, search, ks):
    line = f"{tag:30s}"
    for k in ks:
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
        line += (f" | k={k}: conv {own_ok / own:.2f}"
                 f" Est {law_ok / law:.2f} ambos {both_ok / both:.2f}")
    print(line + f" (n={own}/{law}/{both})", flush=True)



SYN = dict(Tokenizer.SYNONYMS)
for label, bank in (("BANCO ORIGINAL", DEV), ("PREGUNTAS NUEVAS", NEW)):
    Q = bank
    print("==", label)
    for ix, name in ((OLD_INDEX, "chunker antiguo"), (NEW_INDEX, "articulos+titulo")):
        retriever = HybridRetriever(BM25Index.load(ix), VectorIndex.load(ix))
        for syn, tag in (({}, "sin sinonimos"), (SYN, "con sinonimos")):
            Tokenizer.SYNONYMS = syn
            measure(f"{name}, {tag}", split(0.7), (10, 14, 20))
