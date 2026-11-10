"""Compara Gemini Embedding con la busqueda actual. No llama a ninguna API.

Solo con vectores ya guardados. Corpus: convenio TIC + Estatuto, reparto 7/3.
    python data/labour/tools/measure_gemini_embeddings.py <index_dir>
"""
import json
import os
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np

sys.path[:0] = [os.getcwd(), os.path.dirname(__file__)]
from gemini_embed import VECTORS, doc_text, key_of, query_text  # noqa: E402
from question_bank import E, T, Q as DEV  # noqa: E402
from question_bank_new import Q as NEW  # noqa: E402
from src.bm25_index import BM25Index  # noqa: E402
from src.hybrid_retriever import HybridRetriever  # noqa: E402
from src.tokenizer_bm import Tokenizer  # noqa: E402
from src.vector_index import VectorIndex  # noqa: E402

INDEX = Path(sys.argv[1])
ART = re.compile(r"^Art[ií]culo (\d+)(?: bis)?\.", re.M)
spans = {}
for path in Path("data/labour/raw").glob("*.txt"):
    text = path.read_text(encoding="utf-8")
    found = list(ART.finditer(text))
    arts = {}
    for i, m in enumerate(found):
        end = found[i + 1].start() if i + 1 < len(found) else len(text)
        n = int(m.group(1))
        if n not in arts or end - m.start() > arts[n][1] - arts[n][0]:
            arts[n] = (m.start(), end)
    spans[path.stem] = arts

saved = {}
with VECTORS.open(encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        saved[row["key"]] = np.array(row["v"], dtype=np.float32)

bm25 = BM25Index.load(INDEX)
hybrid = HybridRetriever(bm25, VectorIndex.load(INDEX))
stem = lambda c: PurePosixPath(c.file_path).stem  # noqa: E731
by_source = {}
for chunk_id, chunk in enumerate(bm25.chunks):
    if stem(chunk) in (E, T):
        by_source.setdefault(stem(chunk), []).append((chunk_id, chunk))
matrix = {s: np.stack([saved[key_of(doc_text(c))] for _, c in items])
          for s, items in by_source.items()}


def vector_of(q):
    v = saved[key_of(query_text(q))]
    return v / np.linalg.norm(v)


def gemini_scores(q, source):
    return matrix[source] @ vector_of(q)


def only_bm25(q, k, source):
    return [h.chunk for h in bm25.search(q, k, frozenset({source}))]


def current(q, k, source):
    return [h.chunk for h in hybrid.search(q, k, frozenset({source}))]


def only_gemini(q, k, source):
    order = np.argsort(-gemini_scores(q, source))[:k]
    return [by_source[source][i][1] for i in order]


def rerank(q, k, source):
    """Como el hibrido actual, con Gemini en lugar de potion."""
    hits = bm25.search(q, max(30, k), frozenset({source}))
    if not hits:
        return []
    position = {cid: i for i, (cid, _) in enumerate(by_source[source])}
    sims = gemini_scores(q, source)
    top = hits[0].score
    scored = [(0.5 * h.score / top + 0.5 * float(sims[position[h.chunk_id]]),
               h.chunk) for h in hits]
    scored.sort(key=lambda t: -t[0])
    return [c for _, c in scored[:k]]


def union(q, k, source):
    """Fusion por posicion: BM25 y Gemini aportan candidatos."""
    ranks = {}
    chunks = {}
    for i, h in enumerate(bm25.search(q, 30, frozenset({source}))):
        ranks[h.chunk_id] = ranks.get(h.chunk_id, 0) + 1 / (60 + i)
        chunks[h.chunk_id] = h.chunk
    for i, j in enumerate(np.argsort(-gemini_scores(q, source))[:30]):
        cid, chunk = by_source[source][j]
        ranks[cid] = ranks.get(cid, 0) + 1 / (60 + i)
        chunks[cid] = chunk
    best = sorted(ranks, key=lambda c: -ranks[c])[:k]
    return [chunks[c] for c in best]


def split(fn, q, k):
    own, law = fn(q, k, T), fn(q, k, E)
    own_n = min(-(-k * 7 // 10), len(own))
    law_n = min(k - own_n, len(law))
    own_n = min(k - law_n, len(own))
    return own[:own_n] + law[:law_n]


def hit(chunk, name, articles):
    if stem(chunk) != name:
        return False
    first, last = chunk.first_character_index, chunk.last_character_index
    for a in articles:
        s, e = spans[name][a]
        over = min(e, last) - max(s, first)
        if over > 0 and over >= 0.3 * min(last - first, e - s):
            return True
    return False


def measure(tag, fn, bank):
    line = f"{tag:30s}"
    for k in (5, 10):
        own = own_ok = law = law_ok = 0
        for q, gold in bank.items():
            if not ({E, T} & set(gold)):
                continue
            got = split(fn, q, k)
            if T in gold:
                own += 1
                own_ok += any(hit(c, T, gold[T]) for c in got)
            if E in gold:
                law += 1
                law_ok += any(hit(c, E, gold[E]) for c in got)
        line += f" | k={k}: TIC {own_ok}/{own} Estatuto {law_ok}/{law}"
    print(line, flush=True)


SYN = dict(Tokenizer.SYNONYMS)
for label, bank in (("BANCO ORIGINAL", DEV), ("PREGUNTAS NUEVAS", NEW)):
    print("==", label)
    Tokenizer.SYNONYMS = {}
    measure("BM25+potion, sin sinonimos", current, bank)
    measure("BM25 solo, sin sinonimos", only_bm25, bank)
    measure("Gemini solo", only_gemini, bank)
    measure("BM25+Gemini rerank, sin sin.", rerank, bank)
    measure("Union BM25+Gemini, sin sin.", union, bank)
    Tokenizer.SYNONYMS = SYN
    measure("BM25+potion, con sinonimos", current, bank)
    measure("BM25+Gemini rerank, con sin.", rerank, bank)
    measure("Union BM25+Gemini, con sin.", union, bank)
