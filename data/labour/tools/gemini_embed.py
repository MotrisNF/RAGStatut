"""Embebe con Gemini sin pasarse NUNCA de la cuota gratuita.

La API no informa de la cuota restante, asi que los limites se aplican aqui:
cada peticion se apunta en un registro en disco ANTES de enviarla, y el primer
error detiene todo sin reintentar.

Desde la raiz del proyecto:
    python data/labour/tools/gemini_embed.py <index_dir> [--run]
Sin --run es un ensayo: no hace ninguna llamada.
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path, PurePosixPath
from zoneinfo import ZoneInfo

sys.path[:0] = [os.getcwd(), os.path.dirname(__file__)]
from question_bank import E, T, Q as DEV  # noqa: E402
from question_bank_new import Q as NEW  # noqa: E402
from src.bm25_index import BM25Index  # noqa: E402

MODEL = "gemini-embedding-2"
URL = ("https://generativelanguage.googleapis.com/v1beta/models/"
       f"{MODEL}:embedContent")
DIMENSIONS = 768
DAY_CAP = 830          # limite real: 1000 al dia
MINUTE_REQUESTS = 40   # limite real: 100 por minuto
MINUTE_TOKENS = 15000  # limite real: 30000 por minuto
CHARS_PER_TOKEN = 3    # estimacion a la baja: cuenta de mas tokens

OUT = Path("data/labour/embeddings")
LEDGER = OUT / "ledger.jsonl"
VECTORS = OUT / "vectors.jsonl"
PACIFIC = ZoneInfo("America/Los_Angeles")


def key_of(text):
    return hashlib.sha256(f"{MODEL}|{DIMENSIONS}|{text}".encode()).hexdigest()


def tokens_of(text):
    return len(text) // CHARS_PER_TOKEN + 1


def today():
    return datetime.now(PACIFIC).strftime("%Y-%m-%d")


def used_today():
    """Peticiones registradas hoy (el dia de la cuota es el del Pacifico)."""
    if not LEDGER.exists():
        return 0
    with LEDGER.open(encoding="utf-8") as f:
        return sum(json.loads(line)["day"] == today() for line in f)


def doc_text(chunk):
    return f"title: {chunk.title or 'none'} | text: {chunk.text}"


def query_text(question):
    return f"task: search result | query: {question}"


def pending_texts(index_dir):
    chunks = BM25Index.load(Path(index_dir)).chunks
    wanted = {E, T}
    for i, arg in enumerate(sys.argv):
        if arg == "--add":
            wanted.add(sys.argv[i + 1])
    questions = [q for bank in (DEV, NEW) for q, gold in bank.items()
                 if wanted & set(gold)]
    texts = [query_text(q) for q in questions]
    texts += [doc_text(c) for c in chunks
              if PurePosixPath(c.file_path).stem in wanted]
    done = set()
    if VECTORS.exists():
        with VECTORS.open(encoding="utf-8") as f:
            done = {json.loads(line)["key"] for line in f}
    return [t for t in dict.fromkeys(texts) if key_of(t) not in done]


def main():
    texts = pending_texts(sys.argv[1])
    tokens = sum(map(tokens_of, texts))
    used = used_today()
    print(f"Pendientes: {len(texts)} textos, ~{tokens} tokens estimados")
    print(f"Registradas hoy: {used}; tope diario propio: {DAY_CAP}")
    minutes = max(len(texts) / MINUTE_REQUESTS, tokens / MINUTE_TOKENS)
    print(f"Tiempo minimo: ~{minutes:.0f} min")
    if used + len(texts) > DAY_CAP:
        print("NO CABE en el tope de hoy: no se envia nada.")
        return 2
    if "--run" not in sys.argv:
        print("Ensayo: ninguna llamada realizada.")
        return 0

    api_key = os.environ["GEMINI_API_KEY"]
    OUT.mkdir(parents=True, exist_ok=True)
    window = deque()
    sent = 0
    for text in texts:
        cost = tokens_of(text)
        while True:
            now = time.time()
            while window and now - window[0][0] >= 62:
                window.popleft()
            if (len(window) < MINUTE_REQUESTS
                    and sum(c for _, c in window) + cost <= MINUTE_TOKENS):
                break
            time.sleep(0.5)
        if used_today() >= DAY_CAP:
            print("Tope diario alcanzado: parada.")
            return 3

        # Se apunta antes de enviar: una peticion fallida tambien cuenta.
        with LEDGER.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"day": today(), "t": time.time(),
                                "tokens": cost}) + "\n")
        window.append((time.time(), cost))
        body = json.dumps({
            "model": f"models/{MODEL}",
            "content": {"parts": [{"text": text}]},
            "output_dimensionality": DIMENSIONS}).encode()
        request = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json",
                                     "x-goog-api-key": api_key})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                values = json.load(response)["embedding"]["values"]
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: parada sin reintentar. "
                  f"Enviadas: {sent}. {e.read()[:300]!r}")
            return 4
        except (urllib.error.URLError, TimeoutError, KeyError) as e:
            print(f"Error {type(e).__name__}: parada sin reintentar. "
                  f"Enviadas: {sent}.")
            return 5
        with VECTORS.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key_of(text), "v": values}) + "\n")
        sent += 1
        if sent % 50 == 0:
            print(f"{sent}/{len(texts)}", flush=True)
    print(f"Terminado. Enviadas: {sent}. Hoy: {used_today()}/{DAY_CAP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
