"""Mini banco: genera 10 respuestas con Gemini y las guarda.

Una peticion cada 6 s, se detiene al primer error, sin reintentos.
Desde la raiz del proyecto: python data/laboral/tools/mini_bank.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())
from src.answer_cache import AnswerCache  # noqa: E402
from src.commands import RagCLI  # noqa: E402
from src.convenio_search import ConvenioSearch  # noqa: E402
from src.exceptions import RagError  # noqa: E402

INDEX = "data/laboral/processed"
QUESTIONS = [
    ("tic", "¿Cuántos días de vacaciones me corresponden al año?"),
    ("contact_center", "Se ha muerto mi padre, ¿cuántos días de permiso tengo?"),
    ("grandes_almacenes", "¿Cuánto dura el periodo de prueba?"),
    ("contact_center", "¿Cuánto cobro de más por trabajar de noche?"),
    ("tic", "Me han despedido y el juez dice que es improcedente, "
            "¿qué indemnización me toca?"),
    ("tic", "¿Puedo trabajar desde casa? ¿qué dice sobre el teletrabajo?"),
    ("tic", "Si me voy de la empresa, ¿con cuánto tiempo tengo que avisar?"),
    ("grandes_almacenes", "¿Cuántas horas tengo que trabajar al año?"),
    ("tic", "¿Cuántos días de permiso me dan por el cumpleaños de mi perro?"),
    ("tic", "¿Cuál es la capital de Francia?"),
]

ONLY = {int(n) for n in sys.argv[1].split(",")} if len(sys.argv) > 1 else None
retriever = RagCLI._build_retriever(INDEX, "gemini")
generator = RagCLI._gemini_generator(INDEX)
cache = RagCLI._answer_cache(INDEX, "gemini")
results = []
for number, (convenio, question) in enumerate(QUESTIONS, 1):
    if ONLY is not None and number not in ONLY:
        continue
    hits = ConvenioSearch.search(retriever, question, 10, convenio)
    chunks = [hit.chunk for hit in hits]
    key = AnswerCache.key(question, chunks)
    cached = cache.get(key)
    start = time.time()
    if cached is None:
        try:
            answer = generator.generate(question, chunks)
        except RagError as e:
            print(f"PARADA en la pregunta {number}: {e}")
            break
        cache.put(key, answer)
        cache.flush()
    else:
        answer = cached
    seconds = time.time() - start
    results.append({
        "convenio": convenio, "question": question, "answer": answer,
        "seconds": round(seconds, 1),
        "articles": [f"{c.file_path.rsplit('/', 1)[-1][:-4]}: {c.title}"
                     for c in chunks]})
    print(f"\n[{number}] ({convenio}, {seconds:.1f}s) {question}\n{answer}")
    print("  fuentes:", sorted({a.split(': ', 1)[1][:32]
                                for a in results[-1]["articles"]}))
    sys.stdout.flush()
    if cached is None and number < len(QUESTIONS):
        time.sleep(6)
out = Path(f"data/laboral/generation/mini_bank_{generator.model}.json")
out.write_text(json.dumps(results, ensure_ascii=False, indent=1),
               encoding="utf-8")
print(f"\nGuardadas {len(results)} respuestas en {out}")
