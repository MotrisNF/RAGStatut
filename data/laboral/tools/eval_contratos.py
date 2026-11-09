"""Mide si Gemini detecta las discrepancias de los contratos ficticios.

Sin --run es un ensayo: construye las peticiones, comprueba que las respuestas
de _verdad.json no se filtran y NO llama a ninguna API de generacion.
    python data/laboral/tools/eval_contratos.py
    python data/laboral/tools/eval_contratos.py --run   (gasta 72 generaciones)
"""
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path[:0] = [os.getcwd(), "web"]
from app_helpers import split_sections  # noqa: E402
from src.commands import RagCLI  # noqa: E402
from src.contract_rules import check_contract, describe  # noqa: E402
from src.contracts import ContractStore  # noqa: E402
from src.convenio_search import ConvenioSearch  # noqa: E402
from src.exceptions import RagError  # noqa: E402

INDEX = "data/laboral/processed"
FOLDER = Path("data/laboral/contratos")
QUESTION = ("Revisa este contrato: ¿alguna condición es peor que lo que dice "
            "el convenio sobre vacaciones, jornada anual o periodo de prueba?")
FIELDS = {"vacaciones": r"vacaciones", "jornada": r"jornada|horas",
          "prueba": r"prueba"}
# Marcas que solo existen en _verdad.json, no en leyes ni en el prompt.
LEAKS = ('"campo"', "prueba_dias", "_verdad", '"limites"', '"pactado"')


def detected(answer):
    """Campos que la respuesta marca como posible discrepancia."""
    text = " ".join(body for title, body in split_sections(answer)
                    if title == "Posible discrepancia").lower()
    found = set()
    # Cada frase cuenta por el campo que nombra primero: asi "jornada
    # intensiva" dentro de una frase sobre vacaciones no cuenta como jornada.
    for sentence in re.split(r"\n|\.\s", text):
        places = [(m.start(), f) for f, words in FIELDS.items()
                  if (m := re.search(words, sentence))]
        if places:
            found.add(min(places)[1])
    return found


truth = {t["id"]: t for t in json.loads(
    (FOLDER / "_verdad.json").read_text(encoding="utf-8"))}
contracts = ContractStore(FOLDER).all()
retriever = RagCLI._build_retriever(INDEX, "gemini")
generator = RagCLI._gemini_generator(INDEX)

if "--run" not in sys.argv:
    sizes = []
    for contract in contracts:
        hits = ConvenioSearch.search(retriever, QUESTION, 10,
                                     contract.convenio)
        chunks = [hit.chunk for hit in hits]
        context = generator._context(generator._whole_articles(chunks))
        body = generator._body(QUESTION, context, contract.text,
                               describe(check_contract(contract)))
        text = json.dumps(body, ensure_ascii=False)
        bad = [word for word in LEAKS if word in text]
        assert not bad, (contract.id, bad)
        assert contract.name in text, contract.id
        sizes.append(len(text))
    print(f"Ensayo correcto: {len(contracts)} peticiones construidas, "
          "ninguna enviada.")
    print(f"Sin fugas de las respuestas. Tamano medio: "
          f"{sum(sizes) // len(sizes)} caracteres; modelo: {generator.model}.")
    sys.exit(0)

hits_found = tp = fp = fn = clean_ok = clean = 0
rows = []
for number, contract in enumerate(contracts, 1):
    chunks = [h.chunk for h in ConvenioSearch.search(
        retriever, QUESTION, 10, contract.convenio)]
    try:
        answer = generator.generate(QUESTION, chunks, contract.text,
                                    describe(check_contract(contract)))
    except RagError as error:
        print(f"PARADA en {contract.id}: {error}")
        break
    expected = {d["campo"] for d in truth[contract.id]["discrepancias"]}
    found = detected(answer)
    tp += len(expected & found)
    fp += len(found - expected)
    fn += len(expected - found)
    if not expected:
        clean += 1
        clean_ok += not found
    direct = next((b for t, b in split_sections(answer) if t == "Respuesta"),
                  "")
    print("   RESPUESTA:", direct[:220].replace("\n", " "), flush=True)
    rows.append({"id": contract.id, "esperado": sorted(expected),
                 "detectado": sorted(found), "respuesta": answer})
    print(f"[{number}/{len(contracts)}] {contract.id} "
          f"esperado={sorted(expected)} detectado={sorted(found)}",
          flush=True)
    time.sleep(6)
Path(f"data/laboral/generation/eval_contratos_{generator.model}.json"
     ).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(f"\nAciertos {tp}, omitidas {fn}, inventadas {fp}. "
      f"Contratos correctos sin falsa alarma: {clean_ok}/{clean}.")
