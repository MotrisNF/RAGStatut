"""Answer on this computer, without any external service (experimental).

The same idea as ``GeminiGenerator`` with an open model, Qwen3-4B, run by
PyTorch on the CPU. In bfloat16 it needs about 8 GB of memory (16 GB in
float32) and takes several minutes per answer, and a model this size can
still lose a detail or copy a wrong figure: the page says so before using
it. Qwen3-1.7B (about 4.5 GB in bfloat16) mixed up the agreement and the
Statute and lost the units of the figures.
Importing this module loads PyTorch, so it is only imported on demand.
"""
from collections.abc import Iterator
import hashlib
import os
import re
import sys
import time

from .generator import Generator
from .models import Chunk, DtypeMode, SearchHit
from .norm_catalog import NormCatalog


class LocalGenerator(Generator):
    """Qwen3-4B with a prompt written for a small model, in Spanish.

    Measured on three questions of the banks: the earlier English prompt
    with examples made the model copy the example, and a strict "say it
    is not there" rule made it refuse everything. This prompt asks for
    the final format only, so the page can show it like Gemini's.

    Attributes:
        MODEL_NAME: The model, from the Hugging Face cache.
        MAX_CONTEXT_TOKENS: Room for the retrieved texts.
        MAX_NEW_TOKENS: Longest answer; enough for the three sections.
        REFUSAL: What it says when nothing in the texts answers.
        SYSTEM_PROMPT: The rules of every answer.
    """

    MODEL_NAME: str = "Qwen/Qwen3-4B"
    MAX_CONTEXT_TOKENS: int = 2600
    MAX_NEW_TOKENS: int = 500
    # True: three calls (answer, agreement alone, Statute alone), more
    # precise and three times slower. False: the single call of before.
    SPLIT_CALLS: bool = True
    PART_NEW_TOKENS: int = 160
    ABSENT: str = "No lo menciona."
    NOTHING: str = "No aparece en los textos consultados."
    ANSWER_PROMPT: str = (
        "Eres un asistente que explica derechos laborales en España en "
        "castellano sencillo. Recibes lo que dicen sobre la pregunta el "
        "convenio colectivo y el Estatuto de los Trabajadores, ya "
        "resumido.\n"
        "Responde a la pregunta en una o dos frases completas, usando SOLO "
        "esos dos resúmenes: copia sus cifras y sus artículos entre "
        "paréntesis. Si dan cifras distintas, da las dos y di de cuál es "
        "cada una. No contradigas los resúmenes ni añadas nada. Escribe "
        "solo la respuesta, sin títulos.\n"
        "Si los dos dicen «No lo menciona», escribe solo: No aparece en los "
        "textos consultados.")
    PART_PROMPT: str = (
        "Eres un asistente que explica derechos laborales en España en "
        "castellano sencillo. Recibes fragmentos de {law}, cada uno con el "
        "título de su artículo.\n"
        "Explica en una o dos frases completas qué dicen estos fragmentos "
        "sobre la pregunta, con la cifra exacta y su artículo entre "
        "paréntesis, por ejemplo (art. 30). Usa solo estos fragmentos y "
        "escribe solo el texto, sin títulos.\n"
        "Si ningún fragmento trata el tema, escribe solo: No lo menciona.")
    # A part call never sees the contract: without this, "check this
    # contract" made it look for a contract in the legal texts and answer
    # "No lo menciona".
    CONTRACT_PART_RULE: str = (
        "\nLa pregunta habla del contrato de una persona, que tú no "
        "tienes: no lo busques en los fragmentos. Explica qué dicen los "
        "fragmentos sobre cada tema que menciona la pregunta (por ejemplo "
        "vacaciones, jornada anual o periodo de prueba), una frase por "
        "tema, con su cifra exacta y su artículo. Escribe «el convenio "
        "establece» o «el Estatuto establece» según los fragmentos que "
        "tienes; nunca «el contrato establece».")
    CONTRACT_PART_TOKENS: int = 260
    SPLIT_CONTRACT_RULES: str = (
        "Hay además un CONTRATO de una persona: responde para esa persona "
        "con los datos de su contrato. La COMPROBACIÓN es exacta: si algún "
        "resultado dice «peor», dilo en la respuesta como posible "
        "discrepancia, con el artículo; no nombres la comprobación. Cita el "
        "contrato por sus cláusulas, así: (cláusula cuarta); usa «art.» "
        "solo para el convenio y el Estatuto, nunca para el contrato.")
    REFUSAL: str = "No aparece en los textos consultados."
    SYSTEM_PROMPT: str = (
        "Eres un asistente que explica derechos laborales en España en "
        "castellano sencillo. Recibes fragmentos de un convenio colectivo "
        "y del Estatuto de los Trabajadores, cada uno con su título.\n"
        "Cómo responder:\n"
        "- Lee los fragmentos y quédate con el artículo cuyo título trata "
        "el tema de la pregunta.\n"
        "- Escribe tres párrafos de una o dos frases completas cada uno, "
        "en este orden:\n"
        "  «Tu convenio:» y lo que dice el BLOQUE 1, con la cifra exacta, "
        "su unidad (días laborables, días naturales, horas anuales...) y "
        "su artículo entre paréntesis, por ejemplo (art. 30).\n"
        "  «Estatuto de los Trabajadores:» y lo que dice el BLOQUE 2, con "
        "su cifra, su unidad y su artículo. Usa SOLO el BLOQUE 2: el "
        "Estatuto es otra ley y sus cifras y artículos son distintos de los "
        "del convenio; nunca copies aquí lo del BLOQUE 1.\n"
        "  «Respuesta:» al final, la respuesta a la pregunta en una o dos "
        "frases, resumiendo los párrafos anteriores sin contradecirlos: "
        "cada cifra con su unidad, su artículo y de qué norma es. Si la "
        "cifra depende de la categoría o el puesto, dilo y da los extremos "
        "en lugar de toda la lista.\n"
        "- Cada párrafo debe tener contenido: nunca dejes una etiqueta "
        "vacía ni copies el título de un artículo como respuesta.\n"
        "- No uses información que no esté en los fragmentos. Si un texto "
        "no habla del tema, escribe en su párrafo «No lo menciona.». Si "
        "ninguno lo trata, escribe solo: No aparece en los textos "
        "consultados.")
    CONTRACT_RULES: str = (
        "Hay además un CONTRATO de una persona. Empieza con un párrafo «Tu "
        "contrato:» con lo que dice su contrato sobre lo preguntado, citado "
        "por cláusulas, así: (cláusula cuarta); usa «art.» solo para el "
        "convenio y el Estatuto, nunca para el contrato. La COMPROBACIÓN es "
        "exacta: usa sus resultados sin nombrarla. Solo si alguno dice "
        "«peor», añade antes de «Respuesta» un párrafo «Posible "
        "discrepancia:» con lo que dice el contrato, lo que dice el "
        "convenio y su artículo. En «Respuesta» di si hay o no alguna "
        "condición peor y cuál.")
    NOTES_RULE: str = (
        "Si citas un artículo de MODIFICACIONES POSTERIORES, avisa de que "
        "fue modificado y por qué publicación del BOE.")

    #: Environment variable that chooses the precision: ``bfloat16`` (the
    #: default, about 8 GB) or ``float32`` (twice the memory). Generating
    #: on a CPU is limited by the memory read for each word, so bfloat16
    #: reads half and should be faster.
    DTYPE_ENV: str = "MICONVENIO_LOCAL_DTYPE"

    @classmethod
    def dtype(cls) -> DtypeMode:
        """The precision to load the model with, from ``DTYPE_ENV``."""
        value = os.environ.get(cls.DTYPE_ENV, "bfloat16")
        return "float32" if value == "float32" else "bfloat16"

    @classmethod
    def namespace(cls) -> str:
        """Identify everything that shapes an answer, for the cache.

        The model, every prompt, the way of calling it and the context
        sizes: changing any of them must not reuse an old answer (it did,
        once, after a change of the context alone).
        """
        text = "|".join(str(part) for part in (
            cls.MODEL_NAME, cls.dtype(), cls.SYSTEM_PROMPT, cls.CONTRACT_RULES,
            cls.NOTES_RULE, cls.SPLIT_CALLS, cls.ANSWER_PROMPT,
            cls.PART_PROMPT, cls.SPLIT_CONTRACT_RULES, cls.CONTRACT_PART_RULE,
            cls.CONTRACT_PART_TOKENS, cls.MAX_CONTEXT_TOKENS,
            cls.AGREEMENT_SHARE, cls.MAX_NEW_TOKENS, cls.PART_NEW_TOKENS))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_context(self, chunks: list[Chunk]) -> str:
        """The retrieved texts in two blocks: the agreement, then the
        Statute.

        Mixed together, a small model gave the Statute the figures of the
        agreement (measured on two questions). Each block has a clear
        heading and each text its article.

        Each block has its own share of ``MAX_CONTEXT_TOKENS``, so the
        agreement, retrieved first, cannot fill it all and leave the
        Statute out (it did: seven agreement texts used it whole). A text
        that does not fit in its share is skipped and the next ones are
        still tried.
        """
        blocks: dict[str, list[str]] = {"convenio": [], "estatuto": []}
        own = int(self.MAX_CONTEXT_TOKENS * self.AGREEMENT_SHARE)
        room = {"convenio": own,
                "estatuto": self.MAX_CONTEXT_TOKENS - own}
        if getattr(self, "_only", None) is not None:
            room = {"convenio": self.MAX_CONTEXT_TOKENS,
                    "estatuto": self.MAX_CONTEXT_TOKENS}
        used = {"convenio": 0, "estatuto": 0}
        given = getattr(self, "_given", None)
        if given is not None:
            return str(given)
        only = getattr(self, "_only", None)
        for chunk in chunks:
            citation = NormCatalog.citation(
                SearchHit(chunk=chunk, score=0.0, chunk_id=0))
            if only is not None and citation.kind != only:
                continue
            text = f"{chunk.title}\n{chunk.text.strip()}\n\n"
            size = len(self._tokenizer.encode(text,
                                              add_special_tokens=False))
            kind = citation.kind
            if used[kind] + size > room[kind]:
                continue
            blocks[kind].append(text)
            used[kind] += size
        if only is not None:
            # One law alone gets all the room and no block headings.
            return "".join(blocks[only])
        context = ""
        for kind, heading in (("convenio", self.AGREEMENT_HEADING),
                              ("estatuto", self.STATUTE_HEADING)):
            body = "".join(blocks[kind]) or "(Ningún fragmento.)\n\n"
            context += f"{heading}\n{body}"
        return context

    AGREEMENT_SHARE: float = 0.65
    AGREEMENT_HEADING: str = "=== BLOQUE 1: TU CONVENIO ==="
    STATUTE_HEADING: str = "=== BLOQUE 2: ESTATUTO DE LOS TRABAJADORES ==="

    def _build_messages(self, question: str,
                        context: str) -> list[dict[str, str]]:
        """System rules and one user message, without examples.

        A small model copies the examples instead of reading the
        context, so there are none. The question goes first and last.
        """
        system = getattr(self, "_prompt", None) or self.SYSTEM_PROMPT
        extra = getattr(self, "_extra", "")
        if "CONTRATO" in extra and system == self.SYSTEM_PROMPT:
            system += "\n" + self.CONTRACT_RULES
        elif "CONTRATO" in extra:
            system += "\n" + self.SPLIT_CONTRACT_RULES
        if "MODIFICACIONES" in extra:
            system += "\n" + self.NOTES_RULE
        return [{"role": "system", "content": system},
                {"role": "user", "content":
                    f"Pregunta: {question}\n\n{extra}"
                    f"Fragmentos:\n{context}\nPregunta: {question}"}]

    def generate(self, question: str, chunks: list[Chunk],
                 contract: str | None = None, checks: str | None = None,
                 notes: str | None = None) -> str:
        """Answer from the chunks, and a contract if there is one.

        Args:
            question: The question.
            chunks: The retrieved chunks, best first.
            contract: Text of a person's contract.
            checks: Its comparison with the agreement, made by the code.
            notes: Later BOE publications of the retrieved articles.

        Returns:
            The answer, in the sections the page shows.
        """
        answer = ""
        for event in self.iter_generate(question, chunks, contract, checks,
                                        notes):
            if event[0] == "answer":
                answer = event[1]
        return answer

    def iter_generate(self, question: str, chunks: list[Chunk],
                      contract: str | None = None, checks: str | None = None,
                      notes: str | None = None
                      ) -> Iterator[tuple[str, ...]]:
        """Generate step by step, telling what is being written.

        Yields, in order: ``("stage", name)`` when a call starts (name is
        ``convenio``, ``estatuto`` or ``respuesta``), ``("part", title,
        text)`` when it ends, and last ``("answer", text)`` with the whole
        answer, as ``generate`` returns it.
        """
        extra = ""
        if contract:
            extra += f"CONTRATO:\n{contract}\n\n"
        if checks:
            extra += f"COMPROBACIÓN:\n{checks}\n\n"
        if notes:
            extra += f"MODIFICACIONES POSTERIORES:\n{notes}\n\n"
        if not self.SPLIT_CALLS:
            yield ("stage", "todo")
            self._extra = extra
            try:
                answer = self.check_answer(super().generate(question, chunks))
            finally:
                self._extra = ""
            yield ("answer", answer)
            return
        yield from self._iter_split(question, chunks, extra)

    def _call(self, question: str, chunks: list[Chunk], prompt: str,
              only: str | None, extra: str, tokens: int,
              given: str | None = None) -> str:
        """One call to the model with its own prompt and texts.

        ``given`` replaces the retrieved texts as the context (the answer
        call reads the two paragraphs already written).
        """
        self._prompt: str | None = prompt
        self._only: str | None = only
        self._given: str | None = given
        self._extra = extra
        saved = self.MAX_NEW_TOKENS
        self.MAX_NEW_TOKENS = tokens
        try:
            start = time.perf_counter()
            text = super().generate(question, chunks)
            print(f"  {only or 'respuesta'}: "
                  f"{time.perf_counter() - start:.1f} s", file=sys.stderr)
        finally:
            self._prompt, self._only, self._extra = None, None, ""
            self._given = None
            self.MAX_NEW_TOKENS = saved
        # The model sometimes adds a heading anyway: it is removed.
        return re.sub(r"^\W*(Respuesta|Tu convenio|Estatuto de los "
                      r"Trabajadores|Tu contrato)\W*:\s*", "",
                      text.strip())

    def _iter_split(self, question: str, chunks: list[Chunk],
                    extra: str) -> Iterator[tuple[str, ...]]:
        """Three calls: each law alone, then the answer from both.

        A small model mixing both laws gave the Statute the figures of the
        agreement, and an answer made from every text contradicted the
        detail ("not in the Statute"). So each law is written alone, from
        its own texts only, and the answer is written last from those two
        paragraphs, which it can only summarise.

        Decided by the code, not the model: a law with no retrieved text
        is neither asked nor shown (with "only the Statute" there is no
        agreement); with nothing found the answer is ``NOTHING`` without a
        call; and an answer of ``NOTHING`` while a law did say something is
        replaced by what the laws said (it happened once). When only one
        law says something and there is no contract, its paragraph is the
        whole answer: a summary of a single paragraph only repeated it.
        """
        kinds = {NormCatalog.citation(
            SearchHit(chunk=c, score=0.0, chunk_id=0)).kind for c in chunks}
        about_contract = "CONTRATO" in extra
        rule = self.CONTRACT_PART_RULE if about_contract else ""
        tokens = (self.CONTRACT_PART_TOKENS if about_contract
                  else self.PART_NEW_TOKENS)
        parts = []
        said = []
        for kind, title, name in (
                ("convenio", "Tu convenio", "tu convenio colectivo"),
                ("estatuto", "Estatuto de los Trabajadores",
                 "el Estatuto de los Trabajadores")):
            if kind not in kinds:
                continue
            yield ("stage", kind)
            text = self._call(question, chunks,
                              self.PART_PROMPT.format(law=name) + rule,
                              kind, "", tokens)
            yield ("part", title, text)
            parts.append(f"{title}: {text}")
            if not text.startswith(self.ABSENT.rstrip(".")):
                said.append(text)
        if not said:
            answer = self.NOTHING
        elif len(said) == 1 and not about_contract:
            yield ("part", "Respuesta", said[0])
            yield ("answer", self.check_answer(f"Respuesta: {said[0]}"))
            return
        else:
            summaries = "\n\n".join(parts) + "\n\n"
            yield ("stage", "respuesta")
            answer = self._call(question, chunks, self.ANSWER_PROMPT, None,
                                extra, self.PART_NEW_TOKENS, summaries)
            if self.NOTHING.rstrip(".") in answer:
                answer = " ".join(said)
        yield ("part", "Respuesta", answer)
        yield ("answer", self.check_answer(
            "\n\n".join([f"Respuesta: {answer}", *parts])))

    #: Shown instead of an answer the model left empty.
    EMPTY: str = ("El modelo local no ha sabido redactar una respuesta. "
                  "Consulta los artículos encontrados.")

    @classmethod
    def check_answer(cls, answer: str) -> str:
        """Drop the sections the model left empty.

        A small model sometimes copies the format with nothing after the
        labels. Empty sections are removed and, if nothing is left, a
        clear message replaces the answer instead of an empty template.

        Args:
            answer: What the model wrote.

        Returns:
            The answer without empty sections, or ``EMPTY``.
        """
        labels = ("Respuesta", "Tu convenio", "Estatuto de los Trabajadores",
                  "Tu contrato", "Posible discrepancia")
        pattern = re.compile(r"^\W*(" + "|".join(labels) + r")\W*:",
                             re.MULTILINE)
        cuts = list(pattern.finditer(answer))
        if not cuts:
            return answer.strip() or cls.EMPTY
        kept = []
        for found, after in zip(cuts, cuts[1:] + [None]):
            end = after.start() if after else len(answer)
            body = answer[found.end():end].strip(" \n*")
            # A body that is only a citation, like "(art. 37. Descanso…)",
            # says nothing.
            if re.sub(r"\(art\.[^)]*\)", "", body).strip(" .\n*"):
                kept.append(f"{found[1]}: {body}")
        return "\n\n".join(kept) if kept else cls.EMPTY
