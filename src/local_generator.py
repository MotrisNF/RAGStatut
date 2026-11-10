"""Answer on this computer, without any external service (experimental).

The same idea as ``GeminiGenerator`` with a small open model, Qwen3-1.7B,
run by PyTorch on the CPU. It needs about 4.5 GB of memory and takes
around three minutes per answer, and a model this small can still copy
a wrong figure or a wrong paragraph: the page says so before using it.
Importing this module loads PyTorch, so it is only imported on demand.
"""
import hashlib

from .generator import Generator
from .models import Chunk, SearchHit
from .norm_catalog import NormCatalog


class LocalGenerator(Generator):
    """Qwen3-1.7B with a prompt written for a small model, in Spanish.

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

    MODEL_NAME: str = "Qwen/Qwen3-1.7B"
    MAX_CONTEXT_TOKENS: int = 2600
    MAX_NEW_TOKENS: int = 350
    REFUSAL: str = "No aparece en los textos consultados."
    SYSTEM_PROMPT: str = (
        "Eres un asistente que explica derechos laborales en España en "
        "castellano sencillo. Recibes fragmentos de un convenio colectivo "
        "y del Estatuto de los Trabajadores. Busca el artículo que trata "
        "el tema de la pregunta, toma la cifra exacta que dice y cítalo "
        "así: (art. 21).\n"
        "Responde SOLO con este formato, sin numerar pasos ni repetir la "
        "pregunta:\n"
        "Respuesta: una frase con la respuesta directa.\n"
        "Tu convenio: lo que dice el convenio, con su cita.\n"
        "Estatuto de los Trabajadores: lo que dice el Estatuto, con su "
        "cita.\n"
        "No uses información que no esté en los fragmentos. Si ningún "
        "fragmento trata el tema, escribe solo: No aparece en los textos "
        "consultados.")
    CONTRACT_RULES: str = (
        "Hay además un CONTRATO de una persona. Añade antes de «Respuesta» "
        "una línea «Tu contrato:» con lo que dice su contrato. La "
        "COMPROBACIÓN es exacta: usa sus resultados sin nombrarla (cita los "
        "artículos y las cláusulas) y, solo si alguno dice «peor», añade al "
        "final «Posible discrepancia:» con esa condición.")
    NOTES_RULE: str = (
        "Si citas un artículo de MODIFICACIONES POSTERIORES, avisa de que "
        "fue modificado y por qué publicación del BOE.")

    @classmethod
    def namespace(cls) -> str:
        """Identify the model and the prompts, for the answer cache."""
        text = (f"{cls.MODEL_NAME}|{cls.SYSTEM_PROMPT}|{cls.CONTRACT_RULES}"
                f"|{cls.NOTES_RULE}")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _build_context(self, chunks: list[Chunk]) -> str:
        """The retrieved texts, each labelled with its law and article.

        A text that does not fit in ``MAX_CONTEXT_TOKENS`` is skipped and
        the next ones are still tried.
        """
        context, used = "", 0
        for chunk in chunks:
            citation = NormCatalog.citation(
                SearchHit(chunk=chunk, score=0.0, chunk_id=0))
            block = (f"### {citation.norm} — {chunk.title}\n"
                     f"{chunk.text.strip()}\n\n")
            size = len(self._tokenizer.encode(block,
                                              add_special_tokens=False))
            if used + size > self.MAX_CONTEXT_TOKENS:
                continue
            context += block
            used += size
        return context

    def _build_messages(self, question: str,
                        context: str) -> list[dict[str, str]]:
        """System rules and one user message, without examples.

        A small model copies the examples instead of reading the
        context, so there are none. The question goes first and last.
        """
        system = self.SYSTEM_PROMPT
        extra = getattr(self, "_extra", "")
        if "CONTRATO" in extra:
            system += "\n" + self.CONTRACT_RULES
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
        extra = ""
        if contract:
            extra += f"CONTRATO:\n{contract}\n\n"
        if checks:
            extra += f"COMPROBACIÓN:\n{checks}\n\n"
        if notes:
            extra += f"MODIFICACIONES POSTERIORES:\n{notes}\n\n"
        self._extra = extra
        try:
            return super().generate(question, chunks)
        finally:
            self._extra = ""
