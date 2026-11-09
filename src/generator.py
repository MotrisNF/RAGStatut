"""Answer generation with Qwen/Qwen3-0.6B from the retrieved chunks.

Imported only by the ``answer`` commands, so the other commands never
load torch.
"""
from pathlib import Path
import sys
from typing import Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from .exceptions import RagError
from .models import NO_SOURCES, Chunk, DtypeMode


class Generator:
    """Generate an answer grounded in the retrieved chunks.

    The model and its tokenizer are loaded once in ``__init__`` and
    reused for every question.

    Attributes:
        MODEL_NAME: Hugging Face model used for generation.
        MAX_CONTEXT_TOKENS: Token budget of the retrieved context; the
            chunks that do not fit are left out.
        MAX_NEW_TOKENS: Maximum length of a generated answer, in tokens.
        REFUSAL: Sentence the model says when the context does not
            contain the answer.
        NO_SOURCES: Answer when retrieval found nothing; the model is not
            run, since with no context it could only answer from its own
            knowledge.
        SYSTEM_PROMPT: Fixed rules of the system message.
        REMINDER: Rule repeated after the question in each user message.
        EXAMPLE_ANSWER_CONTEXT: Context of the answered few-shot example.
        EXAMPLE_ANSWER_QUESTION: Question of the answered example.
        EXAMPLE_ANSWER: Expected answer of the answered example.
        EXAMPLE_REFUSAL_CONTEXT: Context of the refused few-shot example.
        EXAMPLE_REFUSAL_QUESTION: Question of the refused example.
        FLOAT32_MIN_GB: Available memory, in GB, needed to load the model
            in float32.
        BFLOAT16_MIN_GB: Available memory, in GB, needed to load the
            model in bfloat16.
    """

    MODEL_NAME: str = "Qwen/Qwen3-0.6B"
    MAX_CONTEXT_TOKENS: int = 2200
    MAX_NEW_TOKENS: int = 150
    REFUSAL: str = "The context does not contain the answer."
    NO_SOURCES: str = NO_SOURCES
    SYSTEM_PROMPT: str = (
        "You answer questions about the vLLM codebase. Answer ONLY from "
        "the context. Quote the exact names and values written in the "
        f"context. If the context does not contain the answer, say: "
        f"\"{REFUSAL}\""
    )
    REMINDER: str = (
        "Answer the question using the context above."
    )
    # Few-shot examples: one answered from the context, one refused. Short
    # and invented on purpose (not real vLLM facts), in the same format as
    # the real context, so the model learns the behaviour, not the content.
    EXAMPLE_ANSWER_CONTEXT: str = (
        "[CHUNK 1]\n"
        "data/raw/example/server/config.py\n"
        "class ServerConfig:\n"
        "    max_retries: int = 3\n"
        "    timeout_s: float = 30.0\n"
        "---\n"
    )
    EXAMPLE_ANSWER_QUESTION: str = "What is the default number of retries?"
    # No "(source)" at the end: the model copied the parenthetical into
    # real answers ("`--runner pooling` (ServerConfig)", see DEVLOG).
    EXAMPLE_ANSWER: str = "The default number of retries is `max_retries = 3`."
    EXAMPLE_REFUSAL_CONTEXT: str = (
        "[CHUNK 1]\n"
        "data/raw/example/docs/install.md\n"
        "Install the package with `pip install example`.\n"
        "---\n"
    )
    EXAMPLE_REFUSAL_QUESTION: str = "Who wrote Don Quixote?"
    FLOAT32_MIN_GB: float = 3.5
    BFLOAT16_MIN_GB: float = 2.0

    def __init__(self, dtype: DtypeMode = "auto") -> None:
        """Load the model and its tokenizer.

        Args:
            dtype: ``auto`` chooses float32 or bfloat16 from the available
                memory; ``float32`` or ``bfloat16`` force it (with a
                warning if the memory looks too low).

        Raises:
            RagError: Not enough memory to load the model, or the model
                is not cached and cannot be downloaded.
        """
        selected = self._set_dtype_mode(dtype)
        try:
            self._model = self._from_pretrained(AutoModelForCausalLM,
                                                dtype=selected)
        except (MemoryError, RuntimeError) as e:
            if not self._is_out_of_memory(e):
                raise
            name = str(selected).removeprefix("torch.")
            hint = " Try --dtype bfloat16." if name == "float32" else ""
            raise RagError(f"Not enough memory to load {self.MODEL_NAME} "
                           f"in {name} (allocation failed).{hint}") from e
        self._tokenizer = self._from_pretrained(AutoTokenizer)
        self._model.eval()

    def _from_pretrained(self, loader: Any, **kwargs: Any) -> Any:
        """Load the model or the tokenizer, from the local cache first.

        A plain ``from_pretrained`` asks the Hugging Face Hub for updates
        on every run, even with the files cached: a request that never
        answers left a command hanging for minutes (see DEVLOG). Loading
        with ``local_files_only`` makes no network request; the Hub is
        only used when the files are not cached yet (first run).

        Args:
            loader: ``AutoModelForCausalLM`` or ``AutoTokenizer``.
            **kwargs: Extra arguments for ``from_pretrained`` (``dtype``).

        Returns:
            The loaded model or tokenizer.

        Raises:
            RagError: The files are not cached and cannot be downloaded.
        """
        try:
            loaded = loader.from_pretrained(self.MODEL_NAME,
                                            local_files_only=True, **kwargs)
            # With the weights cached but not the tokenizer files,
            # transformers 5 does not raise: it builds an empty tokenizer
            # (1 token, no chat template). Download the files instead.
            if getattr(loaded, "chat_template", True) is not None:
                return loaded
        except OSError:
            pass
        try:
            return loader.from_pretrained(self.MODEL_NAME, **kwargs)
        except OSError as e:
            raise RagError(
                f"Cannot load {self.MODEL_NAME}: it is not in the local "
                f"cache and the download failed ({e}). Check the internet "
                "connection, or HF_HOME if the model is cached elsewhere."
            ) from e

    @staticmethod
    def _is_out_of_memory(error: BaseException) -> bool:
        """Whether a loading error means that memory could not be reserved.

        ``MemoryError`` always does. torch reports a failed allocation as a
        ``RuntimeError`` whose message says so (``can't allocate memory``,
        ``Cannot allocate memory``); any other ``RuntimeError`` (e.g. a
        corrupted model file) is a different problem and is not hidden.

        Args:
            error: The exception raised while loading the model.

        Returns:
            True if it is an out-of-memory error.
        """
        if isinstance(error, MemoryError):
            return True
        message = str(error).lower()
        return "allocate memory" in message

    def _set_dtype_mode(self, dtype: DtypeMode) -> torch.dtype:
        """Turn the ``--dtype`` option into the torch type to load.

        A forced type is used even with too little memory: only a
        warning is printed. ``auto`` is decided by ``_choose_dtype``.

        Args:
            dtype: ``auto``, ``float32`` or ``bfloat16``.

        Returns:
            ``torch.float32`` or ``torch.bfloat16``.

        Raises:
            RagError: ``auto`` and not enough memory for either type.
        """
        if dtype == "float32":
            gb = self._get_available_ram()
            if gb is None:
                print("WARNING:\n"
                      "The available memory cannot be read. If there is "
                      "not enough, the system may kill the process without "
                      "an error message ('Killed').", file=sys.stderr)
            if gb is not None and gb < self.FLOAT32_MIN_GB:
                print("WARNING:\n"
                      "'--dtype float32' needs at least "
                      f"{self.FLOAT32_MIN_GB} GB of available memory, "
                      "but only "
                      f"{gb:.2f} GB is available.\n"
                      "The system may kill the process without an error "
                      "message ('Killed').",
                      file=sys.stderr
                      )
            print(
                "Torch mode: float32 (chosen by the user)",
                file=sys.stderr
            )
            return torch.float32
        if dtype == "bfloat16":
            gb = self._get_available_ram()
            if gb is None:
                print("WARNING:\n"
                      "The available memory cannot be read. If there is "
                      "not enough, the system may kill the process without "
                      "an error message ('Killed').", file=sys.stderr)
            if gb is not None and gb < self.BFLOAT16_MIN_GB:
                print("WARNING:\n"
                      "'--dtype bfloat16' needs at least "
                      f"{self.BFLOAT16_MIN_GB} GB of available memory, "
                      "but only "
                      f"{gb:.2f} GB is available.\n"
                      "The system may kill the process without an error "
                      "message ('Killed').",
                      file=sys.stderr
                      )
            print(
                "Torch mode: bfloat16 (chosen by the user)",
                file=sys.stderr
            )
            return torch.bfloat16
        return self._choose_dtype(self._get_available_ram())

    def _get_available_ram(self) -> float | None:
        """Read the available memory from ``/proc/meminfo``.

        Returns:
            ``MemAvailable`` in GB, or None if it cannot be read.
        """
        gb: float | None = None
        try:
            for line in Path(
                            "/proc/meminfo"
                            ).read_text().splitlines():

                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 ** 2
        except OSError:
            return None
        return gb

    def _choose_dtype(self, gb: float | None) -> torch.dtype:
        """Choose the torch type for ``auto`` from the available memory.

        float32 when it fits, bfloat16 (half the memory) otherwise, and
        bfloat16 when the memory cannot be read.

        Args:
            gb: Available memory in GB, or None if unknown.

        Returns:
            ``torch.float32`` or ``torch.bfloat16``.

        Raises:
            RagError: Less than ``BFLOAT16_MIN_GB`` available.
        """
        if gb is None:
            print("The available memory cannot be read. "
                  "Torch mode: bfloat16", file=sys.stderr)
            return torch.bfloat16
        if gb >= self.FLOAT32_MIN_GB:
            print("Torch mode: float32", file=sys.stderr)
            return torch.float32
        if gb >= self.BFLOAT16_MIN_GB:
            print("Torch mode: bfloat16", file=sys.stderr)
            return torch.bfloat16
        raise RagError(f"Not enough memory to load {self.MODEL_NAME}: "
                       f"{gb:.2f} GB available, at least "
                       f"{self.BFLOAT16_MIN_GB} GB needed")

    def _build_context(self, chunks: list[Chunk]) -> str:
        """Join the chunks into the context text, within the token budget.

        Each chunk is a block with a ``[CHUNK i]`` header, its file path,
        its text and a ``---`` separator. Chunks are added best first and
        the first one that would exceed ``MAX_CONTEXT_TOKENS`` stops the
        loop, so the best chunks are always kept.

        Args:
            chunks: The retrieved chunks, best first.

        Returns:
            The context text; empty when there are no chunks.
        """
        context: str = ""
        total_context: int = 0

        for i, chunk in enumerate(chunks, 1):
            header = f"[CHUNK {i}]\n"
            path = chunk.file_path + "\n"
            text = chunk.text + "\n"
            separator = "---\n"

            block = header + path + text + separator
            ids = self._tokenizer.encode(block, add_special_tokens=False)
            n_tokens = len(ids)

            if total_context + n_tokens > self.MAX_CONTEXT_TOKENS:
                break
            context += block
            total_context += n_tokens
        return context

    def _build_messages(self,
                        question: str,
                        context: str
                        ) -> list[dict[str, str]]:
        """Build the chat messages the model receives.

        The fixed rules go in the system message. Two few-shot examples
        follow, as earlier turns of the conversation: one question
        answered by quoting the context and one unrelated question
        refused, so a small model learns both behaviours by example
        instead of by a rule. The real question comes last.

        Args:
            question: The user's question.
            context: The retrieved chunks, from ``_build_context``.

        Returns:
            The system message, the two example turns and the real user
            message, in the format of ``apply_chat_template``.
        """
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": self._user_message(
                self.EXAMPLE_ANSWER_QUESTION, self.EXAMPLE_ANSWER_CONTEXT)},
            {"role": "assistant", "content": self.EXAMPLE_ANSWER},
            {"role": "user", "content": self._user_message(
                self.EXAMPLE_REFUSAL_QUESTION,
                self.EXAMPLE_REFUSAL_CONTEXT)},
            {"role": "assistant", "content": self.REFUSAL},
            {"role": "user", "content": self._user_message(question, context)},
        ]

    def _user_message(self, question: str, context: str) -> str:
        """One user turn: the context first, the question LAST.

        A small model attends most to what it reads right before
        answering, and the most important rule is repeated after the
        question. The examples and the real question share this format,
        so the model relates them.

        Args:
            question: The question of this turn.
            context: The chunks of this turn.

        Returns:
            The text of the user message.
        """
        return (f"Context:\n{context}\n"
                f"Question: {question}\n\n"
                f"{self.REMINDER}")

    def generate(self, question: str, chunks: list[Chunk]) -> str:
        """Answer a question from the given chunks.

        Builds the context and the chat messages, then generates
        greedily (``do_sample=False``, so the same input always gives the
        same answer) up to ``MAX_NEW_TOKENS``, with Qwen3's thinking mode
        off.

        Args:
            question: The question to answer.
            chunks: The retrieved chunks, best first.

        Returns:
            The answer, cleaned by ``_clean_answer``. ``NO_SOURCES`` when
            there are no chunks, without running the model: with no
            context it could only answer from its own knowledge.
        """
        if not chunks:
            return self.NO_SOURCES
        context = self._build_context(chunks)
        messages = self._build_messages(question, context)

        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False)
        inputs = self._tokenizer(text, return_tensors="pt")

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.MAX_NEW_TOKENS,
                do_sample=False)

        new_tokens = output[:, inputs["input_ids"].shape[1]:]
        answers = self._tokenizer.batch_decode(new_tokens,
                                               skip_special_tokens=True)
        return self._clean_answer(answers[0])

    def _clean_answer(self, answer: str) -> str:
        """Remove the refusal sentence from an answer that has content.

        The model sometimes answers and then adds ``REFUSAL`` ("`e5be006`.
        The context does not contain the answer."), which contradicts the
        answer. If there is anything besides the refusal, the refusal is
        dropped; an answer that is only the refusal stays as it is.

        Args:
            answer: The decoded answer of the model.

        Returns:
            The answer without surrounding whitespace, and without the
            refusal sentence when it has other content.
        """
        answer = answer.strip()
        content = answer.replace(self.REFUSAL, "").strip()
        return content if content else answer
