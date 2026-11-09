"""A static embedding model run with numpy, without PyTorch.

``minishlab/potion-base-8M`` (Model2Vec) is a tokenizer plus a table with
one vector per token: a text's vector is the mean of its tokens' rows,
normalised. ``sentence-transformers`` gives the same vectors but imports
PyTorch (~6.5 s); this class loads in ~0.3 s. ``index`` saves the model
next to the vector index, so searching never touches the Hugging Face
cache.
"""
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.numpy import load_file
from tokenizers import Tokenizer as HFTokenizer
import numpy as np
from numpy.typing import NDArray

from .exceptions import (IndexNotFoundError, RagError, PathError,
                         VectorIndexError)


class StaticEncoder:
    """Turns texts into normalised vectors with a static embedding model.

    Attributes:
        TABLE_NAME: File with the token vectors inside the index folder.
        TOKENIZER_NAME: File with the tokenizer inside the index folder.
    """

    TABLE_NAME = "embeddings.npy"
    TOKENIZER_NAME = "tokenizer.json"

    def __init__(self, tokenizer: HFTokenizer,
                 table: NDArray[np.float32]) -> None:
        """Store a loaded model.

        Use ``from_pretrained`` or ``load`` instead of calling this
        directly.

        Args:
            tokenizer: The model's tokenizer (text → token ids).
            table: One vector per token id, shape ``(vocabulary,
                dimensions)``.
        """
        self._tokenizer = tokenizer
        self._table = table

    @classmethod
    def from_pretrained(cls, model_name: str) -> "StaticEncoder":
        """Load a model from the Hugging Face cache, downloading it once.

        The local cache is tried first, so with the model cached no
        request is sent to the Hub (a request that never answers would
        hang the command).

        Args:
            model_name: Hugging Face model id (``minishlab/potion-base-8M``).

        Returns:
            The loaded model.

        Raises:
            RagError: The model is not cached and cannot be downloaded.
        """
        try:
            folder = snapshot_download(model_name, local_files_only=True)
        except OSError:
            try:
                folder = snapshot_download(model_name, local_files_only=False)
            except OSError as e:
                raise RagError(
                    f"Cannot load {model_name}: it is not in the local "
                    f"cache and the download failed ({e}). Check the internet "
                    "connection, or HF_HOME if the model is cached elsewhere."
                    ) from e
        tokenizer = HFTokenizer.from_file(f"{folder}/tokenizer.json")
        table = load_file(f"{folder}/model.safetensors")["embeddings"]
        return cls(tokenizer, table)

    def encode(self, texts: list[str]) -> NDArray[np.float32]:
        """Embed texts: the mean of their token vectors, normalised.

        Same result as ``SentenceTransformer.encode(...,
        normalize_embeddings=True)`` for this model (checked on the whole
        corpus: differences below 1e-7).

        Args:
            texts: The texts to embed.

        Returns:
            One vector of length 1 per text, shape ``(len(texts),
            dimensions)``; a text with no token gives a zero vector.
        """
        encodings = self._tokenizer.encode_batch(texts,
                                                 add_special_tokens=False)
        vectors = np.zeros(
            (len(texts),
             self._table.shape[1]),
            dtype=np.float32
            )

        for i, encoding in enumerate(encodings):
            if encoding.ids:
                vectors[i] = self._table[encoding.ids].mean(axis=0)

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors /= np.where(norms == 0, 1.0, norms)

        return vectors

    def save(self, files_dir: Path) -> None:
        """Write the model as ``TABLE_NAME`` and ``TOKENIZER_NAME``.

        The folder is created if needed and a previous model is replaced.

        Args:
            files_dir: Folder where the model is saved (the index folder).

        Raises:
            PathError: The folder or the table cannot be written.
            RagError: The tokenizer cannot be written.
        """
        try:
            files_dir.mkdir(parents=True, exist_ok=True)

            table_path = files_dir / self.TABLE_NAME
            tokenizer_path = files_dir / self.TOKENIZER_NAME

            np.save(table_path, self._table)
        except OSError as e:
            raise PathError(files_dir, e) from e
        try:
            self._tokenizer.save(str(tokenizer_path))
        except Exception as e:
            raise RagError(f"Cannot write '{tokenizer_path}': {e}") from e

    @classmethod
    def load(cls, files_dir: Path) -> "StaticEncoder":
        """Read a model saved by ``save``, without the Hugging Face cache.

        Args:
            files_dir: Folder given to ``save``.

        Returns:
            The same model that was saved.

        Raises:
            IndexNotFoundError: A model file is missing (``index`` was
                never run for this folder, or with an older version).
            PathError: The table exists but cannot be read.
            VectorIndexError: A file is not valid (not a ``.npy``, a table
                that is not 2D, or a broken tokenizer).
        """
        try:
            table = np.load(files_dir / cls.TABLE_NAME)
        except FileNotFoundError as e:
            raise IndexNotFoundError(files_dir) from e
        except OSError as e:
            raise PathError(files_dir, e) from e
        except ValueError as e:
            raise VectorIndexError("cannot read the embedding table."
                                   " Run 'index' again") from e
        tokenizer_path = files_dir / cls.TOKENIZER_NAME
        if not tokenizer_path.is_file():
            raise IndexNotFoundError(files_dir)
        try:
            tokenizer = HFTokenizer.from_file(str(tokenizer_path))
        except Exception as e:  # a Rust error: no more specific type
            raise VectorIndexError("cannot read the tokenizer."
                                   " Run 'index' again") from e
        if table.ndim != 2:
            raise VectorIndexError("the embedding table is corrupted."
                                   " Run 'index' again")

        return cls(tokenizer, table)
