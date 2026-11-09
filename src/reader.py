"""Read the corpus files that will be indexed."""
from collections.abc import Collection
import os
from pathlib import Path

from tqdm import tqdm

from .exceptions import CorpusError
from .models import CorpusReadResult, Document, FileStamp


class CorpusReader:
    """Read the text files of a corpus folder.

    Paths are stored relative to the project root (``data/raw/...``),
    the form the moulinette compares verbatim.

    Paths are normalised without following symbolic links: ``..`` is
    removed as text, so ``--raw_dir ../`` is still outside the project,
    but ``data/raw`` may be a link to a bigger disk (``goinfre`` on 42
    computers) and stays ``data/raw/...``.

    Attributes:
        EXTENSIONS: File types that are read.
        PROJECT_ROOT: Folder the stored paths are relative to.
    """

    EXTENSIONS: frozenset[str] = frozenset({".py", ".md", ".txt"})
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

    def __init__(self, path_to_read: Path | str) -> None:
        """Store the folder to read.

        Args:
            path_to_read: Corpus folder; relative paths are taken from
                the working directory.
        """
        self._path_to_read = Path(os.path.abspath(path_to_read))

    @classmethod
    def project_path(cls, path: Path | str) -> Path:
        """Absolute path of a project file, refusing anything outside.

        A relative path is taken from the project root. ``..`` and
        absolute paths are normalised first, so ``../../.ssh/id_rsa`` or
        ``/etc/passwd`` are rejected; symbolic links are not followed.
        Used for the source files listed in a search results JSON.

        Args:
            path: A path such as ``data/raw/vllm-0.10.1/README.md``.

        Returns:
            The absolute, normalised path inside the project.

        Raises:
            CorpusError: The path is outside the project root.
        """
        absolute = Path(os.path.abspath(cls.PROJECT_ROOT / path))
        if not absolute.is_relative_to(cls.PROJECT_ROOT):
            raise CorpusError(
                f"{path} is outside the project root {cls.PROJECT_ROOT}"
            )
        return absolute

    def _list_files(self) -> list[Path]:
        """Files of the folder, recursively, with a valid extension."""
        return sorted(
            path for path in self._path_to_read.rglob("*")
            if path.suffix in self.EXTENSIONS and path.is_file()
        )

    def _relative_path(self, file: Path) -> str:
        """Path of ``file`` relative to the project root, with ``/``.

        Raises:
            CorpusError: The file is outside the project root.
        """
        try:
            return file.relative_to(self.PROJECT_ROOT).as_posix()
        except ValueError as e:
            raise CorpusError(
                f"{file} is outside the project root "
                f"{self.PROJECT_ROOT}"
            ) from e

    def _read_file(self, file: Path) -> Document | None:
        """Read one file as UTF-8, or return None if it cannot be read."""
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None
        return Document(file_path=self._relative_path(file), text=text)

    def _checked_files(self) -> list[Path]:
        """Check the corpus folder and list its valid files.

        Shared by every method that walks the corpus, so they all accept
        and refuse the same folders.

        Returns:
            The files with a valid extension, sorted (``_list_files``).

        Raises:
            CorpusError: The folder does not exist, is outside the
                project root, or has no file with a valid extension.
        """
        if not self._path_to_read.is_dir():
            raise CorpusError(
                f"Corpus folder not found: {self._path_to_read}"
            )
        if not self._path_to_read.is_relative_to(self.PROJECT_ROOT):
            raise CorpusError(
                f"Corpus folder {self._path_to_read} is outside the "
                f"project root {self.PROJECT_ROOT}"
            )
        files = self._list_files()
        if not files:
            raise CorpusError(
                "No valid files to read "
                f"in {self._path_to_read}"
            )
        return files

    def stamps(self) -> dict[str, FileStamp]:
        """Size and modification time of every valid file, without
        reading it.

        Only ``stat`` is called, so this is much faster than ``read``;
        the incremental ``index`` compares the result with the saved
        manifest to find what changed. Files ``read`` would skip (not
        UTF-8) are included: if they change, they are tried again. A
        file that disappears between listing and ``stat`` is left out,
        so it counts as deleted.

        Returns:
            ``{path relative to the project root: FileStamp}``, the same
            paths as ``Document.file_path``.

        Raises:
            CorpusError: As ``_checked_files``.
        """
        files = self._checked_files()
        files_stamped: dict[str, FileStamp] = {}
        for file in files:
            try:
                st = file.stat()
            except OSError:
                continue
            files_stamped[self._relative_path(file)] = FileStamp(
                mtime_ns=st.st_mtime_ns, size=st.st_size)
        return files_stamped

    def read(self, only: Collection[str] | None = None) -> CorpusReadResult:
        """Read the valid files of the folder, with a progress bar.

        Args:
            only: Paths relative to the project root (the keys of
                ``stamps``) to read; the other files are ignored, and
                paths that are not in the folder are skipped silently.
                ``None`` reads every file. The incremental ``index``
                passes the new and modified files.

        Returns:
            The documents read and the paths of the skipped files.

        Raises:
            CorpusError: The folder does not exist, is outside the
                project root, or has no valid file; or, reading every
                file, none of them is readable. With ``only``, reading
                nothing is not an error (an empty list, or every chosen
                file unreadable, while the rest of the index is fine).
        """
        files = self._checked_files()
        files_read: list[Document] = []
        skipped: list[str] = []
        if only is not None:
            wanted = frozenset(only)
            files = [file for file in files
                     if self._relative_path(file) in wanted]

        # No bar when there is nothing to read (an incremental index
        # with only deleted files).
        for file in tqdm(files, desc="Reading corpus", unit="file",
                         disable=not files):
            document = self._read_file(file)
            if document is not None:
                files_read.append(document)
            else:
                skipped.append(self._relative_path(file))
        if not files_read and only is None:
            raise CorpusError(
                f"No readable files in {self._path_to_read}: "
                f"{len(files)} found, none could be read as UTF-8 text"
            )

        return CorpusReadResult(documents=files_read, skipped=skipped)
