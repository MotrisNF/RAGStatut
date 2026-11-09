"""The fictional contracts that can be asked about.

Only the contract files are read: the answer key next to them
(``_verdad.json``) is for evaluating and must never reach the model.
"""
import json
from pathlib import Path
import re

from pydantic import BaseModel


class Contract(BaseModel):
    """One fictional contract.

    Attributes:
        id: Identifier, such as ``tic_05``.
        name: Name of the person.
        job: Their job.
        convenio: The agreement that applies (``tic``...).
        text: The whole contract.
    """

    id: str
    name: str
    job: str
    convenio: str
    text: str


class ContractStore:
    """The contracts of a folder: one sub-folder per agreement.

    Attributes:
        ACTIVE_NAME: Optional JSON list of the identifiers offered; without
            it, every contract is. The other files stay in the folder.
    """

    ACTIVE_NAME = "_activos.json"

    PERSON = re.compile(r"^(?:La trabajadora|El trabajador): (.+)$", re.M)
    JOB = re.compile(r"Puesto: (.+?)\. (?:Categoría|Clasificación)", re.M)

    def __init__(self, folder: Path) -> None:
        """Read every contract; a missing folder means no contracts.

        Args:
            folder: Folder with one sub-folder per agreement.
        """
        self._contracts: dict[str, Contract] = {}
        active_file = folder / self.ACTIVE_NAME
        active: set[str] | None = None
        if active_file.exists():
            active = set(json.loads(active_file.read_text(encoding="utf-8")))
        for path in sorted(folder.glob("*/*.txt")):
            text = path.read_text(encoding="utf-8")
            person = self.PERSON.search(text)
            job = self.JOB.search(text)
            if person and job:
                identifier = path.stem.split("_")
                key = "_".join(identifier[:-1])
                if active is not None and key not in active:
                    continue
                self._contracts[key] = Contract(
                    id=key, name=person[1], job=job[1],
                    convenio=path.parent.name, text=text)

    def all(self) -> list[Contract]:
        """Every contract, sorted by the name of the person."""
        return sorted(self._contracts.values(), key=lambda c: c.name)

    def get(self, identifier: str) -> Contract | None:
        """The contract with that identifier, if there is one."""
        return self._contracts.get(identifier)
