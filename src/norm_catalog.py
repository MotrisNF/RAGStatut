"""The legal texts of the corpus, and the citations shown to the reader.

The search only knows files and character ranges. This module turns its
hits into something a person can check: which law says it, in which
article, and where to read the official text.
"""
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel

from .convenio_search import ConvenioSearch
from .models import SearchHit

BOE_DOC = "https://www.boe.es/buscar/doc.php?id="
BOE_ACT = "https://www.boe.es/buscar/act.php?id="


class Norm(BaseModel):
    """One legal text of the corpus.

    Attributes:
        id: Value of ``convenio`` that selects it.
        source: Name of its file, without folder or extension.
        name: Name shown to the reader.
        boe_id: Identifier in the Official State Gazette.
        url: Official text.
        validity: Validity as written in the text itself.
    """

    id: str
    source: str
    name: str
    boe_id: str
    url: str
    validity: str


class Citation(BaseModel):
    """One retrieved piece of a legal text.

    Attributes:
        kind: ``convenio`` or ``estatuto``, to show them apart.
        norm: Name of the legal text.
        article: Heading of the article; empty outside an article.
        text: The retrieved text.
        url: Official text; empty for a file not in the catalog.
        file_path: File the text comes from.
        first_character_index: Start of the text in the file.
        last_character_index: End of the text in the file.
        score: Search score; only comparable within the same ``kind``.
    """

    kind: Literal["convenio", "estatuto"]
    norm: str
    article: str
    text: str
    url: str
    file_path: str
    first_character_index: int
    last_character_index: int
    score: float


class CitationResults(BaseModel):
    """What ``/citations`` answers.

    Attributes:
        query: The question.
        convenio: The legal text chosen.
        citations: The agreement's citations first, then the Statute's.
    """

    query: str
    convenio: str
    citations: list[Citation]


class NormCatalog:
    """Look up the legal texts of the corpus.

    Attributes:
        NORMS: Every known legal text.
    """

    NORMS: tuple[Norm, ...] = (
        Norm(id=ConvenioSearch.STATUTE,
             source=ConvenioSearch.STATUTE_SOURCE,
             name="Estatuto de los Trabajadores",
             boe_id="BOE-A-2015-11430",
             url=f"{BOE_ACT}BOE-A-2015-11430",
             validity="Texto consolidado"),
        Norm(id="construccion", source="convenio_construccion",
             name="VII Convenio general del sector de la construcción",
             boe_id="BOE-A-2023-19903",
             url=f"{BOE_DOC}BOE-A-2023-19903",
             validity="Hasta el 31 de diciembre de 2026"),
        Norm(id="contact_center", source="convenio_contact_center",
             name="III Convenio estatal del sector de contact center",
             boe_id="BOE-A-2023-13741",
             url=f"{BOE_DOC}BOE-A-2023-13741",
             validity="Hasta el 31 de diciembre de 2026"),
        Norm(id="grandes_almacenes", source="convenio_grandes_almacenes",
             name="Convenio colectivo del sector de grandes almacenes",
             boe_id="BOE-A-2023-13740",
             url=f"{BOE_DOC}BOE-A-2023-13740",
             validity="Hasta el 31 de diciembre de 2026"),
        Norm(id="tic", source="convenio_tic",
             name="XIX Convenio estatal de consultoría y tecnologías de "
                  "la información",
             boe_id="BOE-A-2025-7766",
             url=f"{BOE_DOC}BOE-A-2025-7766",
             validity="Hasta el 31 de diciembre de 2027"),
    )

    @classmethod
    def available(cls, index_dir: str | None) -> list[Norm]:
        """The legal texts that can be chosen for an index.

        Args:
            index_dir: Folder of the index, or ``None`` if unknown.

        Returns:
            The texts in the index; all of them when it cannot be read.
        """
        names = None if index_dir is None else ConvenioSearch.available(
            index_dir)
        return [norm for norm in cls.NORMS
                if names is None or norm.id in names]

    @classmethod
    def citation(cls, hit: SearchHit) -> Citation:
        """Describe a search hit for the reader.

        Args:
            hit: A hit of the search.

        Returns:
            Its citation. A file not in the catalog keeps its file name
            as the name of the text, and has no link.
        """
        chunk = hit.chunk
        source = PurePosixPath(chunk.file_path).stem
        norm = next((n for n in cls.NORMS if n.source == source), None)
        statute = source == ConvenioSearch.STATUTE_SOURCE
        return Citation(
            kind="estatuto" if statute else "convenio",
            norm=norm.name if norm else source,
            article=chunk.title,
            text=chunk.text,
            url=norm.url if norm else "",
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
            score=hit.score,
        )
