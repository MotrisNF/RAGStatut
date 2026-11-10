"""Later publications that change a collective agreement.

The corpus has each agreement as it was first published. This list says
what the BOE changed afterwards, so an answer never presents as current an
article that a ruling annulled. It only warns and links: the new texts are
not part of the search.
"""
import json
from pathlib import Path
import re

from pydantic import BaseModel

BOE_DOC = "https://www.boe.es/buscar/doc.php?id="


class Amendment(BaseModel):
    """One later publication.

    Attributes:
        boe_id: Identifier in the BOE.
        date: Publication date, ``YYYY-MM-DD``.
        kind: Ruling, salary tables, amendment...
        summary: What it changes, in plain words.
        articles: Articles it affects.
        others: Dispositions or annexes it affects, as titled in the text.
        url: Where to read it.
    """

    boe_id: str
    date: str
    kind: str
    summary: str
    articles: list[int] = []
    others: list[str] = []
    url: str = ""


class AmendmentBook:
    """The later publications of every agreement, read from a JSON file.

    Attributes:
        FILE_NAME: Name of the file, next to the corpus folders.
    """

    FILE_NAME = "amendments.json"

    def __init__(self, path: Path) -> None:
        """Read the file; a missing or broken file means no warnings.

        Args:
            path: The JSON file.
        """
        self.reviewed = ""
        self._items: dict[str, list[Amendment]] = {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.reviewed = str(data.get("reviewed", ""))
            for norm, items in data["agreements"].items():
                self._items[norm] = [
                    Amendment(**item, url=BOE_DOC + item["boe_id"])
                    for item in items]
        except (OSError, ValueError, KeyError, TypeError):
            self._items = {}

    def for_norm(self, norm: str) -> list[Amendment]:
        """Every later publication of an agreement, newest first."""
        return sorted(self._items.get(norm, []), key=lambda a: a.date,
                      reverse=True)

    def for_title(self, norm: str, title: str) -> list[Amendment]:
        """The publications that affect one article, disposition or annex.

        Args:
            norm: The agreement (``contact_center``...).
            title: Heading of the retrieved text ("Artículo 30. ...").

        Returns:
            The publications that name it, newest first.
        """
        number = re.match(r"Art[ií]culo (\d+)\b", title)
        found = []
        for item in self.for_norm(norm):
            if number and int(number[1]) in item.articles:
                found.append(item)
            elif not number and any(self._same(title, other)
                                    for other in item.others):
                found.append(item)
        return found

    @staticmethod
    def _same(title: str, name: str) -> bool:
        """Whether a heading starts with a disposition or annex name."""
        head, name = title.lower(), name.lower()
        return head.startswith(name) and (
            len(head) == len(name) or not head[len(name)].isalnum())
