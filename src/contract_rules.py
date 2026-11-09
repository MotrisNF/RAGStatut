"""Compare a contract with the limits of its agreement, without the model.

Reading a number in a table is where a language model fails, and here a
mistake would accuse a company of something false. These limits are written
by hand from the articles cited, so the result can be tested and is always
the same. The model only explains it.
"""
import re
from typing import Literal

from pydantic import BaseModel

from .contracts import Contract

Status = Literal["correcto", "peor", "depende"]
MONTH = 30


class Check(BaseModel):
    """One condition of a contract compared with its agreement.

    Attributes:
        field: What is compared.
        agreed: What the contract says.
        limit: What the agreement says.
        article: Where it says it.
        status: ``peor`` when the contract is worse; ``depende`` when
            the agreement leaves it open.
    """

    field: str
    agreed: str
    limit: str
    article: str
    status: Status


VACATIONS = {"tic": (23, "días laborables", "art. 21"),
             "grandes_almacenes": (31, "días naturales", "art. 35"),
             "contact_center": (32, "días naturales", "art. 29"),
             "construccion": (30, "días naturales", "art. 76")}
HOURS = {"tic": (1800, "art. 20"), "grandes_almacenes": (1770, "art. 26"),
         "contact_center": (1764, "art. 22"),
         "construccion": (1736, "art. 71")}
PROBATION_ARTICLE = {"tic": "art. 10", "grandes_almacenes": "art. 5",
                     "contact_center": "art. 18", "construccion": "art. 21"}
TIC_MONTHS = {
    (1, 2): {"A": 6, "B": 6, "C": 4, "D": 3, "E": 3},
    (3, 4, 5): {"A": 6, "B": 6, "C": 6, "D": 6, "E": 3},
    (6,): {"A": 6, "B": 6, "C": 3, "D": 3, "E": 3},
}


def _find(pattern: str, text: str) -> str | None:
    """The first group of a pattern in the text, if it appears."""
    found = re.search(pattern, text)
    return found[1] if found else None


def _days(text: str) -> int | None:
    """A duration such as ``5 meses`` or ``15 días``, in days."""
    found = re.match(r"(\d+) (mes|meses|días)", text)
    if found is None:
        return None
    return int(found[1]) * (MONTH if found[2].startswith("mes") else 1)


def _probation_limit(contract: Contract, category: str, grade: str,
                     modality: str) -> int | None:
    """The longest probation the agreement allows, in days."""
    if contract.convenio == "tic":
        area = _find(r"Área (\d)", grade)
        group = _find(r"grupo ([A-E])", grade)
        if area is None or group is None:
            return None
        for areas, months in TIC_MONTHS.items():
            if int(area) in areas:
                return months[group] * MONTH
        return None
    if contract.convenio == "grandes_almacenes":
        return (3 if "discontinuo" in modality.lower() else 6) * MONTH
    if contract.convenio == "contact_center":
        return {"Personal teleoperador": MONTH, "Gestor telefónico": MONTH,
                "Personal técnico titulado": 6 * MONTH,
                "Personal no cualificado": 15}.get(category, 2 * MONTH)
    if contract.convenio == "construccion":
        if category == "Técnico titulado":
            return 6 * MONTH
        if category == "Empleado":
            return 2 * MONTH
        return (MONTH if contract.job.startswith(("Capataz", "Encargad"))
                else 15)
    return None


def _show(days: int) -> str:
    """A number of days as the contracts write it."""
    if days % MONTH:
        return f"{days} días"
    months = days // MONTH
    return "1 mes" if months == 1 else f"{months} meses"


def check_contract(contract: Contract) -> list[Check]:
    """Compare vacations, annual hours and probation with the agreement.

    Args:
        contract: A contract in the known format.

    Returns:
        One check per condition; none for an unknown agreement or a text
        without those clauses, so nothing is ever guessed.
    """
    text = contract.text
    days_off = _find(r"Vacaciones: (\d+) días", text)
    hours = _find(r"pactada: (\d+) horas", text)
    probation = _find(r"Periodo de prueba: ([^.\n]+)", text)
    category = _find(r"Categoría profesional: ([^.\n]+)", text) or ""
    grade = _find(r"Clasificación profesional: ([^\n]+)", text) or ""
    modality = _find(r"Modalidad: ([^\n]+)", text) or ""
    if (contract.convenio not in VACATIONS or days_off is None
            or hours is None or probation is None):
        return []

    results: list[Check] = []
    minimum, unit, article = VACATIONS[contract.convenio]
    agreed = int(days_off)
    status: Status = "correcto" if agreed >= minimum else "peor"
    limit = f"mínimo {minimum} {unit}"
    if contract.convenio == "tic" and agreed == 22:
        status = "depende"
        limit = "23 días laborables, o 22 si hay jornada intensiva"
    results.append(Check(field="Vacaciones", agreed=f"{agreed} {unit}",
                         limit=limit, article=article, status=status))

    maximum, article = HOURS[contract.convenio]
    results.append(Check(
        field="Jornada anual", agreed=f"{hours} horas",
        limit=f"máximo {maximum} horas", article=article,
        status="correcto" if int(hours) <= maximum else "peor"))

    allowed = _probation_limit(contract, category, grade, modality)
    length = _days(probation)
    if allowed is not None and length is not None:
        status = "correcto" if length <= allowed else "peor"
        limit = f"máximo {_show(allowed)}"
        if category == "Gestor telefónico":
            # The agreement does not say if this job counts as teleoperator.
            limit = "1 mes si cuenta como teleoperador; si no, 2 meses"
            if MONTH < length <= 2 * MONTH:
                status = "depende"
        results.append(Check(field="Periodo de prueba", agreed=probation,
                             limit=limit,
                             article=PROBATION_ARTICLE[contract.convenio],
                             status=status))
    return results


def describe(checks: list[Check]) -> str:
    """The checks as text for the model."""
    return "\n".join(f"- {c.field}: contrato {c.agreed}; convenio {c.limit} "
                     f"({c.article}); resultado: {c.status}." for c in checks)
