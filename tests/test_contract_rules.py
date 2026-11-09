"""The comparison of contracts with their agreements, made by the code."""
import json
from pathlib import Path
import shutil

import pytest

from src.contract_rules import check_contract, describe
from src.contracts import Contract, ContractStore

BASE = ("PRIMERA. Modalidad: {modality}.\nSEGUNDA. Puesto: {job}. Categoría "
        "profesional: {category}. Clasificación profesional: {grade}.\n"
        "TERCERA. Jornada: Tiempo completo. Jornada anual a tiempo completo "
        "pactada: {hours} horas.\nCUARTA. Vacaciones: {days} {unit} al año.\n"
        "QUINTA. Periodo de prueba: {probation}.\n")


def contract(convenio: str = "tic", job: str = "Programador",
             category: str = "Área 3, grupo C",
             grade: str = "Área 3, grupo C, nivel 1", hours: int = 1800,
             days: int = 23, unit: str = "días laborables",
             probation: str = "6 meses",
             modality: str = "Indefinido") -> Contract:
    """A contract with the given conditions."""
    text = BASE.format(modality=modality, job=job, category=category,
                       grade=grade, hours=hours, days=days, unit=unit,
                       probation=probation)
    return Contract(id="x", name="N", job=job, convenio=convenio, text=text)


def states(item: Contract) -> dict[str, str]:
    """The result of each comparison."""
    return {c.field: c.status for c in check_contract(item)}


class TestRules:
    """The cases that the model got wrong."""

    def test_area_3_group_c_allows_six_months(self) -> None:
        """Five months in area 3 is fine: the model said the limit was 4."""
        assert states(contract(probation="5 meses"))[
            "Periodo de prueba"] == "correcto"

    def test_area_1_group_c_allows_four(self) -> None:
        """The same group in area 1 has a different limit."""
        item = contract(category="Área 1, grupo C", grade="Área 1, grupo C",
                        probation="5 meses")
        assert states(item)["Periodo de prueba"] == "peor"

    def test_shorter_probation_is_not_worse(self) -> None:
        """Less probation is better."""
        assert states(contract(probation="1 mes"))[
            "Periodo de prueba"] == "correcto"

    def test_fewer_vacations(self) -> None:
        """Fewer days than the agreement is worse."""
        assert states(contract(days=19))["Vacaciones"] == "peor"

    def test_more_vacations(self) -> None:
        """More days than the agreement is fine."""
        assert states(contract(days=25))["Vacaciones"] == "correcto"

    def test_more_hours(self) -> None:
        """More hours than the maximum is worse."""
        assert states(contract(hours=1856))["Jornada anual"] == "peor"

    def test_operative_worker_in_construction(self) -> None:
        """Nivel X operative: fifteen days, not the two months of staff."""
        item = contract("construccion", "Ayudante de oficio",
                        "Personal operario", "Nivel X", 1736, 30,
                        "días naturales", "1 mes")
        assert states(item)["Periodo de prueba"] == "peor"

    def test_ambiguous_job_is_not_accused(self) -> None:
        """Two months for this job depends on how it is classified."""
        item = contract("contact_center", "Gestor telefónico",
                        "Gestor telefónico", "Grupo D, nivel 9", 1764, 32,
                        "días naturales", "2 meses")
        assert states(item)["Periodo de prueba"] == "depende"

    def test_unknown_text_checks_nothing(self) -> None:
        """Nothing is guessed from a text in another format."""
        item = Contract(id="x", name="N", job="J", convenio="tic", text="Hola")
        assert check_contract(item) == []

    def test_description_has_the_result(self) -> None:
        """The model receives the result, not only the numbers."""
        assert "resultado: peor" in describe(
            check_contract(contract(days=19)))


def test_agrees_with_all_72_contracts(tmp_path: Path) -> None:
    """Every planted discrepancy is found, and no other one."""
    root = Path("data/laboral/contratos")
    if not root.exists():
        pytest.skip("contracts not present")
    copy = tmp_path / "all"
    shutil.copytree(root, copy)
    (copy / "_activos.json").unlink(missing_ok=True)
    truth = {t["id"]: t for t in json.loads(
        (root / "_verdad.json").read_text(encoding="utf-8"))}
    names = {"Vacaciones": "vacaciones", "Jornada anual": "jornada",
             "Periodo de prueba": "prueba"}
    found = ContractStore(copy).all()
    assert len(found) == 72
    for item in found:
        flagged = {names[c.field] for c in check_contract(item)
                   if c.status != "correcto"}
        expected = {d["campo"] for d in truth[item.id]["discrepancias"]}
        assert flagged == expected, item.id
