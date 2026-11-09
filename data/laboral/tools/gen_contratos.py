"""Genera contratos de trabajo TOTALMENTE FICTICIOS para probar MiConvenio.

Personas y empresas inventadas, sin DNI ni direcciones. Reproducible (semilla
fija). Las discrepancias plantadas van solo en _verdad.json, que no debe
enviarse nunca al modelo: sirve para comprobar si las detecta.

Desde la raiz del proyecto: python data/laboral/tools/gen_contratos.py
"""
import json
import random
import re
import shutil
import unicodedata
from datetime import date, timedelta
from pathlib import Path

OUT = Path("data/laboral/contratos")
SEED = 20261109
PER_SECTOR = 18
PLANTED = 4  # contratos con alguna clausula peor que el convenio, por sector
M, D = "meses", "dias"

WOMEN = """Lucia Marta Elena Carmen Paula Nuria Ines Sara Alba Irene Laura Rocio
Teresa Julia Claudia Noelia Patricia Beatriz Silvia Aitana Ioana Fatima Mei
Ana-Sofia Yolanda Candela Vega Lorena""".split()
MEN = """Hugo Pablo Javier Adrian Daniel Sergio Mario Alvaro Diego Ruben Ivan
Raul Marcos Oscar Victor Gonzalo Youssef Andrei Wei Bruno Nicolas Unai Joel
Tomas Iker Samuel Fernando""".split()
SURNAMES = """Garcia Fernandez Lopez Martinez Sanchez Perez Gomez Martin Jimenez
Ruiz Hernandez Diaz Moreno Alvarez Romero Navarro Torres Dominguez Vazquez Ramos
Gil Serrano Blanco Molina Morales Ortega Delgado Castro Ortiz Rubio Marin Sanz
Nunez Iglesias Medina Garrido Santos Castillo Cortes Lozano Popescu Benali Chen
Oliveira Etxeberria Puig Vidal""".split()
CITIES = """Madrid Barcelona Valencia Sevilla Zaragoza Malaga Murcia Bilbao
Alicante Cordoba Valladolid Vigo Gijon Granada Pamplona Santander Burgos
Salamanca""".split()
ACCENTS = {"Lucia": "Lucía", "Ines": "Inés", "Rocio": "Rocío", "Adrian": "Adrián",
           "Alvaro": "Álvaro", "Ruben": "Rubén", "Ivan": "Iván", "Raul": "Raúl",
           "Oscar": "Óscar", "Victor": "Víctor", "Nicolas": "Nicolás",
           "Tomas": "Tomás", "Fatima": "Fátima", "Ana-Sofia": "Ana Sofía",
           "Garcia": "García", "Fernandez": "Fernández", "Lopez": "López",
           "Martinez": "Martínez", "Sanchez": "Sánchez", "Perez": "Pérez",
           "Gomez": "Gómez", "Martin": "Martín", "Jimenez": "Jiménez",
           "Hernandez": "Hernández", "Diaz": "Díaz", "Alvarez": "Álvarez",
           "Dominguez": "Domínguez", "Vazquez": "Vázquez", "Marin": "Marín",
           "Nunez": "Núñez", "Cortes": "Cortés", "Malaga": "Málaga",
           "Cordoba": "Córdoba", "Gijon": "Gijón"}


def tilde(text):
    return " ".join(ACCENTS.get(w, w) for w in text.split())


def tic_limit(area, group):
    if area in (1, 2):
        return {"A": 6, "B": 6, "C": 4, "D": 3, "E": 3}[group]
    if area in (3, 4, 5):
        return {"A": 6, "B": 6, "C": 6, "D": 6, "E": 3}[group]
    return {"A": 6, "B": 6, "C": 3, "D": 3, "E": 3}[group]


def tic_roles():
    data = [("Programador/a junior", 3, "C", 3, 24105),
            ("Programador/a", 3, "C", 2, 26577),
            ("Programador/a senior", 3, "C", 1, 28423),
            ("Analista de sistemas", 3, "B", 2, 29080),
            ("Arquitecto/a de software", 3, "B", 1, 29978),
            ("Técnico/a de sistemas", 3, "D", 1, 20436),
            ("Técnico/a de soporte (CAU)", 2, "D", 1, 17403),
            ("Coordinador/a de soporte (CAU)", 2, "C", 1, 17867),
            ("Administrativo/a", 1, "D", 2, 17320),
            ("Auxiliar administrativo/a", 1, "E", 1, 17156),
            ("Consultor/a", 4, "B", 2, 29080),
            ("Consultor/a senior", 4, "B", 1, 29978),
            ("Jefe/a de proyecto", 4, "A", None, 30904),
            ("Analista de ciberseguridad", 5, "C", 1, 28423),
            ("Especialista en ciberseguridad", 5, "B", 2, 29080),
            ("Encuestador/a", 6, "E", 1, 17156),
            ("Técnico/a de campo", 6, "D", 1, 17403)]
    out = []
    for job, area, group, level, ref in data:
        nivel = f", nivel {level}" if level else ""
        out.append({"job": job, "class": f"Área {area}, grupo {group}{nivel}",
                    "limit": (tic_limit(area, group), M), "ref": ref})
    return out


SECTORS = {
    "tic": {
        "name": "XIX Convenio colectivo estatal de empresas de consultoría, "
                "tecnologías de la información y estudios de mercado y de la "
                "opinión pública",
        "vacation": (23, "días laborables"), "hours": 1800,
        "articles": {"vacaciones": "art. 21", "jornada": "art. 20",
                     "prueba": "art. 10"},
        "roles": tic_roles(), "suffix": ["Consulting", "Sistemas", "Tech",
                                         "Digital", "Software"],
        "extras": ["Teletrabajo: {tele}, conforme al acuerdo individual de "
                   "trabajo a distancia."],
    },
    "grandes_almacenes": {
        "name": "Convenio colectivo del sector de grandes almacenes",
        "vacation": (31, "días naturales"), "hours": 1770,
        "articles": {"vacaciones": "art. 35", "jornada": "art. 26",
                     "prueba": "art. 5"},
        "roles": [{"job": j, "class": f"Grupo de {g}", "limit": (6, M),
                   "ref": r} for j, g, r in [
            ("Reponedor/a", "personal base", 19000),
            ("Mozo/a de almacén", "personal base", 19000),
            ("Cajero/a", "profesionales", 21000),
            ("Vendedor/a", "profesionales", 21000),
            ("Atención al cliente", "profesionales", 21000),
            ("Escaparatista", "profesionales", 21500),
            ("Coordinador/a de sección", "profesionales de coordinación", 26000),
            ("Técnico/a de recursos humanos", "técnicos/técnicas", 29000),
            ("Técnico/a de logística", "técnicos/técnicas", 29000),
            ("Jefe/a de departamento", "mandos", 36000)]],
        "suffix": ["Comercial", "Almacenes", "Galerías", "Retail"],
        "extras": ["Prestación de servicios en domingos y festivos de apertura "
                   "autorizada, conforme al convenio."],
    },
    "contact_center": {
        "name": "III Convenio colectivo de ámbito estatal del sector de "
                "contact center",
        "vacation": (32, "días naturales"), "hours": 1764,
        "articles": {"vacaciones": "art. 29", "jornada": "art. 22",
                     "prueba": "art. 18"},
        "roles": [{"job": j, "class": c, "limit": l, "ref": r, "tele": t}
                  for j, c, l, r, t in [
            ("Teleoperador/a", "Grupo D, nivel 11", (1, M), 16100, True),
            ("Teleoperador/a especialista", "Grupo D, nivel 10", (1, M), 16100, True),
            ("Gestor/a telefónico/a", "Grupo D, nivel 9", (1, M), 16990, True),
            ("Coordinador/a", "Grupo D, nivel 8", (2, M), 17600, False),
            ("Formador/a", "Grupo D, nivel 8", (2, M), 17600, False),
            ("Agente de calidad", "Grupo D, nivel 8", (2, M), 17600, False),
            ("Supervisor/a", "Grupo D, nivel 7", (2, M), 18530, False),
            ("Técnico/a de sistemas", "Grupo C, nivel 5", (2, M), 22750, False),
            ("Programador/a senior", "Grupo C, nivel 5", (2, M), 22750, False),
            ("Titulado/a superior", "Grupo B, nivel 4", (6, M), 26000, False),
            ("Conserje", "Grupo E", (15, D), 15500, False)]],
        "suffix": ["Contact", "Atención", "Call Services", "Relación Cliente"],
        "extras": ["Turnos rotativos publicados con la antelación que fija "
                   "el convenio."],
    },
    "construccion": {
        "name": "VII Convenio colectivo general del sector de la construcción",
        "vacation": (30, "días naturales"), "hours": 1736,
        "articles": {"vacaciones": "art. 76", "jornada": "art. 71",
                     "prueba": "art. 21"},
        "roles": [{"job": j, "class": c, "limit": l, "ref": r}
                  for j, c, l, r in [
            ("Peón ordinario", "Nivel XII", (15, D), 18965),
            ("Peón especializado", "Nivel XI", (15, D), 19250),
            ("Ayudante de oficio", "Nivel X", (15, D), 19540),
            ("Oficial de 2.ª", "Nivel IX", (15, D), 19835),
            ("Oficial de 1.ª", "Nivel VIII", (15, D), 20130),
            ("Capataz", "Nivel VII", (1, M), 20430),
            ("Encargado/a general de obra", "Nivel V", (1, M), 21050),
            ("Auxiliar administrativo/a", "Nivel IX", (2, M), 19835),
            ("Delineante de 2.ª", "Nivel VII", (2, M), 20430),
            ("Delineante de 1.ª", "Nivel VI", (2, M), 20740),
            ("Titulado/a medio", "Nivel III", (6, M), 21690)]],
        "suffix": ["Construcciones", "Obras", "Edificación", "Reformas"],
        "extras": ["Los equipos de protección individual los facilita la "
                   "empresa."],
    },
}
PREFIXES = """Nubaria Arventis Olmedia Tarsis Veloria Quirania Solvento Brisalia
Maretia Cortalia Almenara Fontiva Lumera Ordesia Rivanta Selenia Zafira
Bartolia""".split()
START = date(2015, 1, 1)
TODAY = date(2026, 11, 9)


def gendered(job, female):
    """Escribe el puesto en femenino o masculino: «Técnico/a» -> «Técnica»."""
    def replace(found):
        word = found[1]
        if not female:
            return word
        return word[:-1] + "a" if word[-1] in "oe" else word + "a"

    return re.sub(r"([\wáéíóúñ]+)/a\b", replace, job)


def category_of(sid, role):
    """Categoria profesional, con los nombres que usa cada convenio."""
    job, group = role["job"], role["class"]
    if sid == "construccion":
        if job.startswith("Titulado"):
            return "Técnico titulado"
        if job.startswith(("Auxiliar", "Delineante")):
            return "Empleado"
        return "Personal operario"
    if sid == "contact_center":
        if job.startswith("Teleoperador"):
            return "Personal teleoperador"
        if job.startswith("Titulado"):
            return "Personal técnico titulado"
        if job == "Conserje":
            return "Personal no cualificado"
        if job.startswith("Gestor"):
            return "Gestor telefónico"
        return "Restantes niveles"
    if sid == "tic":
        return ", ".join(group.split(", ")[:2])
    return group


def days_of(probation):
    return probation[0] * 30 if probation[1] == M else probation[0]


def show(probation):
    n, unit = probation
    if unit == M:
        return f"{n} mes" if n == 1 else f"{n} meses"
    return f"{n} días"


def names(rng, used):
    while True:
        female = rng.random() < 0.5
        first = rng.choice(WOMEN if female else MEN)
        full = f"{first} {rng.choice(SURNAMES)} {rng.choice(SURNAMES)}"
        if full not in used:
            used.add(full)
            return tilde(full), female


def contract(rng, sector, sid, number, used, planted):
    s = SECTORS[sid]
    role = s["roles"][(number - 1) % len(s["roles"])]
    name, female = names(rng, used)
    company = f"{rng.choice(PREFIXES)} {rng.choice(s['suffix'])}, S.L."
    city = tilde(rng.choice(CITIES))
    temporal = rng.random() < 0.14
    discontinuous = sid == "grandes_almacenes" and not temporal and rng.random() < .2
    if temporal:
        begin = TODAY - timedelta(days=rng.randint(10, 150))
        months = rng.randint(2, 6)
        kind = f"Temporal por circunstancias de la producción, {months} meses"
    else:
        begin = START + timedelta(days=rng.randint(0, (TODAY - START).days - 40))
        kind = "Fijo discontinuo" if discontinuous else "Indefinido"
    pct = 100 if rng.random() < .7 else rng.choice([50, 60, 75, 80])

    vac_n, vac_unit = s["vacation"]
    vacation = vac_n + (rng.choice([1, 2]) if rng.random() < .15 else 0)
    hours = s["hours"] - (rng.choice([16, 24, 40]) if rng.random() < .2 else 0)
    limit = role["limit"]
    if discontinuous:
        limit = (3, M)
    if temporal:
        limit = (min(limit[0], 1), M) if limit[1] == M else limit
    if limit[1] == M:
        probation = (rng.randint(max(1, limit[0] - 2), limit[0]), M)
    else:
        probation = limit

    discrepancies = []
    if planted:
        fields = ["vacaciones", "jornada", "prueba"]
        if temporal or discontinuous:
            fields.remove("prueba")
        for field in rng.sample(fields, rng.choice([1, 1, 2])):
            if field == "vacaciones":
                vacation = vac_n - rng.choice([2, 3, 4])
                pact, norm = f"{vacation} {vac_unit}", f"{vac_n} {vac_unit}"
            elif field == "jornada":
                hours = s["hours"] + rng.choice([36, 56, 80])
                pact, norm = f"{hours} horas anuales", f"{s['hours']} horas anuales"
            else:
                bad = (limit[0] + rng.choice([1, 2]), M) if limit[1] == M else (1, M)
                probation = bad
                pact, norm = show(bad), f"máximo {show(limit)}"
            discrepancies.append({"campo": field, "pactado": pact,
                                  "convenio": norm,
                                  "articulo": s["articles"][field]})

    salary = int(round(role["ref"] * rng.uniform(1.15, 1.40), -1))
    real = int(round(salary * pct / 100, -1))
    days = rng.randint(1, 3)
    extra = rng.choice(s["extras"]).format(
        tele="1 día a la semana" if days == 1 else f"{days} días a la semana")
    job = gendered(role["job"], female)
    who = "La trabajadora" if female else "El trabajador"
    category = category_of(sid, role)
    identifier = f"{sid}_{number:02d}"
    plain = unicodedata.normalize("NFKD", name.lower()).encode("ascii", "ignore")
    slug = "-".join(plain.decode().split()[:2])
    time = ("Tiempo completo" if pct == 100 else
            f"Tiempo parcial, {pct} % de la jornada")
    pay = f"{salary:,} euros brutos anuales".replace(",", ".")
    if pct != 100:
        pay += (f" a jornada completa; {real:,} euros brutos anuales "
                "por la jornada pactada").replace(",", ".")
    text = f"""CONTRATO DE TRABAJO — DOCUMENTO FICTICIO PARA DEMOSTRACIÓN
Persona y empresa inventadas. No corresponde a ninguna relación laboral real.

Empresa: {company} (empresa ficticia)
{who}: {name}
Centro de trabajo: {city}
Fecha de inicio: {begin.strftime('%d/%m/%Y')}

PRIMERA. Modalidad: {kind}.
SEGUNDA. Puesto: {job}. Categoría profesional: {category}. Clasificación profesional: {role['class']}.
TERCERA. Jornada: {time}. Jornada anual a tiempo completo pactada: {hours} horas.
CUARTA. Vacaciones: {vacation} {vac_unit} al año.
QUINTA. Periodo de prueba: {show(probation)}.
SEXTA. Retribución: {pay}, en catorce pagas.
SÉPTIMA. {extra}
OCTAVA. Convenio aplicable: {s['name']}.
"""
    truth = {"id": identifier, "archivo": f"{sid}/{identifier}_{slug}.txt",
             "nombre": name, "sector": sid, "puesto": job,
             "categoria": category,
             "clasificacion": role["class"], "modalidad": kind,
             "fijo_discontinuo": discontinuous, "temporal": temporal,
             "porcentaje_jornada": pct, "salario_anual": salary,
             "pactado": {"vacaciones": vacation, "horas": hours,
                         "prueba_dias": days_of(probation)},
             "limites": {"vacaciones": vac_n, "horas": s["hours"],
                         "prueba_dias": days_of(limit)},
             "discrepancias": discrepancies}
    return truth, text


def check(truth):
    p, l = truth["pactado"], truth["limites"]
    found = {f for f, bad in (("vacaciones", p["vacaciones"] < l["vacaciones"]),
                              ("jornada", p["horas"] > l["horas"]),
                              ("prueba", p["prueba_dias"] > l["prueba_dias"]))
             if bad}
    assert found == {d["campo"] for d in truth["discrepancias"]}, truth["id"]


def main():
    rng = random.Random(SEED)
    active = OUT / "_activos.json"
    kept = active.read_text(encoding="utf-8") if active.exists() else None
    if OUT.exists():
        shutil.rmtree(OUT)
    used, all_truth = set(), []
    for sid in SECTORS:
        (OUT / sid).mkdir(parents=True)
        planted = set(rng.sample(range(1, PER_SECTOR + 1), PLANTED))
        for number in range(1, PER_SECTOR + 1):
            truth, text = contract(rng, SECTORS[sid], sid, number, used,
                                   number in planted)
            check(truth)
            (OUT / truth["archivo"]).write_text(text, encoding="utf-8")
            all_truth.append(truth)
    (OUT / "_verdad.json").write_text(
        json.dumps(all_truth, ensure_ascii=False, indent=1), encoding="utf-8")
    if kept is not None:
        active.write_text(kept, encoding="utf-8")
    (OUT / "LEEME.md").write_text(
        "# Contratos ficticios\n\nGenerados por `tools/gen_contratos.py` para "
        "demostrar MiConvenio. Personas y empresas inventadas.\n\n"
        "`_verdad.json` guarda las discrepancias plantadas y los límites de "
        "cada convenio: solo sirve para evaluar, **nunca debe enviarse al "
        "modelo**. Los salarios son inventados y no están contrastados con "
        "las tablas vigentes.\n", encoding="utf-8")
    print(len(all_truth), "contratos;", sum(bool(t["discrepancias"]) for t in
                                            all_truth), "con discrepancia")


main()
