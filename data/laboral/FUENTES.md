# Corpus laboral (descargado del BOE el 2026-11-09)

| Archivo | Norma | Identificador BOE |
|---|---|---|
| estatuto_trabajadores | Estatuto de los Trabajadores (texto consolidado) | BOE-A-2015-11430 |
| convenio_tic | XIX Convenio consultoría y TIC | BOE-A-2025-7766 |
| convenio_grandes_almacenes | Convenio sector grandes almacenes | BOE-A-2023-13740 |
| convenio_contact_center | III Convenio contact center | BOE-A-2023-13741 |
| convenio_construccion | VII Convenio general construcción | BOE-A-2023-19903 |

- boe_xml/: respuestas originales del BOE.
- raw/: texto convertido con tools/boe2txt.py (última versión de cada bloque en el Estatuto).
- tools/: conversor, banco de preguntas (bank.py), medición (ev.py), prueba de comandos (cli.sh).
- datasets/: banco de preguntas en el formato de search_dataset / evaluate.

Faltan las publicaciones posteriores que modifican estos convenios (tablas salariales,
acuerdos de modificación y sentencias), por ejemplo BOE-A-2026-9024 (TIC),
BOE-A-2024-426, BOE-A-2025-2663 y BOE-A-2025-25234 (construcción),
BOE-A-2024-3432 y BOE-A-2026-17492 (contact center).
