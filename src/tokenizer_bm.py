"""Tokenizer shared by indexing and search."""
import re
import unicodedata


class Tokenizer:
    """Split text into lowercase terms for BM25.

    The text is lowercased and its accents removed (``días`` and ``dias``
    are one term), then every run of letters or digits is a term
    (``max_model_len`` → ``max``, ``model``, ``len``). The same tokenizer
    must be used for the chunks and the queries, or their terms would
    never match: rebuild the index after changing ``tokenize``.

    Attributes:
        PATTERN: A term is a run of lowercase letters or digits.
        STOPWORDS: Words to drop from the QUERY terms only (not from the
            index). Words like ``cuantos`` or ``how`` say nothing about
            the topic, but BM25 would reward the chunks that share them.
        SYNONYMS: For each word or phrase a worker uses, the words the
            legal texts use (``me caso`` → ``matrimonio``), added to the
            query.
    """

    PATTERN = re.compile(r"[a-z0-9]+")
    STOPWORDS: frozenset[str] = frozenset({
        # Question words and auxiliaries
        "what", "why", "where", "when", "which", "who", "how",
        "does", "do", "did", "is", "are", "was",
        "can", "should", "would", "could",
        # Common English words
        "the", "a", "an", "of", "to", "in", "on", "for", "and", "or",
        "with", "by", "it", "its", "this", "that", "be", "i", "you", "my",
        "there", "from", "as", "at", "use", "used", "using",
        # Spanish, without accents: ``tokenize`` removes them before these
        # are compared. "no" and "sin" are kept on purpose: they tell apart
        # terms such as "permisos no retribuidos".
        # Question words
        "que", "cual", "cuales", "quien", "quienes", "como", "donde",
        "cuando", "cuanto", "cuanta", "cuantos", "cuantas", "porque",
        # Articles and contractions
        "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del",
        "lo",
        # Prepositions and conjunctions
        "de", "en", "por", "para", "con", "sobre", "entre", "y", "e", "o",
        "u", "ni", "pero", "si",
        # Pronouns and possessives
        "yo", "me", "mi", "mis", "te", "tu", "tus", "se", "su", "sus",
        "le", "les", "nos", "nuestro", "nuestra", "este", "esta", "esto",
        "estos", "estas", "ese", "esa", "eso", "esos", "esas",
        # Common verbs of a question
        "es", "son", "ser", "soy", "estoy", "estan", "estar", "he", "ha",
        "han", "hay", "tengo", "tiene", "tienen", "tener", "puedo", "puede",
        "pueden", "poder", "debo", "debe", "deben", "quiero", "dice",
        "dicen", "dan", "dar",
        # Other frequent words
        "mas", "muy", "ya", "algo", "alguna", "alguno", "todo", "toda",
        "todos", "todas",
    })

    # What a worker says -> how the legal texts say it. Without accents, like
    # the terms; a key of several words must appear in that order.
    SYNONYMS: dict[str, tuple[str, ...]] = {
        "noche": ("nocturnidad", "nocturno", "nocturnos", "nocturna"),
        "boda": ("matrimonio",), "me caso": ("matrimonio",),
        "casarme": ("matrimonio",), "casarse": ("matrimonio",),
        "muerto": ("fallecimiento", "permisos", "licencias"),
        "muere": ("fallecimiento", "permisos", "licencias"),
        "murio": ("fallecimiento", "permisos", "licencias"),
        "fallecido": ("fallecimiento", "permisos", "licencias"),
        "libres": ("permisos", "licencias", "retribuidos"),
        "permiso": ("permisos", "licencias"),
        "casa": ("distancia", "teletrabajo"),
        "teletrabajo": ("distancia",),
        "horas al ano": ("jornada", "anual"),
        "cuantas horas": ("jornada",),
        "horario": ("jornada", "modificaciones", "sustanciales"),
        "trasladar": ("movilidad", "geografica"),
        "otra ciudad": ("movilidad", "geografica"),
        "echar": ("despido", "extincion"),
        "echado": ("despido", "extincion"),
        "despedir": ("despido", "extincion"),
        "despedido": ("despido", "extincion"),
        "baja": ("incapacidad", "temporal"),
        "sueldo": ("salario", "retribucion", "salarial"),
        "nomina": ("salario", "retribucion"),
        "cobro": ("retribucion", "salario", "complemento"),
        "cobrar": ("retribucion", "salario", "complemento"),
        "pagan": ("retribucion", "salario", "abono"),
        "pagas extra": ("gratificaciones", "extraordinarias"),
        "extra": ("extraordinarias",),
        "avisar": ("preaviso", "dimision"),
        "irme": ("dimision", "preaviso"),
        "me voy": ("dimision", "preaviso"),
        "bebe": ("nacimiento", "maternidad", "suspension"),
        "domingo": ("domingos", "festivos"),
        "festivo": ("festivos", "domingos"),
        "viajo": ("dietas", "desplazamientos"),
        "viaje": ("dietas", "desplazamientos"),
        "sancion": ("sanciones", "faltas"),
        "sancionan": ("sanciones", "faltas"),
        "sindicato": ("sindical", "sindicales"),
        "temporal": ("duracion", "determinada"),
        # Death of a relative: a worker asks about the leave, not about
        # the pension plan, which also talks about "fallecimiento".
        "muerte": ("fallecimiento", "permisos", "licencias"),
        "muerta": ("fallecimiento", "permisos", "licencias"),
        "fallecio": ("fallecimiento", "permisos", "licencias"),
        "fallecida": ("fallecimiento", "permisos", "licencias"),
        "fallece": ("fallecimiento", "permisos", "licencias"),
        # Illness, hospital and medical visits.
        "enfermo": ("enfermedad", "incapacidad", "temporal"),
        "enferma": ("enfermedad", "incapacidad", "temporal"),
        "enfermedad": ("incapacidad", "temporal"),
        "ingresado": ("hospitalizacion", "permisos"),
        "ingresada": ("hospitalizacion", "permisos"),
        "hospital": ("hospitalizacion", "permisos"),
        "operan": ("intervencion", "quirurgica", "permisos"),
        "operacion": ("intervencion", "quirurgica", "permisos"),
        "medico": ("consulta", "medica", "asistencia"),
        # Family.
        "embarazada": ("embarazo", "nacimiento", "lactancia"),
        "parto": ("nacimiento", "embarazo"),
        "cuidar": ("cuidado", "reduccion", "guarda"),
        "mudo": ("traslado", "domicilio"),
        "mudanza": ("traslado", "domicilio"),
        "mudarme": ("traslado", "domicilio"),
        # Money and career.
        "subida": ("incremento", "salarial", "tablas"),
        "aumento": ("incremento", "salarial"),
        "finiquito": ("liquidacion", "extincion"),
        "ascenso": ("ascensos", "promocion"),
        "ascender": ("ascensos", "promocion"),
        "trienio": ("antiguedad", "complemento"),
        "trienios": ("antiguedad", "complemento"),
        "dieta": ("dietas", "desplazamientos"),
        "kilometros": ("kilometraje", "desplazamientos"),
        # Leaving, schedules and the rest.
        "dimitir": ("dimision", "preaviso"),
        "renunciar": ("dimision", "preaviso"),
        "turno": ("turnos", "jornada"),
        "turnos": ("jornada",),
        "estudiar": ("examenes", "formacion"),
        "examen": ("examenes", "formacion"),
        "acoso": ("protocolo", "acoso"),
        "uniforme": ("ropa", "prendas"),
    }

    def tokenize(self, text: str) -> list[str]:
        """Return the terms of ``text``, repetitions included.

        Repetitions matter: they are the term frequency of BM25.

        Args:
            text: Any text.

        Returns:
            The terms in text order. Empty when there are none.
        """
        normalized = unicodedata.normalize("NFKD", text.lower())
        plain = "".join(c for c in normalized if not unicodedata.combining(c))
        return self.PATTERN.findall(plain)

    def tokenize_query(self, text: str) -> list[str]:
        """Return the terms of a query, without the ``STOPWORDS``.

        Only for queries: the chunks are indexed with ``tokenize``, so
        changing ``STOPWORDS`` does not require rebuilding the index.

        Args:
            text: The question to search for.

        Returns:
            The terms of ``tokenize`` in the same order, minus the
            stopwords, then the ``SYNONYMS`` of what the query says. Empty
            when the query has only stopwords.
        """
        terms = self.tokenize(text)
        joined = f" {' '.join(terms)} "
        extra = [synonym
                 for key, synonyms in self.SYNONYMS.items()
                 if f" {key} " in joined
                 for synonym in synonyms]
        return [term for term in terms if term not in self.STOPWORDS] + extra
