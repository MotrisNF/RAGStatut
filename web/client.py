"""Talk to the RAG API. The page only knows these three functions."""
import json
import os
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

API_URL = os.environ.get("MICONVENIO_API", "http://127.0.0.1:8000")
TIMEOUT = 120


class ApiError(Exception):
    """The API could not answer; the message can be shown to the user."""


def _get(path: str, **params: Any) -> Any:
    """Send a GET request and decode its JSON.

    Args:
        path: Endpoint, such as ``/convenios``.
        **params: Query parameters.

    Returns:
        The decoded answer.

    Raises:
        ApiError: The API is down or answered with an error.
    """
    given = {name: value for name, value in params.items()
             if value is not None}
    url = f"{API_URL}{path}?{urllib.parse.urlencode(given)}"
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT) as response:
            return json.load(response)
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("detail", "")
        except ValueError:
            detail = ""
        raise ApiError(str(detail) or f"Error {e.code} del servidor.") from e
    except (OSError, ValueError) as e:
        raise ApiError("No puedo conectar con el servidor. "
                       "¿Has arrancado «make api»?") from e


def norms() -> list[dict[str, Any]]:
    """The legal texts that can be chosen."""
    result: list[dict[str, Any]] = _get("/convenios")
    return result


def citations(query: str, convenio: str) -> dict[str, Any]:
    """Search the articles for a question, without generating."""
    result: dict[str, Any] = _get("/citations", query=query,
                                  convenio=convenio)
    return result


def consult(query: str, convenio: str,
            contract: str | None = None) -> dict[str, Any]:
    """The answer to a question, with the articles it is based on."""
    result: dict[str, Any] = _get("/consult", query=query, convenio=convenio,
                                  contract=contract)
    return result


def contracts() -> list[dict[str, Any]]:
    """The fictional contracts that can be asked about."""
    result: list[dict[str, Any]] = _get("/contracts")
    return result
