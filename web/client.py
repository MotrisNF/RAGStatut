"""Talk to the RAG API. The page only knows these three functions."""
from collections.abc import Iterator
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


def _get(path: str, timeout: int = TIMEOUT, **params: Any) -> Any:
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
        with urllib.request.urlopen(url, timeout=timeout) as response:
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


def status() -> dict[str, Any]:
    """Whether the server has a Gemini key and a local model."""
    result: dict[str, Any] = _get("/status")
    return result


def set_key(key: str) -> dict[str, Any]:
    """Give the server a Gemini key, kept only in its memory.

    Raises:
        ApiError: The server refused it or is down.
    """
    request = urllib.request.Request(
        f"{API_URL}/key", data=json.dumps({"key": key}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            result: dict[str, Any] = json.load(response)
            return result
    except urllib.error.HTTPError as e:
        raise ApiError("La clave no es válida: revisa que la has copiado "
                       "entera.") from e
    except (OSError, ValueError) as e:
        raise ApiError("No puedo conectar con el servidor.") from e


def norms() -> list[dict[str, Any]]:
    """The legal texts that can be chosen."""
    result: list[dict[str, Any]] = _get("/convenios")
    return result


def citations(query: str, convenio: str) -> dict[str, Any]:
    """Search the articles for a question, without generating."""
    result: dict[str, Any] = _get("/citations", query=query,
                                  convenio=convenio)
    return result


# A local answer can take minutes; anything else gets TIMEOUT.
LOCAL_TIMEOUT = 600


def consult(query: str, convenio: str, contract: str | None = None,
            mode: str = "gemini") -> dict[str, Any]:
    """The answer to a question, with the articles it is based on.

    ``mode`` ``sources`` skips the written answer; ``local`` asks the
    model of this computer.
    """
    result: dict[str, Any] = _get(
        "/consult", timeout=LOCAL_TIMEOUT if mode == "local" else TIMEOUT,
        query=query, convenio=convenio, contract=contract, mode=mode)
    return result


def consult_stream(query: str, convenio: str,
                   contract: str | None = None) -> Iterator[dict[str, Any]]:
    """A local answer, step by step: one event per line of the API.

    Yields:
        The events of ``/consult/stream``, the last one ``done``.

    Raises:
        ApiError: The API is down or answered with an error.
    """
    params = {"query": query, "convenio": convenio, "contract": contract}
    given = {name: value for name, value in params.items()
             if value is not None}
    url = f"{API_URL}/consult/stream?{urllib.parse.urlencode(given)}"
    try:
        with urllib.request.urlopen(url, timeout=LOCAL_TIMEOUT) as response:
            for line in response:
                if not line.strip():
                    continue
                event: dict[str, Any] = json.loads(line)
                if event.get("event") == "error":
                    raise ApiError(str(event.get("detail", "")))
                yield event
    except urllib.error.HTTPError as e:
        try:
            detail = json.load(e).get("detail", "")
        except ValueError:
            detail = ""
        raise ApiError(str(detail) or f"Error {e.code} del servidor.") from e
    except (OSError, ValueError) as e:
        raise ApiError("No puedo conectar con el servidor. "
                       "¿Has arrancado «make api»?") from e


def contracts() -> list[dict[str, Any]]:
    """The fictional contracts that can be asked about."""
    result: list[dict[str, Any]] = _get("/contracts")
    return result
