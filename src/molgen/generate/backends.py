"""LLM backends.

Two implementations share one ``LLMBackend`` protocol:

* :class:`OllamaBackend` — local inference (default; Gemma 3 12B). Keeps
  structures on-premise and removes per-token cost.
* :class:`OpenAIBackend` — retained for benchmarking against the v1 baseline.

Backends return raw text. Parsing into SMILES is the caller's job
(:mod:`molgen.generate.generator`), and validation is RDKit's
(:mod:`molgen.chem.validate`). The LLM never manipulates structures.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a medicinal chemist. You reply with valid SMILES strings and concise "
    "design rationale. You never invent experimental data. You output strict JSON "
    "with no markdown fences and no commentary outside the JSON object."
)


class BackendError(RuntimeError):
    """Raised when a backend cannot produce a completion."""


@runtime_checkable
class LLMBackend(Protocol):
    name: str

    def complete(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        ...


@dataclass
class OllamaBackend:
    """Local inference through the Ollama REST API.

    Ollama must be running (``ollama serve``) and the model pulled
    (``ollama pull gemma3:12b``).
    """

    model: str = "gemma3:12b"
    host: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    timeout: int = 600
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    keep_alive: str = "10m"

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def health_check(self) -> None:
        import requests

        try:
            resp = requests.get(f"{self.host}/api/tags", timeout=10)
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network path
            raise BackendError(
                f"cannot reach Ollama at {self.host}. Start it with 'ollama serve'."
            ) from exc

        tags = {m.get("name", "") for m in resp.json().get("models", [])}
        if self.model not in tags and f"{self.model}:latest" not in tags:
            raise BackendError(
                f"model '{self.model}' not pulled. Run: ollama pull {self.model}"
            )

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": self.system_prompt,
            "stream": False,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self.host}/api/generate", json=payload, timeout=self.timeout
            )
            resp.raise_for_status()
        except Exception as exc:  # pragma: no cover - network path
            raise BackendError(f"Ollama request failed: {exc}") from exc

        text = resp.json().get("response", "")
        if not text.strip():
            raise BackendError("Ollama returned an empty completion")
        return text


@dataclass
class OpenAIBackend:
    """Cloud backend. Kept for baseline comparison only.

    Sending receptor context to a third-party API is a data-governance decision;
    the pipeline default is :class:`OllamaBackend`.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = None
    system_prompt: str = DEFAULT_SYSTEM_PROMPT

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise BackendError("OPENAI_API_KEY is not set")

    @property
    def name(self) -> str:
        return f"openai:{self.model}"

    def health_check(self) -> None:
        return None

    def complete(self, prompt: str, *, temperature: float = 0.7, max_tokens: int = 2048) -> str:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise BackendError("pip install openai") from exc

        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""


def build_backend(spec: str, **kwargs) -> LLMBackend:
    """Construct a backend from a ``provider:model`` string.

    >>> build_backend("ollama:gemma3:12b").name
    'ollama:gemma3:12b'
    """
    provider, _, model = spec.partition(":")
    provider = provider.lower()

    if provider == "ollama":
        return OllamaBackend(model=model or "gemma3:12b", **kwargs)
    if provider == "openai":
        return OpenAIBackend(model=model or "gpt-4o-mini", **kwargs)
    raise ValueError(f"unknown backend provider '{provider}' (use ollama or openai)")


def complete_with_retry(
    backend: LLMBackend,
    prompt: str,
    *,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    attempts: int = 3,
) -> str:
    """Call ``backend.complete`` with exponential backoff."""
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return backend.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        except Exception as exc:
            last = exc
            log.warning("backend attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(2 ** attempt)
    raise BackendError(f"all {attempts} attempts failed: {last}")
