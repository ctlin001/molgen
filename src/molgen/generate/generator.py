"""Turn LLM completions into candidate molecule records.

Parsing is defensive by design. Local models wrap JSON in markdown fences,
emit trailing prose, or drop a closing brace. The recovery ladder is:

1. strip fences and parse the outermost JSON object
2. scan for balanced JSON objects anywhere in the text
3. fall back to line-wise SMILES extraction, each candidate verified by
   ``Chem.MolFromSmiles`` — never by regex alone

Nothing here judges chemistry. That is :mod:`molgen.chem.validate`.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .backends import LLMBackend, complete_with_retry

log = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json|python|smiles)?\s*|\s*```\s*$", re.MULTILINE)
_SMILES_CHARS = re.compile(r"^[A-Za-z0-9@+\-\[\]\(\)=#$%/\\.*]{4,300}$")


@dataclass
class Candidate:
    """One generated molecule, before any chemistry validation."""

    smiles: str
    name: str = ""
    rationale: str = ""
    key_features: list[str] = field(default_factory=list)
    source: str = "json"
    round_index: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "smiles": self.smiles,
            "name": self.name,
            "rationale": self.rationale,
            "key_features": self.key_features,
            "source": self.source,
            "round": self.round_index,
        }


def strip_fences(text: str) -> str:
    """Remove markdown code fences that local models add unprompted."""
    return _FENCE.sub("", text).strip()


def _balanced_objects(text: str) -> list[str]:
    """Yield substrings that are balanced ``{...}`` blocks."""
    out, depth, start, in_string, escape = [], 0, None, False, False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                out.append(text[start : i + 1])
                start = None
            elif depth < 0:
                depth = 0
    return out


def _is_plausible_smiles(token: str) -> bool:
    """Structural check delegated to RDKit; pattern match is only a prefilter."""
    if not _SMILES_CHARS.match(token):
        return False
    if token.isalpha() and token.islower():
        return False  # ordinary word
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    return Chem.MolFromSmiles(token) is not None


def parse_completion(text: str, round_index: int = 0) -> list[Candidate]:
    """Extract candidates from a raw completion."""
    cleaned = strip_fences(text)
    records: list[dict[str, Any]] = []

    for blob in [cleaned, *_balanced_objects(cleaned)]:
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("molecules"), list):
            records = [m for m in data["molecules"] if isinstance(m, dict)]
            break
        if isinstance(data, dict) and "smiles" in data:
            records = [data]
            break
        if isinstance(data, list) and data and isinstance(data[0], dict):
            records = [m for m in data if isinstance(m, dict)]
            break

    if records:
        candidates = []
        for i, rec in enumerate(records, 1):
            smiles = str(rec.get("smiles", "")).strip()
            if not smiles:
                continue
            features = rec.get("key_features") or []
            if isinstance(features, str):
                features = [features]
            candidates.append(
                Candidate(
                    smiles=smiles,
                    name=str(rec.get("name") or f"cpd_{round_index}_{i}"),
                    rationale=str(rec.get("rationale", "")),
                    key_features=[str(f) for f in features],
                    source="json",
                    round_index=round_index,
                )
            )
        if candidates:
            return candidates

    log.warning("JSON parse failed; falling back to SMILES scraping")
    return _scrape_smiles(cleaned, round_index)


def _scrape_smiles(text: str, round_index: int) -> list[Candidate]:
    found: list[Candidate] = []
    seen: set[str] = set()
    for raw_line in text.splitlines():
        line = raw_line.strip().strip("-*•`\"' ,")
        if ":" in line:
            line = line.split(":", 1)[1].strip().strip("\"', ")
        for token in line.split():
            token = token.strip("\"', ;")
            if token in seen:
                continue
            if _is_plausible_smiles(token):
                seen.add(token)
                found.append(
                    Candidate(
                        smiles=token,
                        name=f"cpd_{round_index}_{len(found) + 1}",
                        rationale="recovered from unstructured output",
                        source="scraped",
                        round_index=round_index,
                    )
                )
    return found


class MoleculeGenerator:
    """Prompt -> candidates, with retries and optional oversampling."""

    def __init__(self, backend: LLMBackend, temperature: float = 0.8, max_tokens: int = 2048):
        self.backend = backend
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        *,
        round_index: int = 0,
        attempts: int = 3,
    ) -> tuple[list[Candidate], str]:
        """Return ``(candidates, raw_text)`` for one completion."""
        raw = complete_with_retry(
            self.backend,
            prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            attempts=attempts,
        )
        candidates = parse_completion(raw, round_index=round_index)
        log.info("backend %s returned %d candidates", self.backend.name, len(candidates))
        return candidates, raw

    def generate_batch(
        self,
        prompt: str,
        n_calls: int = 1,
        *,
        round_index: int = 0,
        temperature_jitter: float = 0.1,
    ) -> list[Candidate]:
        """Several completions of the same prompt at jittered temperature.

        Oversampling is the cheapest lever against the mode collapse measured in
        v1 (43% duplicate SMILES); the deduplication happens in
        :mod:`molgen.chem.diversity`.
        """
        base_t = self.temperature
        out: list[Candidate] = []
        try:
            for call in range(n_calls):
                self.temperature = min(1.2, base_t + call * temperature_jitter)
                candidates, _ = self.generate(prompt, round_index=round_index)
                out.extend(candidates)
        finally:
            self.temperature = base_t
        return out
