"""Diversity, deduplication and novelty.

Benchmarking v1 produced 1000 molecules that were 100% Lipinski-compliant,
43% duplicated, and shared a single Bemis-Murcko scaffold. Compliance is not
diversity. This module makes diversity a measured, enforced property rather
than an assumed one, and it runs on every batch.

Three jobs:

* **dedupe** — collapse identical structures (InChIKey, not raw SMILES; the
  same molecule has many valid SMILES spellings)
* **diversity report** — scaffold counts, Tanimoto distribution, internal
  diversity, scaffold entropy
* **novelty filter** — Tanimoto screen against a reference library, so docking
  compute is not spent rediscovering known compounds
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(mol: Chem.Mol):
    """Morgan (ECFP4-equivalent) count-free bit fingerprint."""
    return _MORGAN.GetFingerprint(mol)


def fingerprints(smiles: Iterable[str]) -> tuple[list, list[str]]:
    """Return ``(fps, kept_smiles)``, silently dropping unparseable entries."""
    fps, kept = [], []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fps.append(fingerprint(mol))
        kept.append(smi)
    return fps, kept


def murcko_scaffold(smiles: str, generic: bool = False) -> str | None:
    """Bemis-Murcko scaffold SMILES.

    ``generic=True`` strips atom and bond types, collapsing e.g. pyridine and
    benzene to the same ring skeleton — the stricter diversity view.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            scaffold = MurckoScaffold.MakeScaffoldGeneric(scaffold)
        smi = Chem.MolToSmiles(scaffold)
        return smi or None
    except Exception:
        return None


# ------------------------------------------------------------------ dedupe


def deduplicate(smiles: Sequence[str]) -> tuple[list[str], dict[str, int]]:
    """Collapse duplicates by InChIKey.

    Returns ``(unique_smiles, counts)`` where ``counts`` maps the retained
    canonical SMILES to how many times it was generated. Comparing raw SMILES
    strings under-counts duplicates badly — that was part of why v1's rate
    looked acceptable until it was measured properly.
    """
    seen: dict[str, str] = {}
    counts: Counter[str] = Counter()

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        key = Chem.MolToInchiKey(mol)
        canonical = Chem.MolToSmiles(mol, canonical=True)
        if key not in seen:
            seen[key] = canonical
        counts[seen[key]] += 1

    return list(seen.values()), dict(counts)


# -------------------------------------------------------------- statistics


@dataclass
class DiversityReport:
    n_input: int = 0
    n_unique: int = 0
    duplicate_rate: float = 0.0
    n_scaffolds: int = 0
    n_generic_scaffolds: int = 0
    scaffold_diversity: float = 0.0
    scaffold_entropy: float = 0.0
    largest_scaffold_share: float = 0.0
    internal_diversity: float = 0.0
    mean_pairwise_tanimoto: float = 0.0
    tanimoto_histogram: dict[str, int] = field(default_factory=dict)
    top_scaffolds: list[tuple[str, int]] = field(default_factory=list)

    def verdict(self) -> str:
        """One-line quality call, using the v1 run as the reference failure."""
        problems = []
        if self.duplicate_rate > 0.15:
            problems.append(f"duplicate rate {self.duplicate_rate:.0%}")
        if self.scaffold_diversity < 0.30:
            problems.append(f"scaffold diversity {self.scaffold_diversity:.2f}")
        if self.largest_scaffold_share > 0.40:
            problems.append(f"one scaffold covers {self.largest_scaffold_share:.0%}")
        if not problems:
            return "PASS - diverse output"
        return "FAIL - " + "; ".join(problems)

    def to_dict(self) -> dict[str, Any]:
        return {**self.__dict__, "verdict": self.verdict()}


def pairwise_tanimoto(fps: Sequence) -> np.ndarray:
    """Condensed upper-triangle vector of pairwise similarities."""
    if len(fps) < 2:
        return np.zeros(0)
    out = []
    for i in range(len(fps) - 1):
        out.extend(DataStructs.BulkTanimotoSimilarity(fps[i], list(fps[i + 1 :])))
    return np.asarray(out, dtype=float)


def analyse_diversity(smiles: Sequence[str], generic_scaffolds: bool = True) -> DiversityReport:
    """Full diversity report for one batch."""
    report = DiversityReport(n_input=len(smiles))
    if not smiles:
        return report

    unique, counts = deduplicate(smiles)
    report.n_unique = len(unique)
    total = sum(counts.values()) or 1
    report.duplicate_rate = round(1 - len(unique) / total, 3)

    scaffolds = [s for s in (murcko_scaffold(x) for x in unique) if s]
    scaffold_counts = Counter(scaffolds)
    report.n_scaffolds = len(scaffold_counts)
    report.top_scaffolds = scaffold_counts.most_common(10)

    if generic_scaffolds:
        generic = [s for s in (murcko_scaffold(x, generic=True) for x in unique) if s]
        report.n_generic_scaffolds = len(set(generic))

    if unique:
        report.scaffold_diversity = round(report.n_scaffolds / len(unique), 3)
    if scaffold_counts:
        n = sum(scaffold_counts.values())
        report.largest_scaffold_share = round(max(scaffold_counts.values()) / n, 3)
        report.scaffold_entropy = round(
            -sum((c / n) * math.log2(c / n) for c in scaffold_counts.values()), 3
        )

    fps, _ = fingerprints(unique)
    sims = pairwise_tanimoto(fps)
    if sims.size:
        report.mean_pairwise_tanimoto = round(float(sims.mean()), 3)
        report.internal_diversity = round(1 - float(sims.mean()), 3)
        edges = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
        labels = ["0.0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
        hist, _ = np.histogram(sims, bins=edges)
        report.tanimoto_histogram = dict(zip(labels, (int(h) for h in hist), strict=True))

    return report


# ------------------------------------------------------------------ novelty


class NoveltyFilter:
    """Tanimoto screen against a reference library.

    Runs *before* docking: a fingerprint comparison costs microseconds, a Vina
    run costs seconds to minutes. Anything above ``threshold`` similarity to a
    known compound is a near-duplicate and is deprioritised.
    """

    def __init__(self, reference_smiles: Sequence[str], threshold: float = 0.85):
        self.threshold = threshold
        self.reference_fps, self.reference_smiles = fingerprints(reference_smiles)
        log.info("novelty filter loaded with %d reference compounds", len(self.reference_fps))

    @classmethod
    def from_file(cls, path: str | Path, threshold: float = 0.85, column: str = "smiles"):
        """Load references from ``.smi``, ``.csv`` or ``.sdf``."""
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".sdf":
            supplier = Chem.SDMolSupplier(str(path))
            smiles = [Chem.MolToSmiles(m) for m in supplier if m is not None]
        elif suffix == ".csv":
            import csv

            with path.open(newline="") as fh:
                rows = list(csv.DictReader(fh))
            key = column if rows and column in rows[0] else (list(rows[0]) if rows else [""])[0]
            smiles = [r[key] for r in rows if r.get(key)]
        else:
            smiles = [
                line.split()[0]
                for line in path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        return cls(smiles, threshold=threshold)

    def score(self, smiles: str) -> tuple[float, str | None]:
        """Return ``(max_similarity, nearest_reference_smiles)``."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or not self.reference_fps:
            return 0.0, None
        sims = DataStructs.BulkTanimotoSimilarity(fingerprint(mol), self.reference_fps)
        idx = int(np.argmax(sims))
        return round(float(sims[idx]), 3), self.reference_smiles[idx]

    def annotate(self, smiles_list: Sequence[str]) -> list[dict[str, Any]]:
        """Per-molecule novelty record."""
        out = []
        for smi in smiles_list:
            sim, nearest = self.score(smi)
            out.append({
                "smiles": smi,
                "max_similarity": sim,
                "nearest_reference": nearest,
                "novel": sim < self.threshold,
            })
        return out

    def filter(self, smiles_list: Sequence[str]) -> tuple[list[str], list[dict[str, Any]]]:
        """Return ``(novel_smiles, all_annotations)``."""
        annotations = self.annotate(smiles_list)
        return [a["smiles"] for a in annotations if a["novel"]], annotations


def pick_diverse_subset(smiles: Sequence[str], k: int) -> list[str]:
    """MaxMin selection of ``k`` maximally dissimilar molecules.

    Used to cap how many candidates go to docking while keeping chemical space
    coverage, instead of taking the first ``k`` the model happened to emit.
    """
    fps, kept = fingerprints(smiles)
    if len(kept) <= k:
        return list(kept)

    picked = [0]
    min_sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[0], fps), dtype=float)
    while len(picked) < k:
        min_sims[picked] = 1.1
        nxt = int(np.argmin(min_sims))
        picked.append(nxt)
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[nxt], fps), dtype=float)
        min_sims = np.minimum(min_sims, sims)
    return [kept[i] for i in picked]
