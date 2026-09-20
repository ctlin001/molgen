"""Protein-ligand interaction fingerprinting (simplified PLIP-style).

Detects hydrophobic contacts, putative hydrogen bonds and aromatic stacking
from a co-crystal structure. Geometry only — no energy model — so results are
used as *prompt context*, never as a binding prediction.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import NeighborSearch

from .parser import PocketExtractor

log = logging.getLogger(__name__)

HYDROPHOBIC_RES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO"}
AROMATIC_RES = {"PHE", "TYR", "TRP", "HIS"}
HBOND_ELEMENTS = {"N", "O", "S"}

AROMATIC_RING_ATOMS = {
    "PHE": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TYR": ["CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "TRP": ["CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "HIS": ["CG", "ND1", "CD2", "CE1", "NE2"],
}


@dataclass
class InteractionCutoffs:
    hydrophobic: float = 4.0
    hbond: float = 3.5
    stacking: float = 5.5


@dataclass
class InteractionProfile:
    hydrophobic: list[dict[str, Any]] = field(default_factory=list)
    hbond: list[dict[str, Any]] = field(default_factory=list)
    stacking: list[dict[str, Any]] = field(default_factory=list)

    def counts(self) -> dict[str, int]:
        return {
            "hydrophobic": len(self.hydrophobic),
            "hbond": len(self.hbond),
            "stacking": len(self.stacking),
        }

    @property
    def total(self) -> int:
        return sum(self.counts().values())

    def hot_residues(self, top_n: int = 8) -> list[tuple[str, int]]:
        counter: Counter[str] = Counter()
        for group in (self.hydrophobic, self.hbond, self.stacking):
            for item in group:
                counter[f"{item['res_name']}{item['res_id']}"] += 1
        return counter.most_common(top_n)

    def to_dict(self) -> dict[str, Any]:
        return {
            "counts": self.counts(),
            "total": self.total,
            "hot_residues": self.hot_residues(),
            "interactions": {
                "hydrophobic": self.hydrophobic,
                "hbond": self.hbond,
                "stacking": self.stacking,
            },
        }


class InteractionAnalyzer:
    """Geometric interaction detection between receptor and bound ligand."""

    def __init__(
        self,
        structure_file: str | Path,
        ligand_resname: str | None = None,
        cutoffs: InteractionCutoffs | None = None,
    ):
        self.extractor = PocketExtractor(structure_file, ligand_resname)
        self.cutoffs = cutoffs or InteractionCutoffs()

    @staticmethod
    def _record(prot_atom, lig_atom, distance: float, kind: str) -> dict[str, Any]:
        residue = prot_atom.get_parent()
        return {
            "type": kind,
            "res_name": residue.get_resname().strip(),
            "res_id": residue.id[1],
            "chain": residue.get_parent().id,
            "protein_atom": prot_atom.get_name(),
            "ligand_atom": getattr(lig_atom, "get_name", lambda: None)(),
            "distance": round(float(distance), 2),
        }

    def analyse(self) -> InteractionProfile:
        protein_atoms, ligand_atoms, ligand_residues = self.extractor._partition_atoms()
        profile = InteractionProfile()

        if not ligand_atoms:
            log.warning("no ligand present; interaction profile will be empty")
            return profile

        search = NeighborSearch(protein_atoms)
        seen_hydrophobic: set[tuple] = set()

        for lig_atom in ligand_atoms:
            element = (lig_atom.element or "").strip().upper()

            if element == "C":
                for prot_atom in search.search(lig_atom.coord, self.cutoffs.hydrophobic):
                    if (prot_atom.element or "").strip().upper() != "C":
                        continue
                    residue = prot_atom.get_parent()
                    if residue.get_resname().strip() not in HYDROPHOBIC_RES:
                        continue
                    key = (residue.get_full_id(), lig_atom.get_name())
                    if key in seen_hydrophobic:
                        continue
                    seen_hydrophobic.add(key)
                    d = np.linalg.norm(lig_atom.coord - prot_atom.coord)
                    profile.hydrophobic.append(
                        self._record(prot_atom, lig_atom, d, "hydrophobic")
                    )

            if element in HBOND_ELEMENTS:
                for prot_atom in search.search(lig_atom.coord, self.cutoffs.hbond):
                    if (prot_atom.element or "").strip().upper() not in HBOND_ELEMENTS:
                        continue
                    d = np.linalg.norm(lig_atom.coord - prot_atom.coord)
                    profile.hbond.append(self._record(prot_atom, lig_atom, d, "hbond"))

        profile.stacking = self._stacking(protein_atoms, ligand_residues)
        return profile

    def _stacking(self, protein_atoms, ligand_residues) -> list[dict[str, Any]]:
        """Aromatic ring-centroid proximity.

        Ring centroids are computed from named ring atoms rather than whole-residue
        centroids, which otherwise drift toward the backbone and inflate counts.
        """
        results: list[dict[str, Any]] = []
        if not ligand_residues:
            return results

        residues = {a.get_parent().get_full_id(): a.get_parent() for a in protein_atoms}
        for residue in residues.values():
            resname = residue.get_resname().strip()
            if resname not in AROMATIC_RES:
                continue
            ring = [residue[n].coord for n in AROMATIC_RING_ATOMS[resname] if n in residue]
            if len(ring) < 5:
                continue
            prot_centre = np.mean(ring, axis=0)

            for lig_res in ligand_residues:
                coords = np.asarray([a.coord for a in lig_res if a.element != "H"])
                if coords.size == 0:
                    continue
                lig_centre = coords.mean(axis=0)
                d = float(np.linalg.norm(prot_centre - lig_centre))
                if d <= self.cutoffs.stacking:
                    results.append({
                        "type": "stacking",
                        "res_name": resname,
                        "res_id": residue.id[1],
                        "chain": residue.get_parent().id,
                        "protein_atom": "ring-centroid",
                        "ligand_atom": lig_res.get_resname().strip(),
                        "distance": round(d, 2),
                    })
        return results


def describe_interactions(profile: InteractionProfile, examples: int = 3) -> str:
    """Natural-language interaction summary for the generation prompt."""
    if profile.total == 0:
        return "No ligand-derived interaction data available for this pocket."

    counts = profile.counts()
    lines = [
        f"Observed interactions with the co-crystallised ligand ({profile.total} total):",
        f"  hydrophobic contacts: {counts['hydrophobic']}",
        f"  hydrogen bonds:       {counts['hbond']}",
        f"  aromatic stacking:    {counts['stacking']}",
        "",
        "Most engaged residues:",
    ]
    for label, n in profile.hot_residues():
        lines.append(f"  {label}: {n} contacts")

    lines.append("")
    lines.append("Representative contacts:")
    for kind, group in (
        ("hydrophobic", profile.hydrophobic),
        ("hbond", profile.hbond),
        ("stacking", profile.stacking),
    ):
        for item in group[:examples]:
            lines.append(
                f"  {kind}: {item['res_name']}{item['res_id']}"
                f" - {item['ligand_atom'] or 'ligand'} ({item['distance']} A)"
            )
    return "\n".join(lines)
