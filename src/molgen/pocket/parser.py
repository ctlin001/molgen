"""Binding-pocket extraction from PDB/mmCIF receptor structures.

The pocket is defined geometrically: every standard residue with at least one
atom within ``cutoff`` angstroms of a heteroatom (the co-crystallised ligand).
Waters and common crystallisation additives are excluded.

This module is deliberately deterministic — no LLM touches structural data.
See docs/architecture.md.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from Bio.PDB import MMCIFParser, NeighborSearch, PDBParser
from Bio.PDB.Polypeptide import protein_letters_3to1

log = logging.getLogger(__name__)

HYDROPHOBIC = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "PRO", "GLY"}
CHARGED = {"LYS", "ARG", "HIS", "ASP", "GLU"}
POLAR = {"SER", "THR", "ASN", "GLN", "CYS", "TYR"}

#: Heteroatoms that are solvent/buffer rather than a real ligand.
SOLVENT = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "GOL", "EDO", "PEG", "MPD", "ACT",
    "DMS", "TRS", "IMD", "FMT", "CL", "NA", "K", "MG", "CA", "ZN", "MN",
}


@dataclass
class PocketFeatures:
    """Composition and geometry of a binding pocket."""

    residues: list[dict[str, Any]] = field(default_factory=list)
    coordinates: list[list[float]] = field(default_factory=list)
    residue_counts: dict[str, int] = field(default_factory=dict)
    centroid: list[float] = field(default_factory=list)
    bounding_box: list[float] = field(default_factory=list)
    volume: float = 0.0
    n_hydrophobic: int = 0
    n_charged: int = 0
    n_polar: int = 0

    @property
    def n_residues(self) -> int:
        return len(self.residues)

    def composition_pct(self) -> dict[str, float]:
        n = max(self.n_residues, 1)
        return {
            "hydrophobic": 100 * self.n_hydrophobic / n,
            "charged": 100 * self.n_charged / n,
            "polar": 100 * self.n_polar / n,
        }

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["n_residues"] = self.n_residues
        d["composition_pct"] = self.composition_pct()
        return d


class PocketExtractor:
    """Extract pocket residues and descriptive features from a receptor file.

    Parameters
    ----------
    structure_file:
        Path to a ``.pdb`` or ``.cif`` file containing receptor + ligand.
    ligand_resname:
        Restrict pocket detection to this HET residue name. When ``None``
        every non-solvent heteroatom is treated as ligand.
    """

    def __init__(self, structure_file: str | Path, ligand_resname: str | None = None):
        self.path = Path(structure_file)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        parser = MMCIFParser(QUIET=True) if self.path.suffix.lower() in {".cif", ".mmcif"} \
            else PDBParser(QUIET=True)
        self.structure = parser.get_structure(self.path.stem, str(self.path))
        self.ligand_resname = ligand_resname

    # ---------------------------------------------------------------- atoms

    def _partition_atoms(self):
        """Split atoms into (protein_atoms, ligand_atoms, ligand_residues)."""
        protein_atoms, ligand_atoms, ligand_residues = [], [], []

        for residue in self.structure.get_residues():
            hetflag = residue.id[0]
            resname = residue.get_resname().strip()

            if hetflag == " ":
                protein_atoms.extend(residue.get_atoms())
                continue

            if resname in SOLVENT:
                continue
            if self.ligand_resname and resname != self.ligand_resname:
                continue

            ligand_residues.append(residue)
            ligand_atoms.extend(residue.get_atoms())

        return protein_atoms, ligand_atoms, ligand_residues

    # --------------------------------------------------------------- pocket

    def pocket_residues(self, cutoff: float = 5.0) -> list:
        """Residues with any atom within ``cutoff`` A of a ligand atom.

        Uses a KD-tree (``NeighborSearch``) rather than the O(n*m) double loop
        of the original implementation — roughly 100x faster on a full receptor.
        """
        protein_atoms, ligand_atoms, _ = self._partition_atoms()
        if not ligand_atoms:
            log.warning("no ligand heteroatoms found in %s", self.path.name)
            return []
        if not protein_atoms:
            raise ValueError(f"no standard protein residues in {self.path.name}")

        search = NeighborSearch(protein_atoms)
        hits = {}
        for lig_atom in ligand_atoms:
            for atom in search.search(lig_atom.coord, cutoff):
                residue = atom.get_parent()
                hits[residue.get_full_id()] = residue

        return sorted(hits.values(), key=lambda r: (r.get_parent().id, r.id[1]))

    def features(self, residues: list) -> PocketFeatures:
        """Summarise composition and geometry of the pocket residues."""
        feats = PocketFeatures()

        for residue in residues:
            resname = residue.get_resname().strip()
            feats.residues.append({
                "name": resname,
                "id": residue.id[1],
                "chain": residue.get_parent().id,
                "one_letter": protein_letters_3to1.get(resname, "X"),
            })
            feats.residue_counts[resname] = feats.residue_counts.get(resname, 0) + 1

            if resname in HYDROPHOBIC:
                feats.n_hydrophobic += 1
            elif resname in CHARGED:
                feats.n_charged += 1
            elif resname in POLAR:
                feats.n_polar += 1

            if "CA" in residue:
                feats.coordinates.append(residue["CA"].get_coord().tolist())

        if feats.coordinates:
            coords = np.asarray(feats.coordinates)
            lo, hi = coords.min(axis=0), coords.max(axis=0)
            feats.centroid = coords.mean(axis=0).tolist()
            feats.bounding_box = (hi - lo).tolist()
            feats.volume = float(np.prod(hi - lo))

        return feats

    def ligand_centroid(self) -> list[float] | None:
        """Centroid of ligand heavy atoms — the preferred docking box centre."""
        _, ligand_atoms, _ = self._partition_atoms()
        heavy = [a.coord for a in ligand_atoms if a.element != "H"]
        if not heavy:
            return None
        return np.asarray(heavy).mean(axis=0).tolist()

    # ------------------------------------------------------------- metadata

    def metadata(self) -> dict[str, Any]:
        """Header fields from the structure file, best-effort."""
        meta: dict[str, Any] = {
            "id": self.path.stem.upper(),
            "title": None,
            "classification": None,
            "method": None,
            "resolution": None,
        }

        if self.path.suffix.lower() not in {".pdb", ".ent"}:
            header = getattr(self.structure, "header", {}) or {}
            meta.update({
                "title": header.get("name"),
                "method": header.get("structure_method"),
                "resolution": header.get("resolution"),
            })
            return meta

        title_lines = []
        with self.path.open() as fh:
            for line in fh:
                tag = line[:6]
                if tag == "HEADER":
                    meta["classification"] = line[10:50].strip() or None
                    if line[62:66].strip():
                        meta["id"] = line[62:66].strip()
                elif tag == "TITLE ":
                    title_lines.append(line[10:].strip())
                elif tag == "EXPDTA":
                    meta["method"] = line[10:].strip()
                elif line.startswith("REMARK   2 RESOLUTION"):
                    for token in line.split():
                        try:
                            meta["resolution"] = float(token)
                            break
                        except ValueError:
                            continue
                elif tag in {"ATOM  ", "HETATM"}:
                    break

        if title_lines:
            meta["title"] = " ".join(title_lines)
        return meta


def describe_pocket(meta: dict[str, Any], feats: PocketFeatures, top_n: int = 12) -> str:
    """Natural-language pocket summary used as LLM prompt context."""
    pct = feats.composition_pct()
    lines = [
        f"Target: {meta.get('id', 'unknown')}",
        f"Title: {meta.get('title') or 'n/a'}",
        f"Classification: {meta.get('classification') or 'n/a'}",
        f"Method: {meta.get('method') or 'n/a'}  Resolution: {meta.get('resolution') or 'n/a'} A",
        "",
        "Binding pocket:",
        f"  residues: {feats.n_residues}",
        f"  bounding-box volume: {feats.volume:.1f} A^3",
        f"  hydrophobic: {feats.n_hydrophobic} ({pct['hydrophobic']:.0f}%)",
        f"  charged:     {feats.n_charged} ({pct['charged']:.0f}%)",
        f"  polar:       {feats.n_polar} ({pct['polar']:.0f}%)",
        "",
        "Lining residues:",
    ]
    for res in feats.residues[:top_n]:
        lines.append(f"  {res['name']}{res['id']} (chain {res['chain']})")
    if feats.n_residues > top_n:
        lines.append(f"  ... and {feats.n_residues - top_n} more")
    return "\n".join(lines)
