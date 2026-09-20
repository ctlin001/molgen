"""Symmetry-aware RMSD between a docked pose and a reference ligand.

Atom correspondence is solved as a linear assignment problem (Hungarian
algorithm) within each element type, so chemically equivalent atoms — the two
oxygens of a carboxylate, the four CH positions of a para-substituted ring —
are matched by geometry rather than by file order. Naive index-order RMSD on
such molecules is inflated by several angstroms for no physical reason.

Not to be confused with Vina's own ``rmsd_lb`` / ``rmsd_ub``
-----------------------------------------------------------
Those columns in Vina output compare each alternative pose to the *best pose of
the same run*. They are a pose-diversity measure. They say nothing about how
close the prediction is to the crystal structure. Redocking accuracy needs the
reference-based RMSD computed here, against an experimentally determined pose.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from scipy.optimize import linear_sum_assignment

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

#: Conventional redocking success threshold.
SUCCESS_THRESHOLD_A = 2.0


@dataclass
class RMSDResult:
    rmsd: float
    n_atoms_matched: int
    method: str
    reference_atoms: int
    pose_atoms: int
    note: str = ""

    @property
    def is_success(self) -> bool:
        """True when the pose reproduces the reference within 2.0 A."""
        return self.rmsd <= SUCCESS_THRESHOLD_A


def heavy_atom_coords(mol: Chem.Mol, conf_id: int = -1) -> tuple[np.ndarray, list[str]]:
    """Return ``(coords, elements)`` for heavy atoms of one conformer."""
    if mol.GetNumConformers() == 0:
        raise ValueError("molecule has no 3D conformer")
    conf = mol.GetConformer(conf_id)
    coords, elements = [], []
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() <= 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        coords.append([pos.x, pos.y, pos.z])
        elements.append(atom.GetSymbol())
    return np.asarray(coords, dtype=float), elements


def hungarian_rmsd(
    ref_coords: np.ndarray,
    ref_elements: list[str],
    pose_coords: np.ndarray,
    pose_elements: list[str],
) -> tuple[float, int]:
    """Minimum RMSD over element-wise optimal atom assignment.

    Atoms are grouped by element; within each group the cost matrix of pairwise
    squared distances is solved with ``linear_sum_assignment``. Unequal group
    sizes are handled by matching the smaller group — the surplus atoms are
    reported but excluded rather than forced into a spurious pairing.
    """
    total_sq = 0.0
    matched = 0

    for element in set(ref_elements) & set(pose_elements):
        ref_idx = [i for i, e in enumerate(ref_elements) if e == element]
        pose_idx = [i for i, e in enumerate(pose_elements) if e == element]
        if not ref_idx or not pose_idx:
            continue

        a = ref_coords[ref_idx]
        b = pose_coords[pose_idx]
        cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
        rows, cols = linear_sum_assignment(cost)

        total_sq += float(cost[rows, cols].sum())
        matched += len(rows)

    if matched == 0:
        return float("nan"), 0
    return float(np.sqrt(total_sq / matched)), matched


def naive_rmsd(ref_coords: np.ndarray, pose_coords: np.ndarray) -> float:
    """Index-order RMSD. Only valid when atom ordering is known identical."""
    n = min(len(ref_coords), len(pose_coords))
    diff = ref_coords[:n] - pose_coords[:n]
    return float(np.sqrt((diff ** 2).sum() / n))


def compute_rmsd(
    reference: Chem.Mol,
    pose: Chem.Mol,
    *,
    method: str = "hungarian",
    ref_conf: int = -1,
    pose_conf: int = -1,
) -> RMSDResult:
    """RMSD between a reference ligand and a docked pose.

    ``method`` is ``"hungarian"`` (default), ``"naive"``, or ``"best_rms"``
    which defers to RDKit's ``GetBestRMS`` when the two molecules are the same
    species and substructure matching succeeds.
    """
    ref_coords, ref_elements = heavy_atom_coords(reference, ref_conf)
    pose_coords, pose_elements = heavy_atom_coords(pose, pose_conf)

    if method == "best_rms":
        from rdkit.Chem import rdMolAlign

        try:
            value = rdMolAlign.GetBestRMS(Chem.RemoveHs(pose), Chem.RemoveHs(reference))
            return RMSDResult(
                rmsd=round(value, 3),
                n_atoms_matched=min(len(ref_coords), len(pose_coords)),
                method="best_rms",
                reference_atoms=len(ref_coords),
                pose_atoms=len(pose_coords),
            )
        except Exception as exc:
            log.debug("GetBestRMS failed (%s); falling back to Hungarian", exc)
            method = "hungarian"

    if method == "naive":
        value = naive_rmsd(ref_coords, pose_coords)
        matched = min(len(ref_coords), len(pose_coords))
    else:
        value, matched = hungarian_rmsd(ref_coords, ref_elements, pose_coords, pose_elements)
        method = "hungarian"

    note = ""
    if len(ref_coords) != len(pose_coords):
        note = (
            f"atom-count mismatch (ref {len(ref_coords)}, pose {len(pose_coords)}); "
            "RMSD computed over matched atoms only"
        )

    return RMSDResult(
        rmsd=round(value, 3),
        n_atoms_matched=matched,
        method=method,
        reference_atoms=len(ref_coords),
        pose_atoms=len(pose_coords),
        note=note,
    )


def load_reference_ligand(path: str | Path, resname: str | None = None) -> Chem.Mol:
    """Load a reference ligand from SDF, MOL2, PDB or PDBQT.

    For PDB input, HETATM records are extracted; ``resname`` narrows to one
    ligand when several are present.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".sdf":
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True)
        mols = [m for m in supplier if m is not None]
        if not mols:
            raise ValueError(f"no readable molecule in {path}")
        return mols[0]

    if suffix in {".mol2"}:
        mol = Chem.MolFromMol2File(str(path), removeHs=False)
        if mol is None:
            raise ValueError(f"could not parse {path}")
        return mol

    if suffix in {".pdb", ".ent", ".pdbqt"}:
        lines = [
            line[:66] if suffix == ".pdbqt" else line
            for line in path.read_text().splitlines()
            if line.startswith("HETATM") or (suffix == ".pdbqt" and line.startswith("ATOM"))
        ]
        if resname:
            lines = [line for line in lines if line[17:20].strip() == resname]
        if not lines:
            raise ValueError(f"no HETATM records found in {path}")
        block = "\n".join(lines) + "\nEND\n"
        mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        if mol is None:
            raise ValueError(f"could not parse ligand records from {path}")
        return mol

    raise ValueError(f"unsupported reference format: {suffix}")


def rmsd_to_reference(
    reference_path: str | Path,
    pose_path: str | Path,
    *,
    resname: str | None = None,
    method: str = "hungarian",
) -> RMSDResult:
    """Convenience wrapper: load both files and compute RMSD."""
    reference = load_reference_ligand(reference_path, resname)
    pose = load_reference_ligand(pose_path)
    return compute_rmsd(reference, pose, method=method)
