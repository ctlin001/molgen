"""Search-box derivation for docking.

Manually drawing a grid box in a GUI is the least reproducible step in a
docking protocol and does not survive being handed to a colleague. The box here
is derived from coordinates, so the same inputs always give the same box.

Preference order:
    1. reference ligand extent (best — the box matches the known binding site)
    2. pocket residue CA coordinates (when no ligand is present)
    3. explicit centre and size from config
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

#: Vina slows sharply on large boxes and pose quality degrades.
MAX_RECOMMENDED_EDGE_A = 30.0


@dataclass
class DockingBox:
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    source: str = "unspecified"

    @property
    def volume(self) -> float:
        return self.size_x * self.size_y * self.size_z

    def warnings(self) -> list[str]:
        out = []
        for axis in ("x", "y", "z"):
            edge = getattr(self, f"size_{axis}")
            if edge > MAX_RECOMMENDED_EDGE_A:
                out.append(f"size_{axis} = {edge:.1f} A exceeds {MAX_RECOMMENDED_EDGE_A} A")
            if edge < 10.0:
                out.append(f"size_{axis} = {edge:.1f} A may be too small for a drug-like ligand")
        return out

    def to_vina_args(self) -> list[str]:
        return [
            "--center_x", f"{self.center_x:.3f}",
            "--center_y", f"{self.center_y:.3f}",
            "--center_z", f"{self.center_z:.3f}",
            "--size_x", f"{self.size_x:.3f}",
            "--size_y", f"{self.size_y:.3f}",
            "--size_z", f"{self.size_z:.3f}",
        ]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["volume"] = round(self.volume, 1)
        d["warnings"] = self.warnings()
        return d

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DockingBox:
        data = json.loads(Path(path).read_text())
        fields = {k: data[k] for k in (
            "center_x", "center_y", "center_z", "size_x", "size_y", "size_z"
        )}
        return cls(**fields, source=data.get("source", "loaded"))


def box_from_coords(
    coords: np.ndarray,
    padding: float = 5.0,
    min_edge: float = 15.0,
    cubic: bool = False,
    source: str = "coordinates",
) -> DockingBox:
    """Axis-aligned box around ``coords`` with ``padding`` on every side."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[0] == 0:
        raise ValueError("need an (N, 3) array of coordinates")

    lo, hi = coords.min(axis=0), coords.max(axis=0)
    centre = (lo + hi) / 2.0
    size = np.maximum(hi - lo + 2 * padding, min_edge)
    if cubic:
        size = np.full(3, size.max())

    box = DockingBox(
        center_x=float(centre[0]),
        center_y=float(centre[1]),
        center_z=float(centre[2]),
        size_x=float(size[0]),
        size_y=float(size[1]),
        size_z=float(size[2]),
        source=source,
    )
    for warning in box.warnings():
        log.warning("docking box: %s", warning)
    return box


def box_from_reference_ligand(
    reference_path: str | Path,
    padding: float = 5.0,
    resname: str | None = None,
    cubic: bool = False,
) -> DockingBox:
    """Box around a reference ligand's heavy atoms — the preferred route."""
    from .rmsd import heavy_atom_coords, load_reference_ligand

    mol = load_reference_ligand(reference_path, resname)
    coords, _ = heavy_atom_coords(mol)
    return box_from_coords(coords, padding, cubic=cubic, source="reference_ligand")


def box_from_pocket(pocket_features, padding: float = 4.0, cubic: bool = False) -> DockingBox:
    """Box around pocket residue CA atoms.

    Fallback for apo receptors. CA coordinates trace the pocket wall rather than
    the cavity, so the default padding is smaller than for a reference ligand.
    """
    coords = np.asarray(getattr(pocket_features, "coordinates", pocket_features), dtype=float)
    if coords.size == 0:
        raise ValueError("pocket has no coordinates; supply an explicit box")
    return box_from_coords(coords, padding, cubic=cubic, source="pocket_residues")


def box_from_config(center: tuple[float, float, float], size: tuple[float, float, float]):
    """Explicit box from configuration."""
    return DockingBox(*center, *size, source="config")
