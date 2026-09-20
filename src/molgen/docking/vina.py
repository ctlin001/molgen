"""AutoDock Vina docking, including Vina-GPU 2.1.

Notes carried over from building this on Windows
------------------------------------------------
* Parallelism uses ``ThreadPoolExecutor``, not ``ProcessPoolExecutor``. Vina
  runs in a subprocess so the GIL is not the bottleneck, and process pools
  raise ``BrokenProcessPool`` under Jupyter on Windows.
* Vina-GPU 2.1 runs through WSL2. Paths must be translated to ``/mnt/c/...``
  form; :func:`to_wsl_path` handles that.
* ``rmsd_lb`` / ``rmsd_ub`` in Vina's output table are pose-diversity values
  relative to the best pose of the same run, not accuracy against a crystal
  structure. Reference RMSD comes from :mod:`molgen.docking.rmsd`.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .box import DockingBox

log = logging.getLogger(__name__)

_SCORE_ROW = re.compile(r"^\s*(\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$")


@dataclass
class Pose:
    mode: int
    affinity: float
    rmsd_lb: float
    rmsd_ub: float


@dataclass
class DockingResult:
    ligand_id: str
    success: bool = False
    poses: list[Pose] = field(default_factory=list)
    output_path: str | None = None
    runtime_s: float = 0.0
    error: str | None = None
    reference_rmsd: float | None = None

    @property
    def best_affinity(self) -> float | None:
        return self.poses[0].affinity if self.poses else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ligand_id": self.ligand_id,
            "success": self.success,
            "best_affinity": self.best_affinity,
            "n_poses": len(self.poses),
            "poses": [p.__dict__ for p in self.poses],
            "output_path": self.output_path,
            "runtime_s": round(self.runtime_s, 2),
            "reference_rmsd": self.reference_rmsd,
            "error": self.error,
        }


def to_wsl_path(path: str | Path) -> str:
    """Translate a Windows path to its WSL2 equivalent.

    ``C:\\Users\\x\\data`` becomes ``/mnt/c/Users/x/data``.
    """
    text = str(Path(path).resolve())
    if len(text) > 1 and text[1] == ":":
        drive, rest = text[0].lower(), text[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    return text.replace("\\", "/")


def cif_to_pdb(cif_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Convert mmCIF to PDB.

    Tries ``gemmi`` first (fast, handles large entries), then Biopython.
    Structures downloaded from RCSB default to mmCIF while most docking tooling
    still expects PDB.
    """
    cif_path = Path(cif_path)
    out_path = Path(out_path) if out_path else cif_path.with_suffix(".pdb")

    try:
        import gemmi

        structure = gemmi.read_structure(str(cif_path))
        structure.setup_entities()
        structure.write_pdb(str(out_path))
        return out_path
    except ImportError:
        pass

    from Bio.PDB import PDBIO, MMCIFParser

    structure = MMCIFParser(QUIET=True).get_structure(cif_path.stem, str(cif_path))
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path))
    return out_path


class VinaDocker:
    """Wrapper around the Vina or Vina-GPU executable.

    Parameters
    ----------
    executable:
        ``vina``, or the path to ``AutoDock-Vina-GPU-2-1``.
    use_wsl:
        Run the executable through ``wsl`` and translate paths. Required for
        Vina-GPU on Windows.
    """

    def __init__(
        self,
        executable: str = "vina",
        *,
        use_wsl: bool = False,
        exhaustiveness: int = 16,
        num_modes: int = 9,
        cpu: int | None = None,
        seed: int | None = 42,
        timeout: int = 900,
    ):
        self.executable = executable
        self.use_wsl = use_wsl
        self.exhaustiveness = exhaustiveness
        self.num_modes = num_modes
        self.cpu = cpu
        self.seed = seed
        self.timeout = timeout

    # ----------------------------------------------------------- environment

    def check(self) -> str:
        """Verify the executable is reachable; returns its version banner."""
        cmd = (["wsl"] if self.use_wsl else []) + [self.executable, "--version"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"'{self.executable}' not found. Install AutoDock Vina "
                "(conda install -c conda-forge autodock-vina) or set docking.executable."
            ) from exc
        return (proc.stdout or proc.stderr).strip().splitlines()[0] if (proc.stdout or proc.stderr) else "unknown"

    def _path(self, path: str | Path) -> str:
        return to_wsl_path(path) if self.use_wsl else str(Path(path).resolve())

    # -------------------------------------------------------------- receptor

    def prepare_receptor(self, structure_path: str | Path, out_path: str | Path | None = None) -> Path:
        """Produce a receptor PDBQT.

        Uses ``prepare_receptor`` (ADFR suite) when available, then Open Babel,
        and refuses to guess beyond that. A hand-rolled PDB-to-PDBQT copy loses
        partial charges and atom typing, which silently degrades every score.
        """
        structure_path = Path(structure_path)
        if structure_path.suffix.lower() in {".cif", ".mmcif"}:
            structure_path = cif_to_pdb(structure_path)

        out_path = Path(out_path) if out_path else structure_path.with_suffix(".pdbqt")
        if out_path.exists():
            log.info("reusing existing receptor %s", out_path.name)
            return out_path

        if shutil.which("prepare_receptor"):
            subprocess.run(
                ["prepare_receptor", "-r", str(structure_path), "-o", str(out_path), "-A", "hydrogens"],
                check=True, capture_output=True,
            )
            return out_path

        if shutil.which("obabel"):
            log.warning("ADFR prepare_receptor not found; using Open Babel")
            subprocess.run(
                ["obabel", str(structure_path), "-O", str(out_path), "-xr", "-p", "7.4"],
                check=True, capture_output=True,
            )
            return out_path

        raise RuntimeError(
            "No receptor preparation tool found. Install ADFR suite (prepare_receptor) "
            "or Open Babel (conda install -c conda-forge openbabel)."
        )

    def prepare_ligand(self, ligand_path: str | Path, out_path: str | Path | None = None) -> Path:
        """Convert an SDF/PDB ligand to PDBQT with hydrogens at pH 7.4."""
        ligand_path = Path(ligand_path)
        out_path = Path(out_path) if out_path else ligand_path.with_suffix(".pdbqt")

        if shutil.which("obabel") is None:
            raise RuntimeError("Open Babel required: conda install -c conda-forge openbabel")

        subprocess.run(
            ["obabel", str(ligand_path), "-O", str(out_path), "-p", "7.4", "--partialcharge", "gasteiger"],
            check=True, capture_output=True,
        )
        if not out_path.exists() or out_path.stat().st_size == 0:
            raise RuntimeError(f"ligand preparation produced no output for {ligand_path.name}")
        return out_path

    # ---------------------------------------------------------------- docking

    @staticmethod
    def parse_output(stdout: str) -> list[Pose]:
        """Parse Vina's result table."""
        poses = []
        for line in stdout.splitlines():
            match = _SCORE_ROW.match(line)
            if match:
                mode, affinity, lb, ub = match.groups()
                poses.append(Pose(int(mode), float(affinity), float(lb), float(ub)))
        return sorted(poses, key=lambda p: p.affinity)

    def dock(
        self,
        receptor_pdbqt: str | Path,
        ligand_pdbqt: str | Path,
        box: DockingBox,
        out_path: str | Path | None = None,
        ligand_id: str | None = None,
    ) -> DockingResult:
        """Dock one ligand into one receptor."""
        import time

        ligand_pdbqt = Path(ligand_pdbqt)
        ligand_id = ligand_id or ligand_pdbqt.stem
        out_path = Path(out_path) if out_path else ligand_pdbqt.with_name(f"{ligand_id}_docked.pdbqt")

        cmd = (["wsl"] if self.use_wsl else []) + [
            self.executable,
            "--receptor", self._path(receptor_pdbqt),
            "--ligand", self._path(ligand_pdbqt),
            "--out", self._path(out_path),
            "--exhaustiveness", str(self.exhaustiveness),
            "--num_modes", str(self.num_modes),
            *box.to_vina_args(),
        ]
        if self.cpu:
            cmd += ["--cpu", str(self.cpu)]
        if self.seed is not None:
            cmd += ["--seed", str(self.seed)]

        started = time.perf_counter()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return DockingResult(ligand_id, error=f"timed out after {self.timeout}s",
                                 runtime_s=time.perf_counter() - started)

        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            return DockingResult(
                ligand_id,
                error=(proc.stderr or proc.stdout or "vina returned non-zero").strip()[:500],
                runtime_s=elapsed,
            )

        poses = self.parse_output(proc.stdout)
        if not poses:
            return DockingResult(ligand_id, error="no poses parsed from vina output", runtime_s=elapsed)

        return DockingResult(
            ligand_id=ligand_id,
            success=True,
            poses=poses,
            output_path=str(out_path),
            runtime_s=elapsed,
        )

    def dock_many(
        self,
        receptor_pdbqt: str | Path,
        ligand_paths: Sequence[str | Path],
        box: DockingBox,
        out_dir: str | Path,
        max_workers: int = 4,
    ) -> list[DockingResult]:
        """Dock a batch of ligands.

        Thread-based on purpose — see the module docstring.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        results: list[DockingResult] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {}
            for ligand in ligand_paths:
                ligand = Path(ligand)
                stem = ligand.stem
                try:
                    pdbqt = ligand if ligand.suffix == ".pdbqt" else self.prepare_ligand(
                        ligand, out_dir / f"{stem}.pdbqt"
                    )
                except Exception as exc:
                    results.append(DockingResult(stem, error=f"ligand prep failed: {exc}"))
                    continue
                futures[pool.submit(
                    self.dock, receptor_pdbqt, pdbqt, box, out_dir / f"{stem}_docked.pdbqt", stem
                )] = stem

            for future in as_completed(futures):
                stem = futures[future]
                try:
                    results.append(future.result())
                except Exception as exc:
                    results.append(DockingResult(stem, error=str(exc)))

        results.sort(key=lambda r: (not r.success, r.best_affinity if r.best_affinity is not None else 0))
        return results


def poses_to_sdf(docked_pdbqt: str | Path, out_sdf: str | Path, keep: int = 1) -> Path:
    """Export docked poses as SDF for viewing in DS Visualizer or PyMOL."""
    docked_pdbqt, out_sdf = Path(docked_pdbqt), Path(out_sdf)
    if shutil.which("obabel") is None:
        raise RuntimeError("Open Babel required for pose export")
    cmd = ["obabel", str(docked_pdbqt), "-O", str(out_sdf)]
    if keep == 1:
        cmd += ["-f", "1", "-l", "1"]
    subprocess.run(cmd, check=True, capture_output=True)
    return out_sdf
