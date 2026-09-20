"""Configuration.

All run parameters live in one YAML file so a run is reproducible from a single
artifact. Every field has a default; the CLI overrides the file, and the file
overrides the defaults.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

Mode = Literal["ligand", "protac"]


@dataclass
class PathConfig:
    data_dir: Path = Path("data")
    output_dir: Path = Path("runs")
    receptor: Path | None = None
    reference_ligand: Path | None = None
    known_actives: Path | None = None
    novelty_library: Path | None = None

    def resolve(self) -> None:
        """Fill in conventional sub-paths and expand user/environment vars."""
        self.data_dir = Path(os.path.expandvars(str(self.data_dir))).expanduser()
        self.output_dir = Path(os.path.expandvars(str(self.output_dir))).expanduser()

        for name in ("receptor", "reference_ligand", "known_actives", "novelty_library"):
            value = getattr(self, name)
            if value is None:
                continue
            path = Path(os.path.expandvars(str(value))).expanduser()
            if not path.is_absolute() and not path.exists():
                candidate = self.data_dir / path
                if candidate.exists():
                    path = candidate
            setattr(self, name, path)


@dataclass
class GenerationConfig:
    backend: str = "ollama:gemma3:12b"
    temperature: float = 0.8
    max_tokens: int = 2048
    n_molecules: int = 20
    n_calls: int = 3
    rounds: int = 1
    property_focus: str | None = None
    diversity_hint: bool = True
    retry_attempts: int = 3


@dataclass
class ValidationConfig:
    max_lipinski_violations: int = 1
    allow_alerts: bool = False
    embed_3d: bool = True
    min_heavy_atoms: int = 6


@dataclass
class DiversityConfig:
    enabled: bool = True
    novelty_threshold: float = 0.85
    max_to_dock: int | None = 50
    maxmin_selection: bool = True
    min_scaffold_diversity: float = 0.30


@dataclass
class DockingConfig:
    enabled: bool = True
    executable: str = "vina"
    use_wsl: bool = False
    exhaustiveness: int = 16
    num_modes: int = 9
    cpu: int | None = None
    seed: int | None = 42
    max_workers: int = 4
    box_padding: float = 5.0
    cubic_box: bool = False
    box_center: tuple[float, float, float] | None = None
    box_size: tuple[float, float, float] | None = None
    compute_rmsd: bool = True


@dataclass
class ProtacConfig:
    warhead_smiles: str | None = None
    e3_ligand_smiles: str | None = None
    parent_protac_smiles: str | None = None
    warhead_anchor: int | None = None
    e3_anchor: int | None = None
    anchor_distance: float | None = None
    min_linker_heavy_atoms: int = 4
    max_linker_heavy_atoms: int = 30


@dataclass
class Config:
    mode: Mode = "ligand"
    run_name: str = "run"
    ligand_resname: str | None = None
    pocket_cutoff: float = 5.0
    paths: PathConfig = field(default_factory=PathConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    diversity: DiversityConfig = field(default_factory=DiversityConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    protac: ProtacConfig = field(default_factory=ProtacConfig)

    # ------------------------------------------------------------------ io

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        import yaml

        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Config:
        sections = {
            "paths": PathConfig,
            "generation": GenerationConfig,
            "validation": ValidationConfig,
            "diversity": DiversityConfig,
            "docking": DockingConfig,
            "protac": ProtacConfig,
        }
        kwargs: dict[str, Any] = {
            k: v for k, v in data.items() if k not in sections and k in cls.__annotations__
        }
        for name, klass in sections.items():
            section = data.get(name) or {}
            valid = {k: v for k, v in section.items() if k in klass.__annotations__}
            kwargs[name] = klass(**valid)

        config = cls(**kwargs)
        config.paths.resolve()
        return config

    def to_dict(self) -> dict[str, Any]:
        def encode(obj):
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, dict):
                return {k: encode(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [encode(v) for v in obj]
            return obj

        return encode(asdict(self))

    def save(self, path: str | Path) -> None:
        import yaml

        Path(path).write_text(yaml.safe_dump(self.to_dict(), sort_keys=False))

    # -------------------------------------------------------------- checks

    def validate(self) -> list[str]:
        """Return a list of configuration problems; empty means usable."""
        problems = []

        if self.paths.receptor is None:
            problems.append("paths.receptor is required")
        elif not Path(self.paths.receptor).exists():
            problems.append(f"receptor not found: {self.paths.receptor}")

        if self.mode == "protac":
            if not self.protac.warhead_smiles and not self.protac.parent_protac_smiles:
                problems.append(
                    "protac mode needs protac.warhead_smiles or protac.parent_protac_smiles"
                )
            if not self.protac.e3_ligand_smiles and not self.protac.parent_protac_smiles:
                problems.append(
                    "protac mode needs protac.e3_ligand_smiles or protac.parent_protac_smiles"
                )
            if self.protac.parent_protac_smiles and (
                self.protac.warhead_anchor is None or self.protac.e3_anchor is None
            ):
                problems.append(
                    "excising a parent PROTAC needs protac.warhead_anchor and protac.e3_anchor"
                )

        if self.docking.enabled and self.docking.box_center and not self.docking.box_size:
            problems.append("docking.box_center given without docking.box_size")

        if not 0.0 <= self.generation.temperature <= 2.0:
            problems.append("generation.temperature must be between 0 and 2")

        return problems
