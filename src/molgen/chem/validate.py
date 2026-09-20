"""Deterministic validation of generated SMILES.

Every candidate passes through here before a chemist or a docking run sees it.
Properties are recomputed with RDKit and the model's own claims are discarded:
LLMs state molecular weights confidently and wrongly.

Filters applied, in order:
    parse -> sanitise -> canonicalise -> properties -> Ro5 -> alerts -> 3D embed
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import QED, AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

_CATALOG: FilterCatalog | None = None
_SASCORER = None


def _catalog() -> FilterCatalog:
    """PAINS (A/B/C) + Brenk structural alert catalog, built once."""
    global _CATALOG
    if _CATALOG is None:
        params = FilterCatalogParams()
        for entry in (
            FilterCatalogParams.FilterCatalogs.PAINS_A,
            FilterCatalogParams.FilterCatalogs.PAINS_B,
            FilterCatalogParams.FilterCatalogs.PAINS_C,
            FilterCatalogParams.FilterCatalogs.BRENK,
        ):
            params.AddCatalog(entry)
        _CATALOG = FilterCatalog(params)
    return _CATALOG


def _sascorer():
    global _SASCORER
    if _SASCORER is None:
        try:
            import sys

            from rdkit.Chem import RDConfig

            sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
            import sascorer  # type: ignore

            _SASCORER = sascorer
        except Exception:  # pragma: no cover - optional contrib module
            log.warning("SA_Score contrib module unavailable; sa_score will be None")
            _SASCORER = False
    return _SASCORER or None


@dataclass
class Properties:
    mw: float = 0.0
    logp: float = 0.0
    hbd: int = 0
    hba: int = 0
    tpsa: float = 0.0
    rotatable_bonds: int = 0
    aromatic_rings: int = 0
    rings: int = 0
    heavy_atoms: int = 0
    fraction_csp3: float = 0.0
    formal_charge: int = 0
    qed: float = 0.0
    sa_score: float | None = None


@dataclass
class ValidationResult:
    smiles: str
    canonical_smiles: str = ""
    inchikey: str = ""
    valid: bool = False
    error: str | None = None
    properties: Properties | None = None
    lipinski_violations: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)
    passes_ro5: bool = False
    passes_alerts: bool = False
    has_3d: bool = False
    sdf_path: str | None = None

    @property
    def passes_all(self) -> bool:
        return self.valid and self.passes_ro5 and self.passes_alerts

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passes_all"] = self.passes_all
        return d


def compute_properties(mol: Chem.Mol) -> Properties:
    """Physicochemical descriptors for one sanitised molecule."""
    sascorer = _sascorer()
    return Properties(
        mw=round(Descriptors.MolWt(mol), 2),
        logp=round(Crippen.MolLogP(mol), 2),
        hbd=Lipinski.NumHDonors(mol),
        hba=Lipinski.NumHAcceptors(mol),
        tpsa=round(Descriptors.TPSA(mol), 2),
        rotatable_bonds=Lipinski.NumRotatableBonds(mol),
        aromatic_rings=Lipinski.NumAromaticRings(mol),
        rings=Chem.rdMolDescriptors.CalcNumRings(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        fraction_csp3=round(Chem.rdMolDescriptors.CalcFractionCSP3(mol), 3),
        formal_charge=Chem.GetFormalCharge(mol),
        qed=round(QED.qed(mol), 3),
        sa_score=round(sascorer.calculateScore(mol), 2) if sascorer else None,
    )


def lipinski_violations(props: Properties) -> list[str]:
    """Rule-of-five violations. Drug-like conventionally means at most one."""
    out = []
    if props.mw > 500:
        out.append(f"MW {props.mw} > 500")
    if props.logp > 5:
        out.append(f"cLogP {props.logp} > 5")
    if props.hbd > 5:
        out.append(f"HBD {props.hbd} > 5")
    if props.hba > 10:
        out.append(f"HBA {props.hba} > 10")
    return out


def structural_alerts(mol: Chem.Mol) -> list[str]:
    """PAINS and Brenk matches by description."""
    return [m.GetDescription() for m in _catalog().GetMatches(mol)]


def embed_3d(
    mol: Chem.Mol,
    *,
    optimise: bool = True,
    seed: int = 0xF00D,
    max_attempts: int = 10,
) -> Chem.Mol | None:
    """Generate and minimise a single 3D conformer.

    Returns ``None`` when the molecule cannot be given usable 3D coordinates.

    Dummy (``*``) atoms are rejected up front. ETKDG will happily embed them on
    most RDKit builds, which is the trap: neither MMFF nor UFF has parameters
    for atom type ``*_``, so minimisation is skipped with only a logger warning
    and the molecule travels onward with unrefined geometry, before Open Babel
    and Vina reject it further downstream. Failing here makes the problem
    visible at its source. Cap linkers first with
    :func:`molgen.chem.protac.cap_wildcards`.
    """
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        log.warning(
            "refusing to embed a molecule with dummy atoms; cap attachment points first"
        )
        return None

    mol_h = Chem.AddHs(Chem.Mol(mol))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.useSmallRingTorsions = True
    params.maxIterations = 200 * max_attempts

    if AllChem.EmbedMolecule(mol_h, params) != 0:
        params.useRandomCoords = True
        if AllChem.EmbedMolecule(mol_h, params) != 0:
            return None

    if optimise:
        try:
            if AllChem.MMFFHasAllMoleculeParams(mol_h):
                AllChem.MMFFOptimizeMolecule(mol_h, maxIters=1000)
            else:
                AllChem.UFFOptimizeMolecule(mol_h, maxIters=1000)
        except Exception as exc:
            log.debug("minimisation failed, keeping embedded coords: %s", exc)
    return mol_h


class StructureValidator:
    """Batch validation with configurable thresholds.

    Parameters
    ----------
    max_lipinski_violations:
        Candidates above this are marked failing (default 1, the usual convention).
    allow_alerts:
        When ``True``, PAINS/Brenk matches are recorded but not disqualifying —
        appropriate for PROTAC mode, where the property window is different.
    """

    def __init__(
        self,
        max_lipinski_violations: int = 1,
        allow_alerts: bool = False,
        embed: bool = True,
        sdf_dir: str | Path | None = None,
    ):
        self.max_lipinski_violations = max_lipinski_violations
        self.allow_alerts = allow_alerts
        self.embed = embed
        self.sdf_dir = Path(sdf_dir) if sdf_dir else None
        if self.sdf_dir:
            self.sdf_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, smiles: str, name: str | None = None) -> ValidationResult:
        result = ValidationResult(smiles=smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            result.error = "unparseable SMILES"
            return result
        try:
            Chem.SanitizeMol(mol)
        except Exception as exc:
            result.error = f"sanitisation failed: {exc}"
            return result
        if mol.GetNumHeavyAtoms() < 6:
            result.error = "fewer than 6 heavy atoms"
            return result

        result.valid = True
        result.canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        result.inchikey = Chem.MolToInchiKey(mol)

        props = compute_properties(mol)
        result.properties = props
        result.lipinski_violations = lipinski_violations(props)
        result.passes_ro5 = len(result.lipinski_violations) <= self.max_lipinski_violations
        result.alerts = structural_alerts(mol)
        result.passes_alerts = self.allow_alerts or not result.alerts

        if self.embed:
            mol_3d = embed_3d(mol)
            result.has_3d = mol_3d is not None
            if mol_3d is not None and self.sdf_dir:
                stem = name or result.inchikey[:14]
                path = self.sdf_dir / f"{stem}.sdf"
                mol_3d.SetProp("_Name", stem)
                mol_3d.SetProp("SMILES", result.canonical_smiles)
                with Chem.SDWriter(str(path)) as writer:
                    writer.write(mol_3d)
                result.sdf_path = str(path)

        return result

    def validate_many(self, items: Iterable[tuple[str, str]]) -> list[ValidationResult]:
        """Validate ``(smiles, name)`` pairs."""
        return [self.validate(smi, name) for smi, name in items]


def summarise(results: list[ValidationResult]) -> dict[str, Any]:
    """Aggregate pass rates — the headline numbers for a run report."""
    n = len(results)
    if n == 0:
        return {"n": 0}
    valid = [r for r in results if r.valid]
    return {
        "n": n,
        "valid": len(valid),
        "validity_rate": round(len(valid) / n, 3),
        "ro5_pass": sum(r.passes_ro5 for r in valid),
        "alert_free": sum(r.passes_alerts for r in valid),
        "embedded_3d": sum(r.has_3d for r in valid),
        "passes_all": sum(r.passes_all for r in results),
        "mean_qed": round(
            sum(r.properties.qed for r in valid if r.properties) / max(len(valid), 1), 3
        ),
    }
