"""Fragment decomposition with parent-atom provenance.

Backs the chemist-facing workflow: a molecule is decomposed into labelled
fragments, the chemist marks which to keep and which to redesign, and only the
marked regions are regenerated.

Critical implementation note
----------------------------
Mapping fragment atoms back to the parent molecule uses ``FragmentOnBonds``
with ``fragsMolAtomMapping``. Post-hoc ``GetSubstructMatches`` is wrong here:
on a molecule with symmetric groups — two identical methoxys, a para-substituted
ring — it returns several matches and there is no way to tell which one produced
the fragment. That silently mislabels provenance and the chemist edits the wrong
part of the molecule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import BRICS, Recap
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)

#: Common functional groups, as SMARTS, for chemist-readable labelling.
FUNCTIONAL_GROUPS: dict[str, str] = {
    "carboxylic_acid": "[CX3](=O)[OX2H1]",
    "ester": "[CX3](=O)[OX2H0][#6]",
    "amide": "[NX3][CX3](=[OX1])",
    "primary_amine": "[NX3;H2;!$(NC=O)]",
    "secondary_amine": "[NX3;H1;!$(NC=O)]",
    "tertiary_amine": "[NX3;H0;!$(NC=O);!$(N=*)]",
    "alcohol": "[OX2H][CX4]",
    "phenol": "[OX2H][c]",
    "ether": "[OD2]([#6])[#6]",
    "sulfonamide": "[SX4](=[OX1])(=[OX1])([NX3])",
    "nitrile": "[NX1]#[CX2]",
    "nitro": "[$([NX3](=O)=O)]",
    "halogen": "[F,Cl,Br,I]",
    "trifluoromethyl": "[CX4](F)(F)F",
    "aromatic_ring": "c1ccccc1",
    "pyridine": "n1ccccc1",
    "piperazine": "C1CNCCN1",
    "piperidine": "C1CCNCC1",
    "morpholine": "C1COCCN1",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "sulfone": "[SX4](=[OX1])(=[OX1])([#6])[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
    "aldehyde": "[CX3H1](=O)[#6]",
}


@dataclass
class Fragment:
    """One labelled fragment with provenance back to the parent."""

    fragment_id: str
    smiles: str
    parent_atom_indices: list[int] = field(default_factory=list)
    attachment_points: list[int] = field(default_factory=list)
    label: str = ""
    method: str = "brics"
    n_heavy_atoms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _frag_atom_mapping(mol: Chem.Mol, bond_indices: list[int]) -> list[Fragment]:
    """Cut ``mol`` at ``bond_indices`` and keep exact parent-atom provenance."""
    if not bond_indices:
        return []

    mapping: list[list[int]] = []
    fragmented = Chem.FragmentOnBonds(mol, bond_indices, addDummies=True)
    pieces = Chem.GetMolFrags(
        fragmented, asMols=True, sanitizeFrags=False, fragsMolAtomMapping=mapping
    )

    fragments: list[Fragment] = []
    for i, (piece, atom_map) in enumerate(zip(pieces, mapping, strict=True), start=1):
        try:
            Chem.SanitizeMol(piece)
        except Exception:
            continue
        # Dummy atoms introduced by the cut carry index -1 style placeholders;
        # keep only indices that exist in the parent.
        parent_indices = [idx for idx in atom_map if 0 <= idx < mol.GetNumAtoms()]
        attachment = [a.GetIdx() for a in piece.GetAtoms() if a.GetAtomicNum() == 0]
        fragments.append(
            Fragment(
                fragment_id=f"F{i}",
                smiles=Chem.MolToSmiles(piece),
                parent_atom_indices=parent_indices,
                attachment_points=attachment,
                n_heavy_atoms=sum(1 for a in piece.GetAtoms() if a.GetAtomicNum() > 1),
            )
        )
    return fragments


def brics_fragments(smiles: str, min_fragment_size: int = 2) -> list[Fragment]:
    """BRICS decomposition with provenance preserved."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")

    bonds = [bond[0] for bond in BRICS.FindBRICSBonds(mol)]
    bond_indices = [
        mol.GetBondBetweenAtoms(a, b).GetIdx()
        for a, b in bonds
        if mol.GetBondBetweenAtoms(a, b) is not None
    ]
    fragments = _frag_atom_mapping(mol, sorted(set(bond_indices)))
    for frag in fragments:
        frag.method = "brics"
        frag.label = classify_fragment(frag.smiles)
    return [f for f in fragments if f.n_heavy_atoms >= min_fragment_size]


def recap_fragments(smiles: str) -> list[Fragment]:
    """RECAP decomposition — retrosynthetically motivated cuts."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")

    tree = Recap.RecapDecompose(mol)
    fragments = []
    for i, (frag_smiles, _) in enumerate(tree.GetLeaves().items(), start=1):
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        if frag_mol is None:
            continue
        fragments.append(
            Fragment(
                fragment_id=f"R{i}",
                smiles=frag_smiles,
                method="recap",
                label=classify_fragment(frag_smiles),
                n_heavy_atoms=frag_mol.GetNumHeavyAtoms(),
            )
        )
    return fragments


def scaffold_and_sidechains(smiles: str) -> dict[str, Any]:
    """Split into Murcko scaffold plus side chains, with atom indices."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")

    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_atoms: list[int] = []
    if scaffold.GetNumAtoms():
        match = mol.GetSubstructMatch(scaffold)
        scaffold_atoms = list(match)

    sidechain_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in scaffold_atoms]
    return {
        "scaffold_smiles": Chem.MolToSmiles(scaffold) if scaffold.GetNumAtoms() else "",
        "generic_scaffold_smiles": (
            Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(scaffold))
            if scaffold.GetNumAtoms()
            else ""
        ),
        "scaffold_atom_indices": scaffold_atoms,
        "sidechain_atom_indices": sidechain_atoms,
    }


def find_functional_groups(smiles: str) -> list[dict[str, Any]]:
    """Locate known functional groups with their parent atom indices."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")

    found = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            found.append({"group": name, "atom_indices": list(match)})
    return found


def classify_fragment(fragment_smiles: str) -> str:
    """Chemist-readable label for a fragment, from its functional groups."""
    mol = Chem.MolFromSmiles(fragment_smiles)
    if mol is None:
        return "unknown"

    labels = []
    for name, smarts in FUNCTIONAL_GROUPS.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            labels.append(name)
    if not labels:
        heavy = mol.GetNumHeavyAtoms()
        return "ring system" if mol.GetRingInfo().NumRings() else f"alkyl C{heavy}"
    return ", ".join(labels[:3])


def decompose(smiles: str, method: str = "brics") -> dict[str, Any]:
    """Full decomposition record for the fragment-selection UI."""
    if method == "brics":
        fragments = brics_fragments(smiles)
    elif method == "recap":
        fragments = recap_fragments(smiles)
    else:
        raise ValueError(f"unknown method '{method}' (use brics or recap)")

    return {
        "input_smiles": smiles,
        "method": method,
        "fragments": [f.to_dict() for f in fragments],
        "scaffold": scaffold_and_sidechains(smiles),
        "functional_groups": find_functional_groups(smiles),
    }


def highlight_fragment(smiles: str, atom_indices: list[int], size: tuple[int, int] = (500, 400)):
    """Render the parent molecule with ``atom_indices`` highlighted.

    Uses the stable ``Draw.MolToImage`` highlight API. The newer
    ``DrawMoleculeWithHighlights`` path expects colour values as lists rather
    than tuples and broke on several RDKit builds.
    """
    from rdkit.Chem import AllChem, Draw

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")
    AllChem.Compute2DCoords(mol)

    colours = {idx: (1.0, 0.55, 0.15) for idx in atom_indices}
    return Draw.MolToImage(
        mol,
        size=size,
        highlightAtoms=list(atom_indices),
        highlightColor=(1.0, 0.55, 0.15),
        highlightMap=colours,
    )
