"""PROTAC linker excision, assembly and repair.

Workflow (AIMLinker-style):

1. chemist picks one anchor atom on the warhead and one on the E3 ligand
2. ``GetShortestPath`` between anchors defines the existing linker
3. ``FragmentOnBonds`` excises it, leaving two capped anchor fragments
4. the LLM proposes replacement linkers as ``[*]...[*]``
5. linkers are stitched back onto the anchors and validated

The wildcard trap
-----------------
Generated linkers carry ``*`` dummy atoms at the attachment points. RDKit will
parse and even canonicalise these, but ``EmbedMolecule`` fails on them and
AutoDock Vina rejects the resulting file outright. Any linker that is going to
be embedded or docked standalone must first be capped — :func:`cap_wildcards`
replaces each dummy with a methyl group. Assembled PROTACs have no dummies left
and need no capping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

RDLogger.DisableLog("rdApp.*")
log = logging.getLogger(__name__)


@dataclass
class LinkerExcision:
    """Result of cutting an existing linker out of a PROTAC."""

    warhead_smiles: str = ""
    e3_smiles: str = ""
    linker_smiles: str = ""
    linker_path_length: int = 0
    warhead_anchor: int | None = None
    e3_anchor: int | None = None
    cut_bonds: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def cap_wildcards(smiles: str, cap: str = "C") -> str:
    """Replace ``*`` attachment atoms with a real substituent.

    Required before 3D embedding, property calculation or docking of a bare
    linker. Without it ``EmbedMolecule`` returns -1 and Vina refuses the ligand.

    >>> cap_wildcards("[*]CCOCC[*]")
    'CCOCCC'
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smiles}")

    cap_atom = Chem.MolFromSmiles(cap)
    if cap_atom is None or cap_atom.GetNumAtoms() != 1:
        raise ValueError("cap must be a single-atom SMILES, e.g. 'C'")
    cap_num = cap_atom.GetAtomWithIdx(0).GetAtomicNum()

    editable = Chem.RWMol(mol)
    for atom in editable.GetAtoms():
        if atom.GetAtomicNum() == 0:
            atom.SetAtomicNum(cap_num)
            atom.SetIsotope(0)
            atom.SetNoImplicit(False)
            atom.SetNumExplicitHs(0)
            atom.SetFormalCharge(0)

    capped = editable.GetMol()
    Chem.SanitizeMol(capped)
    return Chem.MolToSmiles(capped)


def count_attachment_points(smiles: str) -> int:
    """Number of ``*`` dummy atoms in a linker SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 0)


def validate_linker(smiles: str, min_heavy: int = 2, max_heavy: int = 40) -> tuple[bool, str]:
    """Check a generated linker is usable before assembly."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, "unparseable SMILES"

    n_star = count_attachment_points(smiles)
    if n_star != 2:
        return False, f"expected 2 attachment points, found {n_star}"

    heavy = sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
    if heavy < min_heavy:
        return False, f"linker too short ({heavy} heavy atoms)"
    if heavy > max_heavy:
        return False, f"linker too long ({heavy} heavy atoms)"
    return True, "ok"


def _first_acyclic_bond(mol: Chem.Mol, path: list[int]) -> tuple[int, int, int] | None:
    """First non-ring bond walking inward along ``path``.

    Returns ``(bond_index, outer_atom, inner_atom)`` in path order, or ``None``.

    Cutting a ring bond would open the ring, leaving aromatic flags on atoms
    that are no longer in a ring and raising ``AtomKekulizeException`` during
    sanitisation. It is also chemically meaningless — a linker is excised at
    acyclic connections. When an anchor sits inside a ring, walk along the path
    until leaving it.
    """
    for a, b in zip(path, path[1:], strict=False):
        bond = mol.GetBondBetweenAtoms(a, b)
        if bond is not None and not bond.IsInRing():
            return bond.GetIdx(), a, b
    return None


def excise_linker(
    protac_smiles: str, warhead_anchor: int, e3_anchor: int
) -> LinkerExcision:
    """Cut an existing PROTAC into warhead, linker and E3 ligand.

    ``warhead_anchor`` and ``e3_anchor`` are atom indices in ``protac_smiles``.
    The shortest path between them defines the linker; the bonds immediately
    inside each anchor are the cut points.
    """
    mol = Chem.MolFromSmiles(protac_smiles)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {protac_smiles}")
    n = mol.GetNumAtoms()
    for idx, label in ((warhead_anchor, "warhead"), (e3_anchor, "E3")):
        if not 0 <= idx < n:
            raise IndexError(f"{label} anchor {idx} outside molecule (0-{n - 1})")

    path = list(Chem.GetShortestPath(mol, warhead_anchor, e3_anchor))
    if len(path) < 3:
        raise ValueError("anchors are adjacent; no linker to excise")

    head = _first_acyclic_bond(mol, path)
    tail = _first_acyclic_bond(mol, list(reversed(path)))
    if head is None or tail is None:
        raise ValueError(
            "no acyclic bond on the path between the anchors; the anchors sit "
            "inside a fused ring system and cannot be separated without opening a ring"
        )

    head_bond, _, linker_start = head
    tail_bond, _, linker_end = tail
    if head_bond == tail_bond:
        raise ValueError("anchors are separated by a single bond; no linker to excise")
    cut_bonds = sorted({head_bond, tail_bond})

    # Atoms strictly between the two cuts, in path order, are the linker.
    start_i, end_i = path.index(linker_start), path.index(linker_end)
    linker_atoms = set(path[start_i : end_i + 1])
    warhead_atoms = set(path[:start_i])
    e3_atoms = set(path[end_i + 1 :])

    mapping: list[list[int]] = []
    fragmented = Chem.FragmentOnBonds(mol, cut_bonds, addDummies=True)
    pieces = Chem.GetMolFrags(
        fragmented, asMols=True, sanitizeFrags=True, fragsMolAtomMapping=mapping
    )

    result = LinkerExcision(
        warhead_anchor=warhead_anchor,
        e3_anchor=e3_anchor,
        cut_bonds=cut_bonds,
        linker_path_length=len(linker_atoms),
    )

    for piece, atom_map in zip(pieces, mapping, strict=True):
        smiles = Chem.MolToSmiles(piece)
        atoms = {i for i in atom_map if 0 <= i < n}
        if atoms & linker_atoms:
            result.linker_smiles = smiles
        elif atoms & warhead_atoms or warhead_anchor in atoms:
            result.warhead_smiles = smiles
        elif atoms & e3_atoms or e3_anchor in atoms:
            result.e3_smiles = smiles

    return result


def assemble_protac(warhead: str, linker: str, e3_ligand: str) -> str | None:
    """Stitch ``[*]``-terminated fragments into one PROTAC SMILES.

    Each of the three parts must carry the dummy atoms that mark where it joins.
    Returns ``None`` if the pieces cannot be combined.
    """
    parts = [Chem.MolFromSmiles(s) for s in (warhead, linker, e3_ligand)]
    if any(p is None for p in parts):
        log.warning("assembly aborted: one or more fragments unparseable")
        return None

    combined = parts[0]
    for part in parts[1:]:
        combined = Chem.CombineMols(combined, part)

    editable = Chem.RWMol(combined)
    dummies = [a.GetIdx() for a in editable.GetAtoms() if a.GetAtomicNum() == 0]
    if len(dummies) < 2:
        return None

    # Pair dummies: warhead dummy to first linker dummy, second linker dummy to E3.
    pairs = [(dummies[i], dummies[i + 1]) for i in range(0, len(dummies) - 1, 2)]
    to_remove: set[int] = set()

    for a_idx, b_idx in pairs:
        a_nbrs = [n.GetIdx() for n in editable.GetAtomWithIdx(a_idx).GetNeighbors()]
        b_nbrs = [n.GetIdx() for n in editable.GetAtomWithIdx(b_idx).GetNeighbors()]
        if not a_nbrs or not b_nbrs:
            continue
        if editable.GetBondBetweenAtoms(a_nbrs[0], b_nbrs[0]) is None:
            editable.AddBond(a_nbrs[0], b_nbrs[0], Chem.BondType.SINGLE)
        to_remove.update({a_idx, b_idx})

    for idx in sorted(to_remove, reverse=True):
        editable.RemoveAtom(idx)

    try:
        product = editable.GetMol()
        Chem.SanitizeMol(product)
    except Exception as exc:
        log.warning("assembled PROTAC failed sanitisation: %s", exc)
        return None
    return Chem.MolToSmiles(product)


def linker_descriptors(linker_smiles: str) -> dict[str, Any]:
    """Descriptors for a bare linker.

    Wildcards are capped first, otherwise every descriptor is either wrong or
    raises.
    """
    capped = cap_wildcards(linker_smiles)
    mol = Chem.MolFromSmiles(capped)
    if mol is None:
        return {}

    from rdkit.Chem import Crippen, Descriptors, Lipinski

    heavy_path = 0
    star_neighbours = []
    raw = Chem.MolFromSmiles(linker_smiles)
    if raw is not None:
        star_neighbours = [
            n.GetIdx()
            for a in raw.GetAtoms()
            if a.GetAtomicNum() == 0
            for n in a.GetNeighbors()
        ]
        if len(star_neighbours) >= 2:
            heavy_path = max(
                0, len(Chem.GetShortestPath(raw, star_neighbours[0], star_neighbours[-1])) - 0
            )

    return {
        "capped_smiles": capped,
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "path_length": heavy_path,
        "mw": round(Descriptors.MolWt(mol), 2),
        "logp": round(Crippen.MolLogP(mol), 2),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "tpsa": round(Descriptors.TPSA(mol), 2),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "class": classify_linker(capped),
    }


def classify_linker(smiles: str) -> str:
    """Coarse chemotype label used to check the set is not all PEGs."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    patterns = [
        ("PEG", "[#6][OX2][#6][#6][OX2][#6]"),
        ("triazole", "c1cn[nH0]n1"),
        ("piperazine", "C1CNCCN1"),
        ("piperidine", "C1CCNCC1"),
        ("amide", "[NX3][CX3](=[OX1])"),
        ("aryl", "c1ccccc1"),
    ]
    for label, smarts in patterns:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None and mol.HasSubstructMatch(pattern):
            return label
    return "alkyl"


def protac_property_window(smiles: str) -> dict[str, Any]:
    """Degrader-appropriate property check.

    Rule-of-five is the wrong yardstick for PROTACs — they sit in "beyond Ro5"
    space by construction. These bounds are advisory rather than pass/fail.
    """
    from rdkit.Chem import Crippen, Descriptors, Lipinski

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False}

    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    rotb = Lipinski.NumRotatableBonds(mol)

    flags = []
    if not 700 <= mw <= 1100:
        flags.append(f"MW {mw:.0f} outside 700-1100")
    if not 2 <= logp <= 7:
        flags.append(f"cLogP {logp:.1f} outside 2-7")
    if tpsa > 250:
        flags.append(f"TPSA {tpsa:.0f} > 250")
    if rotb > 20:
        flags.append(f"{rotb} rotatable bonds > 20")

    return {
        "valid": True,
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "rotatable_bonds": rotb,
        "flags": flags,
        "in_window": not flags,
    }
