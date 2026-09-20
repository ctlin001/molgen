import pytest

from molgen.chem.protac import (
    cap_wildcards,
    classify_linker,
    count_attachment_points,
    excise_linker,
    linker_descriptors,
    validate_linker,
)

pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


def test_cap_wildcards_removes_dummies():
    capped = cap_wildcards("[*]CCOCC[*]")
    assert "*" not in capped
    assert Chem.MolFromSmiles(capped) is not None


def test_uncapped_linker_rejected_capped_succeeds():
    """The v2 bug: `*` atoms parse and even embed, but have no force-field type."""
    from rdkit.Chem import AllChem

    from molgen.chem.validate import embed_3d

    raw = Chem.MolFromSmiles("[*]CCOCCOCC[*]")
    assert raw is not None, "wildcards parse — that is what makes the trap subtle"

    # Neither force field can type a dummy atom, so minimisation is silently skipped.
    with_h = Chem.AddHs(Chem.Mol(raw))
    AllChem.EmbedMolecule(with_h, AllChem.ETKDGv3())
    assert not AllChem.MMFFHasAllMoleculeParams(with_h)
    assert not AllChem.UFFHasAllMoleculeParams(with_h)

    # So embed_3d refuses it rather than passing unrefined geometry downstream.
    assert embed_3d(raw) is None

    capped = Chem.MolFromSmiles(cap_wildcards("[*]CCOCCOCC[*]"))
    assert embed_3d(capped) is not None


def test_attachment_point_counting():
    assert count_attachment_points("[*]CCC[*]") == 2
    assert count_attachment_points("CCC") == 0


def test_linker_validation_rules():
    assert validate_linker("[*]CCOCCOCC[*]")[0]
    assert not validate_linker("[*]CCC")[0]          # one attachment point
    assert not validate_linker("[*]C[*]", min_heavy=4)[0]   # too short


def test_excise_linker_splits_three_ways():
    # phenyl - PEG linker - pyridine
    protac = "c1ccccc1CCOCCOCCc1ccncc1"
    mol = Chem.MolFromSmiles(protac)
    assert mol is not None
    result = excise_linker(protac, warhead_anchor=0, e3_anchor=mol.GetNumAtoms() - 1)
    assert result.linker_smiles
    assert result.warhead_smiles
    assert result.e3_smiles
    assert result.linker_path_length > 0


def test_linker_descriptors_cap_first():
    desc = linker_descriptors("[*]CCOCCOCC[*]")
    assert "*" not in desc["capped_smiles"]
    assert desc["heavy_atoms"] > 0
    assert desc["class"] in {"PEG", "alkyl", "aryl", "amide", "triazole", "piperazine", "piperidine"}


def test_linker_classification():
    assert classify_linker("CCOCCOCC") == "PEG"
    assert classify_linker("CCCCCC") == "alkyl"
