import pytest

from molgen.chem.fragments import (
    brics_fragments,
    classify_fragment,
    decompose,
    find_functional_groups,
    scaffold_and_sidechains,
)

pytest.importorskip("rdkit")


def test_brics_returns_fragments_with_provenance():
    frags = brics_fragments("CC(=O)Oc1ccccc1C(=O)O")
    assert frags
    for frag in frags:
        assert frag.smiles
        assert all(i >= 0 for i in frag.parent_atom_indices)


def test_symmetric_molecule_provenance_is_unambiguous():
    """Two identical methoxys: substructure matching cannot tell them apart,
    FragmentOnBonds atom mapping can."""
    frags = brics_fragments("COc1ccc(OC)cc1C(=O)NC")
    index_sets = [set(f.parent_atom_indices) for f in frags]
    for i, a in enumerate(index_sets):
        for b in index_sets[i + 1:]:
            assert not (a & b), "fragments must not claim the same parent atoms"


def test_scaffold_split(aspirin):
    result = scaffold_and_sidechains(aspirin)
    assert result["scaffold_smiles"] == "c1ccccc1"
    assert result["scaffold_atom_indices"]


def test_functional_groups_found(aspirin):
    names = {g["group"] for g in find_functional_groups(aspirin)}
    assert "carboxylic_acid" in names
    assert "ester" in names


def test_classify_fragment():
    assert "carboxylic_acid" in classify_fragment("CC(=O)O")


def test_decompose_shape(aspirin):
    result = decompose(aspirin)
    assert set(result) == {"input_smiles", "method", "fragments", "scaffold", "functional_groups"}


def test_bad_smiles_raises():
    with pytest.raises(ValueError):
        brics_fragments("not-a-molecule((")
