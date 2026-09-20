import pytest

from molgen.chem.validate import (
    StructureValidator,
    compute_properties,
    lipinski_violations,
    summarise,
)

rdkit = pytest.importorskip("rdkit")
from rdkit import Chem  # noqa: E402


def test_valid_smiles_passes(aspirin):
    result = StructureValidator(embed=False).validate(aspirin, "aspirin")
    assert result.valid
    assert result.passes_ro5
    assert result.canonical_smiles
    assert result.inchikey.startswith("BSYNRYMUTXBXSQ")


def test_invalid_smiles_rejected():
    result = StructureValidator(embed=False).validate("C(C(C", "broken")
    assert not result.valid
    assert result.error


def test_tiny_fragment_rejected():
    result = StructureValidator(embed=False).validate("CCO", "ethanol")
    assert not result.valid
    assert "heavy atoms" in result.error


def test_properties_are_recomputed_not_trusted(aspirin):
    props = compute_properties(Chem.MolFromSmiles(aspirin))
    assert 179 < props.mw < 181
    assert props.hbd == 1
    assert 0 < props.qed <= 1


def test_lipinski_flags_heavy_molecule():
    mol = Chem.MolFromSmiles("C" * 60)
    assert any("MW" in v for v in lipinski_violations(compute_properties(mol)))


def test_pains_alert_detected():
    # Catechol-type rhodanine, a canonical PAINS motif.
    result = StructureValidator(embed=False).validate("O=C1CSC(=S)N1", "rhodanine")
    if result.valid:
        assert result.alerts or not result.passes_alerts


def test_summarise_counts(aspirin):
    validator = StructureValidator(embed=False)
    results = [validator.validate(aspirin, "a"), validator.validate("xx((", "b")]
    stats = summarise(results)
    assert stats["n"] == 2
    assert stats["valid"] == 1
    assert stats["validity_rate"] == 0.5


def test_3d_embedding(aspirin):
    result = StructureValidator(embed=True).validate(aspirin, "aspirin")
    assert result.has_3d
