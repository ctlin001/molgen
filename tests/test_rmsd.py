import numpy as np
import pytest

from molgen.docking.rmsd import compute_rmsd, hungarian_rmsd, naive_rmsd

pytest.importorskip("rdkit")
pytest.importorskip("scipy")
from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402


def _embed(smiles, seed=42):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def test_identical_pose_is_zero():
    mol = _embed("CC(=O)Oc1ccccc1C(=O)O")
    assert compute_rmsd(mol, mol).rmsd == pytest.approx(0.0, abs=1e-6)


def test_translated_pose_matches_offset():
    mol = _embed("c1ccccc1")
    shifted = Chem.Mol(mol)
    conf = shifted.GetConformer()
    for i in range(shifted.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, (pos.x + 1.0, pos.y, pos.z))
    assert compute_rmsd(mol, shifted).rmsd == pytest.approx(1.0, abs=1e-3)


def test_hungarian_beats_naive_on_symmetric_relabeling():
    """Benzene with atoms listed in reversed order: same structure, wrong order."""
    coords = np.array([
        [1.4, 0.0, 0.0], [0.7, 1.2, 0.0], [-0.7, 1.2, 0.0],
        [-1.4, 0.0, 0.0], [-0.7, -1.2, 0.0], [0.7, -1.2, 0.0],
    ])
    elements = ["C"] * 6
    reversed_coords = coords[::-1]

    hungarian, matched = hungarian_rmsd(coords, elements, reversed_coords, elements)
    naive = naive_rmsd(coords, reversed_coords)

    assert matched == 6
    assert hungarian == pytest.approx(0.0, abs=1e-6)
    assert naive > 1.0, "index-order RMSD is inflated by relabeling alone"


def test_success_threshold():
    mol = _embed("c1ccccc1")
    assert compute_rmsd(mol, mol).is_success
