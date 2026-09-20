import pytest

from molgen.config import Config


def test_defaults():
    cfg = Config()
    assert cfg.mode == "ligand"
    assert cfg.generation.backend.startswith("ollama:")
    assert cfg.docking.enabled


def test_missing_receptor_is_a_problem():
    problems = Config().validate()
    assert any("receptor" in p for p in problems)


def test_protac_mode_requires_binders(tmp_path):
    receptor = tmp_path / "r.pdb"
    receptor.write_text("ATOM\n")
    cfg = Config.from_dict({"mode": "protac", "paths": {"receptor": str(receptor)}})
    problems = cfg.validate()
    assert any("warhead" in p for p in problems)


def test_roundtrip_yaml(tmp_path):
    pytest.importorskip("yaml")
    receptor = tmp_path / "r.pdb"
    receptor.write_text("ATOM\n")
    cfg = Config.from_dict({
        "run_name": "t",
        "paths": {"receptor": str(receptor)},
        "generation": {"n_molecules": 7},
    })
    path = tmp_path / "c.yaml"
    cfg.save(path)
    assert Config.from_yaml(path).generation.n_molecules == 7


def test_unknown_keys_ignored(tmp_path):
    cfg = Config.from_dict({"generation": {"n_molecules": 5, "nonsense": True}})
    assert cfg.generation.n_molecules == 5
