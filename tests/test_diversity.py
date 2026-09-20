import pytest

from molgen.chem.diversity import (
    NoveltyFilter,
    analyse_diversity,
    deduplicate,
    murcko_scaffold,
    pick_diverse_subset,
)

pytest.importorskip("rdkit")


def test_dedupe_catches_alternate_spellings():
    # Same molecule, three valid SMILES spellings.
    variants = ["c1ccccc1C(=O)O", "OC(=O)c1ccccc1", "C1=CC=CC=C1C(O)=O"]
    unique, counts = deduplicate(variants)
    assert len(unique) == 1
    assert sum(counts.values()) == 3


def test_collapsed_set_fails_diversity(collapsed_set):
    report = analyse_diversity(collapsed_set)
    assert report.verdict().startswith("FAIL")
    assert report.duplicate_rate > 0


def test_diverse_set_passes(diverse_set):
    report = analyse_diversity(diverse_set)
    assert report.n_scaffolds >= 3
    assert report.scaffold_diversity > 0.3


def test_scaffold_extraction(aspirin):
    assert murcko_scaffold(aspirin) == "c1ccccc1"
    assert murcko_scaffold("CCCC") in (None, "")


def test_novelty_filter_flags_known(aspirin):
    filt = NoveltyFilter([aspirin], threshold=0.85)
    similarity, nearest = filt.score(aspirin)
    assert similarity == 1.0
    assert nearest == aspirin
    novel, annotations = filt.filter([aspirin, "Cn1c(=O)c2c(ncn2C)n(C)c1=O"])
    assert aspirin not in novel
    assert len(annotations) == 2


def test_maxmin_subset_size(diverse_set):
    picked = pick_diverse_subset(diverse_set, 3)
    assert len(picked) == 3
    assert len(set(picked)) == 3
