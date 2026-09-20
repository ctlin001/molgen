import pytest

from molgen.generate.generator import parse_completion, strip_fences

pytest.importorskip("rdkit")

CLEAN = '{"molecules":[{"smiles":"CC(=O)Oc1ccccc1C(=O)O","name":"A","rationale":"r"}]}'


def test_clean_json():
    got = parse_completion(CLEAN)
    assert len(got) == 1
    assert got[0].name == "A"


def test_markdown_fences_stripped():
    assert strip_fences("```json\n{}\n```") == "{}"
    assert len(parse_completion(f"```json\n{CLEAN}\n```")) == 1


def test_preamble_and_trailing_prose():
    text = f"Sure! Here are the molecules:\n{CLEAN}\nLet me know if you need more."
    assert len(parse_completion(text)) == 1


def test_bare_list_of_objects():
    text = '[{"smiles":"c1ccccc1O","name":"phenol"}]'
    assert len(parse_completion(text)) == 1


def test_fallback_smiles_scraping():
    text = "1. CC(=O)Oc1ccccc1C(=O)O\n2. Cn1c(=O)c2c(ncn2C)n(C)c1=O\nThose are my picks."
    got = parse_completion(text)
    assert len(got) == 2
    assert all(c.source == "scraped" for c in got)


def test_scraping_rejects_prose():
    got = parse_completion("I could not think of any suitable molecules for this target.")
    assert got == []


def test_empty_completion():
    assert parse_completion("") == []
