import pytest

ASPIRIN = "CC(=O)Oc1ccccc1C(=O)O"
IMATINIB = "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


@pytest.fixture
def aspirin():
    return ASPIRIN


@pytest.fixture
def diverse_set():
    """Different scaffolds — should pass the diversity gate."""
    return [ASPIRIN, IMATINIB, CAFFEINE, "c1ccc2[nH]ccc2c1", "C1CCNCC1", "O=C(N)c1ccncc1"]


@pytest.fixture
def collapsed_set():
    """One scaffold, substituent variation only — the v1 failure mode."""
    return [
        "CC(=O)Oc1ccccc1C(=O)O",
        "CCC(=O)Oc1ccccc1C(=O)O",
        "CCCC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CC(=O)Oc1ccc(C)cc1C(=O)O",
        "CC(=O)Oc1ccccc1C(=O)O",
    ]
