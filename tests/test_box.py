import numpy as np
import pytest

from molgen.docking.box import DockingBox, box_from_coords


def test_box_centres_on_coords():
    coords = np.array([[0.0, 0, 0], [10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]])
    box = box_from_coords(coords, padding=5.0)
    assert box.center_x == pytest.approx(5.0)
    assert box.size_x == pytest.approx(20.0)


def test_min_edge_enforced():
    box = box_from_coords(np.array([[0.0, 0, 0], [1.0, 0, 0]]), padding=1.0, min_edge=15.0)
    assert box.size_x >= 15.0


def test_cubic_box_is_cubic():
    coords = np.array([[0.0, 0, 0], [20.0, 2.0, 1.0]])
    box = box_from_coords(coords, cubic=True)
    assert box.size_x == box.size_y == box.size_z


def test_oversized_box_warns():
    coords = np.array([[0.0, 0, 0], [60.0, 0, 0]])
    assert box_from_coords(coords).warnings()


def test_vina_args_complete():
    box = DockingBox(1, 2, 3, 20, 20, 20)
    args = box.to_vina_args()
    assert len(args) == 12
    assert "--center_x" in args


def test_save_load_roundtrip(tmp_path):
    box = DockingBox(1.5, 2.5, 3.5, 20, 22, 24, source="test")
    path = tmp_path / "box.json"
    box.save(path)
    assert DockingBox.load(path).center_y == pytest.approx(2.5)
