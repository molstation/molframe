"""Partner-omitted surface patches: balls, probe and the closed-form two-sphere cap."""

import math

import molframe
import numpy as np
import pytest
from molframe import surface

PROBE = 1.4


def _pair(distance: float, radius: float = 1.7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]], dtype=np.float32)
    radii = np.full(2, radius, dtype=np.float32)
    pairs = np.array([[0, 1]], dtype=np.uint64)
    return coordinates, radii, pairs


def test_isolated_atom_has_no_partner_patch() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    radii = np.array([1.7], dtype=np.float32)
    pairs = np.zeros((0, 2), dtype=np.uint64)
    points = surface.surface_points_excluding_pairs(coordinates, radii, pairs, probe=PROBE)
    assert len(points) == 0


def test_far_apart_atoms_share_no_patch() -> None:
    coordinates, radii, pairs = _pair(distance=3.0 * (2 * (1.7 + PROBE)))
    points = surface.surface_points_excluding_pairs(coordinates, radii, pairs, probe=PROBE)
    assert len(points) == 0
    areas = surface.atom_contact_areas(coordinates, radii, probe=PROBE)
    assert len(areas) == 0


def test_two_sphere_cap_matches_the_closed_form() -> None:
    """Equal balls of expanded radius R at distance d: the cap of atom i inside ball j has
    area 2*pi*R*(R - d/2). Sampling converges to it at high density."""
    radius = 1.7
    distance = 3.0
    expanded = radius + PROBE
    coordinates, radii, pairs = _pair(distance, radius)
    areas = surface.atom_contact_areas(coordinates, radii, probe=PROBE, density=400.0)
    expected = 2.0 * math.pi * expanded * (expanded - distance / 2.0)
    assert areas.area[0] == pytest.approx(expected, rel=0.02)


def test_directed_patches_sum_to_twice_the_pair_area() -> None:
    coordinates, radii, pairs = _pair(distance=3.2)
    excluded = surface.surface_points_excluding_pairs(coordinates, radii, pairs, probe=PROBE, density=200.0)
    pair = surface.atom_contact_areas(coordinates, radii, probe=PROBE, density=200.0)
    assert float(excluded.areas.sum()) == pytest.approx(2.0 * float(pair.area[0]), rel=1e-9)


def test_points_lie_on_the_owner_ball_and_inside_the_partner_ball() -> None:
    coordinates, radii, pairs = _pair(distance=3.0)
    points = surface.surface_points_excluding_pairs(coordinates, radii, pairs, probe=PROBE, density=50.0)
    assert len(points) > 0
    owner = coordinates[points.atoms.astype(np.intp)]
    partner = coordinates[points.excluded.astype(np.intp)]
    positions = points.positions.astype(np.float64)
    expanded = radii.astype(np.float64) + PROBE
    owner_distance = np.linalg.norm(positions - owner, axis=1)
    partner_distance = np.linalg.norm(positions - partner, axis=1)
    assert np.allclose(owner_distance, expanded[points.atoms.astype(np.intp)], atol=1e-3)
    assert (partner_distance <= expanded[points.excluded.astype(np.intp)] + 1e-3).all()


def test_pairs_must_be_two_columns() -> None:
    coordinates, radii, _ = _pair(distance=3.0)
    with pytest.raises(ValueError, match=r"\(m, 2\)"):
        surface.surface_points_excluding_pairs(
            coordinates, radii, np.zeros((1, 3), dtype=np.uint64), probe=PROBE
        )


def test_molframe_reexports_the_new_api() -> None:
    assert molframe.surface.surface_points_excluding_pairs is surface.surface_points_excluding_pairs
    assert molframe.surface.atom_contact_areas is surface.atom_contact_areas
    assert molframe.surface.partner_covered_points is surface.partner_covered_points


def test_a_pair_naming_one_atom_twice_is_refused() -> None:
    coordinates, radii, _ = _pair(distance=3.0)
    with pytest.raises(molframe.MolframeError):
        surface.surface_points_excluding_pairs(
            coordinates, radii, np.array([[0, 0]], dtype=np.uint64), probe=PROBE
        )


def test_partner_group_patch_equals_pair_patch_for_one_partner() -> None:
    coordinates, radii, _ = _pair(distance=3.0)
    owners = np.array([0], dtype=np.uint64)
    partners = np.array([1], dtype=np.uint64)
    group = surface.partner_covered_points(
        coordinates, radii, owners, partners, probe=PROBE, density=200.0
    )
    expected = 2.0 * math.pi * (1.7 + PROBE) * ((1.7 + PROBE) - 1.5)
    assert float(group.areas.sum()) == pytest.approx(expected, rel=0.02)


def test_a_blocker_between_the_centres_hides_the_covered_patch() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32)
    radii = np.full(3, 1.7, dtype=np.float32)
    owners = np.array([0], dtype=np.uint64)
    partners = np.array([1], dtype=np.uint64)
    points = surface.partner_covered_points(
        coordinates, radii, owners, partners, probe=PROBE, density=200.0
    )
    assert len(points) == 0


def test_overlapping_groups_are_refused() -> None:
    coordinates, radii, _ = _pair(distance=3.0)
    with pytest.raises(molframe.MolframeError):
        surface.partner_covered_points(
            coordinates,
            radii,
            np.array([0], dtype=np.uint64),
            np.array([0], dtype=np.uint64),
            probe=PROBE,
        )
