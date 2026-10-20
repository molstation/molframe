use super::{atom_contact_areas, surface_points_excluding_pairs};
use crate::accessible_area::SasaError;
use core::f64::consts::PI;
use molframe_core::ExecutionContext;

fn context() -> ExecutionContext {
    ExecutionContext::default()
}

/// Two equal balls of expanded radius `R` at separation `d` share a spherical cap of
/// area `2 pi R (R - d/2)` on each ball, the closed form the sampler converges to.
fn cap_area(radius: f64, distance: f64) -> f64 {
    2.0 * PI * radius * (radius - distance / 2.0)
}

#[test]
fn an_isolated_atom_has_no_partner_patch() {
    let Ok(points) =
        surface_points_excluding_pairs(&[[0.0, 0.0, 0.0]], &[1.7], 1.4, 4.0, &[], &context())
    else {
        panic!("a lone atom is a valid input");
    };
    assert!(points.is_empty());
}

#[test]
fn atoms_beyond_contact_share_no_patch() {
    let positions = [[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]];
    let Ok(areas) = atom_contact_areas(&positions, &[1.7, 1.7], 1.4, 50.0, &context()) else {
        panic!("valid input");
    };
    assert!(
        areas.is_empty(),
        "no candidate pairs beyond the neighbour radius"
    );
}

#[test]
fn two_sphere_cap_matches_the_closed_form() {
    let positions = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let Ok(areas) = atom_contact_areas(&positions, &[1.7, 1.7], 1.4, 400.0, &context()) else {
        panic!("valid input");
    };
    let [pair] = areas.as_slice() else {
        panic!("one candidate pair expected, found {}", areas.len());
    };
    let expected = cap_area(1.7 + 1.4, 3.0);
    assert!(
        (pair.area - expected).abs() / expected < 0.02,
        "cap {} differs from closed form {expected}",
        pair.area
    );
}

#[test]
fn directed_patches_sum_to_twice_the_pair_area() {
    let positions = [[0.0, 0.0, 0.0], [3.2, 0.0, 0.0]];
    let radii = [1.7, 1.7];
    let Ok(points) =
        surface_points_excluding_pairs(&positions, &radii, 1.4, 200.0, &[(0, 1)], &context())
    else {
        panic!("valid input");
    };
    let Ok(areas) = atom_contact_areas(&positions, &radii, 1.4, 200.0, &context()) else {
        panic!("valid input");
    };
    let directed: f64 = points.iter().map(|point| point.area).sum();
    assert!((directed - 2.0 * areas[0].area).abs() < 1e-9 * directed.max(1.0));
}

#[test]
fn sampled_points_lie_on_the_owner_ball_and_inside_the_partner_ball() {
    let positions = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let radii = [1.7, 1.7];
    let probe = 1.4;
    let Ok(points) =
        surface_points_excluding_pairs(&positions, &radii, probe, 50.0, &[(0, 1)], &context())
    else {
        panic!("valid input");
    };
    assert!(!points.is_empty());
    let expanded = f64::from(1.7_f32) + f64::from(probe);
    for point in &points {
        let owner = positions[point.atom];
        let partner = positions[point.excluded];
        let distance = |centre: [f64; 3]| -> f64 {
            point
                .position
                .iter()
                .zip(centre)
                .map(|(value, c)| (f64::from(*value) - c).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let owner_centre = owner.map(f64::from);
        let partner_centre = partner.map(f64::from);
        assert!((distance(owner_centre) - expanded).abs() < 1e-3);
        assert!(distance(partner_centre) <= expanded + 1e-3);
    }
}

#[test]
fn a_pair_naming_one_atom_twice_is_rejected() {
    let result = surface_points_excluding_pairs(
        &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        &[1.7, 1.7],
        1.4,
        4.0,
        &[(0, 0)],
        &context(),
    );
    assert!(matches!(
        result,
        Err(SasaError::InvalidPair {
            first: 0,
            second: 0,
            atoms: 2
        })
    ));
}

#[test]
fn a_pair_outside_the_structure_is_rejected() {
    let result = surface_points_excluding_pairs(
        &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        &[1.7, 1.7],
        1.4,
        4.0,
        &[(0, 5)],
        &context(),
    );
    assert!(matches!(
        result,
        Err(SasaError::InvalidPair { second: 5, .. })
    ));
}

#[test]
fn a_non_positive_density_is_rejected() {
    let result = atom_contact_areas(
        &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        &[1.7, 1.7],
        1.4,
        0.0,
        &context(),
    );
    assert!(matches!(result, Err(SasaError::InvalidDensity)));
}
