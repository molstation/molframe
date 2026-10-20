use super::partner_covered_points;
use crate::accessible_area::SasaError;
use core::f64::consts::PI;
use molframe_core::ExecutionContext;

fn context() -> ExecutionContext {
    ExecutionContext::default()
}

/// Equal balls of expanded radius `R` at separation `d`: the cap that one covers on the other
/// has area `2 pi R (R - d/2)`, the closed form the sampler converges to.
fn cap_area(radius: f64, distance: f64) -> f64 {
    2.0 * PI * radius * (radius - distance / 2.0)
}

fn total(points: &[super::ExcludedSurfacePoint]) -> f64 {
    points.iter().map(|point| point.area).sum()
}

#[test]
fn an_empty_partner_group_covers_nothing() {
    let Ok(points) =
        partner_covered_points(&[[0.0, 0.0, 0.0]], &[1.7], 1.4, 4.0, &[0], &[], &context())
    else {
        panic!("valid input");
    };
    assert!(points.is_empty());
}

#[test]
fn a_single_partner_reproduces_the_two_sphere_cap() {
    let positions = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let Ok(points) =
        partner_covered_points(&positions, &[1.7, 1.7], 1.4, 400.0, &[0], &[1], &context())
    else {
        panic!("valid input");
    };
    let expected = cap_area(1.7 + 1.4, 3.0);
    assert!(
        (total(&points) - expected).abs() / expected < 0.02,
        "cap {} differs from closed form {expected}",
        total(&points)
    );
    assert!(
        points
            .iter()
            .all(|point| point.excluded == 1 && point.atom == 0)
    );
}

#[test]
fn coincident_partners_do_not_double_count_the_patch() {
    let single = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let doubled = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let Ok(one) = partner_covered_points(&single, &[1.7, 1.7], 1.4, 200.0, &[0], &[1], &context())
    else {
        panic!("valid input");
    };
    let Ok(two) = partner_covered_points(
        &doubled,
        &[1.7, 1.7, 1.7],
        1.4,
        200.0,
        &[0],
        &[1, 2],
        &context(),
    ) else {
        panic!("valid input");
    };
    assert!((total(&one) - total(&two)).abs() < 1e-9 * total(&one).max(1.0));
}

#[test]
fn a_non_partner_atom_between_owner_and_partner_hides_the_patch() {
    let open = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let blocked = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [1.5, 0.0, 0.0]];
    let Ok(without) =
        partner_covered_points(&open, &[1.7, 1.7], 1.4, 200.0, &[0], &[1], &context())
    else {
        panic!("valid input");
    };
    let Ok(with) = partner_covered_points(
        &blocked,
        &[1.7, 1.7, 1.7],
        1.4,
        200.0,
        &[0],
        &[1],
        &context(),
    ) else {
        panic!("valid input");
    };
    // The blocker lies between the centres, so every sample the partner covers lies within the
    // blocker's grown sphere too: the covered patch is hidden entirely.
    assert!(total(&without) > 0.0);
    assert!(
        total(&with) < 1e-9,
        "blocked patch kept area {}",
        total(&with)
    );
}

#[test]
fn sampled_points_lie_on_the_owner_expanded_sphere() {
    let positions = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]];
    let Ok(points) =
        partner_covered_points(&positions, &[1.7, 1.7], 1.4, 50.0, &[0], &[1], &context())
    else {
        panic!("valid input");
    };
    assert!(!points.is_empty());
    for point in &points {
        let norm = point
            .position
            .iter()
            .map(|value| f64::from(*value).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - (1.7 + 1.4)).abs() < 1e-3, "norm {norm}");
    }
}

#[test]
fn an_atom_in_both_groups_is_rejected() {
    let result = partner_covered_points(
        &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        &[1.7, 1.7],
        1.4,
        4.0,
        &[0],
        &[0],
        &context(),
    );
    assert!(matches!(
        result,
        Err(SasaError::InvalidGroup { atom: 0, atoms: 2 })
    ));
}

#[test]
fn a_repeated_owner_is_rejected() {
    let result = partner_covered_points(
        &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        &[1.7, 1.7],
        1.4,
        4.0,
        &[0, 0],
        &[1],
        &context(),
    );
    assert!(matches!(
        result,
        Err(SasaError::InvalidGroup { atom: 0, .. })
    ));
}

#[test]
fn a_partner_outside_the_structure_is_rejected() {
    let result = partner_covered_points(
        &[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
        &[1.7, 1.7],
        1.4,
        4.0,
        &[0],
        &[7],
        &context(),
    );
    assert!(matches!(
        result,
        Err(SasaError::InvalidGroup { atom: 7, atoms: 2 })
    ));
}
