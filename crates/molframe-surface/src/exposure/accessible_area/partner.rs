//! Probe patches on each owner atom that a partner group covers and nothing else hides.
//!
//! This is the burial rule of the legacy Connolly-surface programme, realised with probe-grown
//! balls. A probe centre on an owner's grown sphere is buried by the partner group when it lies
//! inside a partner's grown sphere and inside no other atom's grown sphere. Ligand atoms that are
//! bonded to one another all sit in the partner group, so their shading does not hide a patch
//! the partner really covers. Each returned sample is attributed to the nearest covering partner.

use super::contact::ExcludedSurfacePoint;
use super::{SasaError, point_on_sphere, samples_for_density, validate_density};
use crate::neighbourhood::{self, Coverage, Neighbourhood};
use crate::numeric::f64_to_f32;
use crate::sampling::fibonacci_sphere;
use molframe_core::ExecutionContext;
use std::collections::BTreeMap;

/// Owner-attributed probe patches buried by a partner group.
///
/// # Examples
///
/// ```
/// use molframe_core::ExecutionContext;
/// use molframe_surface::partner_covered_points;
///
/// // Two equal atoms 3 Å apart: the partner covers a cap of the owner's grown sphere.
/// let positions = [[0.0_f32, 0.0, 0.0], [3.0, 0.0, 0.0]];
/// let radii = [1.7_f32, 1.7];
/// let points = partner_covered_points(
///     &positions, &radii, 1.4, 50.0, &[0], &[1], &ExecutionContext::default(),
/// )?;
/// assert!(!points.is_empty());
/// assert!(points.iter().all(|point| point.atom == 0 && point.excluded == 1));
/// # Ok::<(), molframe_surface::SasaError>(())
/// ```
///
/// `owners` and `partners` must each list distinct atoms of the structure, and no atom may be in
/// both. Each owner contributes the samples of its grown sphere that the partner group covers,
/// with the nearest covering partner recorded as `excluded`. An empty owner or partner list is
/// valid and yields no points.
///
/// # Errors
///
/// Returns invalid geometry, sampling controls, or a group that names an atom outside the
/// structure, repeats one, or appears in both groups.
pub fn partner_covered_points(
    positions: &[[f32; 3]],
    radii: &[f32],
    probe: f32,
    density: f32,
    owners: &[usize],
    partners: &[usize],
    context: &ExecutionContext,
) -> Result<Vec<ExcludedSurfacePoint>, SasaError> {
    validate_density(density)?;
    let flags = partner_flags(positions.len(), owners, partners)?;
    let Some(hood) = neighbourhood::build(positions, radii, probe, context)? else {
        return Ok(Vec::new());
    };
    let mut directions = BTreeMap::new();
    let mut points = Vec::new();
    for &owner in owners {
        sample_owner(&hood, owner, &flags, density, &mut directions, &mut points)?;
    }
    Ok(points)
}

/// Builds the membership flags of the partner group, refusing every malformed group.
fn partner_flags(
    atoms: usize,
    owners: &[usize],
    partners: &[usize],
) -> Result<Vec<bool>, SasaError> {
    let mut owner_flags = vec![false; atoms];
    for &owner in owners {
        if owner >= atoms || owner_flags[owner] {
            return Err(SasaError::InvalidGroup { atom: owner, atoms });
        }
        owner_flags[owner] = true;
    }
    let mut partner_flags = vec![false; atoms];
    for &partner in partners {
        if partner >= atoms || partner_flags[partner] || owner_flags[partner] {
            return Err(SasaError::InvalidGroup {
                atom: partner,
                atoms,
            });
        }
        partner_flags[partner] = true;
    }
    Ok(partner_flags)
}

fn sample_owner(
    hood: &Neighbourhood,
    owner: usize,
    partner: &[bool],
    density: f32,
    directions: &mut BTreeMap<u16, Vec<[f64; 3]>>,
    output: &mut Vec<ExcludedSurfacePoint>,
) -> Result<(), SasaError> {
    let radius = hood.expanded[owner];
    if radius <= 0.0 {
        return Ok(());
    }
    let centre = hood.centre(owner);
    let samples = samples_for_density(radius, density)?;
    let area = 4.0 * core::f64::consts::PI * radius * radius / f64::from(samples);
    let directions = directions
        .entry(samples)
        .or_insert_with(|| fibonacci_sphere(samples));
    for &direction in directions.iter() {
        let point = point_on_sphere(centre, radius, direction);
        if let Coverage::Covered(covering) = hood.classify_against_partners(point, owner, partner) {
            output.push(ExcludedSurfacePoint {
                atom: owner,
                excluded: covering,
                position: point.map(f64_to_f32),
                area,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "partner_tests.rs"]
mod tests;
