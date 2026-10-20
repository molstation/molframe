//! Owner-attributed surface points and pairwise contact areas with one partner omitted.
//!
//! Each atom is a probe-grown ball. A point on that ball is exposed when a probe centred on it
//! touches no other ball, except the partner that is omitted from the test. The points and the
//! areas are returned as `NumPy` arrays so that the Python layer copies nothing per point.

use crate::bindings::array::{Array1, Array2};
use molframe::surface::{
    AtomContactArea, ExcludedSurfacePoint, atom_contact_areas as kernel_atom_contact_areas,
    partner_covered_points as kernel_partner_covered_points,
    surface_points_excluding_pairs as kernel_surface_points_excluding_pairs,
};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

/// Points exposed when one partner is omitted, with their owning atom and omitted partner.
#[derive(Clone, Debug)]
#[pyclass(
    name = "ExcludedSurfacePoints",
    frozen,
    skip_from_py_object,
    module = "molframe.surface"
)]
pub(crate) struct PyExcludedSurfacePoints {
    points: Vec<ExcludedSurfacePoint>,
}

#[pymethods]
impl PyExcludedSurfacePoints {
    /// Point positions on the probe-grown ball, `(n, 3)`.
    #[getter]
    fn positions<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let flat: Vec<f32> = self
            .points
            .iter()
            .flat_map(|point| point.position)
            .collect();
        flat.into_pyarray(py).reshape((self.points.len(), 3))
    }

    /// The atom carrying each point.
    #[getter]
    fn atoms<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let atoms: Vec<u64> = self.points.iter().map(|point| point.atom as u64).collect();
        atoms.into_pyarray(py)
    }

    /// The partner omitted from the accessibility test for each point.
    #[getter]
    fn excluded<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let excluded: Vec<u64> = self
            .points
            .iter()
            .map(|point| point.excluded as u64)
            .collect();
        excluded.into_pyarray(py)
    }

    /// Surface area each sample represents, in square ångström.
    #[getter]
    fn areas<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let areas: Vec<f64> = self.points.iter().map(|point| point.area).collect();
        areas.into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.points.len()
    }
}

/// Solvent-excluded contact area for each unordered atom pair, as two arrays and an area array.
#[derive(Clone, Debug)]
#[pyclass(
    name = "AtomContactAreas",
    frozen,
    skip_from_py_object,
    module = "molframe.surface"
)]
pub(crate) struct PyAtomContactAreas {
    areas: Vec<AtomContactArea>,
}

#[pymethods]
impl PyAtomContactAreas {
    /// The lower atom index of each pair.
    #[getter]
    fn first<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let values: Vec<u64> = self.areas.iter().map(|row| row.first as u64).collect();
        values.into_pyarray(py)
    }

    /// The higher atom index of each pair.
    #[getter]
    fn second<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u64>> {
        let values: Vec<u64> = self.areas.iter().map(|row| row.second as u64).collect();
        values.into_pyarray(py)
    }

    /// Mean of the two directed buried patch areas, in square ångström.
    #[getter]
    fn area<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        let values: Vec<f64> = self.areas.iter().map(|row| row.area).collect();
        values.into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.areas.len()
    }
}

/// Points on each atom's probe-grown ball that are exposed when one partner is omitted.
///
/// ``pairs`` is an ``(m, 2)`` integer array of atom indices. Each unordered pair contributes
/// points from both atoms, each omitting the other.
#[pyfunction]
#[pyo3(signature = (coordinates, radii, pairs, *, probe=1.4, density=4.0, context=None))]
fn surface_points_excluding_pairs(
    py: Python<'_>,
    coordinates: Array2<'_, '_, f32>,
    radii: Array1<'_, '_, f32>,
    pairs: Array2<'_, '_, u64>,
    probe: f32,
    density: f32,
    context: Option<&crate::execution::PyExecutionContext>,
) -> PyResult<PyExcludedSurfacePoints> {
    let positions_view = coordinates.readonly();
    let radii_view = radii.readonly();
    let positions = crate::bindings::coordinates(&positions_view)?;
    let radii = contiguous_radii(&radii_view)?;
    let pairs = pair_list(&pairs.readonly())?;
    let points = crate::execution::run(py, context, |context| {
        kernel_surface_points_excluding_pairs(positions, radii, probe, density, &pairs, context)
    })?
    .map_err(crate::error::kernel)?;
    Ok(PyExcludedSurfacePoints { points })
}

/// Points on each owner atom's probe-grown ball that the partner group covers and nothing else
/// hides. `owners` and `partners` are one-dimensional arrays of atom indices, each atom in at
/// most one group. Each sample records its owner and nearest covering partner.
#[pyfunction]
#[pyo3(signature = (coordinates, radii, owners, partners, *, probe=1.4, density=4.0, context=None))]
fn partner_covered_points(
    py: Python<'_>,
    coordinates: Array2<'_, '_, f32>,
    radii: Array1<'_, '_, f32>,
    owners: Array1<'_, '_, u64>,
    partners: Array1<'_, '_, u64>,
    probe: f32,
    density: f32,
    context: Option<&crate::execution::PyExecutionContext>,
) -> PyResult<PyExcludedSurfacePoints> {
    let positions_view = coordinates.readonly();
    let radii_view = radii.readonly();
    let positions = crate::bindings::coordinates(&positions_view)?;
    let radii = contiguous_radii(&radii_view)?;
    let owners = index_list(&owners.readonly())?;
    let partners = index_list(&partners.readonly())?;
    let points = crate::execution::run(py, context, |context| {
        kernel_partner_covered_points(
            positions, radii, probe, density, &owners, &partners, context,
        )
    })?
    .map_err(crate::error::kernel)?;
    Ok(PyExcludedSurfacePoints { points })
}

/// Solvent-excluded contact area for each unordered atom pair of the structure.
#[pyfunction]
#[pyo3(signature = (coordinates, radii, *, probe=1.4, density=4.0, context=None))]
fn atom_contact_areas(
    py: Python<'_>,
    coordinates: Array2<'_, '_, f32>,
    radii: Array1<'_, '_, f32>,
    probe: f32,
    density: f32,
    context: Option<&crate::execution::PyExecutionContext>,
) -> PyResult<PyAtomContactAreas> {
    let positions_view = coordinates.readonly();
    let radii_view = radii.readonly();
    let positions = crate::bindings::coordinates(&positions_view)?;
    let radii = contiguous_radii(&radii_view)?;
    let areas = crate::execution::run(py, context, |context| {
        kernel_atom_contact_areas(positions, radii, probe, density, context)
    })?
    .map_err(crate::error::kernel)?;
    Ok(PyAtomContactAreas { areas })
}

fn contiguous_radii<'a>(array: &'a numpy::PyReadonlyArray1<'_, f32>) -> PyResult<&'a [f32]> {
    array.as_slice().map_err(|_| {
        crate::error::value("radii must be C-contiguous; call numpy.ascontiguousarray")
    })
}

fn index_list(array: &numpy::PyReadonlyArray1<'_, u64>) -> PyResult<Vec<usize>> {
    array
        .as_slice()
        .map_err(|_| crate::error::value("atom index arrays must be C-contiguous"))?
        .iter()
        .map(|&index| {
            usize::try_from(index)
                .map_err(|_| crate::error::value("atom indices must fit in the platform size"))
        })
        .collect()
}

fn pair_list(array: &numpy::PyReadonlyArray2<'_, u64>) -> PyResult<Vec<(usize, usize)>> {
    let view = array.as_array();
    if view.ncols() != 2 {
        return Err(crate::error::value("pairs must have shape (m, 2)"));
    }
    view.rows()
        .into_iter()
        .map(|row| {
            let first = usize::try_from(row[0]);
            let second = usize::try_from(row[1]);
            match (first, second) {
                (Ok(first), Ok(second)) => Ok((first, second)),
                _ => Err(crate::error::value(
                    "atom indices must fit in the platform size",
                )),
            }
        })
        .collect()
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(surface_points_excluding_pairs, module)?)?;
    module.add_function(wrap_pyfunction!(atom_contact_areas, module)?)?;
    module.add_function(wrap_pyfunction!(partner_covered_points, module)?)?;
    module.add_class::<PyExcludedSurfacePoints>()?;
    module.add_class::<PyAtomContactAreas>()?;
    Ok(())
}
