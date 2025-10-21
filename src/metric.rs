use std::{
    f64::{self, consts::PI},
    iter,
    num::NonZeroUsize,
};

use kiddo::{ImmutableKdTree, SquaredEuclidean};
use nalgebra::{Point3, Vector2};
use parry3d_f64::{query::PointQuery, shape::Triangle};
use rand::{
    Rng, SeedableRng,
    distr::{Distribution, weighted::WeightedIndex},
};
use rand_chacha::ChaCha8Rng;

use crate::{mesh::Mesh, ops};

const RV_K: f64 = 0.3;

/// The number of samples per unit of surface area for computing the Hausdorff
/// distance.
const SAMPLING_RESOLUTION: f64 = 3000.0;

/// The minimum number of samples to be considered when computing the Hausdorff
/// distance.
const MIN_SAMPLES: usize = 1000;

/// Wei et al. 2022, equation (5)
pub fn compute_rv(a: &Mesh, b: &Mesh) -> f64 {
    let a_vol = ops::volume(a);
    let b_vol = ops::volume(b);

    let diff_vol = a_vol.abs() - b_vol.abs();

    f64::powf(3. * diff_vol.abs() / (4. * PI), 1. / 3.)
}

/// Return a uniformly sampled point on the triangle's surface.
///
/// Implemented via the parallelogram method: we sample a point on a
/// parallelogram which contains the triangle. If the point is outside the
/// triangle then we transform it to be inside, preserving the uniformity.
fn sample_triangle_surface<R: Rng>(tri: &Triangle, rng: &mut R) -> Point3<f64> {
    let mut u: Vector2<f64> = rng.random();

    // Sample is in part of the parallelogram which is outside the triangle,
    // shift it back into the triangle.
    if u.sum() > 1.0 {
        u = Vector2::repeat(1.0) - u;
    }

    let ab = tri.b - tri.a;
    let ac = tri.c - tri.a;

    let off = u[0] * ab + u[1] * ac;
    tri.a + off
}

/// A point sampled on the surface of a triangle which remembers its source
/// triangle.
struct PointInTriangle(Triangle, Point3<f64>);

/// Sample uniform points from the surface of a mesh.
///
/// The number of points is determined by the total surface area divided by the
/// resolution.
fn sample<R: Rng>(mesh: &Mesh, resolution: f64, rng: &mut R) -> Vec<PointInTriangle> {
    let mut total_area = 0.0;

    // Compute a distribution weighted by face area.
    let mut weights = Vec::with_capacity(mesh.faces.len());
    for triangle in mesh.triangles() {
        let area = triangle.area();
        weights.push(area);
        total_area += area;
    }

    if total_area == 0.0 {
        return vec![];
    }

    let n_samples = usize::max(MIN_SAMPLES, (total_area * resolution) as usize);
    let distribution = WeightedIndex::new(&weights).unwrap();

    // Draw random triangles from the distribution, and then uniformly sample
    // across the triangle surface.
    iter::repeat_with(|| {
        let face_index = distribution.sample(rng);
        let triangle = mesh.triangle(face_index);
        let point = sample_triangle_surface(&triangle, rng);
        PointInTriangle(triangle, point)
    })
    .take(n_samples)
    .collect()
}

/// Compute the component of the Hausdorff distance to find the least upper
/// bound on the shortest Euclidean distance from each point a ∈ Sample(A) to
/// Sample(B).
fn hausdorff_element(a: &[PointInTriangle], b: &[PointInTriangle]) -> f64 {
    const N_LOOKUP: NonZeroUsize = NonZeroUsize::new(10).unwrap();

    // We will be querying a lot of points in B, so build an acceleration
    // structure.
    let b_index = {
        let v: Vec<_> = b.iter().map(|s| [s.1.x, s.1.y, s.1.z]).collect();
        ImmutableKdTree::new_from_slice(&v)
    };

    // For each vertex a ∈ Sample(A), compute the minimum distance to B. The
    // Hausdorff distance from A to B is the supremum of these minimums.
    a.iter()
        .flat_map(|a_sample| {
            // Find the point on the surface of B which is closest to a.
            let a_pt = a_sample.1;

            // The closest point in B is probably within one of the faces of the
            // nearest N vertices.
            let b_samples =
                b_index.nearest_n::<SquaredEuclidean>(&[a_pt.x, a_pt.y, a_pt.z], N_LOOKUP);
            b_samples
                .into_iter()
                .map(|b_sample| {
                    let tri_in_b = b[b_sample.item as usize].0;
                    tri_in_b.distance_to_local_point(&a_pt, true)
                })
                .min_by(|a, b| a.total_cmp(b))
        })
        .max_by(|a, b| a.total_cmp(b))
        .unwrap_or(0.0)
}

/// Wei et al. 2022, section (4)
fn compute_hb(ma: &Mesh, mb: &Mesh) -> f64 {
    let mut rng = ChaCha8Rng::seed_from_u64(0); // TODO: use global rng

    // Sample points on the surface of each mesh.
    let samples_a = sample(ma, SAMPLING_RESOLUTION, &mut rng);
    let samples_b = sample(mb, SAMPLING_RESOLUTION, &mut rng);

    f64::max(
        hausdorff_element(&samples_a, &samples_b),
        hausdorff_element(&samples_b, &samples_a),
    )
}

/// Wei et al. 2022, equation (6)
pub fn concavity_metric(a: &Mesh, b: &Mesh, exact: bool) -> f64 {
    // R_v is always computed. H_b is only computed when an exact measurement is
    // required.
    let rv = compute_rv(a, b);
    let hb = if exact {
        compute_hb(a, b)
    } else {
        f64::NEG_INFINITY
    };

    f64::max(RV_K * rv, hb)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    use super::*;

    #[test]
    fn test_triangle_sampling() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        for _ in 0..100 {
            let triangle = Triangle::new(rng.random(), rng.random(), rng.random());
            let p = sample_triangle_surface(&triangle, &mut rng);

            assert!(triangle.distance_to_local_point(&p, true).abs() < 1e-10);
        }
    }

    #[test]
    fn test_hausdorff_element() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let tri1 = Triangle::new(
            Point3::new(-1., -1., 0.),
            Point3::new(0., -0.1, 0.),
            Point3::new(1., -1., 0.),
        );
        let tri2 = Triangle::new(
            Point3::new(-1., 1., 0.),
            Point3::new(0., 0.1, 0.),
            Point3::new(1., 1., 0.),
        );

        // Nearest point on tri2 from any point in tri1 is (0, 0.1).
        for _ in 0..100 {
            let p1 = sample_triangle_surface(&tri1, &mut rng);
            let p2 = sample_triangle_surface(&tri2, &mut rng);
            assert_relative_eq!(
                hausdorff_element(&[PointInTriangle(tri1, p1)], &[PointInTriangle(tri2, p2)]),
                nalgebra::distance(&p1, &Point3::new(0., 0.1, 0.))
            );
        }
    }
}
