use std::collections::{VecDeque, hash_map::Entry};

use ahash::AHashMap;
use glamx::DVec3;
use parry3d_f64::utils::remove_unused_points;
use spade::{ConstrainedDelaunayTriangulation, Triangulation};

use crate::{mesh::Mesh, util::cantor_point_hash};

/// Distance threshold in which vertices on the slice cap will be merged.
const DEDUP_THRESHOLD: f64 = 1e-6;

/// A plane with a normal pointing in one of the standard basis vectors, offset
/// from the origin according to the bias.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct CanonicalPlane {
    pub axis: usize,
    pub bias: f64,
}

impl CanonicalPlane {
    /// Return a plane with the given bias.
    pub fn with_bias(&self, bias: f64) -> Self {
        CanonicalPlane {
            axis: self.axis,
            bias,
        }
    }

    /// If this plane's bias is on the interval (0, 1), returns a plane with a
    /// bias on the interval (lb, ub).
    pub fn denormalize(&self, lb: f64, ub: f64) -> Self {
        Self {
            axis: self.axis,
            bias: self.bias * (ub - lb) + lb,
        }
    }

    /// Find the intersection point between the plane and a line segment defined
    /// by two endpoints, if there is one.
    ///
    /// We can simplify this calculation because we are intersecting the line
    /// with a canonical plane.
    pub fn line_intersection(&self, pa: &DVec3, pb: &DVec3) -> Option<DVec3> {
        let l = pb[self.axis] - pa[self.axis];

        // Line is parallel to plane?
        if l.abs() != 0.0 {
            let d = (self.bias - pa[self.axis]) / l;
            // Intersection is within the two endpoints?
            if (0.0..=1.0).contains(&d) {
                return Some(pa + d * (pb - pa));
            }
        }

        None
    }

    /// Returns true if the point p is on the positive side of the plane.
    pub fn above(&self, p: &DVec3) -> bool {
        p[self.axis] > self.bias
    }
}

/// If the triangle defined by (v1, v2, v3) crosses the plane, split it into a
/// quad and a triangle which are each placed into one of the top or bottom
/// meshes.
///
// Ref: https://github.com/dgreenheck/three-pinata/blob/dd44fce55fa20e1c859a6f59a7f6242e19503720/lib/src/fracture/SliceFragment.ts#L232
#[allow(clippy::too_many_arguments)]
fn split_triangle(
    plane: &CanonicalPlane,
    v1_idx: u32,
    v2_idx: u32,
    v3_idx: u32,
    v3_below_plane: bool,
    input_mesh: &Mesh,
    top_mesh: &mut Mesh,
    bot_mesh: &mut Mesh,
    edge_constraints: &mut Vec<(u32, u32)>,
) {
    let v1 = input_mesh.vertices[v1_idx as usize];
    let v2 = input_mesh.vertices[v2_idx as usize];
    let v3 = input_mesh.vertices[v3_idx as usize];

    let e13 = plane.line_intersection(&v1, &v3);
    let e23 = plane.line_intersection(&v2, &v3);

    if let Some(v13) = e13
        && let Some(v23) = e23
    {
        top_mesh.vertices.push(v13);
        top_mesh.vertices.push(v23);

        bot_mesh.vertices.push(v13);
        bot_mesh.vertices.push(v23);

        let idx13_top = (top_mesh.vertices.len() - 2) as u32;
        let idx23_top = (top_mesh.vertices.len() - 1) as u32;
        let idx13_bot = (bot_mesh.vertices.len() - 2) as u32;
        let idx23_bot = (bot_mesh.vertices.len() - 1) as u32;

        if v3_below_plane {
            // Triangle above the plane has two points, so it becomes a quad in
            // the top mesh and a triangle in the bottom.
            top_mesh.faces.push([idx23_top, idx13_top, v2_idx]);
            top_mesh.faces.push([idx13_top, v1_idx, v2_idx]);

            bot_mesh.faces.push([v3_idx, idx13_bot, idx23_bot]);

            edge_constraints.push((idx13_top, idx23_top));
        } else {
            // Triangle above the plane is a simple triangle.
            top_mesh.faces.push([idx13_top, idx23_top, v3_idx]);

            bot_mesh.faces.push([v1_idx, v2_idx, idx13_bot]);
            bot_mesh.faces.push([v2_idx, idx23_bot, idx13_bot]);

            edge_constraints.push((idx23_top, idx13_top));
        }
    }
}

/// Deduplicate vertices of the given mesh, from the given start vertex index to
/// the end. Vertices which are within +/- 0.5 * tol will be merged.
///
/// Because this will shift the vertex indexing, the face and edge constraint
/// arrays will be updated to not change the overall mesh topology.
#[allow(clippy::needless_range_loop)]
fn dedup_vertices(
    start_idx: u32,
    mesh: &mut Mesh,
    edge_constraints: Option<&mut [(u32, u32)]>,
    tol: f64,
) {
    let n_cut_vertices = mesh.vertices.len() - start_idx as usize;

    let mut index_map = vec![0; n_cut_vertices];
    let mut welded_verts = vec![];
    let mut adjacency_map = AHashMap::new();

    let mut k = start_idx;
    for i in 0..n_cut_vertices {
        let v = mesh.vertices[i + start_idx as usize];
        let key = cantor_point_hash(v, tol);

        match adjacency_map.entry(key) {
            Entry::Occupied(occupied_entry) => {
                // Remap this vertex to the one we've already seen.
                index_map[i] = *occupied_entry.get();
            }
            Entry::Vacant(entry) => {
                // Haven't seen this vertex, add it to the output.
                entry.insert(k);
                index_map[i] = k;
                welded_verts.push(v);
                k += 1;
            }
        }
    }

    // Replace the updated portion of the vertex buffer.
    mesh.vertices.truncate(start_idx as usize);
    mesh.vertices.append(&mut welded_verts);

    // Remap the face indices.
    mesh.faces.iter_mut().for_each(|face| {
        for i in 0..3 {
            if face[i] > start_idx {
                face[i] = index_map[(face[i] - start_idx) as usize]
            }
        }
    });

    // Remap the edge constraint indices.
    if let Some(edge_constraints) = edge_constraints {
        edge_constraints.iter_mut().for_each(|constraint| {
            constraint.0 = index_map[(constraint.0 - start_idx) as usize];
            constraint.1 = index_map[(constraint.1 - start_idx) as usize];
        });
    }
}

/// Slice an input mesh along the given canonical axis (X, Y, Z) and offset from
/// the origin.
///
/// Returned are the top and bottom slices of the input mesh. If the slice plane
/// does not intersect the input then one of the meshes will be empty.
pub fn slice(input_mesh: &Mesh, plane: &CanonicalPlane) -> (Mesh, Mesh) {
    // Colorize each vertex based on which side of the plane it is one.
    let above: Vec<bool> = input_mesh.vertices.iter().map(|p| plane.above(p)).collect();

    let mut top_mesh = Mesh {
        vertices: input_mesh.vertices.clone(),
        faces: vec![],
    };
    let mut bot_mesh = Mesh {
        vertices: input_mesh.vertices.clone(),
        faces: vec![],
    };

    let mut edge_constraints = vec![];

    // For each face in the input, there are generally three possibilities:
    //
    // 1. The triangle is entirely contained in the top mesh.
    // 2. The triangle is entirely contained in the bottom mesh.
    // 3. The triangle intersects the slice plane, so it must be divided into
    //    the top and bottom meshes.
    //
    // Reference: https://github.com/dgreenheck/three-pinata
    for face in &input_mesh.faces {
        let a = face[0] as usize;
        let b = face[1] as usize;
        let c = face[2] as usize;

        // Entirely above
        if above[a] && above[b] && above[c] {
            top_mesh.faces.push([a as u32, b as u32, c as u32]);
        }
        // Entirely below
        else if !above[a] && !above[b] && !above[c] {
            bot_mesh.faces.push([a as u32, b as u32, c as u32]);
        }
        // Partially above and below
        else {
            // Two vertices are above and one is below
            let (i, j, k, v3_below_plane) = if above[b] && above[c] && !above[a] {
                (b, c, a, true)
            } else if above[c] && above[a] && !above[b] {
                (c, a, b, true)
            } else if above[a] && above[b] && !above[c] {
                (a, b, c, true)
            }
            // Two vertices are below and one is above
            else if !above[b] && !above[c] && above[a] {
                (b, c, a, false)
            } else if !above[c] && !above[a] && above[b] {
                (c, a, b, false)
            } else if !above[a] && !above[b] && above[c] {
                (a, b, c, false)
            } else {
                unreachable!("not all vertex coloring cases handled")
            };

            split_triangle(
                plane,
                i as u32,
                j as u32,
                k as u32,
                v3_below_plane,
                input_mesh,
                &mut top_mesh,
                &mut bot_mesh,
                &mut edge_constraints,
            );
        }
    }

    // Eliminate any duplicate vertices in the cut face only. Remap the cut face
    // indices and edge constraint indices to correctly point to the newly
    // merged vertices.
    dedup_vertices(
        input_mesh.vertices.len() as u32,
        &mut top_mesh,
        Some(&mut edge_constraints),
        1e-6,
    );
    dedup_vertices(
        input_mesh.vertices.len() as u32,
        &mut bot_mesh,
        None,
        DEDUP_THRESHOLD,
    );

    // Compute the triangulation for the "cap" which is placed at the
    // intersection of each mesh and the plane. We only compute it for the top
    // mesh, as it will be the exact same (with flipped normals) for the bottom.
    let mut triangulation = ConstrainedDelaunayTriangulation::<spade::Point2<f64>>::with_capacity(
        top_mesh.vertices.len() - input_mesh.vertices.len(),
        edge_constraints.len(),
        0,
    );

    let mut mesh2spade = AHashMap::new();
    let mut spade2mesh = AHashMap::new();
    for i in input_mesh.vertices.len()..top_mesh.vertices.len() {
        let point3d = top_mesh.vertices[i];

        let x = point3d[(plane.axis + 1) % 3];
        let y = point3d[(plane.axis + 2) % 3];
        let handle = triangulation
            .insert(spade::Point2::new(x, y))
            .expect("could not insert triangulation point");
        mesh2spade.insert(i as u32, handle);
        spade2mesh.insert(handle, i as u32);
    }

    for edge in &edge_constraints {
        triangulation.try_add_constraint(mesh2spade[&edge.0], mesh2spade[&edge.1]);
    }

    // Remove any faces of the triangulation which are outside of the
    // boundaries. Think of this as a flood fill from within the boundary,
    // which is not allowed to leave any constrained edge.
    let mut skip = vec![true; triangulation.num_all_faces()];
    let mut visited = vec![false; triangulation.num_all_faces()];

    for (v1, v2) in &edge_constraints {
        if let Some(edge) = triangulation.get_edge_from_neighbors(mesh2spade[v1], mesh2spade[v2])
            && edge.is_constraint_edge()
        {
            let mut frontier = VecDeque::new();

            let right_face = edge.rev().face();
            frontier.push_back(right_face);

            while let Some(face) = frontier.pop_front() {
                if visited[face.index()] {
                    continue;
                }

                skip[face.index()] = false;
                visited[face.index()] = true;

                if let Some(frontier_face) = face.as_inner() {
                    for edge in frontier_face.adjacent_edges() {
                        if !edge.is_constraint_edge() {
                            let right_face = edge.rev().face();
                            frontier.push_back(right_face);
                        }
                    }
                }
            }
        }
    }

    // Attach the cap to the mesh. One of the top/bottom meshes needs to have
    // the normals reversed. Otherwise the cap is the same for the top and
    // bottom, so we reuse it.
    for (output_mesh, flip_normals) in [(&mut top_mesh, true), (&mut bot_mesh, false)] {
        for face in triangulation.inner_faces() {
            let [i, j, k] = if flip_normals { [2, 1, 0] } else { [0, 1, 2] };

            if !skip[face.index()] {
                output_mesh.faces.push([
                    spade2mesh[&face.vertices()[i].fix()],
                    spade2mesh[&face.vertices()[j].fix()],
                    spade2mesh[&face.vertices()[k].fix()],
                ]);
            }
        }
    }

    // We may have left vertices from the original mesh which are no longer used
    // by faces in the sliced mesh. Clean them up or else they will affect the
    // convex hull, bbox, etc. calculations later on.
    remove_unused_points(&mut top_mesh.vertices, &mut top_mesh.faces);
    remove_unused_points(&mut bot_mesh.vertices, &mut bot_mesh.faces);

    (top_mesh, bot_mesh)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;

    use super::*;

    #[test]
    fn test_plane_denormalize() {
        let unit_plane = CanonicalPlane { axis: 0, bias: 0.5 };
        assert_relative_eq!(unit_plane.denormalize(1.0, 2.0).bias, 1.5);
        assert_relative_eq!(unit_plane.denormalize(-5.0, 0.0).bias, -2.5);
    }

    #[test]
    fn test_plane_intersection() {
        let plane = CanonicalPlane {
            axis: 2,
            bias: 0.25,
        };

        assert!(
            plane
                .line_intersection(&DVec3::new(0.0, 0.0, 0.0), &DVec3::new(1.0, 0.0, 0.0))
                .is_none()
        );
        assert_relative_eq!(
            plane
                .line_intersection(&DVec3::new(0.0, 0.0, -0.5), &DVec3::new(0.0, 0.0, 0.5))
                .unwrap(),
            DVec3::new(0.0, 0.0, 0.25)
        );
        assert_relative_eq!(
            plane
                .line_intersection(&DVec3::new(1.0, 2.0, -0.5), &DVec3::new(1.0, 2.0, 0.5))
                .unwrap(),
            DVec3::new(1.0, 2.0, 0.25)
        );
        assert_relative_eq!(
            plane
                .line_intersection(
                    &DVec3::new(-0.75, -0.75, -0.75),
                    &DVec3::new(1.25, 1.25, 1.25)
                )
                .unwrap(),
            DVec3::new(0.25, 0.25, 0.25)
        );
    }

    #[test]
    fn test_dedup_vertices() {
        let mut mesh = Mesh::new(
            vec![
                DVec3::new(0., 0., 0.),
                DVec3::new(0., 0., 1e-10),
                DVec3::new(1e-10, 0., 0.0),
            ],
            vec![[0, 1, 2]],
        );
        let mut edge_constraints = vec![(0, 1), (1, 2)];

        dedup_vertices(0, &mut mesh, Some(&mut edge_constraints), 1e-4);

        assert_eq!(mesh.vertices.len(), 1);
        assert_eq!(mesh.faces, vec![[0, 0, 0]]);
        assert_eq!(edge_constraints, vec![(0, 0), (0, 0)]);
    }
}
