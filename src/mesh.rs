use glamx::{DAffine3, DVec3};
use parry3d_f64::shape::Triangle;

use crate::ops;

/// A triangle mesh.
#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<DVec3>,
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn empty() -> Mesh {
        Mesh {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn new(vertices: Vec<DVec3>, faces: Vec<[u32; 3]>) -> Mesh {
        Mesh { vertices, faces }
    }

    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.faces.is_empty()
    }

    pub fn triangle(&self, face_index: usize) -> Triangle {
        let face = self.faces[face_index];

        Triangle::new(
            self.vertices[face[0] as usize],
            self.vertices[face[1] as usize],
            self.vertices[face[2] as usize],
        )
    }

    pub fn triangles(&self) -> impl Iterator<Item = Triangle> {
        self.faces.iter().map(|face| {
            Triangle::new(
                self.vertices[face[0] as usize],
                self.vertices[face[1] as usize],
                self.vertices[face[2] as usize],
            )
        })
    }

    /// Merge the vertices and faces from a mesh into this one.
    pub fn merge(&mut self, mut other: Mesh) {
        let offset = self.vertices.len() as u32;

        // Vertices from the second mesh are tacked on to the end of the vertex
        // buffer.
        self.vertices.append(&mut other.vertices);

        // Indices from the second mesh must be offset to point to the new
        // vertex locations in the combined mesh.
        self.faces
            .extend(other.faces.into_iter().map(|f| f.map(|f| f + offset)));
    }

    /// Transform such that the mesh is centered and on the range (-1, 1) along
    /// the longest extent.
    pub fn normalization_transform(&self) -> DAffine3 {
        let bbox = ops::bbox(self);

        DAffine3::from_scale(DVec3::splat(2. / bbox.extent().max_element()))
            * DAffine3::from_translation(-bbox.center())
    }

    pub fn transform(self, tfm: &DAffine3) -> Mesh {
        Mesh {
            vertices: self
                .vertices
                .into_iter()
                .map(|pt| tfm.transform_point3(pt))
                .collect(),
            faces: self.faces,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use glamx::dvec3;

    use super::*;

    #[test]
    fn test_normalization_transform() {
        let vertices = vec![dvec3(1.0, 2.0, 4.0), dvec3(0.0, -2.0, -4.0)];
        let triangles = vec![[0, 1, 2], [1, 2, 3]];

        let mesh = Mesh::new(vertices, triangles);
        let tfm = mesh.normalization_transform();

        let unit_mesh = mesh.transform(&tfm);

        assert_relative_eq!(&unit_mesh.vertices[0], &dvec3(0.125, 0.5, 1.0));
        assert_relative_eq!(&unit_mesh.vertices[1], &dvec3(-0.125, -0.5, -1.0));
    }
}
