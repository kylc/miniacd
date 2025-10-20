use nalgebra::{Matrix4, Point3};
use parry3d_f64::shape::Triangle;

use crate::ops;

/// A triangle mesh.
#[derive(Clone)]
pub struct Mesh {
    pub vertices: Vec<Point3<f64>>,
    pub faces: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn empty() -> Mesh {
        Mesh {
            vertices: Vec::new(),
            faces: Vec::new(),
        }
    }

    pub fn new(vertices: Vec<Point3<f64>>, faces: Vec<[u32; 3]>) -> Mesh {
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

    /// Transform such that the mesh is centered and on the range (-1, 1) along
    /// the longest extent.
    pub fn normalization_transform(&self) -> Matrix4<f64> {
        let bbox = ops::bbox(self);

        let tfm = Matrix4::identity();
        let tfm = tfm.append_translation(&-bbox.center());
        tfm.append_scaling(2. / bbox.extent().max())
    }

    pub fn transform(self, tfm: &Matrix4<f64>) -> Mesh {
        Mesh {
            vertices: self
                .vertices
                .into_iter()
                .map(|pt| tfm.transform_point(&pt))
                .collect(),
            faces: self.faces,
        }
    }
}
