use nalgebra::{Point3, Vector3};

use crate::mesh::Mesh;

#[derive(Clone)]
pub struct Aabb {
    pub min: Point3<f64>,
    pub max: Point3<f64>,
}

impl Aabb {
    pub fn empty() -> Aabb {
        Aabb {
            min: Point3::origin(),
            max: Point3::origin(),
        }
    }

    pub fn extent(&self) -> Vector3<f64> {
        self.max - self.min
    }

    pub fn center(&self) -> Vector3<f64> {
        (self.max.coords + self.min.coords) / 2.
    }
}

/// Compute the axis-aligned bounding box of a mesh. If the mesh is empty, a
/// zero volume bounding box centered at the origin is returned.
pub fn bbox(mesh: &Mesh) -> Aabb {
    if mesh.is_empty() {
        return Aabb::empty();
    }

    let mut min: Point3<f64> = mesh.vertices[0];
    let mut max: Point3<f64> = mesh.vertices[0];

    for v in &mesh.vertices {
        min = min.inf(v);
        max = max.sup(v);
    }

    Aabb { min, max }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use nalgebra::{point, vector};
    use parry3d_f64::shape::Cuboid;

    use super::*;

    #[test]
    fn test_empty_bbox() {
        let mesh = Mesh::empty();
        let aabb = bbox(&mesh);
        assert_relative_eq!(aabb.min, point![0.0, 0.0, 0.0]);
        assert_relative_eq!(aabb.max, point![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_bbox() {
        let shape = Cuboid::new(vector![0.5, 1.0, 2.0]).to_trimesh();
        let mesh = Mesh::new(shape.0, shape.1);
        let aabb = bbox(&mesh);
        assert_relative_eq!(aabb.min, point![-0.5, -1.0, -2.0]);
        assert_relative_eq!(aabb.max, point![0.5, 1.0, 2.0]);
    }
}
