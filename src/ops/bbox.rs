use glamx::DVec3;
use parry3d_f64::bounding_volume::BoundingVolume;

use crate::mesh::Mesh;

#[derive(Clone)]
pub struct Aabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl Aabb {
    pub fn empty() -> Aabb {
        Aabb {
            min: DVec3::ZERO,
            max: DVec3::ZERO,
        }
    }

    pub fn extent(&self) -> DVec3 {
        self.max - self.min
    }

    pub fn center(&self) -> DVec3 {
        (self.max + self.min) / 2.
    }

    pub fn intersects(&self, other: &Aabb, eps: f64) -> bool {
        let a = parry3d_f64::bounding_volume::Aabb::new(self.min, self.max).loosened(eps / 2.0);
        let b = parry3d_f64::bounding_volume::Aabb::new(other.min, other.max).loosened(eps / 2.0);

        a.intersects(&b)
    }
}

/// Compute the axis-aligned bounding box of a mesh. If the mesh is empty, a
/// zero volume bounding box centered at the origin is returned.
pub fn bbox(mesh: &Mesh) -> Aabb {
    if mesh.is_empty() {
        return Aabb::empty();
    }

    let mut min = mesh.vertices[0];
    let mut max = mesh.vertices[0];

    for v in &mesh.vertices {
        min = min.min(*v);
        max = max.max(*v);
    }

    Aabb { min, max }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use parry3d_f64::shape::Cuboid;

    use super::*;

    #[test]
    fn test_empty_bbox() {
        let mesh = Mesh::empty();
        let aabb = bbox(&mesh);
        assert_relative_eq!(aabb.min, DVec3::new(0.0, 0.0, 0.0));
        assert_relative_eq!(aabb.max, DVec3::new(0.0, 0.0, 0.0));
    }

    #[test]
    fn test_bbox() {
        let shape = Cuboid::new(DVec3::new(0.5, 1.0, 2.0)).to_trimesh();
        let mesh = Mesh::new(shape.0, shape.1);
        let aabb = bbox(&mesh);
        assert_relative_eq!(aabb.min, DVec3::new(-0.5, -1.0, -2.0));
        assert_relative_eq!(aabb.max, DVec3::new(0.5, 1.0, 2.0));
    }
}
