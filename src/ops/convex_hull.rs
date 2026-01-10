use glamx::DVec3;

use crate::mesh::Mesh;

pub fn convex_hull(mesh: &Mesh) -> Mesh {
    convex_hull_from_points(&mesh.vertices)
}

pub fn convex_hull_from_points(points: &[DVec3]) -> Mesh {
    if points.len() < 3 {
        Mesh::empty()
    } else {
        let (vertices, faces) = parry3d_f64::transformation::convex_hull(points);
        Mesh { vertices, faces }
    }
}

#[cfg(test)]
mod tests {
    use parry3d_f64::shape::Cuboid;

    use super::*;

    #[test]
    fn test_convex_hull() {
        let shape = Cuboid::new(DVec3::new(0.5, 0.5, 0.5)).to_trimesh();
        let mesh = Mesh::new(shape.0, shape.1);
        let _hull = convex_hull(&mesh);

        // TODO: assert
    }
}
