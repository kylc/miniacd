use parry3d_f64::mass_properties::details::trimesh_signed_volume_and_center_of_mass;

use crate::mesh::Mesh;

/// Computes the signed volume of the mesh.
pub fn volume(mesh: &Mesh) -> f64 {
    if mesh.is_empty() {
        return 0.0;
    }

    let (volume, _com) = trimesh_signed_volume_and_center_of_mass(&mesh.vertices, &mesh.faces);
    volume
}

#[cfg(test)]
mod tests {
    use std::f64;

    use approx::assert_relative_eq;
    use glamx::DVec3;
    use parry3d_f64::shape::{Ball, Cuboid};

    use crate::mesh::Mesh;

    use super::*;

    #[test]
    fn test_primitive_volumes() {
        let cube = Cuboid::new(DVec3::new(1.0, 2.0, 3.0)).to_trimesh();
        let cube = Mesh::new(cube.0, cube.1);
        assert_relative_eq!(volume(&cube), 2.0 * 4.0 * 6.0, max_relative = 0.01);

        let ball = Ball::new(0.75).to_trimesh(32, 32);
        let ball = Mesh::new(ball.0, ball.1);
        assert_relative_eq!(
            volume(&ball),
            4. / 3. * f64::consts::PI * f64::powi(0.75, 3),
            max_relative = 0.01
        );
    }
}
