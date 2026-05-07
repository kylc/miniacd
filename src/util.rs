use glamx::DVec3;

// TODO: May break with big numbers due to checked arithmetic.
// TODO: unsigned ints?
pub fn cantor_pairing(a: i64, b: i64) -> i64 {
    ((a + b) * (a + b + 1) / 2) + b
}

// TODO: tol acts like a diameter, allowing for 0.5 * tol on either side.
pub fn cantor_point_hash(v: DVec3, tol: f64) -> i64 {
    let qx = (v.x / tol).round() as i64;
    let qy = (v.y / tol).round() as i64;
    let qz = (v.z / tol).round() as i64;

    cantor_pairing(cantor_pairing(qx, qy), qz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cantor_pairing() {
        let k1 = cantor_pairing(1, 2);
        let k2 = cantor_pairing(2, 1);
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cantor_point_hash() {
        let tol = 0.1;

        assert_eq!(
            cantor_point_hash(DVec3::new(1.0, 2.0, 3.5), tol),
            cantor_point_hash(DVec3::new(1.0, 2.0, 3.54), tol)
        );
        assert_eq!(
            cantor_point_hash(DVec3::new(-1.0, 2.0, -1.04), tol),
            cantor_point_hash(DVec3::new(-1.0, 2.0, -1.0), tol)
        );
        assert_ne!(
            cantor_point_hash(DVec3::new(1.0, 2.0, 3.0), tol),
            cantor_point_hash(DVec3::new(-1.0, -2.0, -3.0), tol)
        );
        assert_ne!(
            cantor_point_hash(DVec3::new(1.0, 2.0, 3.5), tol),
            cantor_point_hash(DVec3::new(1.0, 2.0, 3.56), tol)
        );
        assert_ne!(
            cantor_point_hash(DVec3::new(0.0, 0.0, 3.5), tol),
            cantor_point_hash(DVec3::new(0.0, 0.0, -3.5), tol)
        );
    }
}
