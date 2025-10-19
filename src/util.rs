use nalgebra::Point3;

// TODO: May break with big numbers due to checked arithmetic.
// TODO: unsigned ints?
pub fn cantor_pairing(k1: i64, k2: i64) -> i64 {
    i64::wrapping_mul((k1 + k2) / 2, (k1 + k2 + 1) + k2)
}

// TODO: tol acts like a diameter, allowing for 0.5 * tol on either side.
pub fn cantor_point_hash(point: Point3<f64>, tol: f64) -> i64 {
    let x = (point.x / tol).round() as i64;
    let y = (point.y / tol).round() as i64;
    let z = (point.z / tol).round() as i64;

    cantor_pairing(x, cantor_pairing(y, z))
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
            cantor_point_hash(Point3::new(1.0, 2.0, 3.5), tol),
            cantor_point_hash(Point3::new(1.0, 2.0, 3.54), tol)
        );
        assert_eq!(
            cantor_point_hash(Point3::new(-1.0, 2.0, -1.04), tol),
            cantor_point_hash(Point3::new(-1.0, 2.0, -1.0), tol)
        );
        assert_ne!(
            cantor_point_hash(Point3::new(1.0, 2.0, 3.5), tol),
            cantor_point_hash(Point3::new(1.0, 2.0, 3.56), tol)
        );
        assert_ne!(
            cantor_point_hash(Point3::new(0.0, 0.0, 3.5), tol),
            cantor_point_hash(Point3::new(0.0, 0.0, -3.5), tol)
        );
    }
}
