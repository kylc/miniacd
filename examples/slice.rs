use std::{env::args, hint::black_box, time::Instant};

use miniacd::{
    io::{self, load_obj},
    ops::{CanonicalPlane, slice},
};

const N: u128 = 100;
const PLANE: CanonicalPlane = CanonicalPlane { axis: 2, bias: 0.0 };

fn main() {
    let input_path = args().next_back().expect("no input mesh");
    let meshes = load_obj(input_path);
    let mesh = meshes[0].clone(); // TODO: support multiple objects

    let t0 = Instant::now();
    for _ in 0..N {
        black_box(slice(&mesh, &PLANE));
    }
    let tf = Instant::now();

    println!("slice:\t{}us / iter", (tf - t0).as_micros() / N);

    let (top, bot) = slice(&mesh, &PLANE);
    io::write_meshes_to_obj("meshes/output/slice.obj", &[top, bot]).unwrap();
}
