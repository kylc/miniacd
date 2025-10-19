use std::{env::args, hint::black_box, time::Instant};

use miniacd::{
    io::{self, load_obj},
    ops,
};

const N: u128 = 100;

fn main() {
    let input_path = args().next_back().expect("no input mesh");
    let mesh = load_obj(input_path);

    let t0 = Instant::now();
    for _ in 0..N {
        black_box(ops::convex_hull(&mesh));
    }
    let tf = Instant::now();
    println!("convex_hull:\t{}us / iter", (tf - t0).as_micros() / N);

    let hull = ops::convex_hull(&mesh);
    io::write_meshes_to_obj("meshes/output/convex_hull.obj", &[hull]).unwrap();
}
