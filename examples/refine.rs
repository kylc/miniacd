use std::{env::args, hint::black_box, time::Instant};

use miniacd::{
    io::{self, load_obj},
    mcts::{self, Part},
    ops::{self, CanonicalPlane},
};

const N: u128 = 100;
const PLANE: CanonicalPlane = CanonicalPlane { axis: 2, bias: 0.5 };

fn main() {
    let input_path = args().next_back().expect("no input mesh");
    let mesh = load_obj(input_path);
    let part = Part::from_mesh(mesh.clone());
    let aabb = ops::bbox(&mesh);

    let t0 = Instant::now();
    for _ in 0..N {
        black_box(mcts::refine(&part, PLANE, 0.5));
    }

    let tf = Instant::now();
    println!("refine:\t{}ms / iter", (tf - t0).as_millis() / N);

    let refined_plane = mcts::refine(&part, PLANE, 0.5)
        .unwrap()
        .denormalize(aabb.min[PLANE.axis], aabb.max[PLANE.axis]);

    let (lhs, rhs) = ops::slice(&mesh, &refined_plane);
    io::write_meshes_to_obj("meshes/output/refine.obj", &[lhs, rhs]).unwrap();
}
