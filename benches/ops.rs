use std::hint::black_box;

use divan::Bencher;
use miniacd::{
    io::load_obj,
    metric,
    ops::{self, CanonicalPlane},
};

#[divan::bench]
fn concavity_metric_approx(bencher: Bencher) {
    let mesh = load_obj("benches/bunny.obj");
    let hull = ops::convex_hull(&mesh);
    bencher.bench(|| {
        black_box(metric::concavity_metric(&mesh, &hull, false));
    });
}

#[divan::bench]
fn concavity_metric_exact(bencher: Bencher) {
    let mesh = load_obj("benches/bunny.obj");
    let hull = ops::convex_hull(&mesh);
    bencher.bench(|| {
        black_box(metric::concavity_metric(&mesh, &hull, true));
    });
}

#[divan::bench]
fn slice(bencher: Bencher) {
    const PLANE: CanonicalPlane = CanonicalPlane { axis: 2, bias: 0.0 };

    let mesh = load_obj("benches/bunny.obj");
    bencher.bench(|| {
        black_box(ops::slice(&mesh, &PLANE));
    });
}

#[divan::bench]
fn convex_hull(bencher: Bencher) {
    let mesh = load_obj("benches/bunny.obj");
    bencher.bench(|| {
        black_box(ops::convex_hull(&mesh));
    });
}

#[divan::bench]
fn volume(bencher: Bencher) {
    let mesh = load_obj("benches/bunny.obj");
    bencher.bench(|| {
        black_box(ops::volume(&mesh));
    });
}

#[divan::bench]
fn bbox(bencher: Bencher) {
    let mesh = load_obj("benches/bunny.obj");
    bencher.bench(|| {
        black_box(ops::bbox(&mesh));
    });
}

#[divan::bench(sample_count = 3)]
fn full_run(bencher: Bencher) {
    let mesh = load_obj("benches/bunny.obj");
    let config = miniacd::Config {
        print: false,
        ..Default::default()
    };
    bencher.with_inputs(|| mesh.clone()).bench_values(|mesh| {
        black_box(miniacd::run(mesh, &config));
    });
}

fn main() {
    divan::main();
}
