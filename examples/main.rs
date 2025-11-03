use std::{env, path::PathBuf, time::Instant};

use miniacd::{
    Config,
    io::{self, load_obj},
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let input_path = args.get(1).expect("no input path");
    let output_path = PathBuf::from(&args.get(2).expect("no output path"));
    let threshold: f64 = args
        .get(3)
        .unwrap_or(&"0.1".to_string())
        .parse()
        .expect("invalid threshold");

    let config = Config {
        threshold,
        print: true,
        ..Default::default()
    };
    let meshes = load_obj(input_path);
    let mesh = meshes[0].clone(); // TODO: support multiple objects

    let t0 = Instant::now();
    let components = miniacd::run(mesh, &config);
    let tf = Instant::now();
    println!("main:\t{:.2}s", (tf - t0).as_secs_f64());

    io::write_meshes_to_obj(&output_path, &components).unwrap();
}
