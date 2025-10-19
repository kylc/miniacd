use std::{env, fs, path::PathBuf};

use clap::{Command, arg, builder::styling, crate_description, crate_version, value_parser};
use miniacd::{
    Config,
    io::{self, load_obj},
    mcts::Part,
    mesh::Mesh,
    ops::{self},
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

const CLI_STYLE: styling::Styles = styling::Styles::styled()
    .header(styling::AnsiColor::Green.on_default().bold())
    .usage(styling::AnsiColor::Green.on_default().bold())
    .literal(styling::AnsiColor::Cyan.on_default().bold())
    .placeholder(styling::AnsiColor::Cyan.on_default());

fn main() {
    let args = Command::new("miniacd")
        .about(crate_description!())
        .arg_required_else_help(true)
        .styles(CLI_STYLE)
        .version(crate_version!())
        .arg(
            arg!(<INPUT_PATH> "Path to the input mesh file")
                .value_parser(value_parser!(PathBuf)).required(true)
        )
        .arg(
            arg!(--"output-dir" <OUTPUT_DIR> "Directory to place convex part files [default: cwd]")
                .value_parser(value_parser!(PathBuf))
        )
        .arg(
            arg!(--threshold <THRESHOLD> "Concavity threshold at which mesh parts will be accepted")
                .value_parser(value_parser!(f64))
                .default_value("0.1"),
        )
        .arg(
            arg!(--"mcts-depth" <MCTS_DEPTH> "Tree search depth")
                .value_parser(value_parser!(usize))
                .default_value("3"),
        )
        .arg(
            arg!(--"mcts-iterations" <MCTS_ITERATIONS> "Tree search iterations")
                .value_parser(value_parser!(usize))
                .default_value("150"),
        )
        .arg(
            arg!(--seed <SEED> "Random generator seed for deterministic output")
                .value_parser(value_parser!(u64))
                .default_value("42"),
        )
        .get_matches();

    let input_path = args.get_one::<PathBuf>("INPUT_PATH").unwrap();
    let output_dir = args
        .get_one("output-dir")
        .cloned()
        .unwrap_or_else(|| env::current_dir().expect("could not get cwd"));

    if !output_dir.exists() {
        fs::create_dir(&output_dir).expect("could not create output dir");
    }

    let config = Config {
        threshold: *args.get_one("threshold").unwrap(),
        iterations: *args.get_one("mcts-iterations").unwrap(),
        max_depth: *args.get_one("mcts-depth").unwrap(),
        exploration_param: f64::sqrt(2.0),
        num_nodes: 20,
        random_seed: *args.get_one("seed").unwrap(),
        print: true,
    };

    let mesh: Mesh = load_obj(input_path);
    let components: Vec<Part> = miniacd::run(mesh, &config);

    // PARALLEL: compute the output convex hulls in parallel.
    let convex_meshes: Vec<Mesh> = components
        .par_iter()
        .map(|c| ops::convex_hull(&c.mesh))
        .collect();

    let mut output_path = output_dir.clone();
    output_path.push("output.obj");
    io::write_meshes_to_obj(&output_path, &convex_meshes).unwrap();
}
