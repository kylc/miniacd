pub mod io;
pub mod mcts;
pub mod mesh;
pub mod metric;
pub mod ops;
pub mod py;
pub mod util;

use std::{f64, sync::Arc, time::Duration};

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressFinish, ProgressStyle};
use mcts::Part;
use mesh::Mesh;
use metric::concavity_metric;
use mimalloc::MiMalloc;

// Replacing the default allocator with MiMalloc results in about a 5-10%
// decrease in overall runtime.
#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// Tolerance for the change in threshold between successive cuts at which to
/// give up, as no forward progress is being made.
const TOLERANCE_DIFF: f64 = 1e-10;

pub struct Config {
    /// The minimum acceptable concavity metric for an individual part to be
    /// accepted.
    pub threshold: f64,
    /// The number of iterations taken for each MCTS step which chooses a single
    /// slicing plane.
    pub mcts_iterations: usize,
    /// The depth of the MCTS, i.e. the number of lookahead moves when deciding
    /// on the next move.
    pub mcts_depth: usize,
    /// The exploration parameter for computing the upper confidence bound
    /// (UCB).
    pub mcts_exploration: f64,
    /// The number of discrete slices taken per axis at each node in the MCTS.
    pub mcts_grid_nodes: usize,
    /// A seed for the deterministic RNG.
    pub mcts_random_seed: u64,
    /// Print the progress bar? Enable for human users, disable for tests etc.
    pub print: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            threshold: 0.1,
            mcts_iterations: 150,
            mcts_depth: 3,
            mcts_exploration: f64::sqrt(2.0),
            mcts_grid_nodes: 20,
            mcts_random_seed: 0,
            print: false,
        }
    }
}

fn run_inner(input: Part, config: &Config, progress: &ProgressBar, prev_cost: f64) -> Vec<Part> {
    // Base case: the input mesh already meets the threshold, so no further
    // slicing is necessary.
    //
    // NOTE: We use the expensive concavity check here (includes the Hausdorff
    // distance calculation), whereas the inner loops will use the cheaper
    // approximation.
    let cost = concavity_metric(&input.mesh, &input.convex_hull, true);
    if cost < config.threshold || (cost - prev_cost).abs() < TOLERANCE_DIFF {
        progress.inc(1);
        return vec![input];
    }

    // Further slicing is necessary. Run the tree search to find the slice plane
    // which maximizes the future reward and also refine a more precise plane.
    let optimal_plane = mcts::run(&input, config).expect("no action");
    let (lhs, rhs) = input.slice(optimal_plane);

    // The input mesh is no longer required. Drop it to save on memory usage.
    drop(input);

    // PARALLEL: Launch parallel threads to continue slicing the left and right
    // results of this iteration. If the components are within the threshold
    // then the recursion will immediately dead end. Otherwise, either side
    // could generate any number of parts.
    progress.inc_length(1);
    let (mut components_l, mut components_r) = rayon::join(
        || run_inner(lhs, config, progress, cost),
        || run_inner(rhs, config, progress, cost),
    );

    let mut components = vec![];
    components.append(&mut components_l);
    components.append(&mut components_r);
    components
}

pub fn run(input: Mesh, config: &Config) -> Vec<Mesh> {
    let progress_bar = ProgressBar::new(1)
        .with_message("Slicing parts...")
        .with_style(
            ProgressStyle::with_template("[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}")
                .unwrap(),
        )
        .with_finish(ProgressFinish::AndLeave);
    progress_bar.enable_steady_tick(Duration::from_secs(1));
    progress_bar.set_position(0);
    if !config.print {
        progress_bar.set_draw_target(ProgressDrawTarget::hidden());
    }

    // Apply normalization so that the input is always centered and unit size.
    // Some algorithm parameters depend on the mesh scaling, so it's important
    // to normalize.
    let normalization_tfm = input.normalization_transform();
    let normalization_tfm_inv = normalization_tfm.try_inverse().unwrap();
    let normalized_input = input.transform(&normalization_tfm);

    // Run the slicing algorithm.
    let initial_part = Part::from_mesh(normalized_input);
    let output_parts = run_inner(initial_part, config, &progress_bar, f64::INFINITY);

    // Unapply the transform so the outputs part positions match the input.
    output_parts
        .into_iter()
        .map(|p| Arc::unwrap_or_clone(p.convex_hull).transform(&normalization_tfm_inv))
        .filter(|m| !m.is_empty())
        .collect()
}
