use std::sync::Arc;

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    Config,
    mesh::Mesh,
    metric::concavity_metric,
    ops::{self, Aabb, CanonicalPlane},
};

/// Mesh and accompanying data for a single part.
#[derive(Clone)]
pub struct Part {
    /// The bounding box of the part.
    pub bounds: Aabb,
    /// The approximate concavity calculated from only the R_v metric.
    pub approx_concavity: f64,
    /// The part mesh data.
    pub mesh: Arc<Mesh>,
    /// The convex hull of the part's mesh data.
    pub convex_hull: Arc<Mesh>,
}

impl Part {
    pub fn from_mesh(mesh: Mesh) -> Part {
        let hull = ops::convex_hull(&mesh);
        Part {
            bounds: ops::bbox(&mesh),
            approx_concavity: concavity_metric(&mesh, &hull, false),
            mesh: Arc::new(mesh),
            convex_hull: Arc::new(hull),
        }
    }

    pub fn slice(&self, plane: CanonicalPlane) -> (Part, Part) {
        let (lhs, rhs) = ops::slice(&self.mesh, &plane);

        let build_part = |mesh| {
            let convex_hull = ops::convex_hull(&mesh);
            Part {
                bounds: ops::bbox(&mesh),
                approx_concavity: concavity_metric(&mesh, &convex_hull, false),
                mesh: Arc::new(mesh),
                convex_hull: Arc::new(convex_hull),
            }
        };

        // PARALLEL: Computing the convex hull is the most expensive operation
        // in the pipeline, and this is an easy place to parallelize the lhs/rhs
        // computation.
        //
        // TODO: is there some way to accelerate the convex hull calculation,
        // instead of recomputing from scratch for each side? We are slicing the
        // mesh by a plane, maybe we can also slice the hull?
        rayon::join(|| build_part(lhs), || build_part(rhs))
    }
}

/// An action which can be taken by the MCTS. References a part index in the
/// state part vector to be sliced at the given normalized slicing plane.
#[derive(Copy, Clone)]
pub struct Action {
    pub part_idx: usize,
    pub unit_plane: CanonicalPlane,
}

/// The state at a particular node in the tree.
#[derive(Clone)]
struct MctsState {
    parts: Vec<Part>,
    parent_rewards: Vec<f64>,
    depth: usize,
}

impl MctsState {
    /// Returns the index of the part with the highest concavity.
    fn worst_part_index(&self) -> usize {
        (0..self.parts.len())
            .max_by(|&a, &b| {
                let ca = self.parts[a].approx_concavity;
                let cb = self.parts[b].approx_concavity;
                ca.total_cmp(&cb)
            })
            .expect("no parts")
    }

    /// Computes all of the various slicing planes which can be used by the tree
    /// search. The worst part in the parts list is always operated on.
    ///
    /// The number of slicing planes is dictated by the `num_nodes` parameter.
    fn all_actions(&self, num_nodes: usize) -> Vec<Action> {
        let part_idx = self.worst_part_index();

        (0..num_nodes)
            .flat_map(|i| {
                (0..3)
                    .map(|axis| {
                        // Splits should not occur right at the edge of the mesh
                        // (e.g. normalized bias=0.0 or bias=1.0) as they would
                        // be one-sided.
                        let ratio = (i + 1) as f64 / (num_nodes + 1) as f64;

                        Action {
                            part_idx,
                            unit_plane: CanonicalPlane { axis, bias: ratio },
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    /// Apply the given slicing plane to the current state, returning a new
    /// state with one part replaced by two.
    fn step(&self, action: Action) -> Self {
        let part_idx = action.part_idx;
        let part = &self.parts[part_idx];

        // Convert the slice ratio to an absolute bias.
        let lb = part.bounds.min[action.unit_plane.axis];
        let ub = part.bounds.max[action.unit_plane.axis];
        let plane = action.unit_plane.denormalize(lb, ub);

        let (lhs, rhs) = part.slice(plane);

        Self {
            parts: {
                let mut parts = self.parts.clone();
                parts.remove(part_idx);
                parts.push(lhs);
                parts.push(rhs);
                parts
            },
            parent_rewards: {
                let mut rewards = self.parent_rewards.clone();
                rewards.push(self.reward());
                rewards
            },
            depth: self.depth + 1,
        }
    }

    /// The reward is the inverse of the concavity of the worst part, i.e. a
    /// smaller concavity gives a higher reward. We aim to maximize the reward.
    fn reward(&self) -> f64 {
        let max_concavity = self.parts[self.worst_part_index()].approx_concavity;
        -max_concavity
    }

    /// The quality of this node is the average of its reward and the rewards of
    /// its parents.
    fn quality(&self) -> f64 {
        let sum = self.parent_rewards.iter().sum::<f64>() + self.reward();
        let d = (self.parent_rewards.len() + 1) as f64;
        sum / d
    }
}

struct MctsNode {
    state: MctsState,

    action: Option<Action>,
    remaining_actions: Vec<Action>,

    parent: Option<usize>,
    children: Vec<usize>,
    n: usize, // times visited
    q: f64,   // average reward
}

impl MctsNode {
    fn depth(&self) -> usize {
        self.state.depth
    }
}

struct Mcts {
    nodes: Vec<MctsNode>,
}

impl Mcts {
    /// Upper confidence estimate of the given node's reward.
    fn ucb(&self, v: usize, c: f64) -> f64 {
        if self.nodes[v].n == 0 {
            return f64::INFINITY;
        }

        let node = &self.nodes[v];
        let parent = &self.nodes[node.parent.unwrap()];
        let parent_n = parent.n as f64;

        self.nodes[v].q + c * (2. * parent_n.ln() / self.nodes[v].n as f64).sqrt()
    }

    /// The next child to explore, based on the tradeoff of exploration and
    /// exploitation.
    fn best_child(&self, v: usize, c: f64) -> usize {
        let node = &self.nodes[v];
        assert!(!node.children.is_empty());

        node.children
            .iter()
            .copied()
            .max_by(|&a, &b| {
                let ucb_a = self.ucb(a, c);
                let ucb_b = self.ucb(b, c);
                ucb_a.total_cmp(&ucb_b)
            })
            .unwrap()
    }

    /// The tree policy attempts to exhaust all of the slicing options for the
    /// current index before moving on to explore new nodes.
    fn tree_policy<R: Rng>(
        &mut self,
        mut v: usize,
        d: usize,
        c: f64,
        num_nodes: usize,
        rng: &mut R,
    ) -> usize {
        // TODO: it should be possible to parallelize the tree policy for a big
        // overall speedup. See the parallelization in the default policy, for
        // example.
        while self.nodes[v].depth() < d {
            if !self.nodes[v].remaining_actions.is_empty() {
                // Choose a random slicing plane which has not been used before.
                let remaining_actions = &mut self.nodes[v].remaining_actions;
                let rand_idx = rng.random_range(0..remaining_actions.len());
                let rand_action = remaining_actions.remove(rand_idx);

                // Insert the result as a new node in the tree as a child of
                // this node.
                let next_state = self.nodes[v].state.step(rand_action);
                self.nodes.push(MctsNode {
                    remaining_actions: next_state.all_actions(num_nodes),
                    q: next_state.reward(),
                    state: next_state,
                    action: Some(rand_action),
                    parent: Some(v),
                    children: vec![],
                    n: 0,
                });

                let n = self.nodes.len();
                self.nodes[v].children.push(n - 1);
                return v;
            } else {
                v = self.best_child(v, c);
            }
        }

        v
    }

    /// The default policy chooses the highest reward among splitting the part
    /// directly at the center along one of the three axes.
    fn default_policy(&self, v: usize, d: usize) -> MctsState {
        let mut state = self.nodes[v].state.clone();

        while state.depth < d {
            let part_idx = state.worst_part_index();

            // PARALLEL: evaluate the axes in parallel.
            let (_, new_state) = (0..3)
                .into_par_iter()
                .map(|axis| {
                    let action = Action {
                        part_idx,
                        unit_plane: CanonicalPlane { axis, bias: 0.5 },
                    };

                    let new_state = state.step(action);
                    let new_reward = new_state.reward();

                    (new_reward, new_state)
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap();

            state = new_state;
        }

        state
    }

    /// Propagate rewards at the leaf nodes back up through the tree.
    fn backup(&mut self, mut v: usize, q: f64) {
        // Move upward until the root node is reached.
        loop {
            self.nodes[v].n += 1;
            self.nodes[v].q = f64::max(self.nodes[v].q, q);

            if let Some(parent) = self.nodes[v].parent {
                v = parent;
            } else {
                break;
            }
        }
    }
}

/// Binary search for a refined cutting plane. Iteratively try cutting the input
/// to the left and to the right of the initial plane. Whichever cut side
/// results in a higher reward is recursively refined.
pub fn refine(
    input_part: &Part,
    initial_unit_plane: CanonicalPlane,
    unit_radius: f64,
) -> Option<CanonicalPlane> {
    const EPS: f64 = 1e-5;

    let state = MctsState {
        parts: vec![input_part.clone()],
        parent_rewards: vec![],
        depth: 0,
    };

    let mut lb = initial_unit_plane.bias - unit_radius;
    let mut ub = initial_unit_plane.bias + unit_radius;
    let mut best_action = None;

    // Each iteration cuts the search plane in half, so even in the worst case
    // (traversing the entire unit interval) this should converge in ~20 steps.
    while (ub - lb) > EPS {
        let pivot = (lb + ub) / 2.0;

        let lhs = Action {
            part_idx: 0,
            unit_plane: initial_unit_plane.with_bias((lb + pivot) / 2.),
        };

        let rhs = Action {
            part_idx: 0,
            unit_plane: initial_unit_plane.with_bias((ub + pivot) / 2.),
        };

        if state.step(lhs).reward() > state.step(rhs).reward() {
            // Move left
            ub = pivot;
            best_action = Some(lhs);
        } else {
            // Move right
            lb = pivot;
            best_action = Some(rhs);
        }
    }

    best_action.map(|a| a.unit_plane)
}

/// An implementation of Monte Carlo Tree Search for the approximate convex
/// decomposition via mesh slicing problem.
///
/// A run of the tree search returns the slice with the highest estimated
/// probability to lead to a large reward when followed by more slices.
pub fn run(input_part: Part, config: &Config) -> Option<CanonicalPlane> {
    // A deterministic random number generator.
    let mut rng = ChaCha8Rng::seed_from_u64(config.random_seed);

    let root_state = MctsState {
        parts: vec![input_part],
        parent_rewards: vec![],
        depth: 0,
    };
    let root_node = MctsNode {
        state: root_state.clone(),
        parent: None,
        action: None,
        remaining_actions: root_state.all_actions(config.num_nodes),
        children: vec![],
        n: 0,
        q: f64::NEG_INFINITY,
    };

    let mut mcts = Mcts {
        nodes: vec![root_node],
    };
    for _ in 0..config.iterations {
        let v = mcts.tree_policy(
            0,
            config.max_depth,
            config.exploration_param,
            config.num_nodes,
            &mut rng,
        );
        let state = mcts.default_policy(v, config.max_depth);
        mcts.backup(v, state.quality());
    }

    // For the final result, we only care about the best node. We never want to
    // return an exploratory node. Set the exploration parameter to zero.
    let best_node = &mcts.nodes[mcts.best_child(0, 0.0)];
    best_node.action.map(|a| a.unit_plane)
}
