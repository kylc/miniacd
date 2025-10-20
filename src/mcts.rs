use std::sync::Arc;

use rand::{
    Rng, SeedableRng,
    seq::{IndexedRandom, SliceRandom},
};
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

        // PARALLEL: Computing the convex hull is the most expensive operation
        // in the pipeline, and this is an easy place to parallelize the lhs/rhs
        // computation.
        //
        // TODO: is there some way to accelerate the convex hull calculation,
        // instead of recomputing from scratch for each side? We are slicing the
        // mesh by a plane, maybe we can also slice the hull?
        rayon::join(|| Part::from_mesh(lhs), || Part::from_mesh(rhs))
    }
}

/// An action which can be taken by the MCTS. References a part index in the
/// state part vector to be sliced at the given normalized slicing plane.
#[derive(Copy, Clone)]
pub struct Action {
    pub unit_plane: CanonicalPlane,
}

impl Action {
    fn new(unit_plane: CanonicalPlane) -> Self {
        Self { unit_plane }
    }
}

/// Computes all of the various slicing planes which can be used by the tree
/// search. The worst part in the parts list is always operated on.
///
/// The number of slicing planes is dictated by the `num_nodes` parameter.
///
/// Returns a shuffled vector of actions.
fn all_actions<R: Rng>(num_nodes: usize, rng: &mut R) -> Vec<Action> {
    let mut actions: Vec<_> = (0..num_nodes)
        .flat_map(|node_idx| {
            (0..3).map(move |axis| {
                // Splits should not occur right at the edge of the mesh
                // (e.g. normalized bias=0.0 or bias=1.0) as they would
                // be one-sided.
                let ratio = (node_idx + 1) as f64 / (num_nodes + 1) as f64;

                Action::new(CanonicalPlane { axis, bias: ratio })
            })
        })
        .collect();

    actions.shuffle(rng);
    actions
}

/// The state at a particular node in the tree.
#[derive(Clone)]
struct MctsState {
    /// Parts sorted by concavity in ascending order (worst is last).
    parts: Vec<Part>,
    parent_rewards: Vec<f64>,
    depth: usize,
}

impl MctsState {
    /// Returns the index of the part with the highest concavity.
    fn worst_part_index(&self) -> usize {
        // Parts are kept in order such that the last element is the worst.
        self.parts.len() - 1
    }

    /// Apply the given slicing plane to the current state, returning a new
    /// state with one part replaced by two. The worst part (by concavity) is
    /// always chosen as the action target.
    fn step(&self, action: Action) -> Self {
        let part_idx = self.worst_part_index();
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

                let mut insert_sorted = |part: Part| {
                    let pos = parts
                        .binary_search_by(|probe| {
                            probe.approx_concavity.total_cmp(&part.approx_concavity)
                        })
                        .unwrap_or_else(|e| e);
                    parts.insert(pos, part);
                };

                insert_sorted(lhs);
                insert_sorted(rhs);

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

    fn is_terminal(&self, max_depth: usize) -> bool {
        self.depth >= max_depth
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
    fn new(
        state: MctsState,
        actions: Vec<Action>,
        action: Option<Action>,
        parent: Option<usize>,
    ) -> Self {
        let q = state.reward();
        Self {
            state,
            action,
            parent,
            children: vec![],
            remaining_actions: actions,
            n: 0,
            q,
        }
    }

    fn is_leaf(&self) -> bool {
        !self.remaining_actions.is_empty()
    }

    fn is_terminal(&self, max_depth: usize) -> bool {
        self.state.is_terminal(max_depth)
    }
}

struct Mcts {
    nodes: Vec<MctsNode>,
}

impl Mcts {
    fn new(root: MctsNode) -> Self {
        Mcts { nodes: vec![root] }
    }

    /// Select the leaf node with the highest UCB to explore next.
    fn select(&self, c: f64) -> usize {
        let mut v = 0;
        loop {
            let node = &self.nodes[v];
            if node.is_leaf() {
                return v;
            }

            v = self.best_child(v, c);
        }
    }

    /// Expand the given node by choosing a random action from its list of
    /// unplayed actions. Add the result as a child to this node.
    fn expand<R: Rng>(&mut self, v: usize, num_nodes: usize, rng: &mut R) {
        let random_action_idx = rng.random_range(..self.nodes[v].remaining_actions.len());
        let random_action = self.nodes[v].remaining_actions.remove(random_action_idx);
        let new_state = self.nodes[v].state.step(random_action);

        self.nodes.push(MctsNode::new(
            new_state,
            all_actions(num_nodes, rng),
            Some(random_action),
            Some(v),
        ));

        let child = self.nodes.len() - 1;
        self.nodes[v].children.push(child);
    }

    /// The default policy chooses the highest reward among splitting the part
    /// directly at the center along one of the three axes. The game is played
    /// out until the maximum depth is reached.
    fn simulate(&self, v: usize, max_depth: usize) -> f64 {
        let mut current_state = self.nodes[v].state.clone();
        while !current_state.is_terminal(max_depth) {
            let default_planes = [
                CanonicalPlane { axis: 0, bias: 0.5 },
                CanonicalPlane { axis: 1, bias: 0.5 },
                CanonicalPlane { axis: 2, bias: 0.5 },
            ];

            // PARALLEL: evaluate the axes in parallel.
            let (_, state_to_play) = default_planes
                .into_par_iter()
                .map(|plane| {
                    let action = Action::new(plane);

                    let new_state = current_state.step(action);
                    let new_reward = new_state.reward();

                    (new_reward, new_state)
                })
                .max_by(|a, b| a.0.total_cmp(&b.0))
                .unwrap();

            current_state = state_to_play;
        }

        current_state.quality()
    }

    /// Upper confidence estimate of the given node's reward.
    fn ucb(&self, v: usize, c: f64) -> f64 {
        if self.nodes[v].n == 0 {
            return f64::INFINITY;
        }

        let node = &self.nodes[v];
        let n = node.n as f64;
        let parent = &self.nodes[node.parent.unwrap()];
        let parent_n = parent.n as f64;

        self.nodes[v].q + c * (2. * parent_n.ln() / n).sqrt()
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

    /// Propagate rewards at the leaf nodes back up through the tree.
    fn backprop(&mut self, mut v: usize, q: f64) {
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

        let lhs = Action::new(initial_unit_plane.with_bias((lb + pivot) / 2.));
        let rhs = Action::new(initial_unit_plane.with_bias((ub + pivot) / 2.));

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

    let root_node = MctsNode::new(
        MctsState {
            parts: vec![input_part],
            parent_rewards: vec![],
            depth: 0,
        },
        all_actions(config.num_nodes, &mut rng),
        None,
        None,
    );

    let mut mcts = Mcts::new(root_node);
    for _ in 0..config.iterations {
        let mut v = mcts.select(config.exploration_param);

        if !mcts.nodes[v].is_terminal(config.max_depth) {
            mcts.expand(v, config.num_nodes, &mut rng);
            let children = &mcts.nodes[v].children;
            v = *children.choose(&mut rng).unwrap();
        }

        let reward = mcts.simulate(v, config.max_depth);
        mcts.backprop(v, reward);
    }

    // For the final result, we only care about the best node. We never want to
    // return an exploratory node. Set the exploration parameter to zero.
    let best_node = &mcts.nodes[mcts.best_child(0, 0.0)];
    best_node.action.map(|a| a.unit_plane)
}
