use std::f64;

use rand::{
    Rng, SeedableRng,
    seq::{IndexedRandom, SliceRandom},
};
use rand_chacha::ChaCha8Rng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{
    Config,
    ops::CanonicalPlane,
    search::{Action, Part, State, refine::refine},
};

struct Node {
    state: State,

    action: Option<Action>,
    remaining_actions: Vec<Action>,

    parent: Option<usize>,
    children: Vec<usize>,
    n: usize,      // times visited
    cost_sum: f64, // sum cost
}

impl Node {
    fn new(
        state: State,
        actions: Vec<Action>,
        action: Option<Action>,
        parent: Option<usize>,
    ) -> Self {
        Self {
            state,
            action,
            parent,
            children: vec![],
            remaining_actions: actions,
            n: 0,
            cost_sum: 0.0,
        }
    }

    fn simulate(&self, max_depth: usize) -> f64 {
        let default_planes = [
            CanonicalPlane { axis: 0, bias: 0.5 },
            CanonicalPlane { axis: 1, bias: 0.5 },
            CanonicalPlane { axis: 2, bias: 0.5 },
        ];

        let mut current_state = self.state.clone();
        while current_state.depth < max_depth {
            // PARALLEL: evaluate the axes in parallel.
            let (state_to_play, _) = default_planes
                .into_par_iter()
                .map(|plane| {
                    let action = Action::new(plane);

                    let new_state = current_state.step(action);
                    let new_cost = new_state.cost();

                    (new_state, new_cost)
                })
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .unwrap();

            current_state = state_to_play;
        }

        current_state.cost()
    }

    fn is_leaf(&self) -> bool {
        !self.remaining_actions.is_empty()
    }

    fn is_terminal(&self, max_depth: usize) -> bool {
        self.state.depth >= max_depth
    }
}

struct Mcts {
    nodes: Vec<Node>,
}

impl Mcts {
    fn new(root: Node) -> Self {
        Mcts { nodes: vec![root] }
    }

    /// Expand the given node by choosing a random action from its list of
    /// unplayed actions. Add the result as a child to this node.
    fn expand<R: Rng>(&mut self, v: usize, num_nodes: usize, rng: &mut R) {
        // remaining_actions is shuffled so this returns a random action.
        let random_action = self.nodes[v]
            .remaining_actions
            .pop()
            .expect("no more actions");

        let new_state = self.nodes[v].state.step(random_action);
        let mut actions: Vec<_> = Action::valid_actions(num_nodes).collect();
        actions.shuffle(rng);

        self.nodes
            .push(Node::new(new_state, actions, Some(random_action), Some(v)));

        let child = self.nodes.len() - 1;
        self.nodes[v].children.push(child);
    }

    /// Upper confidence estimate of the given node's cost.
    fn ucb(&self, v: usize, c: f64) -> f64 {
        let node = &self.nodes[v];
        let node_n = node.n as f64;

        if node_n == 0.0 {
            return f64::INFINITY;
        }

        let parent = &self.nodes[node.parent.unwrap()];
        let parent_n = parent.n as f64;

        let avg_cost = node.cost_sum / node_n;
        -avg_cost + c * f64::sqrt(parent_n.ln() / node_n)
    }

    /// The next child to explore, based on the tradeoff of exploration and
    /// exploitation. Returns None if there are no children of v.
    fn best_child(&self, v: usize, c: f64) -> Option<usize> {
        let node = &self.nodes[v];

        node.children.iter().copied().max_by(|&a, &b| {
            let ucb_a = self.ucb(a, c);
            let ucb_b = self.ucb(b, c);
            ucb_a.total_cmp(&ucb_b)
        })
    }

    /// Select the leaf node with the highest UCB to explore next.
    fn select(&self, c: f64) -> usize {
        let mut v = 0;
        loop {
            let node = &self.nodes[v];
            if node.is_leaf() {
                return v;
            }

            v = self
                .best_child(v, c)
                .expect("selected leaf node must have parent");
        }
    }

    /// Propagate costs at the leaf nodes back up through the tree.
    fn backup(&mut self, mut v: usize, cost: f64) {
        // Move upward until the root node is reached.
        loop {
            self.nodes[v].n += 1;
            self.nodes[v].cost_sum += cost;

            if let Some(parent) = self.nodes[v].parent {
                v = parent;
            } else {
                break;
            }
        }
    }

    /// Returns the action path from the root to the lowest cost terminal node.
    fn best_path_from_root(&self) -> Vec<Action> {
        let mut best_path = vec![];
        let mut v = 0;
        while let Some(child) = self.best_child(v, 0.0) {
            if let Some(action) = self.nodes[child].action {
                best_path.push(action);
            }

            v = child;
        }

        best_path
    }
}

/// An implementation of Monte Carlo Tree Search for the approximate convex
/// decomposition via mesh slicing problem.
///
/// A run of the tree search returns the slice with the highest estimated
/// probability to lead to a low cost when followed by more slices.
pub fn run(input_part: &Part, config: &Config) -> Option<CanonicalPlane> {
    // A deterministic random number generator.
    let mut rng = ChaCha8Rng::seed_from_u64(config.mcts_random_seed);

    // The root MCTS node contains just the input part, unmodified.
    let mut root_actions: Vec<_> = Action::valid_actions(config.mcts_grid_nodes).collect();
    root_actions.shuffle(&mut rng);

    let root_node = Node::new(
        State {
            parts: vec![input_part.clone()],
            depth: 0,
        },
        root_actions,
        None,
        None,
    );

    // Run the MCTS algorithm for the specified compute time to compute a
    // probabilistic best path.
    let mut mcts = Mcts::new(root_node);
    for _ in 0..config.mcts_iterations {
        let mut v = mcts.select(config.mcts_exploration);

        while !mcts.nodes[v].is_terminal(config.mcts_depth) {
            if !mcts.nodes[v].is_leaf() {
                v = mcts.best_child(v, config.mcts_exploration).unwrap();
            } else {
                mcts.expand(v, config.mcts_grid_nodes, &mut rng);
                let children = &mcts.nodes[v].children;
                v = *children.choose(&mut rng).unwrap();
            }
        }

        let cost = mcts.nodes[v].simulate(config.mcts_depth);
        mcts.backup(v, cost);
    }

    // Take the discrete best path from MCTS and refine it.
    let best_path = mcts.best_path_from_root();
    let refined_plane = refine(
        &mcts.nodes[0].state,
        &best_path,
        0.5 / (config.mcts_grid_nodes + 1) as f64,
    );

    Some(refined_plane)
}
