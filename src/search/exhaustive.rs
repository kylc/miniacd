use std::f64;

use crate::{
    Config,
    ops::CanonicalPlane,
    search::{Action, Part, State, refine::refine},
};

struct Node {
    state: State,

    action: Option<Action>,

    parent: Option<usize>,
    children: Vec<usize>,
    min_cost: f64,
}

impl Node {
    fn new(state: State, action: Option<Action>, parent: Option<usize>) -> Self {
        let cost = state.cost();
        Self {
            state,
            action,
            parent,
            children: vec![],
            min_cost: cost,
        }
    }

    fn is_terminal(&self, max_depth: usize) -> bool {
        self.state.depth >= max_depth
    }
}

struct Exhaustive {
    nodes: Vec<Node>,
}

impl Exhaustive {
    fn new(root: Node) -> Self {
        Exhaustive { nodes: vec![root] }
    }

    /// Exhaustively expand the given node by executing all possible actions
    /// recursively downward until a terminal depth is achieved.
    fn expand(&mut self, v: usize, num_nodes: usize, max_depth: usize) {
        if self.nodes[v].is_terminal(max_depth) {
            return;
        }

        for next_action in Action::valid_actions(num_nodes) {
            let new_state = self.nodes[v].state.step(next_action);

            let new_node = Node::new(new_state, Some(next_action), Some(v));
            let new_cost = new_node.min_cost;
            self.nodes.push(new_node);

            let child = self.nodes.len() - 1;
            self.nodes[v].children.push(child);
            self.backup(child, new_cost);

            // Recursively expand the child node before continuing.
            self.expand(child, num_nodes, max_depth);
        }
    }

    fn best_child(&self, v: usize) -> Option<usize> {
        let node = &self.nodes[v];

        node.children.iter().copied().min_by(|&a, &b| {
            let cost_a = self.nodes[a].min_cost;
            let cost_b = self.nodes[b].min_cost;
            cost_a.total_cmp(&cost_b)
        })
    }

    /// Propagate costs at the leaf nodes back up through the tree.
    fn backup(&mut self, mut v: usize, cost: f64) {
        // Move upward until the root node is reached.
        loop {
            self.nodes[v].min_cost = f64::min(self.nodes[v].min_cost, cost);

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
        while let Some(child) = self.best_child(v) {
            if let Some(action) = self.nodes[child].action {
                best_path.push(action);
            }

            v = child;
        }

        best_path
    }
}

pub fn run(input_part: &Part, config: &Config) -> Option<CanonicalPlane> {
    let root_node = Node::new(
        State {
            parts: vec![input_part.clone()],
            depth: 0,
        },
        None,
        None,
    );

    // Run the exhaustive search over all possible actions from the root node to
    // find the optimal slicing path.
    let mut search = Exhaustive::new(root_node);
    search.expand(0, config.mcts_grid_nodes, config.mcts_depth);

    // Take the best result and refine it.
    let best_path = search.best_path_from_root();
    let refined_plane = refine(
        &search.nodes[0].state,
        &best_path,
        0.5 / (config.mcts_grid_nodes + 1) as f64,
    );

    Some(refined_plane)
}
