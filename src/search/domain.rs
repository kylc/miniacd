use crate::{
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
    pub mesh: Mesh,
}

impl Part {
    pub fn from_mesh(mesh: Mesh) -> Part {
        let hull = ops::convex_hull(&mesh);

        Part {
            bounds: ops::bbox(&mesh),
            approx_concavity: concavity_metric(&mesh, &hull, false),
            mesh,
        }
    }

    /// Slice this part into two at the given relative canonical plane.
    pub fn slice(&self, plane: CanonicalPlane) -> (Part, Part) {
        let lb = self.bounds.min[plane.axis];
        let ub = self.bounds.max[plane.axis];
        let abs_plane = plane.denormalize(lb, ub);

        let (lhs, rhs) = ops::slice(&self.mesh, &abs_plane);

        // PARALLEL: Computing the convex hull is the most expensive operation
        // in the pipeline, and this is an easy place to parallelize the lhs/rhs
        // computation.
        //
        // TODO: is there some way to accelerate the convex hull calculation,
        // instead of recomputing from scratch for each side? We are slicing the
        // mesh by a plane, maybe we can also slice the hull?
        // rayon::join(|| Part::from_mesh(lhs), || Part::from_mesh(rhs))
        (Part::from_mesh(lhs), Part::from_mesh(rhs))
    }
}

/// An action which can be taken by the MCTS. References a part index in the
/// state part vector to be sliced at the given normalized slicing plane.
#[derive(Copy, Clone)]
pub struct Action {
    pub plane: CanonicalPlane,
}

impl Action {
    pub fn new(plane: CanonicalPlane) -> Self {
        Self { plane }
    }

    pub fn valid_actions_for_axis(axis: usize, num_nodes: usize) -> impl Iterator<Item = Self> {
        (0..num_nodes).map(move |i| {
            // Splits should not occur right at the edge of the mesh
            // (e.g. normalized bias=0.0 or bias=1.0) as they would
            // be one-sided.
            let ratio = (i + 1) as f64 / (num_nodes + 1) as f64;
            Action::new(CanonicalPlane { axis, bias: ratio })
        })
    }

    /// Computes all of the various slicing planes which can be used by the tree
    /// search. The worst part in the parts list is always operated on.
    ///
    /// The number of slicing planes is dictated by the `num_nodes` parameter.
    ///
    /// Returns a shuffled vector of actions.
    pub fn valid_actions(num_nodes: usize) -> impl Iterator<Item = Self> {
        (0..3).flat_map(move |axis| Action::valid_actions_for_axis(axis, num_nodes))
    }
}

/// The state at a particular node in the tree.
#[derive(Clone)]
pub struct State {
    /// Parts sorted by concavity in ascending order (worst is last).
    pub parts: Vec<Part>,
    pub depth: usize,
}

impl State {
    /// Returns the index of the part with the highest concavity.
    fn worst_part_index(&self) -> usize {
        // Parts are kept in order such that the last element is the worst.
        self.parts.len() - 1
    }

    /// Apply the given slicing plane to the current state, returning a new
    /// state with one part replaced by two. The worst part (by concavity) is
    /// always chosen as the action target.
    pub fn step(&self, action: Action) -> Self {
        let part_idx = self.worst_part_index();
        let part = &self.parts[part_idx];

        // Convert the slice ratio to an absolute bias.
        let (lhs, rhs) = part.slice(action.plane);

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
            depth: self.depth + 1,
        }
    }

    /// The cost is the concavity of the worst part, i.e. a smaller concavity
    /// gives a smaller cost. We aim to minimize the cost.
    pub fn cost(&self) -> f64 {
        assert_eq!(
            self.parts
                .iter()
                .map(|p| p.approx_concavity)
                .max_by(|x, y| x.total_cmp(y))
                .unwrap(),
            self.parts[self.worst_part_index()].approx_concavity
        );

        self.parts[self.worst_part_index()].approx_concavity
    }
}
