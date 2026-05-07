use crate::{
    ops::CanonicalPlane,
    search::{Action, State},
};

/// Compute the quality for a path starting at initial_state, with the first
/// action as replace_initial_action, and the remaining actions as actions[1..].
///
/// Used for refinement to determine if a first step replacement results in a
/// high quality path.
fn cost_for_path(initial_state: &State, actions: &[Action], replace_initial_action: Action) -> f64 {
    let mut state = initial_state.clone();

    state = state.step(replace_initial_action);
    for action in &actions[1..] {
        state = state.step(*action);
    }
    state.cost()
}

/// Binary search for a refined cutting plane. Iteratively try cutting the input
/// to the left and to the right of the initial plane.
///
/// To evaluate the cut, the quality of the entire path with the first cut
/// replaced by the left or right hand side from above is simulated. This
/// prevents the refinement from being too greedy and reducing future cost.
pub fn refine(
    initial_state: &State,
    initial_path: &[Action],
    search_radius: f64,
) -> CanonicalPlane {
    // Each iteration cuts the search plane in half, so even in the worst case
    // (traversing the entire unit interval) this should converge in ~20 steps.
    const EPS: f64 = 1e-6;

    let initial_action = initial_path[0];
    let initial_plane = initial_path[0].plane;
    let initial_q = cost_for_path(initial_state, initial_path, initial_path[0]);

    let mut left = initial_plane.bias - search_radius;
    let mut right = initial_plane.bias + search_radius;
    let mut best_action = initial_action;

    // Iterate until convergence.
    let mut new_q = initial_q;
    while (right - left) > EPS {
        let mid1 = left + (right - left) / 3.0;
        let mid2 = right - (right - left) / 3.0;

        let act1 = Action::new(initial_plane.with_bias(mid1));
        let act2 = Action::new(initial_plane.with_bias(mid2));

        let q1 = cost_for_path(initial_state, initial_path, act1);
        let q2 = cost_for_path(initial_state, initial_path, act2);

        // Is left or right better?
        if q1 < q2 {
            // Move left
            right = mid2;
            best_action = act1;
            new_q = q1;
        } else {
            // Move right
            left = mid1;
            best_action = act2;
            new_q = q2;
        }
    }

    // TODO: Understand why the refined plane could be worse than the initial.
    // Falling into a local minimum?
    if new_q < initial_q {
        best_action.plane
    } else {
        initial_plane
    }
}
