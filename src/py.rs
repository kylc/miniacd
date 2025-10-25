#[pyo3::pymodule(name = "miniacd")]
mod pyminiacd {
    use std::sync::Arc;

    use nalgebra::Point3;
    use pyo3::prelude::*;

    use crate::{Config, mesh::Mesh, ops};

    #[pyo3::pyclass(name = "Mesh")]
    pub struct PyMesh(Mesh);

    #[pymethods]
    impl PyMesh {
        #[new]
        pub fn new(vertices: Vec<[f64; 3]>, faces: Vec<[u32; 3]>) -> PyMesh {
            PyMesh(Mesh {
                vertices: vertices
                    .into_iter()
                    .map(|v| Point3::from_slice(&v))
                    .collect(),
                faces,
            })
        }

        pub fn vertices(&self) -> Vec<[f64; 3]> {
            self.0.vertices.iter().map(|v| [v.x, v.y, v.z]).collect()
        }

        pub fn faces(&self) -> Vec<[u32; 3]> {
            self.0.faces.clone()
        }

        pub fn convex_hull(&self) -> PyMesh {
            PyMesh(ops::convex_hull(&self.0))
        }
    }

    #[pyfunction]
    #[pyo3(signature=(
        mesh: "PyMesh",
        threshold: "float" = 0.1,
        mcts_iterations: "int" = 150,
        mcts_depth: "int" = 3,
        mcts_grid_nodes: "int" = 20,
        mcts_random_seed: "int" = 42,
        print: "bool" = true
    ) -> "list[PyMesh]")]
    fn run(
        mesh: &PyMesh,
        threshold: f64,
        mcts_iterations: usize,
        mcts_depth: usize,
        mcts_grid_nodes: usize,
        mcts_random_seed: u64,
        print: bool,
    ) -> Vec<PyMesh> {
        let config = Config {
            threshold,
            mcts_iterations,
            mcts_depth,
            mcts_exploration: f64::sqrt(2.0),
            mcts_grid_nodes,
            mcts_random_seed,
            print,
        };

        let components = crate::run(mesh.0.clone(), &config);
        components
            .into_iter()
            .map(|c| PyMesh(Arc::unwrap_or_clone(c.mesh)))
            .collect()
    }
}
