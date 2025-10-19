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
    #[pyo3(signature=(mesh,
                      threshold=0.1,
                      iterations=150,
                      max_depth=3,
                      random_seed=42,
                      print=true))]
    fn run(
        mesh: &PyMesh,
        threshold: f64,
        iterations: usize,
        max_depth: usize,
        random_seed: u64,
        print: bool,
    ) -> Vec<PyMesh> {
        let config = Config {
            threshold,
            iterations,
            max_depth,
            exploration_param: f64::sqrt(2.0),
            num_nodes: 20,
            random_seed,
            print,
        };

        let components = crate::run(mesh.0.clone(), &config);
        components
            .into_iter()
            .map(|c| PyMesh(Arc::unwrap_or_clone(c.mesh)))
            .collect()
    }
}
