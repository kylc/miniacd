use std::{
    fmt::Debug,
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use nalgebra::Point3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

use crate::mesh::Mesh;

pub fn load_obj<P: AsRef<Path> + Debug>(path: P) -> Mesh {
    let (models, _) = tobj::load_obj(
        path,
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )
    .expect("invalid obj mesh");

    // TODO: multiple objects
    let mesh = &models[0].mesh;

    let vertices: Vec<Point3<f64>> = mesh
        .positions
        .chunks_exact(3)
        .map(|v| Point3::new(v[0] as f64, v[1] as f64, v[2] as f64))
        .collect();

    let faces: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|i| [i[0], i[1], i[2]])
        .collect();

    Mesh { vertices, faces }
}

fn random_rgb<R: Rng>(rng: &mut R) -> (f32, f32, f32) {
    (rng.random(), rng.random(), rng.random())
}

pub fn write_meshes_to_obj<P: AsRef<Path>>(path: P, meshes: &[Mesh]) -> Result<(), std::io::Error> {
    let mut rng = ChaCha8Rng::seed_from_u64(0);

    let mut obj_file = File::create(path)?;
    let mut writer = BufWriter::new(&mut obj_file);

    write!(writer, "# miniacd convex mesh\n\n")?;

    let mut vertex_offset = 1u32; // OBJ starts indexing faces at 1
    for (i, mesh) in meshes.iter().enumerate() {
        let (r, g, b) = random_rgb(&mut rng);
        writeln!(writer, "o geometry_{}", i)?;

        for v in &mesh.vertices {
            writeln!(
                writer,
                "v {:.6} {:.6} {:.6} {:.6} {:.6} {:.6}",
                v.x, v.y, v.z, r, g, b
            )?;
        }

        for f in &mesh.faces {
            let i1 = f[0] + vertex_offset;
            let i2 = f[1] + vertex_offset;
            let i3 = f[2] + vertex_offset;
            writeln!(writer, "f {} {} {}", i1, i2, i3)?;
        }

        vertex_offset += mesh.vertices.len() as u32;
        writeln!(writer)?;
    }

    Ok(())
}
