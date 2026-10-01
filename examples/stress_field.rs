//! Dump the shape-derived stress field on the outer periclinal face as CSV.
use rusty_tube::cellgen::{build_cell, CellSpec};
use rusty_tube::setup::load_outline;
fn main() {
    let src = std::env::args().nth(1).unwrap_or("procedural:7".into());
    let o = load_outline(&src).unwrap();
    let cell = build_cell(
        &o,
        &CellSpec {
            shape_stress: true,
            ..Default::default()
        },
    )
    .unwrap();
    let f = &cell.mesh.point_vectors["strain_0"];
    let (_, hi) = cell.mesh.bounds();
    println!("x,y,z,fx,fy,fz,nz");
    for (i, p) in cell.mesh.positions.iter().enumerate() {
        let q = (*p - cell.shift) / 125.0;
        let _ = hi;
        println!(
            "{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.3}",
            q.x, q.y, q.z, f[i].x, f[i].y, f[i].z, cell.normals[i].z
        );
    }
}
