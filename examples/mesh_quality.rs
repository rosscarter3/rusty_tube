use rusty_tube::cellgen::{build_cell, CellSpec};
use rusty_tube::setup::load_outline;
fn main() {
    let o = load_outline("procedural:7").unwrap();
    let cell = build_cell(&o, &CellSpec::default()).unwrap();
    let mn = cell.mesh.compute_normals();
    let mut a: Vec<(f64, usize)> = cell
        .normals
        .iter()
        .zip(&mn)
        .enumerate()
        .map(|(i, (n, m))| (n.dot(*m).clamp(-1.0, 1.0).acos().to_degrees(), i))
        .collect();
    a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
    let n = a.len();
    println!(
        "angle sdf-normal vs mesh-normal: median {:.1} p90 {:.1} p99 {:.1} max {:.1}",
        a[n / 2].0,
        a[n * 9 / 10].0,
        a[n * 99 / 100].0,
        a[n - 1].0
    );
    for &(ang, i) in a.iter().rev().take(8) {
        let p = (cell.mesh.positions[i] - cell.shift) / 125.0;
        println!(
            "{ang:.1} deg at {:.2} {:.2} {:.2}  sdf n {:.2?} mesh n {:.2?}",
            p.x, p.y, p.z, cell.normals[i], mn[i]
        );
    }
    let mut flipped = [0usize; 3];
    let mut total = 0;
    for t in &cell.mesh.triangles {
        let p = t.map(|i| cell.mesh.positions[i as usize]);
        let f = (p[1] - p[0]).cross(p[2] - p[0]);
        let n = (cell.normals[t[0] as usize] + cell.normals[t[1] as usize] + cell.normals[t[2] as usize]).normalize();
        total += 1;
        if f.dot(n) < 0.0 {
            let fz = f.normalize();
            let ax = if fz.x.abs() > fz.y.abs() && fz.x.abs() > fz.z.abs() {
                0
            } else if fz.y.abs() > fz.z.abs() {
                1
            } else {
                2
            };
            flipped[ax] += 1;
        }
    }
    let mut r: Vec<f64> = cell.residual.iter().map(|x| x.abs()).collect();
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "vertex residual um: median {:.4} p99 {:.4} max {:.4}",
        r[r.len() / 2],
        r[r.len() * 99 / 100],
        r[r.len() - 1]
    );
    let mut idx: Vec<usize> = (0..cell.residual.len()).collect();
    idx.sort_by(|&a, &b| cell.residual[b].abs().partial_cmp(&cell.residual[a].abs()).unwrap());
    for &i in idx.iter().take(6) {
        let p = (cell.mesh.positions[i] - cell.shift) / 125.0;
        println!("res {:.3} at {:.2} {:.2} {:.2}", cell.residual[i], p.x, p.y, p.z);
    }
    println!("flipped triangles {:?} of {total}", flipped);
    let zs: Vec<f64> = cell
        .mesh
        .positions
        .iter()
        .map(|p| (p.z - cell.shift.z) / 125.0)
        .collect();
    println!(
        "z range {:.2} .. {:.2}",
        zs.iter().cloned().fold(f64::INFINITY, f64::min),
        zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
}
#[allow(dead_code)]
fn _x() {}
