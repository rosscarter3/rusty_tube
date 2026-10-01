//! Statistical regression against the C++ tubulaton.
//!
//! Reference: 8 seeds of the C++ (build-cat, improve-correctness-and-english
//! branch) on the procedural:7 cell with configs/pavement_calibrated.ini,
//! 1000 steps: 294 +/- 13 microtubules and 47 900 +/- 3 600 subunits (SD),
//! cortical fraction 0.990, 1.1 % of subunits beyond the membrane limit. The
//! Rust port on the same mesh: 298 +/- 15 and 49 100 +/- 3 900. The bands
//! below are about 3 SD: they catch a broken port, not a few per cent of
//! drift.

use rusty_tube::cellgen::{build_cell, CellSpec, Outline};
use rusty_tube::params::Params;
use rusty_tube::setup::DEFAULT_CONFIG;
use rusty_tube::sim::Sim;

#[test]
fn matches_cpp_statistics() {
    let outline = Outline::procedural(7, 22.0);
    let cell = build_cell(&outline, &CellSpec::default()).unwrap();
    let mut mesh = cell.mesh;
    mesh.point_vectors.insert("Normals".into(), cell.normals);
    let mut p = Params::from_str_config(DEFAULT_CONFIG).unwrap();
    p.set_s("nom_input_vtk", "generated");
    p.set_i("seed", 4242);
    let mut sim = Sim::new(p, Some((mesh.clone(), mesh))).unwrap();
    for t in 0..=1000 {
        sim.tick(t, false).unwrap();
    }
    let n = sim.mts.len();
    let subunits = sim.n_elements();
    let cortical = sim.cortical_fraction();
    let outside = sim
        .mts
        .values()
        .flat_map(|m| m.body.iter())
        .filter(|&&e| sim.pool.elems[e as usize].d_lim < -1.0)
        .count() as f64
        / subunits as f64;
    eprintln!("MTs {n}, subunits {subunits}, cortical {cortical:.3}, outside {outside:.4}");
    assert!((255..=340).contains(&n), "microtubule count {n}");
    assert!((37_000..=60_000).contains(&subunits), "subunits {subunits}");
    assert!(cortical > 0.97, "cortical fraction {cortical}");
    assert!(outside < 0.03, "fraction outside the membrane {outside}");
}
