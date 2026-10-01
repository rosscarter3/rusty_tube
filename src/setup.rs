//! Building a simulation from a config and, optionally, a generated cell.

use crate::cellgen::{self, CellSpec, Outline, UNITS_PER_UM};
use crate::params::Params;
use crate::sim::Sim;
use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// The calibrated pavement-cell parameter set (pavesim's tubulaton
/// calibration: 1 step = 0.1 s at 0.08 um/s growth).
pub const DEFAULT_CONFIG: &str = include_str!("../configs/pavement_calibrated.ini");

/// Nucleation per um^2 of membrane per step: 0.01 /s/um^2 at 0.1 s/step.
pub const NUCLEATION_PER_UM2_STEP: f64 = 0.001;

/// Named parameter changes on top of the calibrated set.
///
/// `dense` is illustrative, not calibrated: steep encounters cause
/// catastrophe 40 % of the time (Dixit & Cyr 2004) instead of always, and the
/// membrane stops a filament only if it must turn by more than 57 degrees
/// rather than 40. In a lobed pavement cell the 40 degree rule causes most
/// catastrophes, so this gives a much denser array.
pub fn preset(name: &str) -> anyhow::Result<Vec<String>> {
    Ok(match name {
        "calibrated" | "" => vec![],
        "dense" => vec!["proba_crossmicro_shrink=0.4".into(), "Angle_mb_limite=1.0".into()],
        // dense, plus steering by the stress field (use with --shape-stress
        // or another strain_0 field). The steering weight scales linearly
        // with the local field magnitude -- the anisotropy for the shape
        // field -- up to influence_ref = 1. At 0.005 per step the turning
        // length is a few micrometres, comparable to a filament, so
        // alignment grows with anisotropy instead of saturating everywhere:
        // mean cos^2 to the stress axis 0.56 where the anisotropy is below
        // 0.25 and 0.83 where it is above 0.75 (0.5 is random).
        "stress" => vec![
            "proba_crossmicro_shrink=0.4".into(),
            "Angle_mb_limite=1.0".into(),
            "part_influence_influence=0.005".into(),
            "influence_ref=1.0".into(),
        ],
        other => anyhow::bail!("unknown preset {other} (calibrated, dense, stress)"),
    })
}

pub fn load_outline(spec: &str) -> Result<Outline> {
    if let Some(seed) = spec.strip_prefix("procedural:") {
        return Ok(Outline::procedural(seed.parse().context("procedural:SEED")?, 22.0));
    }
    let p = PathBuf::from(spec);
    if p.extension().map(|e| e == "json").unwrap_or(false) {
        Outline::from_pavesim_json(&p)
    } else {
        Outline::from_csv(&p)
    }
}

pub struct SimSetup<'a> {
    pub config: Option<&'a Path>,
    pub outline: Option<&'a str>,
    pub sets: &'a [String],
    pub neck_stress: bool,
    pub shape_stress: bool,
    pub scale: f64,
}

pub fn build(setup: &SimSetup) -> Result<Sim> {
    let mut params = match setup.config {
        Some(p) => Params::from_file(p)?,
        None => Params::from_str_config(DEFAULT_CONFIG)?,
    };
    let mut preloaded = None;
    if let Some(o) = setup.outline {
        let mut outline = load_outline(o)?;
        for p in &mut outline.pts {
            *p *= setup.scale;
        }
        let spec = CellSpec {
            neck_stress: setup.neck_stress,
            shape_stress: setup.shape_stress,
            ..Default::default()
        };
        let cell = cellgen::build_cell(&outline, &spec)?;
        let mut mesh = cell.mesh;
        mesh.point_vectors.insert("Normals".into(), cell.normals);
        let area_um2 = mesh.area() / (UNITS_PER_UM * UNITS_PER_UM);
        params.set_s("nom_input_vtk", "generated");
        params.set_s("nom_input_vtk_ref", "generated");
        params.set_i("Bound_Ind", -1);
        params.set_i("num_input_vtk", 1);
        params.set_d("proba_initialisation_par_step", NUCLEATION_PER_UM2_STEP * area_um2);
        println!(
            "generated cell: {} vertices, {:.0} um^2",
            mesh.positions.len(),
            area_um2
        );
        preloaded = Some((mesh.clone(), mesh));
    }
    for kv in setup.sets {
        let (k, v) = kv.split_once('=').context("--set expects key=value")?;
        params.fill(k, v)?;
    }
    params.apply_time_calibration()?;
    let sim = Sim::new(params, preloaded)?;
    sim.check_supported()?;
    Ok(sim)
}
