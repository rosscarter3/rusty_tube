//! Time the render snapshot and its parts.
use rusty_tube::render::scene;
use rusty_tube::setup::{build, preset, SimSetup};
use std::time::Instant;
fn main() -> anyhow::Result<()> {
    let sets = preset("cellulose")?;
    let mut sim = build(&SimSetup {
        config: None,
        outline: Some("procedural:12"),
        sets: &sets,
        neck_stress: false,
        shape_stress: false,
        scale: 1.0,
    })?;
    for t in 0..12000i64 {
        sim.tick(t, false)?;
    }
    let frame = scene::Frame::from_sim(&sim);
    for _ in 0..3 {
        let t = Instant::now();
        let s = scene::snapshot(&sim, &frame, 3);
        let total = t.elapsed().as_secs_f64() * 1000.0;
        let c = sim.csc.as_ref().unwrap();
        let t = Instant::now();
        let _ = c.wall_summary();
        let wall = t.elapsed().as_secs_f64() * 1000.0;
        let t = Instant::now();
        let mut n = 0usize;
        for _ in 0..10 {
            n += sim.mts.values().map(|m| m.body.len()).sum::<usize>();
        }
        let iter = t.elapsed().as_secs_f64() * 100.0;
        println!(
            "snapshot {total:.1} ms ({} segments): wall_summary {wall:.1} ms, body sweep {iter:.2} ms ({n})",
            s.segments.len()
        );
    }
    Ok(())
}
