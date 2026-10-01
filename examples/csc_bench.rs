//! Time steps with and without the CSC layer.
use rusty_tube::setup::{build, preset, SimSetup};
fn main() -> anyhow::Result<()> {
    for csc in [0, 1] {
        let mut sets = preset("dense")?;
        sets.push(format!("csc_enable={csc}"));
        let mut sim = build(&SimSetup {
            config: None,
            outline: Some("procedural:12"),
            sets: &sets,
            neck_stress: false,
            shape_stress: false,
            scale: 1.0,
        })?;
        let t0 = std::time::Instant::now();
        let mut last = t0;
        for t in 0..14000i64 {
            sim.tick(t, false)?;
            if t % 2000 == 1999 {
                let ms = last.elapsed().as_secs_f64() * 1000.0 / 2000.0;
                eprintln!(
                    "csc={csc} t={t}: {ms:.2} ms/step, fibrils {}",
                    sim.csc.as_ref().map(|c| c.fibrils.len()).unwrap_or(0)
                );
                last = std::time::Instant::now();
            }
        }
    }
    Ok(())
}
