use rusty_tube::setup::{build, SimSetup};
fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let outline = args.first().cloned().unwrap_or("procedural:7".into());
    let sets: Vec<String> = args.iter().skip(1).cloned().collect();
    let mut sim = build(&SimSetup {
        config: None,
        outline: Some(&outline),
        sets: &sets,
        neck_stress: false,
        shape_stress: false,
        scale: 1.0,
    })?;
    for t in 0..15000 {
        sim.tick(t, false)?;
    }
    let c = &sim.counters;
    println!(
        "created {} destroyed {} | catastrophes: spontaneous {} encounter {} membrane {} | bundle_grow {} detach {}",
        c.creation_microtubule,
        c.destruction_microtubule,
        c.shrinkages,
        c.encounter_bundle_shrink,
        c.encounter_mb_shrink,
        c.encounter_bundle_grow,
        c.detachements
    );
    let l = sim.lengths();
    let mean = l.iter().sum::<usize>() as f64 / l.len() as f64 / 125.0;
    println!(
        "MTs {} mean length {:.2} um, subunits {}",
        sim.mts.len(),
        mean,
        sim.n_elements()
    );
    Ok(())
}
