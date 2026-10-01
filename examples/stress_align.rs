//! Alignment of cortical subunits with the local stress axis, binned by the
//! local stress anisotropy.
use rusty_tube::setup::{build, preset, SimSetup};
fn main() -> anyhow::Result<()> {
    // Usage: stress_align [OUTLINE] [key=value ...]
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let outline = if args.first().map(|a| !a.contains('=')).unwrap_or(false) {
        args.remove(0)
    } else {
        "procedural:7".to_string()
    };
    let mut sets = preset("dense")?;
    sets.extend(args.iter().cloned());
    let mut sim = build(&SimSetup {
        config: None,
        outline: Some(&outline),
        sets: &sets,
        neck_stress: false,
        shape_stress: true,
        scale: 1.0,
    })?;
    for t in 0..12000 {
        sim.tick(t, false)?;
    }
    let mut bins = [(0.0f64, 0usize); 4];
    for m in sim.mts.values() {
        for &e in &m.body {
            let el = &sim.pool.elems[e as usize];
            if el.cortical && el.stress_seen {
                let b = ((el.stress_mag as f64 * 4.0) as usize).min(3);
                bins[b].0 += el.stress_align as f64;
                bins[b].1 += 1;
            }
        }
    }
    let l = sim.lengths();
    print!(
        "{:?}: mean length {:.2} um, MTs {} | cos^2 by anisotropy",
        args,
        l.iter().sum::<usize>() as f64 / l.len() as f64 / 125.0,
        l.len()
    );
    for (k, b) in bins.iter().enumerate() {
        print!(
            "  [{:.2}-{:.2}] {:.3} (n={})",
            k as f64 / 4.0,
            (k + 1) as f64 / 4.0,
            b.0 / b.1.max(1) as f64,
            b.1
        );
    }
    println!();
    Ok(())
}
