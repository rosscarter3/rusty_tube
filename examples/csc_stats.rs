//! Cellulose synthase layer: check against measured CSC behaviour and
//! compare the deposited cellulose with the microtubule array.
//!
//!     cargo run --release --example csc_stats -- [OUTLINE] [key=value ...]
use glam::DVec3;
use rusty_tube::setup::{build, preset, SimSetup};

fn main() -> anyhow::Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let outline = if args.first().map(|a| !a.contains('=')).unwrap_or(false) {
        args.remove(0)
    } else {
        "procedural:12".into()
    };
    let mut sets = preset("dense")?;
    sets.push("csc_enable=1".into());
    sets.extend(args.iter().cloned());
    let mut sim = build(&SimSetup {
        config: None,
        outline: Some(&outline),
        sets: &sets,
        neck_stress: false,
        shape_stress: false,
        scale: 1.0,
    })?;
    println!(
        "{:>6} {:>6} {:>8} {:>9} {:>8} {:>8} {:>9} {:>9} {:>8} {:>10}",
        "min", "CSCs", "per um2", "um/min", "coloc", "chance", "cell ord", "um/um2", "cryst", "cell~MT"
    );
    for t in 0..=24_000i64 {
        sim.tick(t, false)?;
        if t % 3000 == 0 && t > 0 {
            let c = sim.csc.as_ref().unwrap();
            let (speed, _guided, coloc) = c.motion_summary();
            let order_1um = c.regional_order(&sim.membrane, 1.0);
            let (_order, density, cryst) = c.wall_summary();
            // Microtubule nematic tensor per membrane vertex from cortical subunits.
            let n = sim.membrane.verts.len();
            let mut mtq = vec![[0.0f64; 6]; n];
            for m in sim.mts.values() {
                for &e in &m.body {
                    let el = &sim.pool.elems[e as usize];
                    if !el.cortical || el.nearest_contour as usize >= n {
                        continue;
                    }
                    let d = el.dir;
                    let q = &mut mtq[el.nearest_contour as usize];
                    q[0] += d.x * d.x;
                    q[1] += d.y * d.y;
                    q[2] += d.z * d.z;
                    q[3] += d.x * d.y;
                    q[4] += d.x * d.z;
                    q[5] += d.y * d.z;
                }
            }
            let principal = |q: [f64; 6]| {
                let [xx, yy, zz, xy, xz, yz] = q;
                let m = |d: DVec3| {
                    DVec3::new(
                        xx * d.x + xy * d.y + xz * d.z,
                        xy * d.x + yy * d.y + yz * d.z,
                        xz * d.x + yz * d.y + zz * d.z,
                    )
                };
                let mut d = DVec3::new(1.0, 0.7, 0.3).normalize();
                for _ in 0..24 {
                    d = m(d).normalize_or(DVec3::X);
                }
                d
            };
            let (mut agree, mut w) = (0.0, 0.0);
            for v in 0..n {
                let tr = mtq[v][0] + mtq[v][1] + mtq[v][2];
                if tr < 20.0 || c.wall.amount[v] <= 0.0 {
                    continue;
                }
                let (cd, ord) = c.wall.principal(v);
                let md = principal(mtq[v]);
                let wt = c.wall.amount[v] * ord;
                agree += wt * cd.dot(md).powi(2);
                w += wt;
            }
            println!(
                "{:>6.0} {:>6} {:>8.2} {:>9.3} {:>8.2} {:>8.2} {:>9.2} {:>9.2} {:>8.2} {:>10.2}",
                t as f64 * 0.1 / 60.0,
                c.cscs.len(),
                c.density_per_um2(),
                speed,
                coloc,
                c.chance_colocalisation(),
                order_1um,
                density,
                cryst,
                if w > 0.0 { agree / w } else { f64::NAN }
            );
        }
    }
    Ok(())
}
