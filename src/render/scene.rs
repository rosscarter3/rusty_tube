//! What the renderer draws, extracted from a simulation state.
//!
//! Render space is micrometres, centred on the cell, z up (sim units are
//! 8 nm subunits, 125 per um).

use crate::cellgen::UNITS_PER_UM;
use crate::sim::Sim;
use bytemuck::{Pod, Zeroable};
use glam::{DVec3, Vec3};

/// One capsule: a run of `stride` subunits of one microtubule.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct Segment {
    pub a: [f32; 3],
    /// Distance from the plus end along the filament, um.
    pub tip_dist: f32,
    pub b: [f32; 3],
    /// Distance from the minus end along the filament, um.
    pub minus_dist: f32,
    /// Per-filament random hash.
    pub hash: u32,
    /// Bit flags, see `FLAG_*`.
    pub flags: u32,
    /// Age of this part of the filament, in steps.
    pub age: f32,
    /// Local stress alignment (cos^2), or -1 if unmeasured.
    pub stress: f32,
}

pub const FLAG_GROWING: u32 = 1;
pub const FLAG_SHRINKING: u32 = 2;
pub const FLAG_CORTICAL: u32 = 4;
pub const FLAG_TREADMILL: u32 = 8;
pub const FLAG_BUNDLED: u32 = 16;
pub const FLAG_PLUS_END: u32 = 32;
/// A stress-field glyph, not a microtubule.
pub const FLAG_GLYPH: u32 = 64;
/// A cellulose synthase complex (drawn as a sphere).
pub const FLAG_CSC: u32 = 128;
/// A piece of cellulose microfibril.
pub const FLAG_FIBRIL: u32 = 256;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable, Default)]
pub struct MeshVertex {
    pub pos: [f32; 3],
    /// Signed outline curvature at this point (1/um), for colouring.
    pub kappa: f32,
    pub normal: [f32; 3],
    /// 1 on the anticlinal wall, 0 on the periclinal faces.
    pub wall: f32,
    /// Stress field direction (unit, in the membrane plane) and magnitude.
    pub stress: [f32; 4],
}

/// The static part: the membrane.
#[derive(Clone, Debug, Default)]
pub struct MembraneGeometry {
    /// Stress-field glyphs: short sticks along the principal stress axis on
    /// the cell surface, as long as the local anisotropy.
    pub glyphs: Vec<Segment>,
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
}

/// Simulation state statistics shown alongside the render.
#[derive(Clone, Debug, Default)]
pub struct SceneStats {
    pub step: i64,
    pub microtubules: usize,
    pub subunits: usize,
    pub growing: usize,
    pub shrinking: usize,
    pub mean_length_um: f64,
    pub cortical_fraction: f64,
    /// Nematic order parameter of the cortical array in the xy plane.
    pub order_xy: f64,
    /// Mean cos^2 between cortical subunits and the local stress axis
    /// (0.5 = random in the plane), or -1 without a stress field.
    pub stress_alignment: f64,
    pub seconds_per_step: f64,
    pub snapshot_ms: f64,
    pub cscs: usize,
    pub csc_density: f64,
    /// Fraction of CSC positions within capture range of a microtubule.
    pub csc_on_mt: f64,
    pub cellulose_um_per_um2: f64,
    pub crystallinity: f64,
}

/// The dynamic part: microtubules as capsules.
#[derive(Clone, Debug, Default)]
pub struct Snapshot {
    pub segments: Vec<Segment>,
    pub stats: SceneStats,
}

#[derive(Clone, Copy, Debug)]
pub struct Frame {
    pub center: DVec3,
}

impl Frame {
    pub fn from_sim(sim: &Sim) -> Self {
        let (lo, hi) = sim.membrane.bounds;
        Frame {
            center: (lo + hi) * 0.5,
        }
    }

    #[inline]
    pub fn to_render(&self, p: DVec3) -> Vec3 {
        ((p - self.center) / UNITS_PER_UM).as_vec3()
    }
}

fn hash32(mut x: u32) -> u32 {
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^ (x >> 16)
}

pub fn membrane_geometry(sim: &Sim, frame: &Frame) -> MembraneGeometry {
    let mut g = MembraneGeometry::default();
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for surf in &sim.membrane.surfaces {
        let base = g.vertices.len() as u32;
        let kappa = surf.mesh.point_scalars.get("kappa");
        let wall = surf.mesh.point_scalars.get("anticlinal");
        let field = surf.mesh.point_vectors.get("strain_0");
        for (k, &vid) in surf.vertices.iter().enumerate() {
            let v = &sim.membrane.verts[vid as usize];
            let p = frame.to_render(v.pos);
            lo = lo.min(p);
            hi = hi.max(p);
            g.vertices.push(MeshVertex {
                pos: p.to_array(),
                kappa: kappa.map(|x| x[k] as f32).unwrap_or(0.0),
                normal: v.normal.as_vec3().to_array(),
                wall: wall.map(|x| x[k] as f32).unwrap_or(0.0),
                stress: field
                    .map(|f| {
                        let n = v.normal;
                        let t = f[k] - f[k].dot(n) * n;
                        let m = t.length();
                        let d = if m > 0.0 { t / m } else { t };
                        [d.x as f32, d.y as f32, d.z as f32, m as f32]
                    })
                    .unwrap_or([0.0; 4]),
            });
        }
        for t in &surf.mesh.triangles {
            g.indices.extend(t.iter().map(|&i| base + i));
        }
    }
    g.bounds_min = lo;
    g.bounds_max = hi;
    g.glyphs = stress_glyphs(&g.vertices, 1.6);
    g
}

/// One glyph per `spacing` um cell of surface where the field is non-zero.
fn stress_glyphs(verts: &[MeshVertex], spacing: f32) -> Vec<Segment> {
    let mut taken = rustc_hash::FxHashSet::default();
    let mut out = Vec::new();
    for v in verts {
        let a = v.stress[3].min(1.0);
        // Outer periclinal face only: the face a microscope looks at.
        if a < 0.05 || v.normal[2] < 0.85 {
            continue;
        }
        let p = Vec3::from(v.pos);
        let cell = (p / spacing).floor().as_ivec3();
        if !taken.insert(cell) {
            continue;
        }
        let n = Vec3::from(v.normal);
        let d = Vec3::new(v.stress[0], v.stress[1], v.stress[2]);
        let c = p + n * 0.1;
        let half = d * 0.42 * spacing * a;
        out.push(Segment {
            a: (c - half).to_array(),
            tip_dist: 10.0,
            b: (c + half).to_array(),
            minus_dist: 10.0,
            hash: 0,
            flags: FLAG_GLYPH,
            age: 0.0,
            stress: a,
        });
    }
    out
}

/// Half-width, in subunits, of the moving average applied to drawn paths.
pub const SMOOTH: usize = 4;

/// Capsules for every microtubule, one per `stride` subunits.
pub fn snapshot(sim: &Sim, frame: &Frame, stride: usize) -> Snapshot {
    let stride = stride.max(1);
    let um = 1.0 / UNITS_PER_UM as f32;
    let mut segs = Vec::with_capacity(sim.n_elements() / stride + sim.mts.len());
    let mut stats = SceneStats {
        step: sim.time,
        microtubules: sim.mts.len(),
        subunits: sim.n_elements(),
        ..Default::default()
    };
    #[derive(Default)]
    struct Part {
        segs: Vec<Segment>,
        growing: usize,
        shrinking: usize,
        q11: f64,
        q12: f64,
        qn: f64,
        cortical: usize,
        sa: f64,
        sn: usize,
    }
    let now = sim.time as f32;
    let mts: Vec<(&u32, &crate::sim::Microtubule)> = sim.mts.iter().collect();
    let per_mt = |(id, mt): &(&u32, &crate::sim::Microtubule)| -> Part {
        let mut part = Part::default();
        let n = mt.body.len();
        let mut base_flags = 0;
        if mt.gr_st_plus == 1 {
            part.growing = 1;
            base_flags |= FLAG_GROWING;
        } else if mt.gr_st_plus == -1 {
            part.shrinking = 1;
            base_flags |= FLAG_SHRINKING;
        }
        if mt.gr_st_moins == -1 {
            base_flags |= FLAG_TREADMILL;
        }
        let hash = hash32(**id);
        // Prefix sums of positions, for O(1) path smoothing.
        let mut prefix = Vec::with_capacity(n + 1);
        prefix.push(DVec3::ZERO);
        for &e in &mt.body {
            let el = &sim.pool.elems[e as usize];
            prefix.push(*prefix.last().unwrap() + el.pos);
            if el.cortical && el.stress_seen && el.stress_mag > 0.3 {
                part.sa += el.stress_align as f64;
                part.sn += 1;
            }
            if el.cortical {
                part.cortical += 1;
                let d = el.dir;
                let l2 = d.x * d.x + d.y * d.y;
                if l2 > 1e-6 {
                    part.q11 += (d.x * d.x - d.y * d.y) / l2;
                    part.q12 += 2.0 * d.x * d.y / l2;
                    part.qn += 1.0;
                }
            }
        }
        if n < 2 {
            return part;
        }
        // Smooth the drawn path over a few subunits: the simulation's
        // per-subunit random walk is a numerical device, and at close range
        // it reads as kinks no stiff filament has.
        let smooth = |k: usize| -> DVec3 {
            let w = SMOOTH.min(k).min(n - 1 - k);
            let (a, b) = (k - w, k + w);
            (prefix[b + 1] - prefix[a]) / (b - a + 1) as f64
        };
        part.segs.reserve(n / stride + 1);
        let mut i = 0;
        while i + 1 < n {
            let j = (i + stride).min(n - 1);
            let ea = &sim.pool.elems[mt.body[i] as usize];
            let eb = &sim.pool.elems[mt.body[j] as usize];
            let mut flags = base_flags;
            if ea.cortical {
                flags |= FLAG_CORTICAL;
            }
            if ea.contact || eb.contact {
                flags |= FLAG_BUNDLED;
            }
            if j == n - 1 {
                flags |= FLAG_PLUS_END;
            }
            part.segs.push(Segment {
                a: frame.to_render(smooth(i)).to_array(),
                tip_dist: (n - 1 - i) as f32 * um,
                b: frame.to_render(smooth(j)).to_array(),
                minus_dist: i as f32 * um,
                hash,
                flags,
                age: now - eb.born as f32,
                stress: if eb.stress_seen { eb.stress_align } else { -1.0 },
            });
            i = j;
        }
        part
    };
    use rayon::prelude::*;
    let parts: Vec<Part> = mts.par_iter().map(per_mt).collect();
    let (mut q11, mut q12, mut qn) = (0.0f64, 0.0f64, 0.0f64);
    let mut cortical = 0usize;
    let (mut sa, mut sn) = (0.0f64, 0usize);
    for p in parts {
        segs.extend_from_slice(&p.segs);
        stats.growing += p.growing;
        stats.shrinking += p.shrinking;
        q11 += p.q11;
        q12 += p.q12;
        qn += p.qn;
        cortical += p.cortical;
        sa += p.sa;
        sn += p.sn;
    }
    // Cellulose synthase complexes and their microfibrils.
    if let Some(c) = sim.csc.as_ref() {
        for (k, csc) in c.cscs.iter().enumerate() {
            let p = frame.to_render(csc.pos).to_array();
            segs.push(Segment {
                a: p,
                tip_dist: if csc.pause > 0 { 1.0 } else { 0.0 },
                b: p,
                minus_dist: (sim.time - csc.born) as f32,
                hash: hash32(k as u32 ^ 0x9e37),
                flags: FLAG_CSC | if csc.on_mt { FLAG_BUNDLED } else { 0 },
                age: 0.0,
                stress: 0.0,
            });
        }
        let tau = (c.p.crystallinity_tau_s / c.p.dt) as f32;
        let fibril_segs: Vec<Vec<Segment>> = c
            .fibrils
            .par_iter()
            .map(|f| {
                // Each piece carries the crystallinity of its age.
                let age = (sim.time as f32 - f.born as f32).max(0.0);
                let cryst = 1.0 - (-age / tau).exp();
                f.points
                    .windows(2)
                    .map(|w| Segment {
                        a: frame.to_render(w[0]).to_array(),
                        tip_dist: 10.0,
                        b: frame.to_render(w[1]).to_array(),
                        minus_dist: 10.0,
                        hash: 0,
                        flags: FLAG_FIBRIL,
                        age,
                        stress: cryst,
                    })
                    .collect()
            })
            .collect();
        for v in fibril_segs {
            segs.extend_from_slice(&v);
        }
        // Join each live fibril's last point to its complex.
        for csc in &c.cscs {
            let f = &c.fibrils[csc.fibril];
            let age = (sim.time as f32 - f.born as f32).max(0.0);
            segs.push(Segment {
                a: frame.to_render(*f.points.last().unwrap()).to_array(),
                tip_dist: 10.0,
                b: frame.to_render(csc.pos).to_array(),
                minus_dist: 10.0,
                hash: 0,
                flags: FLAG_FIBRIL,
                age,
                stress: 1.0 - (-age / tau).exp(),
            });
        }
        stats.cscs = c.cscs.len();
        stats.csc_density = c.density_per_um2();
        let (_, _, coloc) = c.motion_summary();
        stats.csc_on_mt = coloc;
        let (_, density, cryst) = c.wall_summary();
        stats.cellulose_um_per_um2 = density;
        stats.crystallinity = cryst;
    }
    let lens = sim.lengths();
    stats.mean_length_um = lens.iter().sum::<usize>() as f64 / lens.len().max(1) as f64 / UNITS_PER_UM;
    stats.cortical_fraction = cortical as f64 / stats.subunits.max(1) as f64;
    stats.stress_alignment = if sn > 0 { sa / sn as f64 } else { -1.0 };
    stats.order_xy = if qn > 0.0 {
        ((q11 / qn).powi(2) + (q12 / qn).powi(2)).sqrt()
    } else {
        0.0
    };
    Snapshot { segments: segs, stats }
}
