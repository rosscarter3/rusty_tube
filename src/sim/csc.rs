//! Cellulose synthase complexes (CSCs) and the cellulose they lay down.
//!
//! Not part of tubulaton: an extension, off unless `csc_enable=1`.
//!
//! Each CSC is a particle in the plasma membrane. It is inserted with a bias
//! toward cortical microtubules (Gutierrez et al. 2009), sits still for about
//! a minute, then glides at ~0.3 um/min (Paredez et al. 2006) while
//! extruding a cellulose microfibril behind it, until it is removed after an
//! exponentially distributed lifetime with mean ~7 min (Sampathkumar et al.
//! 2013). Steady state is ~1 CSC per um^2.
//!
//! Guidance has two parts (Chan & Coen 2020):
//! * microtubule guidance: a tethered CSC (CSI1 present, `csc_tether=1`)
//!   within `csc_capture_um` of a cortical microtubule glides along it, in
//!   whichever of its two directions is closer to its own heading;
//! * autonomous guidance: otherwise it keeps its heading, nudged toward the
//!   local orientation of the cellulose already deposited
//!   (`csc_fibril_guidance`).
//!
//! The cellulose is accumulated per membrane vertex as a nematic orientation
//! tensor and an amount (um of microfibril per um^2 of wall). Crystallinity
//! is phenomenological: deposited cellulose matures from amorphous to
//! crystalline with time constant `crystallinity_tau_s`. There is no kinetic
//! model of crystallinity to base this on; treat it as a placeholder to
//! compare against bulk measurements.

use super::membrane::Membrane;
use super::{ElementPool, NONE};
use crate::params::Params;
use crate::rng::Rng;
use glam::DVec3;

const UNITS_PER_UM: f64 = 125.0;

#[derive(Clone, Copy, Debug)]
pub struct CscParams {
    pub dt: f64,
    pub insertion_per_um2_s: f64,
    pub lifetime_s: f64,
    pub pause_s: f64,
    pub speed_units_per_step: f64,
    pub mt_insertion_bias: f64,
    pub tether: bool,
    pub capture_units: f64,
    /// cos of the largest angle at which an untethered CSC can be captured.
    pub capture_cos: f64,
    /// Radius for counting a CSC as co-localised with a microtubule, as in
    /// a diffraction-limited image.
    pub coloc_units: f64,
    pub fibril_guidance: f64,
    pub crystallinity_tau_s: f64,
    pub update_steps: i64,
    pub fibril_keep_steps: u32,
}

impl CscParams {
    pub fn from_params(p: &Params) -> Self {
        let dt = if p.get_d("dt_seconds") > 0.0 {
            p.get_d("dt_seconds")
        } else {
            0.1
        };
        CscParams {
            dt,
            insertion_per_um2_s: p.get_d("csc_insertion_rate"),
            lifetime_s: p.get_d("csc_lifetime_s"),
            pause_s: p.get_d("csc_pause_s"),
            speed_units_per_step: p.get_d("csc_speed_um_per_min") / 60.0 * dt * UNITS_PER_UM,
            mt_insertion_bias: p.get_d("csc_mt_insertion_bias"),
            tether: p.get_i("csc_tether") != 0,
            capture_units: p.get_d("csc_capture_um") * UNITS_PER_UM,
            capture_cos: p.get_d("csc_capture_angle").cos(),
            coloc_units: p.get_d("csc_coloc_um") * UNITS_PER_UM,
            fibril_guidance: p.get_d("csc_fibril_guidance"),
            crystallinity_tau_s: p.get_d("crystallinity_tau_s"),
            update_steps: p.get_i("csc_update_steps").max(1),
            fibril_keep_steps: (p.get_d("fibril_keep_s") / dt) as u32,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Csc {
    pub pos: DVec3,
    /// Unit heading in the membrane plane.
    pub dir: DVec3,
    /// Nearest membrane vertex.
    pub vertex: u32,
    pub born: i64,
    /// Steps of stillness left after insertion.
    pub pause: i64,
    /// Guided by a microtubule at the last update.
    pub on_mt: bool,
    /// The microtubule it is tethered to (via CSI1), or NONE.
    pub mt: u32,
    /// Index into `CscLayer::fibrils` of the microfibril being made.
    pub fibril: usize,
    /// Distance moved since the last fibril point (units).
    since_point: f64,
}

/// A microfibril's centre line, for rendering and analysis.
#[derive(Clone, Debug)]
pub struct Fibril {
    pub points: Vec<DVec3>,
    pub born: u32,
    pub finished: bool,
}

/// Per-vertex wall state.
#[derive(Clone, Debug, Default)]
pub struct Wall {
    /// Nematic orientation tensor of deposited cellulose (xx, yy, zz, xy, xz, yz), length-weighted, um.
    pub q: Vec<[f64; 6]>,
    /// um of microfibril deposited per vertex.
    pub amount: Vec<f64>,
    /// Of which crystalline.
    pub crystalline: Vec<f64>,
    /// um^2 of membrane each vertex stands for.
    pub area: Vec<f64>,
}

impl Wall {
    /// Principal fibril direction and order (0 isotropic .. 1 aligned) at a vertex.
    pub fn principal(&self, v: usize) -> (DVec3, f64) {
        let [xx, yy, zz, xy, xz, yz] = self.q[v];
        let tr = xx + yy + zz;
        if tr <= 0.0 {
            return (DVec3::ZERO, 0.0);
        }
        // Power iteration on the 3x3 tensor.
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
        let lambda = d.dot(m(d)) / tr;
        // For in-surface fibrils the tensor is effectively 2D: order 2*lambda - 1.
        (d, (2.0 * lambda - 1.0).clamp(0.0, 1.0))
    }

    /// um of cellulose per um^2 of wall at a vertex.
    pub fn density(&self, v: usize) -> f64 {
        if self.area[v] > 0.0 {
            self.amount[v] / self.area[v]
        } else {
            0.0
        }
    }

    pub fn crystallinity(&self, v: usize) -> f64 {
        if self.amount[v] > 0.0 {
            self.crystalline[v] / self.amount[v]
        } else {
            0.0
        }
    }
}

/// Cortical microtubule subunits in a grid sized to the CSC search radii,
/// rebuilt at each guidance update.
#[derive(Clone, Debug, Default)]
struct CorticalIndex {
    cell: f64,
    map: rustc_hash::FxHashMap<(i32, i32, i32), Vec<u32>>,
}

impl CorticalIndex {
    fn key(&self, p: DVec3) -> (i32, i32, i32) {
        (
            (p.x / self.cell).floor() as i32,
            (p.y / self.cell).floor() as i32,
            (p.z / self.cell).floor() as i32,
        )
    }

    fn rebuild(&mut self, pool: &ElementPool, cell: f64) {
        self.cell = cell;
        for v in self.map.values_mut() {
            v.clear();
        }
        for (i, e) in pool.elems.iter().enumerate() {
            if e.alive && e.cortical {
                let k = self.key(e.pos);
                self.map.entry(k).or_default().push(i as u32);
            }
        }
    }

    fn for_each_near(&self, p: DVec3, mut f: impl FnMut(u32)) {
        let (x, y, z) = self.key(p);
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    if let Some(v) = self.map.get(&(x + i, y + j, z + k)) {
                        for &id in v {
                            f(id);
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct CscStats {
    pub inserted: u64,
    pub removed: u64,
    pub inserted_at_mt: u64,
    /// Sum over moving CSC-steps of distance moved (units), and step count.
    pub moved_units: f64,
    pub moving_steps: u64,
    pub guided_steps: u64,
    /// Updates at which a moving CSC was within the capture radius of a
    /// cortical microtubule, tethered or not (co-localisation), and the
    /// number of updates.
    pub near_mt: u64,
    pub updates: u64,
    /// The same count for random membrane points: the chance level.
    pub random_near_mt: u64,
    pub random_updates: u64,
}

#[derive(Clone, Debug)]
pub struct CscLayer {
    pub p: CscParams,
    pub cscs: Vec<Csc>,
    pub fibrils: Vec<Fibril>,
    pub wall: Wall,
    pub stats: CscStats,
    /// Cumulative membrane area per vertex, for uniform insertion.
    area_cdf: Vec<f64>,
    index: CorticalIndex,
    total_area_um2: f64,
}

impl CscLayer {
    pub fn new(p: CscParams, membrane: &Membrane) -> Self {
        let n = membrane.verts.len();
        let mut area = vec![0.0; n];
        for s in &membrane.surfaces {
            let res = s.mesh.vertex_areas();
            for (k, &v) in s.vertices.iter().enumerate() {
                area[v as usize] = res.vertex_area[k] / (UNITS_PER_UM * UNITS_PER_UM);
            }
        }
        // Only the outer surface (surface 1) carries a wall.
        let mut cdf = Vec::with_capacity(n);
        let mut acc = 0.0;
        for (i, v) in membrane.verts.iter().enumerate() {
            if v.surface == 1 {
                acc += area[i];
            }
            cdf.push(acc);
        }
        CscLayer {
            p,
            cscs: Vec::new(),
            fibrils: Vec::new(),
            wall: Wall {
                q: vec![[0.0; 6]; n],
                amount: vec![0.0; n],
                crystalline: vec![0.0; n],
                area,
            },
            stats: CscStats::default(),
            area_cdf: cdf,
            total_area_um2: acc,
            index: CorticalIndex::default(),
        }
    }

    pub fn density_per_um2(&self) -> f64 {
        self.cscs.len() as f64 / self.total_area_um2.max(1e-9)
    }

    /// Place a point on the membrane: the nearest vertex's tangent plane.
    fn onto_membrane(membrane: &Membrane, p: DVec3) -> Option<(DVec3, u32)> {
        let (_, v) = membrane.nearest(p)?;
        let vert = &membrane.verts[v as usize];
        Some((p - (p - vert.pos).dot(vert.normal) * vert.normal, v))
    }

    fn tangent(membrane: &Membrane, v: u32, d: DVec3) -> DVec3 {
        let n = membrane.verts[v as usize].normal;
        (d - d.dot(n) * n).normalize_or_zero()
    }

    /// Nearest cortical microtubule element by distance within the membrane
    /// plane, within `radius` units, optionally restricted to one
    /// microtubule and to an angle of `heading`. Returns its direction and
    /// microtubule id.
    #[allow(clippy::too_many_arguments)]
    fn mt_near(
        index: &CorticalIndex,
        membrane: &Membrane,
        pool: &ElementPool,
        pos: DVec3,
        vertex: u32,
        radius: f64,
        only_mt: u32,
        heading: Option<(DVec3, f64)>,
    ) -> Option<(DVec3, u32)> {
        let n = membrane.verts[vertex as usize].normal;
        let reach = radius + 20.0; // cortical MTs sit ~10 units under the membrane
        let mut best = (radius * radius, NONE);
        index.for_each_near(pos, |id| {
            let e = &pool.elems[id as usize];
            if only_mt != NONE && e.mt != only_mt {
                return;
            }
            let d = e.pos - pos;
            if d.length_squared() > reach * reach {
                return;
            }
            let lateral = d - d.dot(n) * n;
            let l2 = lateral.length_squared();
            if l2 >= best.0 {
                return;
            }
            if let Some((h, cos_max)) = heading {
                let t = e.dir - e.dir.dot(n) * n;
                if t.normalize_or_zero().dot(h).abs() < cos_max {
                    return;
                }
            }
            best = (l2, id);
        });
        if best.1 == NONE {
            None
        } else {
            let e = &pool.elems[best.1 as usize];
            Some((e.dir, e.mt))
        }
    }

    fn insert(&mut self, membrane: &Membrane, pool: &ElementPool, rng: &mut Rng, t: i64) {
        // At a microtubule, or anywhere on the membrane per unit area.
        let mut site = None;
        if rng.unit() < self.p.mt_insertion_bias && pool.live > 0 {
            for _ in 0..32 {
                let k = rng.index(pool.elems.len());
                let e = &pool.elems[k];
                if e.alive && e.cortical {
                    site = Self::onto_membrane(membrane, e.pos).map(|(p, v)| (p, v, Some((e.dir, e.mt))));
                    break;
                }
            }
        }
        let at_mt = site.is_some();
        if site.is_none() {
            let total = *self.area_cdf.last().unwrap_or(&0.0);
            if total <= 0.0 {
                return;
            }
            let target = rng.unit() * total;
            let v = self
                .area_cdf
                .partition_point(|&c| c < target)
                .min(self.area_cdf.len() - 1) as u32;
            site = Some((membrane.verts[v as usize].pos, v, None));
        }
        let Some((pos, v, axis)) = site else { return };
        let random = DVec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit());
        let mut dir = Self::tangent(membrane, v, axis.map(|a| a.0).unwrap_or(random));
        let mt = if self.p.tether {
            axis.map(|a| a.1).unwrap_or(NONE)
        } else {
            NONE
        };
        if dir == DVec3::ZERO {
            dir = Self::tangent(membrane, v, random);
        }
        // Bidirectional: either way along the microtubule.
        if rng.unit() < 0.5 {
            dir = -dir;
        }
        self.fibrils.push(Fibril {
            points: vec![pos],
            born: t as u32,
            finished: false,
        });
        self.cscs.push(Csc {
            pos,
            dir,
            vertex: v,
            born: t,
            pause: (self.p.pause_s / self.p.dt) as i64,
            on_mt: at_mt && mt != NONE,
            mt,
            fibril: self.fibrils.len() - 1,
            since_point: 0.0,
        });
        self.stats.inserted += 1;
        if at_mt {
            self.stats.inserted_at_mt += 1;
        }
    }

    pub fn step(&mut self, membrane: &Membrane, pool: &ElementPool, rng: &mut Rng, t: i64) {
        let p = self.p;
        // Insertion: Poisson with the per-area rate.
        let expected = p.insertion_per_um2_s * self.total_area_um2 * p.dt;
        let mut n = expected.floor() as i64;
        if rng.unit() < expected - n as f64 {
            n += 1;
        }
        for _ in 0..n {
            self.insert(membrane, pool, rng, t);
        }

        // Removal after an exponential lifetime.
        let p_remove = p.dt / p.lifetime_s;
        let mut i = 0;
        while i < self.cscs.len() {
            if rng.unit() < p_remove {
                let c = self.cscs.swap_remove(i);
                self.fibrils[c.fibril].finished = true;
                self.stats.removed += 1;
            } else {
                i += 1;
            }
        }

        // Movement, guidance and deposition.
        let update = t % p.update_steps == 0;
        if update {
            let cell = p.capture_units.max(p.coloc_units) + 21.0;
            self.index.rebuild(pool, cell);
        }
        let step_um = p.speed_units_per_step / UNITS_PER_UM;
        for ci in 0..self.cscs.len() {
            if self.cscs[ci].pause > 0 {
                self.cscs[ci].pause -= 1;
                continue;
            }
            if update {
                let (pos, v, dir, mt) = {
                    let c = &self.cscs[ci];
                    (c.pos, c.vertex, c.dir, c.mt)
                };
                // Co-localisation as imaging would score it, with no angle
                // condition, and the same score at a random membrane point
                // for the chance level.
                self.stats.updates += 1;
                if Self::mt_near(&self.index, membrane, pool, pos, v, p.coloc_units, NONE, None).is_some() {
                    self.stats.near_mt += 1;
                }
                let total = *self.area_cdf.last().unwrap_or(&0.0);
                if total > 0.0 {
                    let target = rng.unit() * total;
                    let rv = self
                        .area_cdf
                        .partition_point(|&c| c < target)
                        .min(self.area_cdf.len() - 1) as u32;
                    self.stats.random_updates += 1;
                    let rp = membrane.verts[rv as usize].pos;
                    if Self::mt_near(&self.index, membrane, pool, rp, rv, p.coloc_units, NONE, None).is_some() {
                        self.stats.random_near_mt += 1;
                    }
                }
                // Microtubule guidance. A tethered complex follows its own
                // microtubule, through crossovers, until that microtubule
                // is gone from under it; a free one is captured by a
                // microtubule within reach that runs close to its heading.
                let mut guide = None;
                let mut new_mt = NONE;
                if p.tether {
                    if mt != NONE {
                        guide = Self::mt_near(&self.index, membrane, pool, pos, v, p.capture_units, mt, None);
                    }
                    if guide.is_none() {
                        guide = Self::mt_near(
                            &self.index,
                            membrane,
                            pool,
                            pos,
                            v,
                            p.capture_units,
                            NONE,
                            Some((dir, p.capture_cos)),
                        );
                    }
                    if let Some((_, id)) = guide {
                        new_mt = id;
                    }
                }
                let mut nd = dir;
                let mut on_mt = false;
                let axis = guide
                    .map(|(a, _)| Self::tangent(membrane, v, a))
                    .filter(|a| *a != DVec3::ZERO);
                if let Some(a) = axis {
                    nd = if a.dot(dir) >= 0.0 { a } else { -a };
                    on_mt = true;
                } else if p.fibril_guidance > 0.0 {
                    let (fd, order) = self.wall.principal(v as usize);
                    let fd = Self::tangent(membrane, v, fd);
                    if fd != DVec3::ZERO && order > 0.0 {
                        let fd = if fd.dot(dir) >= 0.0 { fd } else { -fd };
                        nd = (dir + fd * p.fibril_guidance * order).normalize_or(dir);
                    }
                }
                let c = &mut self.cscs[ci];
                c.dir = Self::tangent(membrane, v, nd).normalize_or(c.dir);
                c.on_mt = on_mt;
                c.mt = if on_mt { new_mt } else { NONE };
            }
            let c = &mut self.cscs[ci];
            let moved = c.pos + c.dir * p.speed_units_per_step;
            if update {
                if let Some((q, v)) = Self::onto_membrane(membrane, moved) {
                    c.pos = q;
                    c.vertex = v;
                } else {
                    c.pos = moved;
                }
            } else {
                c.pos = moved;
            }
            self.stats.moved_units += p.speed_units_per_step;
            self.stats.moving_steps += 1;
            if c.on_mt {
                self.stats.guided_steps += 1;
            }
            // Deposit: length-weighted orientation tensor at the vertex.
            let v = c.vertex as usize;
            let d = c.dir;
            let q = &mut self.wall.q[v];
            q[0] += step_um * d.x * d.x;
            q[1] += step_um * d.y * d.y;
            q[2] += step_um * d.z * d.z;
            q[3] += step_um * d.x * d.y;
            q[4] += step_um * d.x * d.z;
            q[5] += step_um * d.y * d.z;
            self.wall.amount[v] += step_um;
            c.since_point += p.speed_units_per_step;
            if c.since_point >= 0.1 * UNITS_PER_UM {
                c.since_point = 0.0;
                self.fibrils[c.fibril].points.push(c.pos);
            }
        }

        // Crystallisation of what is already there.
        if update {
            let k = (p.update_steps as f64 * p.dt / p.crystallinity_tau_s).min(1.0);
            for v in 0..self.wall.amount.len() {
                let a = self.wall.amount[v];
                if a > 0.0 {
                    let c = &mut self.wall.crystalline[v];
                    *c += (a - *c) * k;
                }
            }
        }

        // Keep the recent fibrils only; finish the live ones' end points.
        if t % 50 == 0 {
            let keep = p.fibril_keep_steps;
            let now = t as u32;
            let live: Vec<usize> = self.cscs.iter().map(|c| c.fibril).collect();
            let mut map = vec![usize::MAX; self.fibrils.len()];
            let mut kept = Vec::with_capacity(self.fibrils.len());
            for (i, f) in self.fibrils.drain(..).enumerate() {
                if !f.finished || now.saturating_sub(f.born) < keep {
                    map[i] = kept.len();
                    kept.push(f);
                }
            }
            self.fibrils = kept;
            for (c, &old) in self.cscs.iter_mut().zip(&live) {
                c.fibril = map[old];
            }
        }
    }

    /// Mean speed of moving CSCs (um/min), the fraction of moving steps
    /// spent guided by a microtubule, and the fraction of CSC positions
    /// within `csc_coloc_um` of a cortical microtubule.
    pub fn motion_summary(&self) -> (f64, f64, f64) {
        let s = &self.stats;
        if s.moving_steps == 0 {
            return (0.0, 0.0, 0.0);
        }
        let speed = s.moved_units / s.moving_steps as f64 / UNITS_PER_UM / self.p.dt * 60.0;
        let coloc = if s.updates > 0 {
            s.near_mt as f64 / s.updates as f64
        } else {
            0.0
        };
        (speed, s.guided_steps as f64 / s.moving_steps as f64, coloc)
    }

    /// Co-localisation expected by chance: the same score at random
    /// membrane points.
    pub fn chance_colocalisation(&self) -> f64 {
        let s = &self.stats;
        if s.random_updates > 0 {
            s.random_near_mt as f64 / s.random_updates as f64
        } else {
            0.0
        }
    }

    /// Zero the motion and co-localisation counters (e.g. after a burn-in).
    pub fn reset_stats(&mut self) {
        let (ins, rem, at) = (self.stats.inserted, self.stats.removed, self.stats.inserted_at_mt);
        self.stats = CscStats {
            inserted: ins,
            removed: rem,
            inserted_at_mt: at,
            ..Default::default()
        };
    }

    /// Cellulose order over regions of `bin_um` (orientation tensors summed
    /// per cube, order weighted by amount): local alignment at a scale an
    /// imaging study would measure, rather than per mesh vertex where a
    /// single track looks perfectly ordered.
    pub fn regional_order(&self, membrane: &Membrane, bin_um: f64) -> f64 {
        let mut bins: rustc_hash::FxHashMap<(i64, i64, i64), [f64; 6]> = Default::default();
        for (v, q) in self.wall.q.iter().enumerate() {
            if self.wall.amount[v] <= 0.0 {
                continue;
            }
            let p = membrane.verts[v].pos / (bin_um * UNITS_PER_UM);
            let b = bins
                .entry((p.x.floor() as i64, p.y.floor() as i64, p.z.floor() as i64))
                .or_default();
            for k in 0..6 {
                b[k] += q[k];
            }
        }
        let (mut acc, mut w) = (0.0, 0.0);
        for q in bins.values() {
            let tmp = Wall {
                q: vec![*q],
                amount: vec![1.0],
                crystalline: vec![0.0],
                area: vec![1.0],
            };
            let (_, order) = tmp.principal(0);
            let amount = q[0] + q[1] + q[2];
            acc += order * amount;
            w += amount;
        }
        if w > 0.0 {
            acc / w
        } else {
            0.0
        }
    }

    /// Area-weighted mean cellulose order, density and crystallinity over
    /// vertices that have any cellulose.
    pub fn wall_summary(&self) -> (f64, f64, f64) {
        let (mut ow, mut dw, mut cw, mut w, mut wc) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for v in 0..self.wall.amount.len() {
            let a = self.wall.area[v];
            if a <= 0.0 {
                continue;
            }
            w += a;
            dw += a * self.wall.density(v);
            if self.wall.amount[v] > 0.0 {
                wc += a;
                ow += a * self.wall.principal(v).1;
                cw += a * self.wall.crystallinity(v);
            }
        }
        (
            if wc > 0.0 { ow / wc } else { 0.0 },
            if w > 0.0 { dw / w } else { 0.0 },
            if wc > 0.0 { cw / wc } else { 0.0 },
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::cellgen::{build_cell, CellSpec, Outline};
    use crate::params::Params;
    use crate::setup::DEFAULT_CONFIG;
    use crate::sim::Sim;

    #[test]
    fn density_speed_and_membrane_confinement() {
        let cell = build_cell(
            &Outline::procedural(3, 16.0),
            &CellSpec {
                spacing_um: 0.35,
                ..Default::default()
            },
        )
        .unwrap();
        let mut mesh = cell.mesh;
        mesh.point_vectors.insert("Normals".into(), cell.normals);
        let mut p = Params::from_str_config(DEFAULT_CONFIG).unwrap();
        p.set_s("nom_input_vtk", "generated");
        p.set_i("seed", 7);
        p.set_i("csc_enable", 1);
        p.set_d("proba_initialisation_par_step", 0.2);
        let mut sim = Sim::new(p, Some((mesh.clone(), mesh))).unwrap();
        let steps = 6000; // 10 min at 0.1 s
        for t in 0..steps {
            sim.tick(t, false).unwrap();
        }
        let c = sim.csc.as_ref().unwrap();
        // Density approaches rate * lifetime = 1 um^-2 as 1 - exp(-t / 7 min).
        let expected = 1.0 - (-600.0f64 / 420.0).exp();
        let d = c.density_per_um2();
        assert!((d - expected).abs() < 0.15, "density {d}, expected about {expected}");
        let (speed, _, _) = c.motion_summary();
        assert!((speed - 0.35).abs() < 1e-6, "speed {speed}");
        for csc in &c.cscs {
            let v = &sim.membrane.verts[csc.vertex as usize];
            let off = (csc.pos - v.pos).dot(v.normal).abs();
            assert!(off < 2.0, "CSC {off} units off the membrane");
        }
    }
}
