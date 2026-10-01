//! The microtubule simulation: a port of tubulaton's ElementPool,
//! Microtubule and MicrotubulePool.
//!
//! A microtubule is a chain of elements (tubulin subunits, one mesh unit =
//! 8 nm apart). Each step a growing plus end adds one element whose position
//! is the previous one plus the previous direction, and whose direction is
//! chosen by the encounter rules: bundle with or cross a neighbour met at a
//! shallow or steep angle, slide along or be stopped by the membrane, or
//! wander freely. Plus ends switch between growth and shrinkage
//! (catastrophe, rescue), minus ends detach and depolymerise (treadmilling),
//! and filaments are severed at sharp bends or crossovers.
//!
//! The rules follow the C++ line by line; departures are listed in
//! `PORTING.md` and are limited to removing undefined behaviour (unclamped
//! `acos`, normalising zero vectors, dangling pointers).

pub mod csc;
pub mod grid;
pub mod membrane;
pub mod output;

use crate::params::Params;
use crate::rng::Rng;
use anyhow::Result;
use glam::DVec3;
use grid::Grid;
use membrane::Membrane;
use std::collections::{BTreeMap, VecDeque};
use std::f64::consts::PI;

pub const NONE: u32 = u32::MAX;

/// A tubulin subunit belonging to a microtubule.
#[derive(Clone, Debug)]
pub struct Element {
    pub pos: DVec3,
    /// Direction of the next step from here.
    pub dir: DVec3,
    pub prev: u32,
    /// Owning microtubule id, NONE once detached.
    pub mt: u32,
    pub gen: u32,
    pub alive: bool,
    /// 0 until this element has been the plus end that grew.
    pub age: u8,
    pub contact: bool,
    pub to_cut: bool,
    pub crossover_to_cut: bool,
    pub cortical: bool,
    pub cortical_structure: i32,
    pub nearest_contour: u32,
    pub nearest_contour_d: f32,
    /// Signed gap to the membrane limit plane (`D_pos_dir2lim_mbFn`).
    pub d_lim: f32,
    pub stress_align: f32,
    pub stress_mag: f32,
    pub stress_seen: bool,
    pub cross_micro: (u32, u32),
    pub cross_num: i32,
    pub grid_key: u64,
    tocut_token: u32,
    cross_token: u32,
    /// Step at which the element was added (for rendering age).
    pub born: u32,
}

impl Element {
    fn new(pos: DVec3, dir: DVec3, born: u32) -> Self {
        Element {
            pos,
            dir,
            prev: NONE,
            mt: NONE,
            gen: 0,
            alive: true,
            age: 0,
            contact: false,
            to_cut: false,
            crossover_to_cut: false,
            cortical: false,
            cortical_structure: 0,
            nearest_contour: NONE,
            nearest_contour_d: 0.0,
            d_lim: -10.0,
            stress_align: 0.0,
            stress_mag: 0.0,
            stress_seen: false,
            cross_micro: (NONE, 0),
            cross_num: 0,
            grid_key: 0,
            tocut_token: 0,
            cross_token: 0,
            born,
        }
    }
}

/// Arena of elements with slot reuse; handles carry a generation so a stale
/// reference reads as dead rather than as whatever reused the slot.
#[derive(Clone, Debug)]
pub struct ElementPool {
    pub elems: Vec<Element>,
    free: Vec<u32>,
    /// Microtubule-microtubule neighbour grid, sized to the bundling radius.
    pub grid: Grid,
    pub live: usize,
}

impl ElementPool {
    fn alloc(&mut self, mut e: Element) -> u32 {
        let idx = if let Some(i) = self.free.pop() {
            e.gen = self.elems[i as usize].gen.wrapping_add(1);
            e.tocut_token = self.elems[i as usize].tocut_token;
            e.cross_token = self.elems[i as usize].cross_token;
            self.elems[i as usize] = e;
            i
        } else {
            self.elems.push(e);
            (self.elems.len() - 1) as u32
        };
        let p = self.elems[idx as usize].pos;
        self.elems[idx as usize].grid_key = self.grid.insert(p, idx);
        self.live += 1;
        idx
    }

    fn kill(&mut self, idx: u32) {
        let e = &mut self.elems[idx as usize];
        debug_assert!(e.alive);
        e.alive = false;
        e.prev = NONE;
        e.mt = NONE;
        e.to_cut = false;
        e.crossover_to_cut = false;
        let key = e.grid_key;
        self.grid.remove(key, idx);
        self.free.push(idx);
        self.live -= 1;
    }

    #[inline]
    pub fn handle(&self, idx: u32) -> (u32, u32) {
        (idx, self.elems[idx as usize].gen)
    }

    #[inline]
    pub fn is_alive(&self, h: (u32, u32)) -> bool {
        h.0 != NONE && {
            let e = &self.elems[h.0 as usize];
            e.alive && e.gen == h.1
        }
    }
}

#[derive(Clone, Debug)]
pub struct Microtubule {
    pub id: u32,
    pub body: VecDeque<u32>,
    /// 1 growing, 0 paused, -1 shrinking.
    pub gr_st_plus: i32,
    /// 0 anchored, -1 depolymerising (detached, treadmilling).
    pub gr_st_moins: i32,
    pub treadmilling_start: i64,
    pub shrink_beyond: i32,
    pub age: i32,
    pub stress_align: f64,
    pub stress_mag: f64,
    pub stress_seen: bool,
}

impl Microtubule {
    fn new(id: u32) -> Self {
        Microtubule {
            id,
            body: VecDeque::new(),
            gr_st_plus: 1,
            gr_st_moins: 0,
            treadmilling_start: 0,
            shrink_beyond: 0,
            age: 0,
            stress_align: 0.0,
            stress_mag: 0.0,
            stress_seen: false,
        }
    }
}

/// Event counts since the last report (`rapport`).
#[derive(Clone, Debug, Default)]
pub struct Counters {
    pub encounter_bundle_grow: u64,
    pub encounter_bundle_shrink: u64,
    pub encounter_mb_grow: u64,
    pub encounter_mb_shrink: u64,
    pub cross: u64,
    pub left_membrane: u64,
    pub destruction_microtubule: u64,
    pub creation_microtubule: u64,
    pub cut_microtubule: u64,
    pub override_cut_microtubule: u64,
    pub detachements: u64,
    pub shrinkages: u64,
    /// Encounters of a growing tip with its own filament.
    pub self_encounters: u64,
}

/// Parameters read on the hot path, refreshed whenever the config changes.
#[derive(Clone, Copy, Debug)]
pub struct Dyn {
    pub r_mt: f64,
    pub angle_bundle: f64,
    pub icpower: i64,
    pub proba_crossmicro_shrink: f64,
    pub cutgroup: f64,
    pub angle_mb_traj: f64,
    pub angle_mb_lim: f64,
    pub d_mb: f64,
    pub cortical_thickness: f64,
    pub decision_accrochage: i64,
    pub decision_rencontre: i64,
    pub decision_cut: i64,
    pub angle_cut: f64,
    pub fixe: f64,
    pub alea: f64,
    pub influence_normale: i64,
    pub influence_weight: f64,
    pub influence_ref: f64,
    pub proba_shrink_beyond: f64,
    pub xpos_beyond: f64,
    pub xpos_beyond_minus: f64,
    pub shrinking_multiple: i64,
    pub treadmilling_frequency: i64,
    pub proba_tocut: f64,
    pub proba_crossmicro_cut: f64,
    pub stress_catastrophe_bias: f64,
    pub stress_severing_bias: f64,
    pub self_encounters: bool,
    pub proba_detach: f64,
    pub proba_shrink: f64,
    pub proba_rescue: f64,
}

impl Dyn {
    pub fn from_params(p: &Params) -> Self {
        Dyn {
            r_mt: p.get_d("D_bundle") / p.get_d("taille_microtubule"),
            angle_bundle: p.get_d("Angle_bundle"),
            icpower: p.get_i("icpower"),
            proba_crossmicro_shrink: p.get_d("proba_crossmicro_shrink"),
            cutgroup: p.get_d("cutgroup"),
            angle_mb_traj: p.get_d("Angle_mb_trajectoire"),
            angle_mb_lim: p.get_d("Angle_mb_limite"),
            d_mb: p.get_d("d_mb"),
            cortical_thickness: p.get_d("epaisseur_corticale"),
            decision_accrochage: p.get_i("decision_accrochage"),
            decision_rencontre: p.get_i("decision_rencontre"),
            decision_cut: p.get_i("decision_cut"),
            angle_cut: p.get_d("Angle_cut"),
            // The C++ stores these weights in ints.
            fixe: p.get_d("part_alea_fixe").trunc(),
            alea: p.get_d("part_alea_alea").trunc(),
            influence_normale: p.get_d("part_influence_normale").trunc() as i64,
            influence_weight: p.get_d("part_influence_influence"),
            influence_ref: p.get_d("influence_ref"),
            proba_shrink_beyond: p.get_d("proba_shrink_beyond"),
            xpos_beyond: p.get_d("Xpos_beyond"),
            xpos_beyond_minus: p.get_d("Xpos_beyond_minus"),
            shrinking_multiple: p.get_i("ShrinkingMultiple"),
            treadmilling_frequency: p.get_i("TreadmillingFrequency"),
            proba_tocut: p.get_d("proba_tocut"),
            proba_crossmicro_cut: p.get_d("proba_crossmicro_cut"),
            stress_catastrophe_bias: p.get_d("stress_catastrophe_bias"),
            stress_severing_bias: p.get_d("stress_severing_bias"),
            self_encounters: p.get_i("self_encounters") != 0,
            proba_detach: p.get_d("proba_detachement_par_step_par_microtubule"),
            proba_shrink: p.get_d("proba_shrink"),
            proba_rescue: p.get_d("proba_rescue"),
        }
    }
}

#[inline]
fn angle(a: DVec3, b: DVec3) -> f64 {
    let d = (a.length_squared() * b.length_squared()).sqrt();
    if d == 0.0 {
        return 0.0;
    }
    (a.dot(b) / d).clamp(-1.0, 1.0).acos()
}

#[inline]
fn has_nan(v: DVec3) -> bool {
    v.is_nan()
}

/// `normalise` as the C++ has it: divides by the length, so a zero vector
/// becomes NaN. Callers that can see a zero vector test for NaN after.
#[inline]
fn normalise(v: DVec3) -> DVec3 {
    v / v.length()
}

pub struct Sim {
    pub params: Params,
    pub d: Dyn,
    pub rng: Rng,
    pub membrane: Membrane,
    pub pool: ElementPool,
    pub mts: BTreeMap<u32, Microtubule>,
    max_mt_id: u32,
    tocut: Vec<(u32, u32, u32)>,
    crosscut: Vec<(u32, u32, u32)>,
    pub counters: Counters,
    pub time: i64,
    scratch_nn: Vec<Option<(f64, u32)>>,
    /// Cellulose synthase layer (csc_enable=1).
    pub csc: Option<csc::CscLayer>,
}

impl Sim {
    /// Build the simulation from a config. `preloaded` supplies surface 0 (and
    /// its reference surface) when the geometry was generated in memory.
    pub fn new(mut params: Params, preloaded: Option<(crate::mesh::TriMesh, crate::mesh::TriMesh)>) -> Result<Self> {
        let rng = Rng::new(params.get_i("seed"));
        println!("Random seed: {}", rng.seed());
        let membrane = Membrane::load(&mut params, preloaded)?;
        let d = Dyn::from_params(&params);
        let mt_cell = d.r_mt.max(1.0);
        let pool = ElementPool {
            elems: Vec::new(),
            free: Vec::new(),
            grid: Grid::new(mt_cell, membrane.grid.min),
            live: 0,
        };
        let mut sim = Sim {
            params,
            d,
            rng,
            membrane,
            pool,
            mts: BTreeMap::new(),
            max_mt_id: 1,
            tocut: Vec::new(),
            crosscut: Vec::new(),
            counters: Counters::default(),
            time: 0,
            scratch_nn: Vec::new(),
            csc: None,
        };
        sim.initiate()?;
        if sim.params.get_i("csc_enable") != 0 {
            sim.csc = Some(csc::CscLayer::new(
                csc::CscParams::from_params(&sim.params),
                &sim.membrane,
            ));
        }
        Ok(sim)
    }

    pub fn refresh_params(&mut self) {
        self.d = Dyn::from_params(&self.params);
        if let Some(c) = self.csc.as_mut() {
            c.p = csc::CscParams::from_params(&self.params);
        }
    }

    // ------------------------------------------------------------------
    // Microtubule bookkeeping

    fn new_mt_id(&mut self) -> u32 {
        self.max_mt_id += 1;
        self.max_mt_id
    }

    /// `giveMicrotubule(ep, v, cont_num)`: a one-element microtubule.
    fn give_microtubule(&mut self, pos: DVec3, dir: DVec3, cont_num: i32) -> u32 {
        let id = self.new_mt_id();
        let mut e = Element::new(pos, dir, self.time as u32);
        e.cortical_structure = cont_num;
        e.mt = id;
        let idx = self.pool.alloc(e);
        let mut mt = Microtubule::new(id);
        mt.body.push_back(idx);
        self.mts.insert(id, mt);
        id
    }

    /// Initial microtubules (`MicrotubulePool::initiate`).
    fn initiate(&mut self) -> Result<()> {
        let cen = self.params.get_d("Initialisation_Volume_Radius");
        let n_obj = self.membrane.surfaces.len();
        let mut created = 0;
        if self.params.get_i("Sep_Nucleations") == 1 {
            for j in 1..=n_obj {
                let (n, every) = if j == 1 {
                    (
                        self.params.get_i("nb_microtubules_init"),
                        self.params.get_i("Everywhere_nb_microtubules_init"),
                    )
                } else {
                    (
                        self.params.get_i(&format!("nb_microtubules_init_{j}")),
                        self.params.get_i(&format!("Everywhere_nb_microtubules_init_{j}")),
                    )
                };
                for (count, ty) in [(n, 0), (every, 1)] {
                    for _ in 0..count {
                        if let Some((p, d)) = self.choose_new_mt(j as i32, ty, cen)? {
                            self.give_microtubule(p, d, j as i32);
                            created += 1;
                        }
                    }
                }
            }
        } else {
            let cont = if n_obj == 1 {
                1
            } else if cen > 0.0 {
                n_obj as i32 + 1
            } else {
                -1
            };
            let n = self.params.get_i("nb_microtubules_init");
            let every = self.params.get_i("Everywhere_nb_microtubules_init");
            for (count, ty) in [(n, 0), (every, 1)] {
                for _ in 0..count {
                    if let Some((p, d)) = self.choose_new_mt(cont, ty, cen)? {
                        self.give_microtubule(p, d, cont);
                        created += 1;
                    }
                }
            }
        }
        println!("Number of MTs created: {created}");
        Ok(())
    }

    fn choose_new_mt(&mut self, cont: i32, type_nuc: i32, cen_unif: f64) -> Result<Option<(DVec3, DVec3)>> {
        if cen_unif > 0.0 {
            return Ok(Some(if self.params.get_s("Nucleation_Volume_Shape") == "sphere" {
                self.random_in_sphere(cen_unif)
            } else {
                self.random_in_cube(cen_unif)
            }));
        }
        let m = &self.membrane;
        if cont < 0 {
            if type_nuc == 1 {
                m.random_site(&self.params, &mut self.rng, None)
            } else {
                m.nucleation_site(&self.params, &mut self.rng, 0)
            }
        } else {
            let s = (cont - 1) as usize;
            if type_nuc == 1 {
                if s >= m.surfaces.len() {
                    return Ok(None);
                }
                m.random_site(&self.params, &mut self.rng, Some(s))
            } else {
                m.nucleation_site(&self.params, &mut self.rng, s)
            }
        }
    }

    fn centre(&self) -> DVec3 {
        DVec3::new(
            self.params.get_d("CentreX"),
            self.params.get_d("CentreY"),
            self.params.get_d("CentreZ"),
        )
    }

    fn random_dir(&mut self) -> DVec3 {
        let v = DVec3::new(
            (self.rng.unit() - 0.5) * 2.0,
            (self.rng.unit() - 0.5) * 2.0,
            (self.rng.unit() - 0.5) * 2.0,
        );
        v.normalize_or(DVec3::X)
    }

    fn random_in_cube(&mut self, r: f64) -> (DVec3, DVec3) {
        let c = self.centre();
        let p = c - DVec3::splat(r) + DVec3::new(self.rng.unit(), self.rng.unit(), self.rng.unit()) * 2.0 * r;
        (p, self.random_dir())
    }

    fn random_in_sphere(&mut self, r: f64) -> (DVec3, DVec3) {
        let c = self.centre();
        let theta = self.rng.unit() * 2.0 * PI;
        let phi = self.rng.signed_unit().acos();
        let rr = r * self.rng.unit().cbrt();
        let p = c + rr * DVec3::new(theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos());
        (p, self.random_dir())
    }

    /// `InitiateMidway`: the integer part of the rate is a number of
    /// nucleations, the fractional part a probability of one more.
    fn initiate_midway(&mut self, rate: f64, cont: i32, type_nuc: i32, cen_unif: f64) -> Result<()> {
        let proba = self.rng.unit();
        let whole = rate.trunc() as i64;
        let frac = rate - whole as f64;
        let make = |s: &mut Self| -> Result<()> {
            if let Some((p, d)) = s.choose_new_mt(cont, type_nuc, cen_unif)? {
                s.give_microtubule(p, d, cont);
                s.counters.creation_microtubule += 1;
            }
            Ok(())
        };
        for _ in 0..whole.max(0) {
            make(self)?;
        }
        if proba < frac {
            make(self)?;
        }
        Ok(())
    }

    // ------------------------------------------------------------------
    // Severing lists

    fn enlist_tocut(&mut self, idx: u32) {
        let e = &mut self.pool.elems[idx as usize];
        if !e.to_cut {
            e.to_cut = true;
            e.tocut_token = e.tocut_token.wrapping_add(1);
            self.tocut.push((idx, e.gen, e.tocut_token));
        }
    }

    fn enlist_crossover(&mut self, idx: u32) {
        let e = &mut self.pool.elems[idx as usize];
        if !e.crossover_to_cut {
            e.crossover_to_cut = true;
            e.cross_token = e.cross_token.wrapping_add(1);
            self.crosscut.push((idx, e.gen, e.cross_token));
        }
    }

    fn unlist_crossover(&mut self, idx: u32) {
        self.pool.elems[idx as usize].crossover_to_cut = false;
    }

    /// Entries still valid: the element is alive, the same one, and has not
    /// been unlisted (and possibly re-listed) since.
    fn compact_lists(&mut self) {
        let elems = &self.pool.elems;
        self.tocut.retain(|&(i, g, t)| {
            let e = &elems[i as usize];
            e.alive && e.gen == g && e.to_cut && e.tocut_token == t
        });
        self.crosscut.retain(|&(i, g, t)| {
            let e = &elems[i as usize];
            e.alive && e.gen == g && e.crossover_to_cut && e.cross_token == t
        });
    }

    pub fn crossover_list_len(&self) -> usize {
        let elems = &self.pool.elems;
        self.crosscut
            .iter()
            .filter(|&&(i, g, t)| {
                let e = &elems[i as usize];
                e.alive && e.gen == g && e.crossover_to_cut && e.cross_token == t
            })
            .count()
    }

    // ------------------------------------------------------------------
    // Rates (ALIGNMENT.md rules R3 and R4)

    fn shrink_probability(&self, mt: &Microtubule) -> f64 {
        let d = &self.d;
        if d.stress_catastrophe_bias == 0.0 || !mt.stress_seen {
            return d.proba_shrink;
        }
        let mut bias = d.stress_catastrophe_bias;
        if d.influence_ref > 0.0 {
            bias *= (mt.stress_mag / d.influence_ref).min(1.0);
        }
        (d.proba_shrink * (1.0 - bias * mt.stress_align)).max(0.0)
    }

    fn severing_probability(&self, e: &Element) -> f64 {
        let d = &self.d;
        if d.stress_severing_bias == 0.0 || !e.stress_seen {
            return d.proba_crossmicro_cut;
        }
        let p = d.proba_crossmicro_cut * (1.0 + d.stress_severing_bias * (1.0 - 2.0 * e.stress_align as f64));
        p.clamp(0.0, 1.0)
    }

    // ------------------------------------------------------------------
    // One step

    /// `MicrotubulePool::evolve`: severing, detachment, catastrophe and
    /// rescue, nucleation, then every microtubule grows or shrinks.
    pub fn evolve(&mut self) -> Result<()> {
        self.compact_lists();

        // Severing at sharp bends.
        let list = self.tocut.clone();
        for (idx, gen, _) in list {
            let p = self.rng.unit();
            if p < self.d.proba_tocut && self.pool.is_alive((idx, gen)) {
                let mt = self.pool.elems[idx as usize].mt;
                if mt != NONE {
                    self.sever(mt, idx);
                }
            }
        }

        // Severing at crossovers.
        let list = self.crosscut.clone();
        for (idx, gen, tok) in list {
            let e = &self.pool.elems[idx as usize];
            if !(e.alive && e.gen == gen && e.crossover_to_cut && e.cross_token == tok) {
                continue;
            }
            let p = self.rng.unit();
            if p < self.severing_probability(e) {
                let crossed = e.cross_micro;
                if self.pool.is_alive(crossed) {
                    let mt = e.mt;
                    if mt != NONE {
                        self.sever(mt, idx);
                    }
                    self.unlist_crossover(idx);
                    self.pool.elems[idx as usize].cross_micro = (NONE, 0);
                    self.counters.cut_microtubule += 1;
                } else {
                    self.unlist_crossover(idx);
                    self.pool.elems[idx as usize].cross_micro = (NONE, 0);
                    self.counters.override_cut_microtubule += 1;
                }
            }
        }

        // Detachment from the nucleation site.
        let ids: Vec<u32> = self.mts.keys().copied().collect();
        for &id in &ids {
            let p = self.rng.unit();
            if p < self.d.proba_detach {
                let t = self.time;
                let mt = self.mts.get_mut(&id).unwrap();
                if mt.gr_st_moins != -1 {
                    mt.gr_st_moins = -1;
                    mt.treadmilling_start = t;
                    self.counters.detachements += 1;
                }
            }
        }

        // Catastrophe and rescue.
        let beyond = self.d.proba_shrink_beyond;
        for &id in &ids {
            let p = self.rng.unit();
            let mt = &self.mts[&id];
            let (p_cat, cat_ok) = if beyond < 0.0 {
                (self.shrink_probability(mt), mt.gr_st_plus == 1)
            } else if mt.shrink_beyond == 1 {
                (beyond, mt.gr_st_plus != -1)
            } else {
                (self.d.proba_shrink, mt.gr_st_plus == 1)
            };
            let rescue = self.d.proba_rescue;
            let mt = self.mts.get_mut(&id).unwrap();
            if p < p_cat && cat_ok {
                mt.gr_st_plus = -1;
                self.counters.shrinkages += 1;
            } else if p < rescue && mt.gr_st_plus == -1 {
                mt.gr_st_plus = 1;
            }
        }

        // Nucleation.
        let cen = self.params.get_d("Initialisation_Volume_Radius");
        let n_obj = self.membrane.surfaces.len();
        if self.params.get_i("Sep_Nucleations") == 1 {
            for j in 1..=n_obj {
                let (rate, every) = if j == 1 {
                    (
                        self.params.get_d("proba_initialisation_par_step"),
                        self.params.get_d("Everywhere_proba_initialisation_par_step"),
                    )
                } else {
                    (
                        self.params.get_d(&format!("proba_initialisation_par_step_{j}")),
                        self.params
                            .get_d(&format!("Everywhere_proba_initialisation_par_step_{j}")),
                    )
                };
                self.initiate_midway(rate, j as i32, 0, -10.0)?;
                if every > 0.0 {
                    self.initiate_midway(every, j as i32, 1, -10.0)?;
                }
            }
        } else {
            let cont = if n_obj == 1 {
                1
            } else if cen > 0.0 {
                n_obj as i32 + 1
            } else {
                -1
            };
            // The C++ passes the radius through an int parameter.
            let cen_i = cen.trunc();
            let rate = self.params.get_d("proba_initialisation_par_step");
            self.initiate_midway(rate, cont, 0, cen_i)?;
            let every = self.params.get_d("Everywhere_proba_initialisation_par_step");
            if every > 0.0 {
                self.initiate_midway(every, cont, 1, cen_i)?;
            }
        }

        // Growth and shrinkage.
        let ids: Vec<u32> = self.mts.keys().copied().collect();
        for id in ids {
            let Some(mut mt) = self.mts.remove(&id) else { continue };
            if self.evolve_mt(&mut mt) {
                self.mts.insert(id, mt);
            } else {
                self.counters.destruction_microtubule += 1;
            }
        }
        Ok(())
    }

    /// Advance the cellulose synthase layer by one step, if enabled.
    pub fn step_csc(&mut self) {
        if let Some(c) = self.csc.as_mut() {
            c.step(&self.membrane, &self.pool, &mut self.rng, self.time);
        }
    }

    /// `Microtubule::evolve`. Returns false if the microtubule vanished.
    fn evolve_mt(&mut self, mt: &mut Microtubule) -> bool {
        mt.age += 1;
        if mt.gr_st_plus == 1 {
            self.grow_plus(mt);
        }
        // Not `else`: an encounter during growth can switch the plus end to
        // shrinking, and it then shrinks in the same step, as in the C++.
        if mt.gr_st_plus == -1 && !self.shrink_plus(mt) {
            return false;
        }
        if mt.gr_st_moins == -1 {
            let f = self.d.treadmilling_frequency.max(1);
            if f == 1 || (self.time - mt.treadmilling_start) % f == 0 {
                return self.shrink_moins(mt);
            }
        }
        true
    }

    // ------------------------------------------------------------------
    // Neighbour searches

    /// Nearest other microtubule element within the bundling radius,
    /// excluding this filament's last ten subunits (`NN_Mt` + `testRecursif`).
    fn nn_mt(&self, plus: u32) -> Option<(f64, u32)> {
        let elems = &self.pool.elems;
        let self_ok = self.d.self_encounters;
        let own = elems[plus as usize].mt;
        let p = elems[plus as usize].pos;
        let mut ancestors = [NONE; 10];
        let mut a = elems[plus as usize].prev;
        for slot in ancestors.iter_mut() {
            if a == NONE {
                break;
            }
            *slot = a;
            a = elems[a as usize].prev;
        }
        let r2 = self.d.r_mt * self.d.r_mt;
        let mut best = (f64::INFINITY, NONE);
        self.pool.grid.for_each_27(p, |id| {
            if id == plus || (!self_ok && elems[id as usize].mt == own) {
                return;
            }
            let d = (elems[id as usize].pos - p).length_squared();
            if d < best.0 && d < r2 && !ancestors.contains(&id) {
                best = (d, id);
            }
        });
        if best.1 == NONE {
            None
        } else {
            Some((best.0.sqrt(), best.1))
        }
    }

    // ------------------------------------------------------------------
    // Growth

    fn new_element(&mut self, mt: &mut Microtubule, pos: DVec3, dir: DVec3, prev: u32) -> u32 {
        let mut e = Element::new(pos, dir, self.time as u32);
        e.prev = prev;
        e.mt = mt.id;
        let idx = self.pool.alloc(e);
        mt.body.push_back(idx);
        idx
    }

    fn check_beyond(&self, mt: &mut Microtubule, pos: DVec3) {
        if self.d.proba_shrink_beyond >= 0.0 && (pos.x > self.d.xpos_beyond || pos.x < self.d.xpos_beyond_minus) {
            mt.gr_st_plus = 0;
            mt.shrink_beyond = 1;
        }
    }

    /// `tag_tocut`: a sharp bend makes the previous subunit a severing site.
    fn tag_tocut(&mut self, idx: u32) {
        if self.d.decision_cut == 0 {
            return;
        }
        let e = &self.pool.elems[idx as usize];
        if e.prev == NONE {
            return;
        }
        let pe = &self.pool.elems[e.prev as usize];
        let a = angle(e.dir, pe.dir);
        if a >= self.d.angle_cut && !pe.to_cut && !e.to_cut {
            let p = e.prev;
            self.enlist_tocut(p);
        }
    }

    fn grow_plus(&mut self, mt: &mut Microtubule) {
        let prev = *mt.body.back().expect("empty microtubule");
        let near_mt = self.nn_mt(prev);
        let near_c = self.membrane.nearest(self.pool.elems[prev as usize].pos);
        let contact_mt = near_mt.filter(|&(_, e)| e != prev && self.pool.elems[e as usize].age != 0);
        if let Some(n) = contact_mt {
            self.grow_contact_neighbour(mt, prev, n, near_c);
        } else if let Some(c) = near_c.filter(|c| c.0 < 100.0) {
            self.grow_contact_mb(mt, prev, c);
        } else {
            self.grow_free(mt, prev);
        }
        self.pool.elems[prev as usize].age = 1;
    }

    /// The plus end meets another filament: bundle with it at a shallow
    /// angle, otherwise undergo catastrophe or cross over it.
    fn grow_contact_neighbour(
        &mut self,
        mt: &mut Microtubule,
        prev: u32,
        (_, el): (f64, u32),
        near_c: Option<(f64, u32)>,
    ) {
        self.pool.elems[el as usize].contact = true;
        if self.pool.elems[el as usize].mt == mt.id {
            self.counters.self_encounters += 1;
        }
        let (s1, anc) = {
            let p = &self.pool.elems[prev as usize];
            (p.dir, p.pos)
        };
        let s2 = self.pool.elems[el as usize].dir;
        let a = angle(s1, s2);
        let proba = self.rng.unit();
        let icp = self.d.icpower;
        let icpf = icp as f64;
        let b = 2f64.powf(icpf - 2.0) * PI.powf(1.0 - icpf);
        let ab = self.d.angle_bundle;
        let shallow = a < ab;
        let anti = a > PI - ab;
        let bundles = if icp == 0 {
            shallow || anti
        } else {
            (shallow && proba / a.powf(icpf) > b) || (anti && proba / (PI - a).powf(icpf) > b)
        };
        if bundles {
            if !self.pool.elems[prev as usize].contact {
                self.counters.encounter_bundle_grow += 1;
            }
            let dir = if shallow { s2 } else { -s2 };
            let pos = anc + s1;
            let src = self.pool.elems[el as usize].clone();
            let idx = self.new_element(mt, pos, dir, prev);
            {
                let n = &mut self.pool.elems[idx as usize];
                n.contact = true;
                n.cortical_structure = src.cortical_structure;
                n.nearest_contour = src.nearest_contour;
                n.nearest_contour_d = src.nearest_contour_d;
                n.d_lim = src.d_lim;
                n.cortical = src.cortical;
            }
            self.tag_tocut(idx);
            self.check_beyond(mt, pos);
            return;
        }

        // Steep encounter: catastrophe, or cross over and mark the crossover.
        let cross_lim = self.d.cutgroup as i64;
        let pprev = self.pool.elems[prev as usize].prev;
        let cn = if pprev != NONE && cross_lim != 0 {
            (self.pool.elems[pprev as usize].cross_num as i64 % cross_lim) as i32 + 1
        } else {
            1
        };
        self.pool.elems[prev as usize].cross_num = cn;
        let group_ok = cross_lim == 0 || (cross_lim >= 1 && cn as f64 == self.d.cutgroup);
        let shrink = if icp == 0 {
            proba < self.d.proba_crossmicro_shrink && group_ok
        } else {
            let steep = if a < std::f64::consts::FRAC_PI_2 {
                proba / a.powf(icpf) - b < -10000.0
            } else {
                proba / (PI - a).powf(icpf) - b < -10000.0
            };
            steep && group_ok
        };
        if shrink {
            self.counters.encounter_bundle_shrink += 1;
            mt.gr_st_plus = -1;
            return;
        }
        match near_c.filter(|c| c.0 < 100.0) {
            Some(c) => self.grow_contact_mb(mt, prev, c),
            None => self.grow_free(mt, prev),
        }
        if cross_lim == 0 || cn as i64 == cross_lim {
            self.enlist_crossover(prev);
            let h = self.pool.handle(el);
            self.pool.elems[prev as usize].cross_micro = h;
        }
    }

    /// Growth near the membrane: deflect along it, or stop if the turn the
    /// membrane demands exceeds `Angle_mb_limite`.
    fn grow_contact_mb(&mut self, mt: &mut Microtubule, plus: u32, (dist, wall0): (f64, u32)) {
        let d = self.d;
        let (dir_prev, pos_prev, plus_cortical) = {
            let e = &self.pool.elems[plus as usize];
            (e.dir, e.pos, e.cortical)
        };
        let mut min_dist = dist;
        let mut wall = wall0;
        self.membrane.verts[wall as usize].contact = true;
        let mut n = self.membrane.verts[wall as usize].normal;
        let pos_mb = self.membrane.verts[wall as usize].pos;
        let pos_dir = pos_prev + dir_prev;
        let mut gap = (pos_mb - pos_dir).dot(n) - d.d_mb;

        // With several surfaces, prefer one whose limit the tip is about to
        // cross while it is still clear of this one.
        if self.membrane.surfaces.len() > 1 {
            let mut all = std::mem::take(&mut self.scratch_nn);
            self.membrane.nearest_per_surface(pos_prev, &mut all);
            let mut chosen = None;
            for &entry in all.iter() {
                if let Some((dd, v)) = entry {
                    if dd < 100.0 {
                        let g = self.membrane.gap(pos_dir, v, d.d_mb);
                        if g < d.cortical_thickness && gap > d.cortical_thickness {
                            chosen = Some((dd, v, g));
                        }
                    }
                }
            }
            if let Some((dd, v, g)) = chosen {
                min_dist = dd;
                wall = v;
                self.membrane.verts[v as usize].contact = true;
                n = self.membrane.verts[v as usize].normal;
                gap = g;
            }
            self.scratch_nn = all;
        }

        let influence = self.membrane.verts[wall as usize].strain;
        let attach = self.membrane.verts[wall as usize].attach == 1.0;
        let dot = dir_prev.dot(n);
        let dir_prev_perp = normalise(n - dot * dir_prev);
        let n_perp = normalise(dir_prev - dot * n);

        let mut tag_cortical = false;
        let mut dir_cand;
        if gap < 0.0 || d.decision_accrochage != 0 || (attach && plus_cortical) {
            tag_cortical = true;
            let (s, c) = d.angle_mb_traj.sin_cos();
            let extr1 = dir_prev_perp * s + dir_prev * c;
            let extr2 = -dir_prev_perp * s + dir_prev * c;
            let p1 = extr1.dot(n);
            let p2 = extr2.dot(n);
            if (p1 > gap && p2 < gap) || (p1 < gap && p2 > gap) {
                let perp = (1.0 - gap * gap).sqrt();
                dir_cand = gap * n + perp * n_perp;
            } else if (p1 - gap).abs() < (p2 - gap).abs() {
                dir_cand = extr1;
            } else {
                dir_cand = extr2;
            }
            dir_cand = normalise(dir_cand);
        } else if gap < d.cortical_thickness && gap > 0.0 && dot > 0.0 {
            tag_cortical = true;
            dir_cand = normalise(n_perp);
        } else {
            dir_cand = normalise(dir_prev);
        }

        let r = DVec3::new(self.rng.signed_unit(), self.rng.signed_unit(), self.rng.signed_unit());
        dir_cand = normalise(d.fixe * dir_cand + d.alea * r);

        if has_nan(dir_cand) {
            // Growing straight into the membrane: turn in a random direction
            // along it. (The C++ only did this when the projections were
            // NaN and otherwise let the NaN through; any NaN is handled here.)
            let mut v = DVec3::new(self.rng.signed_unit(), self.rng.signed_unit(), self.rng.signed_unit());
            v -= v.dot(n) * n;
            dir_cand = v.normalize_or(n.any_orthonormal_vector());
        }

        let mut stress_here: Option<(f64, f64)> = None;
        if let Some(infl) = influence {
            if d.influence_normale > 0 && infl.length_squared() != 0.0 {
                let f1 = d.influence_normale as f64;
                let mut f2 = d.influence_weight;
                let iv = infl - infl.dot(n) * n;
                let proj = iv.length_squared();
                if proj > 0.0 {
                    let sens = iv.dot(dir_cand);
                    let iv = iv / proj.sqrt();
                    if d.influence_ref > 0.0 {
                        f2 *= (proj.sqrt() / d.influence_ref).min(1.0);
                    }
                    let ca = iv.dot(dir_cand);
                    mt.stress_align = ca * ca;
                    mt.stress_seen = true;
                    mt.stress_mag = proj.sqrt();
                    stress_here = Some((ca * ca, proj.sqrt()));
                    let v = if sens > 0.0 {
                        f1 * dir_cand + f2 * iv
                    } else {
                        f1 * dir_cand - f2 * iv
                    };
                    let v = normalise(v);
                    if !has_nan(v) {
                        dir_cand = v;
                    }
                }
            }
        }

        if angle(dir_cand, dir_prev) > d.angle_mb_lim {
            if mt.gr_st_plus as i64 != d.decision_rencontre {
                self.counters.encounter_mb_shrink += 1;
            }
            mt.gr_st_plus = d.decision_rencontre as i32;
        }

        let pos = pos_prev + dir_prev;
        let idx = self.new_element(mt, pos, dir_cand, plus);
        let surface = self.membrane.verts[wall as usize].surface as i32;
        {
            let e = &mut self.pool.elems[idx as usize];
            e.nearest_contour = wall;
            e.nearest_contour_d = min_dist as f32;
            if let Some((a, m)) = stress_here {
                e.stress_align = a as f32;
                e.stress_mag = m as f32;
                e.stress_seen = true;
            }
            e.d_lim = gap as f32;
            e.cortical = tag_cortical;
            if tag_cortical {
                e.cortical_structure = surface;
            }
        }
        self.check_beyond(mt, pos);
        if !plus_cortical {
            self.counters.encounter_mb_grow += 1;
        }
        if plus_cortical && !tag_cortical {
            self.counters.left_membrane += 1;
        }
        self.tag_tocut(idx);
    }

    /// Free growth in the cytoplasm, with no membrane in range.
    fn grow_free(&mut self, mt: &mut Microtubule, plus: u32) {
        let (dir, pos, cortical) = {
            let e = &self.pool.elems[plus as usize];
            (e.dir, e.pos, e.cortical)
        };
        let r = DVec3::new(self.rng.signed_unit(), self.rng.signed_unit(), self.rng.signed_unit());
        let mut nd = normalise(self.d.fixe * dir + self.d.alea * r);
        if has_nan(nd) {
            nd = dir;
        }
        let p = pos + dir;
        self.new_element(mt, p, nd, plus);
        if cortical {
            self.counters.left_membrane += 1;
        }
        self.check_beyond(mt, p);
    }

    // ------------------------------------------------------------------
    // Shrinkage and severing

    /// Remove `ShrinkingMultiple` subunits from the plus end. False if the
    /// microtubule is gone.
    fn shrink_plus(&mut self, mt: &mut Microtubule) -> bool {
        for _ in 0..self.d.shrinking_multiple {
            if let Some(i) = mt.body.pop_back() {
                self.pool.kill(i);
            }
        }
        !mt.body.is_empty()
    }

    /// Remove one subunit from the minus end. False if the microtubule is gone.
    fn shrink_moins(&mut self, mt: &mut Microtubule) -> bool {
        if let Some(i) = mt.body.pop_front() {
            if let Some(&f) = mt.body.front() {
                self.pool.elems[f as usize].prev = NONE;
            }
            self.pool.kill(i);
        }
        !mt.body.is_empty()
    }

    /// Cut microtubule `mt_id` at `elem`: the part behind it keeps the id and
    /// starts shrinking from its new plus end; the part from `elem` on becomes
    /// a new microtubule that keeps growing and depolymerises from its new
    /// minus end.
    pub fn sever(&mut self, mt_id: u32, elem: u32) {
        let Some(mut mt) = self.mts.remove(&mt_id) else { return };
        let ok = mt.body.len() >= 3 && mt.body.front() != Some(&elem);
        if ok {
            if let Some(k) = mt.body.iter().position(|&e| e == elem) {
                self.pool.elems[elem as usize].prev = NONE;
                let second: VecDeque<u32> = mt.body.split_off(k);
                mt.gr_st_plus = -1;
                if !second.is_empty() {
                    let id = self.new_mt_id();
                    for &e in &second {
                        self.pool.elems[e as usize].mt = id;
                    }
                    let mut nm = Microtubule::new(id);
                    nm.body = second;
                    nm.shrink_beyond = 1;
                    nm.age = mt.age;
                    nm.gr_st_moins = -1;
                    self.mts.insert(id, nm);
                }
            }
        }
        self.mts.insert(mt_id, mt);
    }

    // ------------------------------------------------------------------
    // Observation

    pub fn n_elements(&self) -> usize {
        self.pool.live
    }

    pub fn lengths(&self) -> Vec<usize> {
        self.mts.values().map(|m| m.body.len()).collect()
    }

    /// Fraction of subunits marked cortical.
    pub fn cortical_fraction(&self) -> f64 {
        let (mut c, mut n) = (0usize, 0usize);
        for m in self.mts.values() {
            for &e in &m.body {
                n += 1;
                if self.pool.elems[e as usize].cortical {
                    c += 1;
                }
            }
        }
        if n == 0 {
            0.0
        } else {
            c as f64 / n as f64
        }
    }
}
