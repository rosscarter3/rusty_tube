//! The confining surfaces: a port of tubulaton's `Contour`.
//!
//! The membrane is represented, as in tubulaton, by its mesh vertices with
//! outward unit normals; a growing tip looks for the nearest vertex in the
//! 27 grid cells around it. That is what couples correctness to mesh
//! resolution, so [`Membrane::load`] refuses meshes whose covering radius
//! exceeds the grid cell.

use super::grid::Grid;
use crate::mesh::TriMesh;
use crate::params::Params;
use crate::rng::Rng;
use crate::vtk;
use anyhow::{bail, Context, Result};
use glam::DVec3;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct Vertex {
    pub pos: DVec3,
    pub normal: DVec3,
    /// External field steering microtubules (`strain_0`), if the mesh has it.
    pub strain: Option<DVec3>,
    /// The mesh's `attachement` scalar (1 = attach).
    pub attach: f64,
    /// 1-based surface number (`Structure_Number`).
    pub surface: u16,
    /// Index of this point in its input file.
    pub vtk_id: u32,
    /// Set once any microtubule has touched it.
    pub contact: bool,
}

#[derive(Clone, Debug)]
pub struct Surface {
    pub name: String,
    pub mesh: TriMesh,
    /// Global vertex ids of this surface, in file order.
    pub vertices: Vec<u32>,
    pub area: f64,
}

#[derive(Clone, Debug)]
pub struct Membrane {
    pub verts: Vec<Vertex>,
    pub surfaces: Vec<Surface>,
    pub grid: Grid,
    /// Nucleation sites per surface (or one combined list) and the cumulative
    /// area behind each, so nucleation is uniform per unit area rather than
    /// per vertex.
    nucleation: Vec<Vec<u32>>,
    nucleation_cdf: Vec<Vec<f64>>,
    pub bounds: (DVec3, DVec3),
}

/// The geometry the grid is sized from, see [`Membrane::load`].
pub struct GridSetup {
    pub resolution: f64,
    pub min: DVec3,
}

impl Membrane {
    fn input_path(params: &Params, file: &str) -> PathBuf {
        let folder = params.get_s("nom_folder_input_vtk");
        PathBuf::from(format!("{folder}{file}"))
    }

    /// Read every input surface named in the config. `preloaded` short-cuts
    /// the read for surface 0 (used when the geometry was generated in
    /// memory).
    pub fn load(params: &mut Params, preloaded: Option<(TriMesh, TriMesh)>) -> Result<Self> {
        let n_surf = params.get_i("num_input_vtk").max(1) as usize;
        let mut meshes: Vec<(String, TriMesh, TriMesh)> = Vec::new();
        let mut pre = preloaded;
        for obj in 0..n_surf {
            let file = params.get_s(&Params::surface_key("nom_input_vtk", obj)).to_string();
            let ref_file = params.get_s(&Params::surface_key("nom_input_vtk_ref", obj)).to_string();
            if obj == 0 {
                if let Some((m, r)) = pre.take() {
                    meshes.push((if file.is_empty() { "generated".into() } else { file }, m, r));
                    continue;
                }
            }
            let path = Self::input_path(params, &file);
            let mesh = vtk::read_polydata(&path).with_context(|| format!("surface {}", obj + 1))?;
            let ref_mesh = if ref_file.is_empty() || ref_file == file {
                mesh.clone()
            } else {
                vtk::read_polydata(Self::input_path(params, &ref_file))
                    .with_context(|| format!("reference surface {}", obj + 1))?
            };
            meshes.push((file, mesh, ref_mesh));
        }

        // Bounds: from the config if Bound_Ind >= 0, else from the meshes.
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        if params.get_i("Bound_Ind") >= 0 {
            lo = DVec3::new(
                params.get_d("Bound_Xmin"),
                params.get_d("Bound_Ymin"),
                params.get_d("Bound_Zmin"),
            );
            hi = DVec3::new(
                params.get_d("Bound_Xmax"),
                params.get_d("Bound_Ymax"),
                params.get_d("Bound_Zmax"),
            );
        } else {
            for (_, m, _) in &meshes {
                let (a, b) = m.bounds();
                lo = lo.min(a);
                hi = hi.max(b);
            }
        }
        if params.get_d("Initialisation_Volume_Radius") > 0.0 {
            let c = (lo + hi) * 0.5;
            params.set_d("CentreX", c.x);
            params.set_d("CentreY", c.y);
            params.set_d("CentreZ", c.z);
        }

        let grid = Self::grid_setup(params, lo, hi);
        println!(
            "Bounds: x [{:.1}, {:.1}]  y [{:.1}, {:.1}]  z [{:.1}, {:.1}]  grid cell {} units",
            lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, grid.resolution
        );

        let mut m = Membrane {
            verts: Vec::new(),
            surfaces: Vec::new(),
            grid: Grid::new(grid.resolution, grid.min),
            nucleation: Vec::new(),
            nucleation_cdf: Vec::new(),
            bounds: (lo, hi),
        };
        let sep = params.get_i("Sep_Nucleations") == 1;
        let mut nuc: Vec<u32> = Vec::new();
        let mut cdf: Vec<f64> = Vec::new();
        let mut total = 0.0;

        for (obj, (name, mesh, ref_mesh)) in meshes.into_iter().enumerate() {
            if sep {
                nuc.clear();
                cdf.clear();
                total = 0.0;
            }
            let area = mesh.area();
            let ref_area = ref_mesh.area();
            println!(
                "surface {} ({name}): area {area:.1}, reference area {ref_area:.1}",
                obj + 1
            );
            let k_step = Params::surface_key("proba_initialisation_par_step", obj);
            let k_every = Params::surface_key("Everywhere_proba_initialisation_par_step", obj);
            let k_area = Params::surface_key("proba_initialisation_area", obj);
            let k_area_every = Params::surface_key("Everywhere_proba_initialisation_area", obj);
            let init_type = params.get_i(&Params::surface_key("Type_Input", obj));
            // Normalise nucleation to the reference surface, or convert a
            // per-area rate (per um^2 per s, at ~5 steps per second).
            match init_type {
                0 => {
                    let r = area / ref_area;
                    params.set_d(&k_step, params.get_d(&k_step) * r);
                    params.set_d(&k_every, params.get_d(&k_every) * r);
                }
                1 => {
                    let conv = area * 8.0 * 8.0 / 1000.0 / 1000.0 / 5.0;
                    params.set_d(&k_step, params.get_d(&k_area) * conv);
                    params.set_d(&k_every, params.get_d(&k_area_every) * conv);
                }
                _ => bail!("Type_Input must be 0 or 1"),
            }

            let res = mesh.vertex_areas();
            Self::check_resolution(params, &m.grid, res.covering_radius, res.longest_edge, &name)?;
            let normals: Vec<DVec3> = match mesh.point_vectors.get("Normals") {
                Some(n) if n.len() >= mesh.positions.len() => n.iter().map(|v| v.normalize_or_zero()).collect(),
                _ => {
                    println!(
                        "surface {}: no normals in the file; computing them from the surface",
                        obj + 1
                    );
                    mesh.compute_normals()
                }
            };
            let strain = mesh.point_vectors.get("strain_0");
            let attach = mesh.point_scalars.get("attachement");
            let nucleation_mask = mesh.point_scalars.get("Nucleation");
            if nucleation_mask.is_some() {
                println!("surface {}: nucleation has spatial dependence", obj + 1);
            }
            if mesh.positions.iter().any(|p| p.min_element() < 0.0) {
                eprintln!("warning: surface {} has vertices with negative coordinates", obj + 1);
            }
            let mut ids = Vec::with_capacity(mesh.positions.len());
            let mut n_nuc_here = 0usize;
            for (i, p) in mesh.positions.iter().enumerate() {
                let id = m.verts.len() as u32;
                m.verts.push(Vertex {
                    pos: *p,
                    normal: normals[i],
                    strain: strain.map(|s| s[i]),
                    attach: attach.map(|a| a[i]).unwrap_or(0.0),
                    surface: (obj + 1) as u16,
                    vtk_id: i as u32,
                    contact: false,
                });
                m.grid.insert(*p, id);
                ids.push(id);
                let here = nucleation_mask.map(|v| v[i] == 1.0).unwrap_or(true);
                if here {
                    nuc.push(id);
                    total += res.vertex_area[i];
                    cdf.push(total);
                    n_nuc_here += 1;
                }
            }
            if init_type == 1 {
                let v = params.get_d(&k_step) * n_nuc_here as f64 / mesh.positions.len() as f64;
                params.set_d(&k_step, v);
                println!("Initiation probability: {v}");
            }
            m.surfaces.push(Surface {
                name,
                mesh,
                vertices: ids,
                area,
            });
            if sep {
                m.nucleation.push(nuc.clone());
                m.nucleation_cdf.push(cdf.clone());
            }
        }
        if !sep {
            m.nucleation.push(nuc);
            m.nucleation_cdf.push(cdf);
        }
        Ok(m)
    }

    /// Port of the ElementPool constructor and `spaceContourRegisterLimits`:
    /// the cell resolution may be raised so the packed grid covers the box.
    fn grid_setup(params: &Params, _lo: DVec3, hi: DVec3) -> GridSetup {
        let mut res = params.get_i("m_hashContour_resolution") as f64;
        let mut sizes = [256.0f64; 3];
        if params.get_i("Bound_Ind") == 1 {
            let span = [
                params.get_d("Bound_Xmax") - params.get_d("Bound_Xmin"),
                params.get_d("Bound_Ymax") - params.get_d("Bound_Ymin"),
                params.get_d("Bound_Zmax") - params.get_d("Bound_Zmin"),
            ];
            for k in 0..3 {
                let n = (span[k] / res) as i64 + 1;
                sizes[k] = 2f64.powi((n as f64).log2().ceil() as i32);
            }
        }
        let limit = sizes.iter().cloned().fold(0.0, f64::max);
        let speed = if params.get_d("Max_Movement_Speed") > 0.0 {
            params.get_d("Max_Movement_Speed")
        } else {
            params.get_d("Movement_Speed")
        };
        let max_lim = hi.x.max(hi.y).max(hi.z);
        let needed = ((max_lim + 1000.0 + speed * params.get_i("nb_max_steps") as f64) / limit).ceil();
        res = res.max(needed);
        GridSetup {
            resolution: res,
            min: DVec3::new(
                params.get_d("Bound_Xmin"),
                params.get_d("Bound_Ymin"),
                params.get_d("Bound_Zmin"),
            ),
        }
    }

    fn check_resolution(params: &Params, grid: &Grid, covering: f64, longest: f64, name: &str) -> Result<()> {
        println!(
            "Mesh {name}: longest edge {longest:.1} units, covering radius {covering:.1} units, grid cell {} units",
            grid.res
        );
        if covering > grid.res {
            bail!(
                "mesh {name} is too coarse for the spatial grid: covering radius {covering:.1} units \
                 exceeds the grid cell of {} units. A microtubule only searches the 27 grid cells \
                 around it for the membrane, so over the coarse parts of this surface it would grow \
                 straight through the wall. Refine the mesh or raise m_hashContour_resolution.",
                grid.res
            );
        }
        let bundle = params.get_d("D_bundle") / params.get_d("taille_microtubule");
        if bundle > grid.res {
            bail!(
                "the bundling cutoff D_bundle/taille_microtubule = {bundle} units exceeds the grid \
                 cell of {} units; raise m_hashContour_resolution",
                grid.res
            );
        }
        Ok(())
    }

    /// Nearest vertex in the 27 cells around `p`: tubulaton's `NN_Contour`.
    /// `None` when those cells are empty; otherwise the distance and vertex.
    #[inline]
    pub fn nearest(&self, p: DVec3) -> Option<(f64, u32)> {
        let mut best = (f64::INFINITY, u32::MAX);
        self.grid.for_each_27(p, |id| {
            let d = (self.verts[id as usize].pos - p).length_squared();
            if d < best.0 {
                best = (d, id);
            }
        });
        if best.1 == u32::MAX {
            None
        } else {
            Some((best.0.sqrt(), best.1))
        }
    }

    /// Nearest vertex on each surface: `NN_AllContourOutput`.
    pub fn nearest_per_surface(&self, p: DVec3, out: &mut Vec<Option<(f64, u32)>>) {
        out.clear();
        out.resize(self.surfaces.len(), None);
        let mut best = vec![(f64::INFINITY, u32::MAX); self.surfaces.len()];
        self.grid.for_each_27(p, |id| {
            let v = &self.verts[id as usize];
            let s = v.surface as usize - 1;
            let d = (v.pos - p).length_squared();
            if d < best[s].0 {
                best[s] = (d, id);
            }
        });
        for (s, b) in best.into_iter().enumerate() {
            if b.1 != u32::MAX {
                out[s] = Some((b.0.sqrt(), b.1));
            }
        }
    }

    /// Distance from `pos` to the limit plane of a vertex: positive inside,
    /// beyond `d_mb` of the membrane. `CalcGap`.
    #[inline]
    pub fn gap(&self, pos: DVec3, vertex: u32, d_mb: f64) -> f64 {
        let v = &self.verts[vertex as usize];
        (v.pos - pos).dot(v.normal) - d_mb
    }

    /// A nucleation site and direction (`AdaptElement`): 10 units inside the
    /// membrane, pointing either tangent to it or straight in.
    fn adapt(&self, params: &Params, rng: &mut Rng, vertex: u32, cont_num: usize) -> Result<(DVec3, DVec3)> {
        let v = &self.verts[vertex as usize];
        let pos = v.pos - 10.0 * v.normal;
        let mut dir = DVec3::new(rng.signed_unit(), rng.signed_unit(), rng.signed_unit()).normalize_or_zero();
        let key = Params::surface_key("Nucleation_Direction", cont_num);
        match params.get_i(&key) {
            0 => dir -= dir.dot(v.normal) * v.normal,
            1 => dir = -v.normal,
            other => bail!("{key} should be 0 (tangential) or 1 (normal); got {other}"),
        }
        let dir = dir.normalize_or_zero();
        Ok((
            pos,
            if dir == DVec3::ZERO {
                v.normal.any_orthonormal_vector()
            } else {
                dir
            },
        ))
    }

    /// A site drawn uniformly over the vertices of surface `i` (all surfaces
    /// if `None`): `getElementRandom`.
    pub fn random_site(&self, params: &Params, rng: &mut Rng, i: Option<usize>) -> Result<Option<(DVec3, DVec3)>> {
        let (list, cont): (&[u32], usize) = match i {
            Some(s) => (&self.surfaces[s].vertices, s),
            None => {
                let n = self.verts.len();
                if n == 0 {
                    return Ok(None);
                }
                let v = rng.index(n) as u32;
                return self.adapt(params, rng, v, 0).map(Some);
            }
        };
        if list.is_empty() {
            return Ok(None);
        }
        let v = list[rng.index(list.len())];
        self.adapt(params, rng, v, cont).map(Some)
    }

    /// A nucleation site drawn per unit area: `getElementRandomNucleation`.
    pub fn nucleation_site(&self, params: &Params, rng: &mut Rng, i: usize) -> Result<Option<(DVec3, DVec3)>> {
        let Some(list) = self.nucleation.get(i) else {
            return Ok(None);
        };
        if list.is_empty() {
            return Ok(None);
        }
        let cdf = &self.nucleation_cdf[i];
        let choice = if cdf.len() == list.len() && *cdf.last().unwrap() > 0.0 {
            let target = rng.unit() * cdf.last().unwrap();
            cdf.partition_point(|&c| c < target).min(list.len() - 1)
        } else {
            rng.index(list.len())
        };
        self.adapt(params, rng, list[choice], i).map(Some)
    }

    pub fn total_area(&self) -> f64 {
        self.surfaces.iter().map(|s| s.area).sum()
    }
}
