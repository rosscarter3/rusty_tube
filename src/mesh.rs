//! Closed triangulated surfaces: the cell membrane (and any internal surface
//! such as a nucleus) that confines the microtubules.

use glam::DVec3;
use std::collections::BTreeMap;

#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub positions: Vec<DVec3>,
    pub triangles: Vec<[u32; 3]>,
    /// One-component point arrays, e.g. `attachement`, `Nucleation`.
    pub point_scalars: BTreeMap<String, Vec<f64>>,
    /// Three-component point arrays, e.g. `strain_0`, `Normals`.
    pub point_vectors: BTreeMap<String, Vec<DVec3>>,
}

/// Per-vertex area and resolution statistics, see [`TriMesh::vertex_areas`].
pub struct MeshResolution {
    /// One third of the area of each incident triangle.
    pub vertex_area: Vec<f64>,
    /// The largest distance any point on the surface can be from its nearest
    /// vertex. tubulaton sees the membrane only through the vertices, so this
    /// is the resolution that matters.
    pub covering_radius: f64,
    pub longest_edge: f64,
}

impl TriMesh {
    pub fn area(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.positions[i as usize]);
                0.5 * (b - a).cross(c - a).length()
            })
            .sum()
    }

    pub fn bounds(&self) -> (DVec3, DVec3) {
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for p in &self.positions {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        (lo, hi)
    }

    /// Signed enclosed volume; positive when triangles wind outward.
    pub fn signed_volume(&self) -> f64 {
        self.triangles
            .iter()
            .map(|t| {
                let [a, b, c] = t.map(|i| self.positions[i as usize]);
                a.dot(b.cross(c)) / 6.0
            })
            .sum()
    }

    /// Flip every triangle if the surface winds inward.
    pub fn orient_outward(&mut self) {
        if self.signed_volume() < 0.0 {
            for t in &mut self.triangles {
                t.swap(1, 2);
            }
        }
    }

    /// Angle-weighted vertex normals, pointing out of the enclosed volume.
    pub fn compute_normals(&self) -> Vec<DVec3> {
        let mut n = vec![DVec3::ZERO; self.positions.len()];
        for t in &self.triangles {
            let p = t.map(|i| self.positions[i as usize]);
            let face = (p[1] - p[0]).cross(p[2] - p[0]);
            if face.length_squared() == 0.0 {
                continue;
            }
            let face = face.normalize();
            for k in 0..3 {
                let e1 = (p[(k + 1) % 3] - p[k]).normalize_or_zero();
                let e2 = (p[(k + 2) % 3] - p[k]).normalize_or_zero();
                let angle = e1.dot(e2).clamp(-1.0, 1.0).acos();
                n[t[k] as usize] += face * angle;
            }
        }
        let flip = if self.signed_volume() < 0.0 { -1.0 } else { 1.0 };
        n.into_iter().map(|v| v.normalize_or_zero() * flip).collect()
    }

    pub fn vertex_areas(&self) -> MeshResolution {
        let mut area = vec![0.0; self.positions.len()];
        let mut covering: f64 = 0.0;
        let mut longest: f64 = 0.0;
        for t in &self.triangles {
            let p = t.map(|i| self.positions[i as usize]);
            let e = [(p[1] - p[2]).length(), (p[2] - p[0]).length(), (p[0] - p[1]).length()];
            longest = longest.max(e[0]).max(e[1]).max(e[2]);
            let a = 0.5 * (p[1] - p[0]).cross(p[2] - p[0]).length();
            for &i in t {
                area[i as usize] += a / 3.0;
            }
            let l = e[0].max(e[1]).max(e[2]);
            let s = e.map(|x| x * x);
            let smax = s[0].max(s[1]).max(s[2]);
            let obtuse = smax >= s[0] + s[1] + s[2] - smax;
            let r = if !obtuse && a > 1e-12 {
                e[0] * e[1] * e[2] / (4.0 * a)
            } else {
                l / 2.0
            };
            covering = covering.max(r);
        }
        MeshResolution {
            vertex_area: area,
            covering_radius: covering,
            longest_edge: longest,
        }
    }

    pub fn translate(&mut self, d: DVec3) {
        for p in &mut self.positions {
            *p += d;
        }
    }

    /// Euler characteristic V - E + F; 2 for a closed sphere-like surface.
    pub fn euler_characteristic(&self) -> i64 {
        let mut edges = rustc_hash::FxHashSet::default();
        for t in &self.triangles {
            for k in 0..3 {
                let (a, b) = (t[k], t[(k + 1) % 3]);
                edges.insert((a.min(b), a.max(b)));
            }
        }
        self.positions.len() as i64 - edges.len() as i64 + self.triangles.len() as i64
    }
}
