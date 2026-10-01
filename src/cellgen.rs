//! Pavement-cell geometry: a lobed outline turned into a closed 3D surface.
//!
//! The outline comes from a simulated tissue (pavesim's `movie.json`), from a
//! CSV of x,y in micrometres, or is generated procedurally: an irregular
//! polygon whose walls carry interdigitating lobes, the jigsaw shape of a leaf
//! epidermal pavement cell.
//!
//! The 3D cell is built as a signed distance field -- the outline extruded to
//! the anticlinal depth, with the periclinal/anticlinal junction rounded and
//! the outer periclinal face domed -- and meshed by surface nets. Unlike an
//! extruded prism with ear-clipped caps, this gives near-uniform vertex
//! spacing everywhere, which tubulaton needs twice over: the membrane is seen
//! only through mesh vertices (so the covering radius must be small
//! everywhere), and smooth exact normals come straight from the field's
//! gradient.
//!
//! Units: one mesh unit is one tubulin subunit, 8 nm, so 1 um = 125 units.

use crate::mesh::TriMesh;
use crate::rng::Rng;
use anyhow::{bail, Context, Result};
use glam::{DVec2, DVec3};
use rustc_hash::FxHashMap;

pub const UNITS_PER_UM: f64 = 125.0;

#[derive(Clone, Debug)]
pub struct CellSpec {
    /// Anticlinal depth of the cell (um).
    pub height_um: f64,
    /// Radius of the rounded junction between the periclinal faces and the
    /// anticlinal wall (um).
    pub edge_radius_um: f64,
    /// How far the outer periclinal face domes above the rim (um).
    pub dome_um: f64,
    /// Surface-nets grid spacing, i.e. vertex spacing (um).
    pub spacing_um: f64,
    /// Clearance from the coordinate origin (units); tubulaton wants the
    /// mesh in the positive octant.
    pub margin_units: f64,
    /// Attach an imposed neck-high stress field as `strain_0`.
    pub neck_stress: bool,
    /// Attach the shape-derived (inflated-membrane) stress field as
    /// `strain_0`, with magnitude equal to the local stress anisotropy.
    /// Takes precedence over `neck_stress`.
    pub shape_stress: bool,
}

impl Default for CellSpec {
    fn default() -> Self {
        CellSpec {
            height_um: 5.0,
            edge_radius_um: 1.2,
            dome_um: 1.0,
            spacing_um: 0.3,
            margin_units: 1500.0,
            neck_stress: false,
            shape_stress: false,
        }
    }
}

/// A closed outline in micrometres, counter-clockwise.
#[derive(Clone, Debug)]
pub struct Outline {
    pub pts: Vec<DVec2>,
}

impl Outline {
    pub fn signed_area(&self) -> f64 {
        let n = self.pts.len();
        (0..n).map(|i| self.pts[i].perp_dot(self.pts[(i + 1) % n])).sum::<f64>() * 0.5
    }

    pub fn make_ccw(&mut self) {
        if self.signed_area() < 0.0 {
            self.pts.reverse();
        }
    }

    pub fn perimeter(&self) -> f64 {
        let n = self.pts.len();
        (0..n).map(|i| (self.pts[(i + 1) % n] - self.pts[i]).length()).sum()
    }

    pub fn centroid(&self) -> DVec2 {
        let n = self.pts.len();
        let mut c = DVec2::ZERO;
        let mut a = 0.0;
        for i in 0..n {
            let (p, q) = (self.pts[i], self.pts[(i + 1) % n]);
            let w = p.perp_dot(q);
            a += w;
            c += (p + q) * w;
        }
        c / (3.0 * a)
    }

    /// Resample at uniform arc length.
    pub fn resample(&self, spacing: f64) -> Outline {
        let n = self.pts.len();
        let seg: Vec<f64> = (0..n).map(|i| (self.pts[(i + 1) % n] - self.pts[i]).length()).collect();
        let total: f64 = seg.iter().sum();
        let count = ((total / spacing).round() as usize).max(12);
        let step = total / count as f64;
        let (mut out, mut target, mut acc, mut i) = (Vec::with_capacity(count), 0.0, 0.0, 0);
        for _ in 0..count {
            while acc + seg[i] < target - 1e-12 {
                acc += seg[i];
                i = (i + 1) % n;
            }
            let t = if seg[i] > 0.0 { (target - acc) / seg[i] } else { 0.0 };
            out.push(self.pts[i].lerp(self.pts[(i + 1) % n], t));
            target += step;
        }
        Outline { pts: out }
    }

    /// Cyclic Laplacian smoothing.
    pub fn smooth(&self, passes: usize) -> Outline {
        let mut p = self.pts.clone();
        let n = p.len();
        for _ in 0..passes {
            p = (0..n)
                .map(|i| p[(i + n - 1) % n] * 0.25 + p[i] * 0.5 + p[(i + 1) % n] * 0.25)
                .collect();
        }
        Outline { pts: p }
    }

    /// Signed curvature (1/um), positive on convex wall (lobes), negative on
    /// concave wall (necks), for a CCW outline.
    pub fn curvature(&self) -> Vec<f64> {
        let n = self.pts.len();
        (0..n)
            .map(|i| {
                let (a, b, c) = (self.pts[(i + n - 1) % n], self.pts[i], self.pts[(i + 1) % n]);
                let (u, v) = (b - a, c - b);
                let (lu, lv) = (u.length(), v.length());
                if lu <= 0.0 || lv <= 0.0 {
                    return 0.0;
                }
                let cr = (u.perp_dot(v) / (lu * lv)).clamp(-1.0, 1.0);
                cr.asin() / (0.5 * (lu + lv))
            })
            .collect()
    }

    /// Gaussian smoothing of a per-vertex quantity along arc length.
    pub fn smooth_along(&self, values: &[f64], sigma_um: f64) -> Vec<f64> {
        let n = self.pts.len();
        let mut s = vec![0.0; n];
        for i in 1..n {
            s[i] = s[i - 1] + (self.pts[i] - self.pts[i - 1]).length();
        }
        let total = self.perimeter();
        (0..n)
            .map(|i| {
                let (mut num, mut den) = (0.0, 0.0);
                for j in 0..n {
                    let mut d = (s[j] - s[i]).abs();
                    d = d.min(total - d);
                    if d > 3.0 * sigma_um {
                        continue;
                    }
                    let w = (-0.5 * (d / sigma_um).powi(2)).exp();
                    num += w * values[j];
                    den += w;
                }
                num / den
            })
            .collect()
    }

    pub fn tangents(&self) -> Vec<DVec2> {
        let n = self.pts.len();
        (0..n)
            .map(|i| (self.pts[(i + 1) % n] - self.pts[(i + n - 1) % n]).normalize_or_zero())
            .collect()
    }

    pub fn self_intersects(&self) -> bool {
        let n = self.pts.len();
        let seg_x = |a: DVec2, b: DVec2, c: DVec2, d: DVec2| {
            let d1 = (b - a).perp_dot(c - a);
            let d2 = (b - a).perp_dot(d - a);
            let d3 = (d - c).perp_dot(a - c);
            let d4 = (d - c).perp_dot(b - c);
            (d1 > 0.0) != (d2 > 0.0) && (d3 > 0.0) != (d4 > 0.0)
        };
        for i in 0..n {
            for j in i + 2..n {
                if i == 0 && j == n - 1 {
                    continue;
                }
                if seg_x(self.pts[i], self.pts[(i + 1) % n], self.pts[j], self.pts[(j + 1) % n]) {
                    return true;
                }
            }
        }
        false
    }

    /// Load the outline written by pavesim's `cell_to_vtk.py` (`movie.json`
    /// and friends): `{"outline": [{"x_um":..,"y_um":..}, ...]}`.
    pub fn from_pavesim_json(path: &std::path::Path) -> Result<Outline> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text)?;
        let arr = v["outline"].as_array().context("no \"outline\" array")?;
        let pts = arr
            .iter()
            .map(|p| DVec2::new(p["x_um"].as_f64().unwrap_or(0.0), p["y_um"].as_f64().unwrap_or(0.0)))
            .collect();
        let mut o = Outline { pts };
        o.make_ccw();
        Ok(o)
    }

    /// Two columns, x and y in micrometres; `#` comments and a header allowed.
    pub fn from_csv(path: &std::path::Path) -> Result<Outline> {
        let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let mut pts = Vec::new();
        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            let mut it = line
                .split(|c: char| c == ',' || c.is_whitespace())
                .filter(|s| !s.is_empty());
            if let (Some(a), Some(b)) = (it.next(), it.next()) {
                if let (Ok(x), Ok(y)) = (a.parse::<f64>(), b.parse::<f64>()) {
                    pts.push(DVec2::new(x, y));
                }
            }
        }
        if pts.len() < 3 {
            bail!("outline {} has fewer than three points", path.display());
        }
        let mut o = Outline { pts };
        o.make_ccw();
        Ok(o)
    }

    /// A procedural pavement cell: a central body with six to nine curving
    /// arms, each pinched into a neck at its base and swelling into a lobe
    /// at its tip, some of them branching -- the jigsaw shape of a leaf
    /// epidermal pavement cell. Built as a smooth union of discs and
    /// capsules in a 2D distance field and contoured. `size_um` is the
    /// rough diameter.
    pub fn procedural(seed: u64, size_um: f64) -> Outline {
        let mut rng = Rng::new(seed.max(1) as i64);
        let s = size_um / 22.0;
        let mut shapes: Vec<(DVec2, DVec2, f64)> = Vec::new(); // capsules (a, b, r); discs have a == b
        let body = (3.2 + 1.6 * rng.unit()) * s;
        shapes.push((
            DVec2::ZERO,
            DVec2::new(0.8 * rng.signed_unit() * s, 0.8 * rng.signed_unit() * s),
            body,
        ));
        let n = 6 + rng.index(4);
        let phase = rng.unit() * std::f64::consts::TAU;
        for k in 0..n {
            let theta = phase + (k as f64 + 0.3 * rng.signed_unit()) * std::f64::consts::TAU / n as f64;
            let dir = DVec2::new(theta.cos(), theta.sin());
            let perp = DVec2::new(-dir.y, dir.x);
            let len = (5.5 + 4.0 * rng.unit()) * s;
            let neck = (0.85 + 0.45 * rng.unit()) * s;
            let tip_r = (1.5 + 0.8 * rng.unit()) * s;
            let mid = dir * len * 0.55 + perp * 1.4 * rng.signed_unit() * s;
            let end = dir * len + perp * 1.2 * rng.signed_unit() * s;
            shapes.push((DVec2::ZERO, mid, neck));
            shapes.push((mid, end, neck * 1.15));
            shapes.push((end, end, tip_r));
            if rng.unit() < 0.4 {
                // A side lobe off the arm.
                let side = if rng.unit() < 0.5 { 1.0 } else { -1.0 };
                let bdir = (dir * 0.6 + perp * side).normalize();
                let bend = end + bdir * (2.0 + 1.6 * rng.unit()) * s;
                shapes.push((end, bend, neck));
                shapes.push((bend, bend, tip_r * 0.85));
            }
        }
        let sdf = |p: DVec2| -> f64 {
            // Polynomial smooth minimum: fillets the necks.
            let k = 0.9 * s;
            let mut d = f64::INFINITY;
            for &(a, b, r) in &shapes {
                let ab = b - a;
                let t = if ab.length_squared() > 0.0 {
                    ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let di = (p - (a + ab * t)).length() - r;
                if d.is_infinite() {
                    d = di;
                } else {
                    let h = (0.5 + 0.5 * (di - d) / k).clamp(0.0, 1.0);
                    d = di * (1.0 - h) + d * h - k * h * (1.0 - h);
                }
            }
            d
        };
        let o = contour_zero(&sdf, size_um * 0.75, 0.08 * s.max(0.5));
        let mut o = o.resample(0.15);
        o.make_ccw();
        o
    }
}

/// The longest closed zero contour of a 2D field over [-half, half]^2, by
/// marching squares.
fn contour_zero(f: &dyn Fn(DVec2) -> f64, half: f64, h: f64) -> Outline {
    let n = (2.0 * half / h).ceil() as usize + 1;
    let at = |i: usize, j: usize| DVec2::new(-half + i as f64 * h, -half + j as f64 * h);
    let v: Vec<f64> = (0..n * n).map(|k| f(at(k % n, k / n))).collect();
    let val = |i: usize, j: usize| v[j * n + i];
    // Edge ids: horizontal edge (i,j)-(i+1,j) = 2*(j*n+i), vertical (i,j)-(i,j+1) = 2*(j*n+i)+1.
    let point = |e: usize| -> DVec2 {
        let k = e / 2;
        let (i, j) = (k % n, k / n);
        let (a, b, pa, pb) = if e.is_multiple_of(2) {
            (val(i, j), val(i + 1, j), at(i, j), at(i + 1, j))
        } else {
            (val(i, j), val(i, j + 1), at(i, j), at(i, j + 1))
        };
        pa.lerp(pb, a / (a - b))
    };
    let mut adj: FxHashMap<usize, Vec<usize>> = FxHashMap::default();
    let mut link = |a: usize, b: usize| {
        adj.entry(a).or_default().push(b);
        adj.entry(b).or_default().push(a);
    };
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let c = [val(i, j), val(i + 1, j), val(i + 1, j + 1), val(i, j + 1)];
            let e = [
                2 * (j * n + i),
                2 * (j * n + i + 1) + 1,
                2 * ((j + 1) * n + i),
                2 * (j * n + i) + 1,
            ];
            let mut crossing = Vec::new();
            for k in 0..4 {
                if (c[k] < 0.0) != (c[(k + 1) % 4] < 0.0) {
                    crossing.push(e[k]);
                }
            }
            match crossing.len() {
                2 => link(crossing[0], crossing[1]),
                4 => {
                    // Saddle: decide by the centre value.
                    let centre = c.iter().sum::<f64>() / 4.0;
                    if (centre < 0.0) == (c[0] < 0.0) {
                        link(crossing[0], crossing[3]);
                        link(crossing[1], crossing[2]);
                    } else {
                        link(crossing[0], crossing[1]);
                        link(crossing[2], crossing[3]);
                    }
                }
                _ => {}
            }
        }
    }
    let mut seen = rustc_hash::FxHashSet::default();
    let mut best: Vec<DVec2> = Vec::new();
    let keys: Vec<usize> = adj.keys().copied().collect();
    for start in keys {
        if seen.contains(&start) {
            continue;
        }
        let mut loop_pts = Vec::new();
        let (mut prev, mut cur) = (usize::MAX, start);
        loop {
            seen.insert(cur);
            loop_pts.push(point(cur));
            let next = adj[&cur].iter().copied().find(|&x| x != prev && !seen.contains(&x));
            match next {
                Some(nx) => {
                    prev = cur;
                    cur = nx;
                }
                None => break,
            }
        }
        if loop_pts.len() > best.len() {
            best = loop_pts;
        }
    }
    Outline { pts: best }
}

/// Signed distance to a closed polygon (negative inside).
struct PolySdf {
    a: Vec<DVec2>,
    b: Vec<DVec2>,
    // Uniform grid of segment ids for fast queries.
    lo: DVec2,
    cell: f64,
    dims: (usize, usize),
    bins: Vec<Vec<u32>>,
}

impl PolySdf {
    fn new(o: &Outline) -> Self {
        let n = o.pts.len();
        let a: Vec<DVec2> = o.pts.clone();
        let b: Vec<DVec2> = (0..n).map(|i| o.pts[(i + 1) % n]).collect();
        let mut lo = DVec2::splat(f64::INFINITY);
        let mut hi = DVec2::splat(f64::NEG_INFINITY);
        for p in &a {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        let cell = 1.0;
        lo -= DVec2::splat(cell);
        hi += DVec2::splat(cell);
        let dims = (
            ((hi.x - lo.x) / cell).ceil() as usize + 1,
            ((hi.y - lo.y) / cell).ceil() as usize + 1,
        );
        let mut bins = vec![Vec::new(); dims.0 * dims.1];
        for i in 0..n {
            let (smin, smax) = (a[i].min(b[i]), a[i].max(b[i]));
            let (x0, y0) = (((smin.x - lo.x) / cell) as usize, ((smin.y - lo.y) / cell) as usize);
            let (x1, y1) = (((smax.x - lo.x) / cell) as usize, ((smax.y - lo.y) / cell) as usize);
            for y in y0..=y1.min(dims.1 - 1) {
                for x in x0..=x1.min(dims.0 - 1) {
                    bins[y * dims.0 + x].push(i as u32);
                }
            }
        }
        PolySdf {
            a,
            b,
            lo,
            cell,
            dims,
            bins,
        }
    }

    fn seg_dist2(&self, i: usize, p: DVec2) -> f64 {
        let (a, b) = (self.a[i], self.b[i]);
        let ab = b - a;
        let t = ((p - a).dot(ab) / ab.length_squared().max(1e-30)).clamp(0.0, 1.0);
        (p - (a + ab * t)).length_squared()
    }

    fn inside(&self, p: DVec2) -> bool {
        let mut c = false;
        for i in 0..self.a.len() {
            let (a, b) = (self.a[i], self.b[i]);
            if (a.y > p.y) != (b.y > p.y) {
                let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
                if p.x < x {
                    c = !c;
                }
            }
        }
        c
    }

    /// Unsigned distance and the nearest segment.
    fn nearest(&self, p: DVec2) -> (f64, usize) {
        let gx = ((p.x - self.lo.x) / self.cell).floor() as i64;
        let gy = ((p.y - self.lo.y) / self.cell).floor() as i64;
        let mut best = (f64::INFINITY, 0usize);
        let mut ring = 0i64;
        loop {
            for y in gy - ring..=gy + ring {
                for x in gx - ring..=gx + ring {
                    if (y - gy).abs() != ring && (x - gx).abs() != ring {
                        continue;
                    }
                    if x < 0 || y < 0 || x >= self.dims.0 as i64 || y >= self.dims.1 as i64 {
                        continue;
                    }
                    for &i in &self.bins[y as usize * self.dims.0 + x as usize] {
                        let d = self.seg_dist2(i as usize, p);
                        if d < best.0 {
                            best = (d, i as usize);
                        }
                    }
                }
            }
            // Anything outside the searched square is at least ring*cell away.
            let reach = ring as f64 * self.cell;
            if best.0.is_finite() && best.0.sqrt() <= reach {
                break;
            }
            ring += 1;
            if ring > (self.dims.0 + self.dims.1) as i64 {
                // Far outside the grid: brute force.
                for i in 0..self.a.len() {
                    let d = self.seg_dist2(i, p);
                    if d < best.0 {
                        best = (d, i);
                    }
                }
                break;
            }
        }
        (best.0.sqrt(), best.1)
    }

    fn signed(&self, p: DVec2) -> f64 {
        let (d, _) = self.nearest(p);
        if self.inside(p) {
            -d
        } else {
            d
        }
    }
}

/// The 3D field: rounded extrusion with a domed top face. Coordinates in um,
/// the base periclinal face at z = 0.
struct CellField<'a> {
    bubble: &'a ShapeStress,
    poly: &'a PolySdf,
    spec: &'a CellSpec,
}

impl CellField<'_> {
    fn eval_with(&self, p: DVec3, d2: f64) -> f64 {
        let s = self.spec;
        let r = s.edge_radius_um.min(0.49 * s.height_um);
        // The outer face domes like an inflated membrane: its height follows
        // the Prandtl solution u (laplacian u = -1, u = 0 on the outline),
        // which is smooth everywhere. A dome built from the distance to the
        // wall would carry a crease along the medial axis of every lobe.
        // Dividing the vertical gap by the slope keeps the field close to a
        // true distance (|grad| ~ 1), which the projection below relies on.
        const U: f64 = 1.0;
        let (u, gu) = self.bubble.u_at(p.truncate());
        // Zero slope at the wall, so the dome does not fight the rounded rim.
        let x = u.max(0.0) / U;
        let e = (-x * x).exp();
        // Fade the dome out near the wall using the exact wall distance: the
        // interpolated u is least accurate there, where it has its steepest
        // gradient, and the rim's rounding expects a flat top. The fade has
        // zero slope at both ends, so it adds no crease.
        const FADE: f64 = 1.4;
        let t = ((-d2) / FADE).clamp(0.0, 1.0);
        let fade = t * t * (3.0 - 2.0 * t);
        let dfade = if t > 0.0 && t < 1.0 {
            6.0 * t * (1.0 - t) / FADE
        } else {
            0.0
        };
        let dome = s.dome_um * (1.0 - e) * fade;
        let slope = s.dome_um * (e * 2.0 * x / U * gu * fade + (1.0 - e) * dfade);
        let top = s.height_um + dome;
        let dz = ((p.z - top) / (1.0 + slope * slope).sqrt()).max(-p.z);
        // Rounded box in (wall distance, height) space.
        let q = DVec2::new(d2, dz) + DVec2::splat(r);
        q.max(DVec2::ZERO).length() + q.x.max(q.y).min(0.0) - r
    }

    fn eval(&self, p: DVec3) -> f64 {
        self.eval_with(p, self.poly.signed(p.truncate()))
    }

    fn gradient(&self, p: DVec3) -> DVec3 {
        let h = 1e-3;
        DVec3::new(
            self.eval(p + DVec3::X * h) - self.eval(p - DVec3::X * h),
            self.eval(p + DVec3::Y * h) - self.eval(p - DVec3::Y * h),
            self.eval(p + DVec3::Z * h) - self.eval(p - DVec3::Z * h),
        )
        .normalize_or_zero()
    }
}

/// A generated cell: the surface in mesh units, its outward normals, and the
/// outline it came from (in um, before the shift to mesh units).
pub struct Cell {
    /// Distance-field value at each vertex after projection (um), for diagnostics.
    pub residual: Vec<f64>,
    pub mesh: TriMesh,
    pub normals: Vec<DVec3>,
    pub outline: Outline,
    /// mesh_units = um * UNITS_PER_UM + shift
    pub shift: DVec3,
}

pub fn build_cell(outline: &Outline, spec: &CellSpec) -> Result<Cell> {
    // Resample finely and smooth (sigma ~0.2 um): segmented outlines are
    // polygons with corners every pixel or so, and their distance field's
    // normal jumps at every corner.
    let mut outline = outline.resample(0.1).smooth(8);
    outline.make_ccw();
    // Centre on the origin in xy.
    let c = outline.centroid();
    for p in &mut outline.pts {
        *p -= c;
    }
    let poly = PolySdf::new(&outline);
    let bubble = ShapeStress::solve(&poly, 0.2);
    let field = CellField {
        bubble: &bubble,
        poly: &poly,
        spec,
    };

    let h = spec.spacing_um;
    let (mut lo2, mut hi2) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for p in &outline.pts {
        lo2 = lo2.min(*p);
        hi2 = hi2.max(*p);
    }
    let pad = 2.0 * h;
    // Irrational offsets keep flat faces off the node planes.
    let lo = DVec3::new(
        lo2.x - pad - 0.381966 * h,
        lo2.y - pad - 0.236068 * h,
        -pad - 0.414214 * h,
    );
    let hi = DVec3::new(hi2.x + pad, hi2.y + pad, spec.height_um + spec.dome_um + pad);
    let dims = [
        ((hi.x - lo.x) / h).ceil() as usize + 1,
        ((hi.y - lo.y) / h).ceil() as usize + 1,
        ((hi.z - lo.z) / h).ceil() as usize + 1,
    ];
    let idx = |x: usize, y: usize, z: usize| (z * dims[1] + y) * dims[0] + x;
    let node = |x: usize, y: usize, z: usize| lo + DVec3::new(x as f64, y as f64, z as f64) * h;

    // Sample the field. The 2D distance is shared by every z in a column.
    use rayon::prelude::*;
    let columns: Vec<f64> = (0..dims[0] * dims[1])
        .into_par_iter()
        .map(|k| {
            let (x, y) = (k % dims[0], k / dims[0]);
            poly.signed(node(x, y, 0).truncate())
        })
        .collect();
    let mut values = vec![0.0f64; dims[0] * dims[1] * dims[2]];
    values
        .par_chunks_mut(dims[0] * dims[1])
        .enumerate()
        .for_each(|(z, slab)| {
            for y in 0..dims[1] {
                for x in 0..dims[0] {
                    let d2 = columns[y * dims[0] + x];
                    let mut v = field.eval_with(node(x, y, z), d2);
                    // Keep the field off exact zero so every crossing is clean.
                    if v == 0.0 {
                        v = 1e-9;
                    }
                    slab[y * dims[0] + x] = v;
                }
            }
        });

    // Surface nets: one vertex per cell with a sign change.
    let mut cell_vertex: FxHashMap<usize, u32> = FxHashMap::default();
    let mut positions: Vec<DVec3> = Vec::new();
    const CORNERS: [[usize; 3]; 8] = [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [1, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
    ];
    const EDGES: [[usize; 2]; 12] = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
    ];
    for z in 0..dims[2] - 1 {
        for y in 0..dims[1] - 1 {
            for x in 0..dims[0] - 1 {
                let v: [f64; 8] = CORNERS.map(|c| values[idx(x + c[0], y + c[1], z + c[2])]);
                let neg = v.iter().filter(|&&a| a < 0.0).count();
                if neg == 0 || neg == 8 {
                    continue;
                }
                let mut acc = DVec3::ZERO;
                let mut cnt = 0.0;
                for e in EDGES {
                    let (a, b) = (v[e[0]], v[e[1]]);
                    if (a < 0.0) != (b < 0.0) {
                        let t = a / (a - b);
                        let pa = node(x + CORNERS[e[0]][0], y + CORNERS[e[0]][1], z + CORNERS[e[0]][2]);
                        let pb = node(x + CORNERS[e[1]][0], y + CORNERS[e[1]][1], z + CORNERS[e[1]][2]);
                        acc += pa.lerp(pb, t);
                        cnt += 1.0;
                    }
                }
                cell_vertex.insert(idx(x, y, z), positions.len() as u32);
                positions.push(acc / cnt);
            }
        }
    }

    // Quads across every sign-changing grid edge.
    let mut triangles: Vec<[u32; 3]> = Vec::new();
    let mut emit = |q: [u32; 4], flip: bool, pos: &Vec<DVec3>| {
        let q = if flip { [q[0], q[3], q[2], q[1]] } else { q };
        // Split along the shorter diagonal.
        let d02 = (pos[q[0] as usize] - pos[q[2] as usize]).length_squared();
        let d13 = (pos[q[1] as usize] - pos[q[3] as usize]).length_squared();
        if d02 <= d13 {
            triangles.push([q[0], q[1], q[2]]);
            triangles.push([q[0], q[2], q[3]]);
        } else {
            triangles.push([q[0], q[1], q[3]]);
            triangles.push([q[1], q[2], q[3]]);
        }
    };
    for z in 1..dims[2] - 1 {
        for y in 1..dims[1] - 1 {
            for x in 1..dims[0] - 1 {
                let v0 = values[idx(x, y, z)];
                // +x edge: cells sharing it vary in y and z.
                if x + 1 < dims[0] {
                    let v1 = values[idx(x + 1, y, z)];
                    if (v0 < 0.0) != (v1 < 0.0) {
                        let q = [idx(x, y - 1, z - 1), idx(x, y, z - 1), idx(x, y, z), idx(x, y - 1, z)];
                        if let Some(q) = lookup(&cell_vertex, q) {
                            emit(q, v0 >= 0.0, &positions);
                        }
                    }
                }
                if y + 1 < dims[1] {
                    let v1 = values[idx(x, y + 1, z)];
                    if (v0 < 0.0) != (v1 < 0.0) {
                        let q = [idx(x - 1, y, z - 1), idx(x - 1, y, z), idx(x, y, z), idx(x, y, z - 1)];
                        if let Some(q) = lookup(&cell_vertex, q) {
                            emit(q, v0 >= 0.0, &positions);
                        }
                    }
                }
                if z + 1 < dims[2] {
                    let v1 = values[idx(x, y, z + 1)];
                    if (v0 < 0.0) != (v1 < 0.0) {
                        let q = [idx(x - 1, y - 1, z), idx(x, y - 1, z), idx(x, y, z), idx(x - 1, y, z)];
                        if let Some(q) = lookup(&cell_vertex, q) {
                            emit(q, v0 >= 0.0, &positions);
                        }
                    }
                }
            }
        }
    }

    // Relax the surface-nets vertices along the surface and project them
    // onto the zero set, then take normals from the field's gradient.
    let project = |p0: DVec3| -> DVec3 {
        let mut p = p0;
        for _ in 0..8 {
            let f = field.eval(p);
            if f.abs() < 1e-5 {
                break;
            }
            let g = field.gradient(p);
            if g == DVec3::ZERO {
                break;
            }
            let step = (g * f).clamp_length_max(0.5 * h);
            p -= step;
        }
        p
    };
    let mut neighbours: Vec<Vec<u32>> = vec![Vec::new(); positions.len()];
    for t in &triangles {
        for k in 0..3 {
            let (a, b) = (t[k], t[(k + 1) % 3]);
            if !neighbours[a as usize].contains(&b) {
                neighbours[a as usize].push(b);
            }
            if !neighbours[b as usize].contains(&a) {
                neighbours[b as usize].push(a);
            }
        }
    }
    let mut positions: Vec<DVec3> = positions.par_iter().map(|&p| project(p)).collect();
    for _ in 0..4 {
        let relaxed: Vec<DVec3> = (0..positions.len())
            .into_par_iter()
            .map(|i| {
                let nb = &neighbours[i];
                if nb.is_empty() {
                    return positions[i];
                }
                let avg = nb.iter().map(|&j| positions[j as usize]).sum::<DVec3>() / nb.len() as f64;
                let n = field.gradient(positions[i]);
                let d = avg - positions[i];
                project(positions[i] + 0.6 * (d - d.dot(n) * n))
            })
            .collect();
        positions = relaxed;
    }
    let projected: Vec<(DVec3, DVec3)> = positions.par_iter().map(|&p| (p, field.gradient(p))).collect();
    let residual: Vec<f64> = positions.par_iter().map(|&p| field.eval(p)).collect();

    let mut mesh = TriMesh {
        positions: projected.iter().map(|(p, _)| *p).collect(),
        triangles,
        ..Default::default()
    };
    // Drop degenerate triangles.
    mesh.triangles.retain(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]);
    mesh.orient_outward();
    let mut normals: Vec<DVec3> = projected.iter().map(|(_, n)| *n).collect();
    let mesh_normals = mesh.compute_normals();
    for (n, m) in normals.iter_mut().zip(&mesh_normals) {
        if *n == DVec3::ZERO || n.dot(*m) < 0.0 {
            *n = *m;
        }
    }

    // Per-vertex outline curvature and the optional imposed stress field.
    let ccw = outline.clone();
    let kappa = ccw.smooth_along(&ccw.curvature(), 2.0);
    let tang = ccw.tangents();
    let neck: Vec<f64> = {
        let raw: Vec<f64> = kappa.iter().map(|k| (-k).max(0.0)).collect();
        let m = raw.iter().cloned().fold(0.0, f64::max).max(1e-12);
        raw.iter().map(|r| r / m).collect()
    };
    let mut kappa_v = Vec::with_capacity(mesh.positions.len());
    let mut wall_v = Vec::with_capacity(mesh.positions.len());
    let mut strain = Vec::with_capacity(mesh.positions.len());
    let rim = spec.edge_radius_um + 0.3;
    for p in &mesh.positions {
        let (d, seg) = poly.nearest(p.truncate());
        kappa_v.push(kappa[seg]);
        let on_wall = d < rim;
        wall_v.push(if on_wall { 1.0 } else { 0.0 });
        let t = tang[seg] * neck[seg];
        strain.push(if on_wall {
            DVec3::new(t.x, t.y, 0.0)
        } else {
            DVec3::ZERO
        });
    }
    mesh.point_scalars.insert("kappa".into(), kappa_v);
    mesh.point_scalars.insert("anticlinal".into(), wall_v);
    if spec.shape_stress {
        let field = &bubble;
        let mut strain = Vec::with_capacity(mesh.positions.len());
        let mut aniso = Vec::with_capacity(mesh.positions.len());
        for p in &mesh.positions {
            let (dir, a) = field.at(p.truncate());
            // In-plane principal direction plus a vertical part: the
            // membrane projection keeps the first on the periclinal faces
            // and the second on the anticlinal wall, so a neck's stress runs
            // across the face and down the wall like a hoop.
            strain.push(DVec3::new(dir.x, dir.y, 1.0) * a);
            aniso.push(a);
        }
        mesh.point_vectors.insert("strain_0".into(), strain);
        mesh.point_scalars.insert("stress_anisotropy".into(), aniso);
    } else if spec.neck_stress {
        mesh.point_vectors.insert("strain_0".into(), strain);
    }

    // To mesh units, in the positive octant.
    let (lo_um, _) = mesh.bounds();
    let shift = DVec3::splat(spec.margin_units) - lo_um * UNITS_PER_UM;
    for p in &mut mesh.positions {
        *p = *p * UNITS_PER_UM + shift;
    }
    Ok(Cell {
        mesh,
        normals,
        outline,
        shift,
        residual,
    })
}

/// Stress proxy for the periclinal face: a flat membrane inflated by turgor,
/// solved as the Prandtl problem laplacian(u) = -1 inside the outline with
/// u = 0 on it. The Hessian of u carries the shape of the stress pattern:
/// isotropic in a round lobe (u_xx = u_yy for a disc), fully anisotropic
/// across a narrow neck (u_xx = -1, u_yy = 0 for a strip), with the larger
/// principal stress across the neck, as hoop stress exceeds axial stress in
/// a pressurised cylinder.
struct ShapeStress {
    lo: DVec2,
    h: f64,
    nx: usize,
    ny: usize,
    /// Smoothed Hessian (uxx, uyy, uxy) per node.
    hess: Vec<[f64; 3]>,
    /// The solution itself, for the dome.
    u: Vec<f64>,
}

impl ShapeStress {
    fn solve(poly: &PolySdf, h: f64) -> Self {
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        for p in &poly.a {
            lo = lo.min(*p);
            hi = hi.max(*p);
        }
        lo -= DVec2::splat(2.0 * h);
        hi += DVec2::splat(2.0 * h);
        let nx = ((hi.x - lo.x) / h).ceil() as usize + 1;
        let ny = ((hi.y - lo.y) / h).ceil() as usize + 1;
        let idx = |x: usize, y: usize| y * nx + x;
        let inside: Vec<bool> = (0..nx * ny)
            .map(|k| poly.inside(lo + DVec2::new((k % nx) as f64, (k / nx) as f64) * h))
            .collect();
        // Red-black SOR.
        let mut u = vec![0.0f64; nx * ny];
        let omega = 1.92;
        for _ in 0..4000 {
            let mut change: f64 = 0.0;
            for colour in 0..2 {
                for y in 1..ny - 1 {
                    for x in 1..nx - 1 {
                        if (x + y) % 2 != colour || !inside[idx(x, y)] {
                            continue;
                        }
                        let k = idx(x, y);
                        let g = (u[k - 1] + u[k + 1] + u[k - nx] + u[k + nx] + h * h) * 0.25;
                        let d = omega * (g - u[k]);
                        u[k] += d;
                        change = change.max(d.abs());
                    }
                }
            }
            if change < 1e-9 {
                break;
            }
        }
        let mut hess = vec![[0.0; 3]; nx * ny];
        for y in 1..ny - 1 {
            for x in 1..nx - 1 {
                let k = idx(x, y);
                let uxx = (u[k - 1] - 2.0 * u[k] + u[k + 1]) / (h * h);
                let uyy = (u[k - nx] - 2.0 * u[k] + u[k + nx]) / (h * h);
                let uxy = (u[k + nx + 1] - u[k + nx - 1] - u[k - nx + 1] + u[k - nx - 1]) / (4.0 * h * h);
                hess[k] = [uxx, uyy, uxy];
            }
        }
        if std::env::var_os("STRESS_DEBUG").is_some() {
            let tr: Vec<f64> = (0..nx * ny)
                .filter(|&k| inside[k])
                .map(|k| hess[k][0] + hess[k][1])
                .collect();
            let mut s = tr.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!(
                "grid {nx}x{ny}, trace p10 {:.3} p50 {:.3} p90 {:.3}, umax {:.4}",
                s[s.len() / 10],
                s[s.len() / 2],
                s[s.len() * 9 / 10],
                u.iter().cloned().fold(0.0, f64::max)
            );
        }
        // Light Gaussian smoothing (sigma ~0.8 um) of the inside nodes, so
        // the field is free of grid noise; boundary nodes copy the nearest
        // smoothed interior value through the averaging.
        let r = (1.6 / h).ceil() as i64;
        let sigma = 0.8 / h;
        let smooth: Vec<[f64; 3]> = (0..nx * ny)
            .map(|k| {
                let (x, y) = ((k % nx) as i64, (k / nx) as i64);
                let mut acc = [0.0; 3];
                let mut wsum = 0.0;
                for dy in -r..=r {
                    for dx in -r..=r {
                        let (xx, yy) = (x + dx, y + dy);
                        if xx < 1 || yy < 1 || xx >= nx as i64 - 1 || yy >= ny as i64 - 1 {
                            continue;
                        }
                        let kk = idx(xx as usize, yy as usize);
                        if !inside[kk] {
                            continue;
                        }
                        let w = (-((dx * dx + dy * dy) as f64) / (2.0 * sigma * sigma)).exp();
                        for c in 0..3 {
                            acc[c] += w * hess[kk][c];
                        }
                        wsum += w;
                    }
                }
                if wsum > 0.0 {
                    acc.map(|v| v / wsum)
                } else {
                    [-0.5, -0.5, 0.0]
                }
            })
            .collect();
        ShapeStress {
            lo,
            h,
            nx,
            ny,
            hess: smooth,
            u,
        }
    }

    /// u and |grad u| at a point, by Catmull-Rom bicubic interpolation
    /// (C1, so the dome has smooth normals).
    fn u_at(&self, p: DVec2) -> (f64, f64) {
        let f = |q: DVec2| -> f64 {
            let g = (q - self.lo) / self.h;
            let (ix, iy) = (g.x.floor() as i64, g.y.floor() as i64);
            let (tx, ty) = (g.x - ix as f64, g.y - iy as f64);
            let w = |t: f64| {
                [
                    0.5 * (-t * t * t + 2.0 * t * t - t),
                    0.5 * (3.0 * t * t * t - 5.0 * t * t + 2.0),
                    0.5 * (-3.0 * t * t * t + 4.0 * t * t + t),
                    0.5 * (t * t * t - t * t),
                ]
            };
            let (wx, wy) = (w(tx), w(ty));
            let mut acc = 0.0;
            for (j, wyj) in wy.iter().enumerate() {
                for (i, wxi) in wx.iter().enumerate() {
                    let x = ix + i as i64 - 1;
                    let y = iy + j as i64 - 1;
                    if x < 0 || y < 0 || x >= self.nx as i64 || y >= self.ny as i64 {
                        continue;
                    }
                    acc += wxi * wyj * self.u[y as usize * self.nx + x as usize];
                }
            }
            acc
        };
        let e = 1e-3;
        let gx = (f(p + DVec2::X * e) - f(p - DVec2::X * e)) / (2.0 * e);
        let gy = (f(p + DVec2::Y * e) - f(p - DVec2::Y * e)) / (2.0 * e);
        (f(p), (gx * gx + gy * gy).sqrt())
    }

    /// Principal stress direction and anisotropy in [0, 1] at a point.
    fn at(&self, p: DVec2) -> (DVec2, f64) {
        let g = (p - self.lo) / self.h;
        let x0 = (g.x.floor().max(0.0) as usize).min(self.nx - 2);
        let y0 = (g.y.floor().max(0.0) as usize).min(self.ny - 2);
        let (fx, fy) = ((g.x - x0 as f64).clamp(0.0, 1.0), (g.y - y0 as f64).clamp(0.0, 1.0));
        let mut m = [0.0; 3];
        for (dx, dy, w) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let k = (y0 + dy) * self.nx + x0 + dx;
            for c in 0..3 {
                m[c] += w * self.hess[k][c];
            }
        }
        let [uxx, uyy, uxy] = m;
        let trace = uxx + uyy;
        let diff = ((uxx - uyy).powi(2) + 4.0 * uxy * uxy).sqrt();
        let _ = trace;
        // Bounded anisotropy: 0 for equal principal values, 1 for a strip or
        // a saddle (opposite signs, as at a neck).
        let l1 = 0.5 * (uxx + uyy) + 0.5 * diff;
        let l2 = 0.5 * (uxx + uyy) - 0.5 * diff;
        let aniso = if l1.abs() + l2.abs() > 1e-9 {
            (l1 - l2).abs() / (l1.abs() + l2.abs())
        } else {
            0.0
        };
        // Eigenvector of the most negative eigenvalue: the direction in
        // which the inflated face curves most, i.e. carries most tension.
        let theta = 0.5 * (2.0 * uxy).atan2(uxx - uyy) + std::f64::consts::FRAC_PI_2;
        (DVec2::new(theta.cos(), theta.sin()), aniso)
    }
}

fn lookup(map: &FxHashMap<usize, u32>, q: [usize; 4]) -> Option<[u32; 4]> {
    Some([*map.get(&q[0])?, *map.get(&q[1])?, *map.get(&q[2])?, *map.get(&q[3])?])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_cell_is_closed_and_fine() {
        let o = Outline::procedural(3, 20.0);
        assert!(!o.self_intersects());
        let spec = CellSpec {
            spacing_um: 0.5,
            ..Default::default()
        };
        let cell = build_cell(&o, &spec).unwrap();
        assert_eq!(cell.mesh.euler_characteristic(), 2, "surface should be a closed sphere");
        let res = cell.mesh.vertex_areas();
        assert!(res.covering_radius < 0.6 * UNITS_PER_UM);
        assert!(cell.mesh.signed_volume() > 0.0);
        let (lo, _) = cell.mesh.bounds();
        assert!(lo.min_element() > 0.0);
    }
}

#[cfg(test)]
mod stress_tests {
    use super::*;

    fn disc() -> Outline {
        Outline {
            pts: (0..400)
                .map(|k| {
                    let t = k as f64 / 400.0 * std::f64::consts::TAU;
                    DVec2::new(t.cos(), t.sin()) * 6.0
                })
                .collect(),
        }
    }

    #[test]
    fn bubble_matches_disc_solution() {
        let poly = PolySdf::new(&disc());
        let f = ShapeStress::solve(&poly, 0.2);
        for r in [0.0, 3.0, 5.0, 5.9] {
            let (u, g) = f.u_at(DVec2::new(r, 0.0));
            let exact = (36.0 - r * r) / 4.0;
            eprintln!("r {r}: u {u:.3} exact {exact:.3} |grad| {g:.3} exact {:.3}", r / 2.0);
        }
    }

    #[test]
    fn disc_is_isotropic_strip_is_not() {
        let poly = PolySdf::new(&disc());
        let f = ShapeStress::solve(&poly, 0.2);
        let (_, a0) = f.at(DVec2::ZERO);
        let (_, a1) = f.at(DVec2::new(3.0, 0.0));
        eprintln!("disc centre {a0:.3}, half radius {a1:.3}");
        assert!(a0 < 0.15 && a1 < 0.2);
        let strip = Outline {
            pts: vec![
                DVec2::new(-10.0, -1.5),
                DVec2::new(10.0, -1.5),
                DVec2::new(10.0, 1.5),
                DVec2::new(-10.0, 1.5),
            ],
        }
        .resample(0.1);
        let poly = PolySdf::new(&strip);
        let f = ShapeStress::solve(&poly, 0.2);
        let (d, a) = f.at(DVec2::ZERO);
        eprintln!("strip {a:.3} dir {d:?}");
        assert!(a > 0.8 && d.y.abs() > 0.95, "stress should run across the strip");
    }
}
