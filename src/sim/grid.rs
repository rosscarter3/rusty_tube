//! Sparse uniform spatial hashes.

use glam::DVec3;
use rustc_hash::FxHashMap;

/// Cell index of a coordinate, truncating toward zero as the C++ `int`
/// conversion does.
#[inline]
fn cell_of(v: f64, min: f64, res: f64) -> i32 {
    ((v - min) / res) as i32
}

#[inline]
fn pack(x: i32, y: i32, z: i32) -> u64 {
    const M: u64 = (1 << 21) - 1;
    ((x as u64 & M) << 42) | ((y as u64 & M) << 21) | (z as u64 & M)
}

#[derive(Clone, Debug)]
pub struct Grid {
    pub res: f64,
    pub min: DVec3,
    cells: FxHashMap<u64, Vec<u32>>,
}

impl Grid {
    pub fn new(res: f64, min: DVec3) -> Self {
        Grid {
            res,
            min,
            cells: FxHashMap::default(),
        }
    }

    #[inline]
    pub fn key(&self, p: DVec3) -> u64 {
        pack(
            cell_of(p.x, self.min.x, self.res),
            cell_of(p.y, self.min.y, self.res),
            cell_of(p.z, self.min.z, self.res),
        )
    }

    pub fn insert(&mut self, p: DVec3, id: u32) -> u64 {
        let k = self.key(p);
        self.cells.entry(k).or_default().push(id);
        k
    }

    pub fn remove(&mut self, key: u64, id: u32) {
        if let Some(v) = self.cells.get_mut(&key) {
            if let Some(i) = v.iter().position(|&x| x == id) {
                v.swap_remove(i);
            }
            if v.is_empty() {
                self.cells.remove(&key);
            }
        }
    }

    pub fn clear(&mut self) {
        self.cells.clear();
    }

    /// Visit everything in the 27 cells around `p`, exactly as tubulaton's
    /// `getSpaceContourContent`: offsets of one whole cell, and offsets that
    /// land on a negative coordinate are skipped.
    #[inline]
    pub fn for_each_27(&self, p: DVec3, mut f: impl FnMut(u32)) {
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    let q = p + DVec3::new(i as f64, j as f64, k as f64) * self.res;
                    if q.x >= 0.0 && q.y >= 0.0 && q.z >= 0.0 {
                        if let Some(v) = self.cells.get(&self.key(q)) {
                            for &id in v {
                                f(id);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Visit everything in the one cell containing `p`.
    #[inline]
    pub fn for_each_cell(&self, p: DVec3, mut f: impl FnMut(u32)) {
        if let Some(v) = self.cells.get(&self.key(p)) {
            for &id in v {
                f(id);
            }
        }
    }

    /// Whether any cell of the 27 around `p` is occupied.
    pub fn any_27(&self, p: DVec3) -> bool {
        let mut any = false;
        self.for_each_27(p, |_| any = true);
        any
    }

    pub fn occupied_cells(&self) -> usize {
        self.cells.len()
    }
}
