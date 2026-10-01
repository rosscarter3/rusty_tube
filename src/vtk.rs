//! Legacy VTK polydata, read and written without VTK.
//!
//! Reads ASCII and BINARY (big-endian) files, file versions up to 5.1
//! (the OFFSETS/CONNECTIVITY cell layout), with POINT_DATA arrays given as
//! SCALARS, VECTORS, NORMALS or FIELD. Writes the same binary layout
//! tubulaton's `drawScene::pool2vtk` writes, so existing analysis scripts read
//! the output unchanged.

use crate::mesh::TriMesh;
use anyhow::{anyhow, bail, Context, Result};
use glam::DVec3;
use std::io::Write;
use std::path::Path;

struct Cursor<'a> {
    b: &'a [u8],
    pos: usize,
    binary: bool,
}

impl<'a> Cursor<'a> {
    fn eof(&self) -> bool {
        self.pos >= self.b.len()
    }

    fn skip_ws(&mut self) {
        while self.pos < self.b.len() && self.b[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn token(&mut self) -> Option<&'a str> {
        self.skip_ws();
        let start = self.pos;
        while self.pos < self.b.len() && !self.b[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            None
        } else {
            std::str::from_utf8(&self.b[start..self.pos]).ok()
        }
    }

    fn line(&mut self) -> &'a str {
        let start = self.pos;
        while self.pos < self.b.len() && self.b[self.pos] != b'\n' {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.b[start..self.pos]).unwrap_or("");
        if self.pos < self.b.len() {
            self.pos += 1;
        }
        s.trim_end_matches('\r')
    }

    /// Read the rest of a keyword line, so binary data starts after it.
    fn header_line(&mut self) -> Vec<&'a str> {
        self.skip_ws();
        self.line().split_whitespace().collect()
    }

    fn type_size(ty: &str) -> Result<usize> {
        Ok(match ty {
            "float" => 4,
            "double" => 8,
            "int" | "unsigned_int" => 4,
            "long" | "unsigned_long" | "vtktypeint64" | "vtkIdType" => 8,
            "short" | "unsigned_short" => 2,
            "char" | "unsigned_char" | "bit" => 1,
            _ => bail!("unsupported VTK data type {ty}"),
        })
    }

    fn values(&mut self, n: usize, ty: &str) -> Result<Vec<f64>> {
        let mut out = Vec::with_capacity(n);
        if self.binary {
            let size = Self::type_size(ty)?;
            let need = n * size;
            if self.pos + need > self.b.len() {
                bail!("truncated binary VTK data");
            }
            let data = &self.b[self.pos..self.pos + need];
            for k in 0..n {
                let c = &data[k * size..(k + 1) * size];
                out.push(match ty {
                    "float" => f32::from_be_bytes(c.try_into().unwrap()) as f64,
                    "double" => f64::from_be_bytes(c.try_into().unwrap()),
                    "int" => i32::from_be_bytes(c.try_into().unwrap()) as f64,
                    "unsigned_int" => u32::from_be_bytes(c.try_into().unwrap()) as f64,
                    "long" | "vtktypeint64" | "vtkIdType" => i64::from_be_bytes(c.try_into().unwrap()) as f64,
                    "unsigned_long" => u64::from_be_bytes(c.try_into().unwrap()) as f64,
                    "short" => i16::from_be_bytes(c.try_into().unwrap()) as f64,
                    "unsigned_short" => u16::from_be_bytes(c.try_into().unwrap()) as f64,
                    "char" => c[0] as i8 as f64,
                    _ => c[0] as f64,
                });
            }
            self.pos += need;
        } else {
            for _ in 0..n {
                let t = self.token().ok_or_else(|| anyhow!("unexpected end of VTK data"))?;
                out.push(t.parse::<f64>().with_context(|| format!("bad number {t:?}"))?);
            }
        }
        Ok(out)
    }
}

/// Read a legacy VTK POLYDATA surface. Polygons with more than three corners
/// are fan-triangulated.
pub fn read_polydata(path: impl AsRef<Path>) -> Result<TriMesh> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    parse_polydata(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn parse_polydata(bytes: &[u8]) -> Result<TriMesh> {
    let mut c = Cursor {
        b: bytes,
        pos: 0,
        binary: false,
    };
    let version = c.line();
    if !version.starts_with("# vtk DataFile") {
        bail!("not a legacy VTK file");
    }
    let _title = c.line();
    let fmt = c.line().trim().to_ascii_uppercase();
    c.binary = fmt == "BINARY";
    let mut mesh = TriMesh::default();
    let mut n_points = 0usize;
    let mut in_point_data = false;
    let mut in_cell_data = false;

    while !c.eof() {
        let h = c.header_line();
        if h.is_empty() {
            continue;
        }
        match h[0].to_ascii_uppercase().as_str() {
            "DATASET" => {
                if h.get(1).map(|s| s.to_ascii_uppercase()) != Some("POLYDATA".into()) {
                    bail!("only POLYDATA datasets are supported");
                }
            }
            "POINTS" => {
                n_points = h[1].parse()?;
                let v = c.values(n_points * 3, h[2])?;
                mesh.positions = v.chunks(3).map(|p| DVec3::new(p[0], p[1], p[2])).collect();
            }
            "METADATA" => {
                // Skip to the blank line that ends the block.
                loop {
                    if c.eof() {
                        break;
                    }
                    let l = c.line();
                    if l.trim().is_empty() {
                        break;
                    }
                }
            }
            kw @ ("VERTICES" | "LINES" | "POLYGONS" | "TRIANGLE_STRIPS") => {
                let n_cells: usize = h[1].parse()?;
                let size: usize = h[2].parse()?;
                let save = c.pos;
                let next = c.header_line();
                let cells: Vec<Vec<u32>> = if next.first() == Some(&"OFFSETS") {
                    // Version 5.1: OFFSETS (n_cells+1) then CONNECTIVITY.
                    let offsets = c.values(n_cells + 1, next[1])?;
                    let conn_h = c.header_line();
                    if conn_h.first() != Some(&"CONNECTIVITY") {
                        bail!("expected CONNECTIVITY");
                    }
                    let conn = c.values(size, conn_h[1])?;
                    offsets
                        .windows(2)
                        .map(|w| conn[w[0] as usize..w[1] as usize].iter().map(|&x| x as u32).collect())
                        .collect()
                } else {
                    c.pos = save;
                    let raw = c.values(size, "int")?;
                    let mut cells = Vec::with_capacity(n_cells);
                    let mut k = 0;
                    while k < raw.len() {
                        let m = raw[k] as usize;
                        cells.push(raw[k + 1..k + 1 + m].iter().map(|&x| x as u32).collect());
                        k += m + 1;
                    }
                    cells
                };
                match kw {
                    "POLYGONS" => {
                        for cell in cells {
                            for j in 1..cell.len().saturating_sub(1) {
                                mesh.triangles.push([cell[0], cell[j], cell[j + 1]]);
                            }
                        }
                    }
                    "TRIANGLE_STRIPS" => {
                        for cell in cells {
                            for j in 0..cell.len().saturating_sub(2) {
                                if j % 2 == 0 {
                                    mesh.triangles.push([cell[j], cell[j + 1], cell[j + 2]]);
                                } else {
                                    mesh.triangles.push([cell[j + 1], cell[j], cell[j + 2]]);
                                }
                            }
                        }
                    }
                    _ => {}
                }
            }
            "POINT_DATA" => {
                in_point_data = true;
                in_cell_data = false;
            }
            "CELL_DATA" => {
                in_point_data = false;
                in_cell_data = true;
            }
            "SCALARS" => {
                let name = h[1].to_string();
                let ty = h[2];
                let ncomp: usize = h.get(3).and_then(|s| s.parse().ok()).unwrap_or(1);
                let n = if in_cell_data { 0 } else { n_points };
                // LOOKUP_TABLE line is optional in some writers.
                let save = c.pos;
                let lt = c.header_line();
                if lt.first().map(|s| s.to_ascii_uppercase()) != Some("LOOKUP_TABLE".into()) {
                    c.pos = save;
                }
                if in_cell_data {
                    let n_cells = mesh.triangles.len();
                    c.values(n_cells * ncomp, ty)?;
                    continue;
                }
                let v = c.values(n * ncomp, ty)?;
                store(&mut mesh, name, ncomp, v);
            }
            "VECTORS" | "NORMALS" => {
                let name = h[1].to_string();
                let v = c.values(if in_cell_data { 0 } else { n_points * 3 }, h[2])?;
                if in_point_data {
                    store(&mut mesh, name, 3, v);
                }
            }
            "FIELD" => {
                let narrays: usize = h[2].parse()?;
                for _ in 0..narrays {
                    let ah = c.header_line();
                    if ah.is_empty() {
                        continue;
                    }
                    let name = ah[0].to_string();
                    let ncomp: usize = ah[1].parse()?;
                    let ntup: usize = ah[2].parse()?;
                    let v = c.values(ncomp * ntup, ah[3])?;
                    if in_point_data && ntup == n_points {
                        store(&mut mesh, name, ncomp, v);
                    }
                }
            }
            "LOOKUP_TABLE" => {
                let n: usize = h[2].parse()?;
                if c.binary {
                    c.pos += n * 4;
                } else {
                    c.values(n * 4, "float")?;
                }
            }
            "TEXTURE_COORDINATES" => {
                let ncomp: usize = h[2].parse()?;
                c.values(n_points * ncomp, h[3])?;
            }
            _ => {
                // Unknown keyword: give up on the rest rather than misparse.
                break;
            }
        }
    }
    if mesh.positions.is_empty() {
        bail!("no POINTS in VTK file");
    }
    Ok(mesh)
}

fn store(mesh: &mut TriMesh, name: String, ncomp: usize, v: Vec<f64>) {
    match ncomp {
        1 => {
            mesh.point_scalars.insert(name, v);
        }
        3 => {
            mesh.point_vectors
                .insert(name, v.chunks(3).map(|p| DVec3::new(p[0], p[1], p[2])).collect());
        }
        _ => {}
    }
}

/// Write a surface as ASCII legacy VTK with outward point normals and any
/// point arrays it carries. This is the input format tubulaton reads.
pub fn write_surface(path: impl AsRef<Path>, mesh: &TriMesh, normals: &[DVec3]) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path.as_ref())?);
    writeln!(
        f,
        "# vtk DataFile Version 3.0\nrusty_tube surface\nASCII\nDATASET POLYDATA"
    )?;
    writeln!(f, "POINTS {} float", mesh.positions.len())?;
    for p in &mesh.positions {
        writeln!(f, "{:.4} {:.4} {:.4}", p.x, p.y, p.z)?;
    }
    writeln!(f, "POLYGONS {} {}", mesh.triangles.len(), 4 * mesh.triangles.len())?;
    for t in &mesh.triangles {
        writeln!(f, "3 {} {} {}", t[0], t[1], t[2])?;
    }
    writeln!(f, "POINT_DATA {}", mesh.positions.len())?;
    writeln!(f, "NORMALS Normals float")?;
    for n in normals {
        writeln!(f, "{:.6} {:.6} {:.6}", n.x, n.y, n.z)?;
    }
    for (name, v) in &mesh.point_vectors {
        if name == "Normals" {
            continue;
        }
        writeln!(f, "VECTORS {name} float")?;
        for s in v {
            writeln!(f, "{:.5} {:.5} {:.5}", s.x, s.y, s.z)?;
        }
    }
    for (name, v) in &mesh.point_scalars {
        writeln!(f, "SCALARS {name} float 1\nLOOKUP_TABLE default")?;
        for s in v {
            writeln!(f, "{s}")?;
        }
    }
    Ok(())
}

/// One named point array for [`write_points_binary`].
pub enum Array<'a> {
    Int(&'a str, Vec<i32>),
    Float(&'a str, Vec<f32>),
}

/// Write a point cloud with a 3-component `scalars` array and a field-data
/// block, in the exact binary layout tubulaton writes.
pub fn write_points_binary(
    path: impl AsRef<Path>,
    points: &[[f32; 3]],
    scalars3: &[[f32; 3]],
    arrays: &[Array],
) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path.as_ref())?);
    let n = points.len();
    write!(
        f,
        "# vtk DataFile Version 5.1\nvtk output\nBINARY\nDATASET POLYDATA\nPOINTS {n} float\n"
    )?;
    for p in points {
        for c in p {
            f.write_all(&c.to_be_bytes())?;
        }
    }
    write!(f, "\nPOINT_DATA {n}\nSCALARS scalars float 3\nLOOKUP_TABLE default\n")?;
    for p in scalars3 {
        for c in p {
            f.write_all(&c.to_be_bytes())?;
        }
    }
    write!(f, "\nFIELD FieldData {}\n", arrays.len())?;
    for a in arrays {
        match a {
            Array::Int(name, v) => {
                writeln!(f, "{name} 1 {n} int")?;
                for x in v {
                    f.write_all(&x.to_be_bytes())?;
                }
            }
            Array::Float(name, v) => {
                writeln!(f, "{name} 1 {n} float")?;
                for x in v {
                    f.write_all(&x.to_be_bytes())?;
                }
            }
        }
        f.write_all(b"\n")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_roundtrip() {
        let mut m = TriMesh::default();
        m.positions = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z];
        m.triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
        m.point_vectors.insert("strain_0".into(), vec![DVec3::X; 4]);
        let n = m.compute_normals();
        let dir = std::env::temp_dir().join("rusty_tube_vtk_test.vtk");
        write_surface(&dir, &m, &n).unwrap();
        let r = read_polydata(&dir).unwrap();
        assert_eq!(r.positions.len(), 4);
        assert_eq!(r.triangles.len(), 4);
        assert!(r.point_vectors.contains_key("Normals"));
        assert!(r.point_vectors.contains_key("strain_0"));
    }

    #[test]
    fn binary_points_parse_back() {
        let dir = std::env::temp_dir().join("rusty_tube_vtk_bin.vtk");
        write_points_binary(
            &dir,
            &[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            &[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
            &[Array::Int("identite", vec![7, 8]), Array::Float("d", vec![0.5, 1.5])],
        )
        .unwrap();
        let r = read_polydata(&dir).unwrap();
        assert_eq!(r.positions[1], DVec3::new(4.0, 5.0, 6.0));
        assert_eq!(r.point_scalars["identite"], vec![7.0, 8.0]);
        assert_eq!(r.point_vectors["scalars"][1], DVec3::X);
    }
}
