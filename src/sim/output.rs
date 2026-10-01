//! Report and VTK output in tubulaton's formats, and the main time loop.

use super::{Sim, NONE};
use crate::vtk::{self, Array};
use anyhow::{bail, Result};
use std::io::Write;
use std::path::PathBuf;

impl Sim {
    fn out_path(&self, file: &str) -> PathBuf {
        PathBuf::from(format!("{}{}", self.params.get_s("nom_folder_output_vtk"), file))
    }

    /// Append one line of event counts (and a line of lengths) and reset the
    /// counters: `rapport` in main.cpp.
    pub fn write_report(&mut self) -> Result<()> {
        let name = self.params.get_s("nom_rapport").to_string();
        if name.is_empty() {
            self.counters = Default::default();
            return Ok(());
        }
        let path = self.out_path(&name);
        let c = &self.counters;
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(
            f,
            "{};{};{};{};{};{};{};{};{};{};{};{};{};",
            c.encounter_bundle_grow,
            c.encounter_bundle_shrink,
            c.encounter_mb_grow,
            c.encounter_mb_shrink,
            c.cross,
            c.left_membrane,
            c.destruction_microtubule,
            c.creation_microtubule,
            c.cut_microtubule,
            c.override_cut_microtubule,
            c.detachements,
            self.mts.len(),
            self.crossover_list_len()
        )?;
        let stem = name.split(".txt").next().unwrap_or(&name);
        let mut g = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.out_path(&format!("{stem}_lengths.txt")))?;
        let mut line = String::new();
        for l in self.lengths() {
            line.push_str(&format!("{l};"));
        }
        writeln!(g, "{line}")?;
        let shrinkages = self.counters.shrinkages;
        self.counters = Default::default();
        self.counters.shrinkages = shrinkages;
        Ok(())
    }

    /// The element cloud at step `t`: `drawScene::pool2vtk`. With
    /// `details == 0` membrane vertices are included (type 2).
    pub fn write_vtk(&self, t: i64) -> Result<PathBuf> {
        let path = self.out_path(&format!("{}{t}.vtk", self.params.get_s("nom_output_vtk")));
        let detail = self.params.get_i("details");
        let mut pts = Vec::new();
        let mut dirs = Vec::new();
        let (mut ident, mut ty, mut contact, mut to_cut, mut id_vtk) = (vec![], vec![], vec![], vec![], vec![]);
        let (mut ncd, mut ncv, mut ncd2, mut cort, mut snum) = (vec![], vec![], vec![], vec![], vec![]);
        if detail == 0 {
            for v in &self.membrane.verts {
                pts.push(v.pos.as_vec3().to_array());
                dirs.push(v.normal.as_vec3().to_array());
                ident.push(1);
                ty.push(2);
                contact.push(v.contact as i32);
                to_cut.push(0);
                id_vtk.push(v.vtk_id as i32);
                ncd.push(0.0);
                ncv.push(-1.0);
                ncd2.push(-10.0);
                cort.push(0);
                snum.push(0);
            }
        }
        for (id, mt) in &self.mts {
            for &e in &mt.body {
                let el = &self.pool.elems[e as usize];
                pts.push(el.pos.as_vec3().to_array());
                dirs.push(el.dir.as_vec3().to_array());
                ident.push(*id as i32);
                ty.push(1);
                contact.push(el.contact as i32);
                to_cut.push(el.to_cut as i32);
                id_vtk.push(-1);
                ncd.push(el.nearest_contour_d);
                let vtk_id = if el.nearest_contour != NONE && (el.nearest_contour as usize) < self.membrane.verts.len()
                {
                    self.membrane.verts[el.nearest_contour as usize].vtk_id as f32
                } else {
                    -1.0
                };
                ncv.push(vtk_id);
                ncd2.push(el.d_lim);
                cort.push(el.cortical as i32);
                snum.push(el.cortical_structure);
            }
        }
        vtk::write_points_binary(
            &path,
            &pts,
            &dirs,
            &[
                Array::Int("identite", ident),
                Array::Int("type", ty),
                Array::Int("contact", contact),
                Array::Int("to_cut", to_cut),
                Array::Int("id_vtk", id_vtk),
                Array::Float("nearest_contour_d", ncd),
                Array::Float("nearest_contour_vtk", ncv),
                Array::Float("nearest_contour_d2", ncd2),
                Array::Int("cortical", cort),
                Array::Int("structure_num", snum),
            ],
        )?;
        Ok(path)
    }

    /// One iteration of main.cpp's loop at time `t`. Returns false when the
    /// config asks to stop.
    pub fn tick(&mut self, t: i64, write_files: bool) -> Result<bool> {
        self.time = t;
        self.params.set_i("Time", t);
        let upd = self.params.get_i("Update_Time_Parameters");
        if upd > 0 && upd == t {
            let contour_changed = self.params.update()?;
            self.refresh_params();
            if contour_changed {
                bail!("changing the input surface mid-run (Update with nom_input_vtk) is not ported yet");
            }
        }
        if self.params.get_i("Movement_Only") != 1 {
            self.evolve()?;
        }
        self.step_csc();
        if write_files {
            let save = self.params.get_i("save_events");
            if save > 0 && t % save == 0 {
                self.write_report()?;
            }
            let vs = self.params.get_i("vtk_steps");
            let stop = self.params.get_i("stop") != 0;
            if (vs > 0 && t % vs == 0) || stop {
                self.write_vtk(t)?;
            }
            if stop {
                println!("stop requested by the `stop` parameter");
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Features of the C++ that this port does not run yet.
    pub fn check_supported(&self) -> Result<()> {
        let p = &self.params;
        if p.get_d("Movement_Speed") != 0.0 || p.get_i("UpdateNextTime") > 0 {
            bail!("moving or deforming contours (Movement_Speed, UpdateNextTime) are not ported yet");
        }
        if !p.get_s("Update_file").is_empty() {
            bail!("Update_file is not ported yet");
        }
        Ok(())
    }
}
