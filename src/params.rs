//! Configuration: a port of tubulaton's `Parameters` class.
//!
//! Every key the C++ code knows is predeclared here with its default and its
//! type, so a misspelt key in an `.ini` is an error rather than a silently
//! ignored line. The French key names are the ones stored, because every
//! published parameter set uses them; English aliases resolve to the same
//! entries (see [`key_aliases`]).
//!
//! The `.ini` format is `key=value`, `#` starts a comment, and a line reading
//! exactly `Update` ends the initial block. Lines after it are applied later
//! by [`Params::update`] when the simulation reaches `Update_Time_Parameters`.

use anyhow::{anyhow, bail, Context, Result};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Params {
    d: HashMap<String, f64>,
    i: HashMap<String, i64>,
    s: HashMap<String, String>,
    /// Lines after the first `Update` marker, consumed block by block.
    pending_update: Vec<String>,
}

impl Default for Params {
    fn default() -> Self {
        Self::with_defaults()
    }
}

/// English spellings for the French config keys.
pub fn key_aliases() -> &'static [(&'static str, &'static str)] {
    &[
        ("input_vtk_folder", "nom_folder_input_vtk"),
        ("input_vtk", "nom_input_vtk"),
        ("input_vtk_ref", "nom_input_vtk_ref"),
        ("output_vtk_folder", "nom_folder_output_vtk"),
        ("output_vtk", "nom_output_vtk"),
        ("report_file", "nom_rapport"),
        ("config_name", "nom_config"),
        ("n_input_vtk", "num_input_vtk"),
        ("subunit_size", "taille_microtubule"),
        ("cortical_thickness", "epaisseur_corticale"),
        ("membrane_angle_limit", "Angle_mb_limite"),
        ("membrane_trajectory_angle", "Angle_mb_trajectoire"),
        ("tan_membrane_angle", "tan_Angle_mb"),
        ("attach_decision", "decision_accrochage"),
        ("encounter_decision", "decision_rencontre"),
        ("cut_decision", "decision_cut"),
        ("prob_catastrophe", "proba_shrink"),
        ("prob_catastrophe_beyond", "proba_shrink_beyond"),
        ("prob_rescue", "proba_rescue"),
        ("prob_detachment_per_step", "proba_detachement_par_step_par_microtubule"),
        ("prob_severing", "proba_tocut"),
        ("prob_crossover_severing", "proba_crossmicro_cut"),
        ("nucleation_per_step", "proba_initialisation_par_step"),
        ("nucleation_per_area", "proba_initialisation_area"),
        (
            "nucleation_per_step_everywhere",
            "Everywhere_proba_initialisation_par_step",
        ),
        ("nucleation_per_area_everywhere", "Everywhere_proba_initialisation_area"),
        ("n_microtubules_init", "nb_microtubules_init"),
        ("max_steps", "nb_max_steps"),
        ("random_weight", "part_alea_alea"),
        ("fixed_weight", "part_alea_fixe"),
        ("influence_weight", "part_influence_influence"),
        ("normal_weight", "part_influence_normale"),
        ("influence_ref", "influence_ref"),
        ("grid_cell_size", "m_hashContour_resolution"),
    ]
}

/// C `atof`: the longest numeric prefix, or 0.
fn atof(s: &str) -> f64 {
    let s = s.trim();
    if let Ok(v) = s.parse::<f64>() {
        return v;
    }
    let bytes = s.as_bytes();
    for end in (1..=bytes.len()).rev() {
        if let Ok(v) = s[..end].parse::<f64>() {
            return v;
        }
    }
    0.0
}

/// C `atoi`: optional sign and leading digits, or 0. "1.5" is 1.
fn atoi(s: &str) -> i64 {
    let s = s.trim();
    let mut end = 0;
    for (k, c) in s.char_indices() {
        if c.is_ascii_digit() || (k == 0 && (c == '-' || c == '+')) {
            end = k + c.len_utf8();
        } else {
            break;
        }
    }
    s[..end].parse().unwrap_or(0)
}

impl Params {
    /// Every key tubulaton predeclares, at its default.
    pub fn with_defaults() -> Self {
        let mut d: HashMap<String, f64> = HashMap::new();
        let mut i: HashMap<String, i64> = HashMap::new();
        let mut s: HashMap<String, String> = HashMap::new();
        for (k, v) in [
            ("Kgr_sh_Plus", 0.0),
            ("Ksh_gr_Plus", 0.0),
            ("V_gr_Plus", 0.0),
            ("V_sh_Plus", 0.0),
            ("V_sh_Moins", 0.0),
            ("Angle_bundle", 0.7),
            ("K_nobundle_shrink", 0.0),
            ("K_nobundle_cross", 0.0),
            ("K_nobundle_catastrophe", 0.0),
            ("D_bundle", 49.0),
            ("Angle_mb_limite", 0.0),
            ("Angle_mb_trajectoire", 0.7),
            ("Angle_cut", 0.0),
            ("tan_Angle_mb", 0.839099631177),
            ("epaisseur_corticale", 1.0),
            ("poids_previous_vect", 0.0),
            ("d_mb", 10.0),
            ("part_alea_alea", 1.0),
            ("part_alea_fixe", 39.0),
            ("proba_detachement_par_step_par_microtubule", 0.0),
            ("proba_initialisation_par_step", -1.0),
            ("proba_initialisation_area", -1.0),
            ("Everywhere_proba_initialisation_par_step", -1.0),
            ("Everywhere_proba_initialisation_area", -1.0),
            ("Initialisation_Volume_Radius", -1.0),
            ("proba_tocut", 0.0),
            ("proba_rescue", 0.0),
            ("proba_shrink", 0.0),
            ("proba_crossmicro_cut", 0.0),
            ("stress_catastrophe_bias", 0.0),
            ("stress_severing_bias", 0.0),
            ("proba_crossmicro_shrink", 1.0),
            ("cutgroup", 0.0),
            ("part_influence_influence", 0.0),
            ("part_influence_normale", 1.0),
            ("influence_ref", 0.0),
            ("taille_microtubule", 8.0),
            ("Movement_Speed", 0.0),
            ("Max_Movement_Speed", -1.0),
            ("proba_shrink_beyond", -1.0),
            ("Xpos_beyond", 0.0),
            ("Xpos_beyond_minus", -1e12),
            ("Bound_Xmax", -1.0),
            ("Bound_Xmin", 0.0),
            ("Bound_Ymax", -1.0),
            ("Bound_Ymin", 0.0),
            ("Bound_Zmax", -1.0),
            ("Bound_Zmin", 0.0),
            ("growth_speed_um_per_s", 0.0),
            ("dt_seconds", 0.0),
            ("rate_catastrophe_per_s", -1.0),
            ("rate_rescue_per_s", -1.0),
            ("rate_detachment_per_s", -1.0),
            ("rate_severing_per_s", -1.0),
            // Written by Contour::open when nucleating in a volume.
            ("CentreX", 0.0),
            ("CentreY", 0.0),
            ("CentreZ", 0.0),
        ] {
            d.insert(k.to_string(), v);
        }
        for (k, v) in [
            ("gr_st_plus", 1),
            ("gr_st_moins", 0),
            ("Everywhere_nb_microtubules_init", 0),
            ("nb_microtubules_init", 0),
            ("nb_max_steps", 0),
            ("garbage_steps", 500),
            ("vtk_steps", 0),
            ("stop", 0),
            ("details", 1),
            ("num_input_vtk", 1),
            ("Sep_Nucleations", 0),
            ("Output_Contour", 0),
            ("Type_Input", 0),
            ("UpdateNextTime", -1),
            ("Time", 0),
            ("Update_Time_Parameters", -1),
            ("Update_Contour_Index", -1),
            ("Movement_Only", -1),
            ("decision_rencontre", -1),
            ("decision_accrochage", 0),
            ("decision_cut", 0),
            ("Nucleation_Direction", 0),
            ("cortical", 0),
            ("save_events", 100),
            ("icpower", 0),
            ("ShrinkingMultiple", 1),
            ("TreadmillingFrequency", 1),
            ("Bound_Ind", -1),
            ("m_hashContour_resolution", 25),
            ("seed", 0),
            // 1 (tubulaton's behaviour): a growing tip can meet and bundle
            // with older parts of its own filament, beyond its last ten
            // subunits. That lets a filament curved round a lobe tip hairpin
            // back along itself and coil, which a stiff microtubule cannot
            // do. 0 excludes the filament's own lattice from the search.
            ("self_encounters", 1),
        ] {
            i.insert(k.to_string(), v);
        }
        for (k, v) in [
            ("nom_output_vtk", ""),
            ("nom_input_vtk", ""),
            ("nom_rapport", ""),
            ("nom_input_vtk_ref", ""),
            ("nom_folder_input_vtk", ""),
            ("nom_folder_output_vtk", ""),
            ("nom_config", ""),
            ("Update_file", ""),
            ("Nucleation_Volume_Shape", "cube"),
        ] {
            s.insert(k.to_string(), v.to_string());
        }
        Params {
            d,
            i,
            s,
            pending_update: Vec::new(),
        }
    }

    /// Read a config file, validate it, and apply the time calibration.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("cannot read configuration file {}", path.display()))?;
        Self::from_str_config(&text)
    }

    pub fn from_str_config(text: &str) -> Result<Self> {
        let mut p = Self::with_defaults();
        p.load(text)?;
        p.apply_time_calibration()?;
        Ok(p)
    }

    fn strip(line: &str) -> String {
        let mut line = line.trim_end_matches('\r').to_string();
        if let Some(pos) = line.find('#') {
            line.truncate(pos);
        }
        line
    }

    fn trim(s: &str) -> &str {
        s.trim_matches(|c| c == ' ' || c == '\t')
    }

    fn load(&mut self, text: &str) -> Result<()> {
        let mut lines = text.lines();
        for raw in lines.by_ref() {
            let line = Self::strip(raw);
            if line.is_empty() {
                continue;
            }
            if line == "Update" {
                break;
            }
            let Some(pos) = line.find('=') else { continue };
            let key = Self::trim(&line[..pos]).to_string();
            let value = Self::trim(&line[pos + 1..]).to_string();
            // num_input_vtk extends the key set with the per-surface keys; it
            // must precede them in the file, as in the C++.
            if key == "num_input_vtk" || self.resolve_key(&key) == "num_input_vtk" {
                let n = atoi(&value);
                for k in 2..=n {
                    self.declare_surface_keys(k);
                }
            }
            let resolved = self.resolve_key(&key);
            if resolved == "proba_initialisation_area" {
                self.i.insert("Type_Input".into(), 1);
            }
            for k in 2..=self.get_i("num_input_vtk") {
                if resolved == format!("proba_initialisation_area_{k}") {
                    self.i.insert(format!("Type_Input_{k}"), 1);
                }
            }
            if !key.is_empty() && !value.is_empty() {
                self.fill(&key, &value)?;
            }
        }
        self.pending_update = lines.map(|s| s.to_string()).collect();
        self.validate()
    }

    fn declare_surface_keys(&mut self, k: i64) {
        self.s.insert(format!("nom_input_vtk_{k}"), String::new());
        self.s.insert(format!("nom_input_vtk_ref_{k}"), String::new());
        self.i.insert(format!("nb_microtubules_init_{k}"), -1);
        self.i.insert(format!("Everywhere_nb_microtubules_init_{k}"), -1);
        self.d.insert(format!("proba_initialisation_area_{k}"), -1.0);
        self.d.insert(format!("Everywhere_proba_initialisation_area_{k}"), -1.0);
        self.i.insert(format!("Type_Input_{k}"), 0);
        self.d.insert(format!("proba_initialisation_par_step_{k}"), -1.0);
        self.d
            .insert(format!("Everywhere_proba_initialisation_par_step_{k}"), -1.0);
        self.i.insert(format!("Output_Contour_{k}"), 0);
        self.i.insert(format!("Nucleation_Direction_{k}"), 0);
        self.d.insert(format!("Movement_Speed_{k}"), 0.0);
    }

    fn validate(&mut self) -> Result<()> {
        let n = self.get_i("num_input_vtk");
        let mut count = 0;
        for k in 2..=n {
            if self.get_i(&format!("nb_microtubules_init_{k}")) >= 0 {
                count += 1;
            }
            let step = self.get_d(&format!("proba_initialisation_par_step_{k}"));
            let area = self.get_d(&format!("proba_initialisation_area_{k}"));
            if step >= 0.0 {
                count += 1;
            }
            if area >= 0.0 {
                count += 1;
            }
            if area > 0.0 && step >= 0.0 {
                bail!(
                    "surface {k}: give the nucleation rate EITHER per step \
                     (proba_initialisation_par_step_{k}) OR per area \
                     (proba_initialisation_area_{k}), not both"
                );
            }
        }
        let step = self.get_d("proba_initialisation_par_step");
        let area = self.get_d("proba_initialisation_area");
        if (step >= 0.0) == (area >= 0.0) {
            bail!(
                "exactly one of proba_initialisation_par_step or \
                 proba_initialisation_area must be given"
            );
        }
        if count > 0 {
            if count == 2 * (n - 1) {
                self.i.insert("Sep_Nucleations".into(), 1);
            } else {
                bail!(
                    "nb_microtubules_init and proba_initialisation_par_step must be \
                     set on every surface individually, or once for all surfaces"
                );
            }
        }
        Ok(())
    }

    /// Convert any per-second rate into the per-step probability the
    /// simulation reads, and report the conversion.
    pub fn apply_time_calibration(&mut self) -> Result<()> {
        let v = self.get_d("growth_speed_um_per_s");
        let mut dt = self.get_d("dt_seconds");
        if v > 0.0 {
            let step_um = self.get_d("taille_microtubule") / 1000.0 * self.get_i("gr_st_plus") as f64;
            let derived = step_um / v;
            if dt > 0.0 && (derived - dt).abs() > 1e-9 * dt.max(derived) {
                println!(
                    "Note: dt_seconds={dt} is overridden by growth_speed_um_per_s={v}, \
                     which implies {derived} s per step."
                );
            }
            dt = derived;
            self.d.insert("dt_seconds".into(), dt);
        }
        const TABLE: [(&str, &str); 4] = [
            ("rate_catastrophe_per_s", "proba_shrink"),
            ("rate_rescue_per_s", "proba_rescue"),
            ("rate_detachment_per_s", "proba_detachement_par_step_par_microtubule"),
            ("rate_severing_per_s", "proba_tocut"),
        ];
        let any = TABLE.iter().any(|(r, _)| self.get_d(r) >= 0.0);
        if !any {
            if dt > 0.0 {
                println!("Timestep: {dt} s per step");
            }
            return Ok(());
        }
        if dt <= 0.0 {
            bail!(
                "a per-second rate was given but the duration of a step is unknown; \
                 set growth_speed_um_per_s or dt_seconds"
            );
        }
        println!("Timestep: {dt} s per step");
        for (rate, proba) in TABLE {
            let r = self.get_d(rate);
            if r < 0.0 {
                continue;
            }
            let p = r * dt;
            if p > 0.1 {
                eprintln!(
                    "warning: {rate}={r} /s at {dt} s per step gives a per-step \
                     probability of {p}; above ~0.1 the discretisation stops \
                     approximating a Poisson process"
                );
            }
            println!("  {rate}={r} /s -> {proba}={p} per step");
            self.d.insert(proba.into(), p);
        }
        Ok(())
    }

    /// Apply the next `Update` block of the config file. Returns true if the
    /// block changed the input surface.
    pub fn update(&mut self) -> Result<bool> {
        let mut contour_changed = false;
        let lines = std::mem::take(&mut self.pending_update);
        let mut rest = lines.into_iter();
        for raw in rest.by_ref() {
            let line = Self::strip(&raw);
            if line == "Update" {
                break;
            }
            let Some(pos) = line.find('=') else { continue };
            let key = Self::trim(&line[..pos]).to_string();
            let value = Self::trim(&line[pos + 1..]).to_string();
            if !key.is_empty() && !value.is_empty() {
                self.fill(&key, &value)?;
            }
            let r = self.resolve_key(&key);
            if r == "nom_input_vtk" || r == "nom_input_vtk_ref" {
                contour_changed = true;
            }
        }
        self.pending_update = rest.collect();
        Ok(contour_changed)
    }

    pub fn resolve_key(&self, key: &str) -> String {
        if self.d.contains_key(key) || self.i.contains_key(key) || self.s.contains_key(key) {
            return key.to_string();
        }
        let aliases = key_aliases();
        if let Some((_, to)) = aliases.iter().find(|(from, _)| *from == key) {
            return to.to_string();
        }
        if let Some(us) = key.rfind('_') {
            let suffix = &key[us + 1..];
            if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
                if let Some((_, to)) = aliases.iter().find(|(from, _)| *from == &key[..us]) {
                    return format!("{to}{}", &key[us..]);
                }
            }
        }
        key.to_string()
    }

    /// Set a key from its textual value, with the type the key was declared as.
    pub fn fill(&mut self, raw_key: &str, value: &str) -> Result<()> {
        let key = self.resolve_key(raw_key);
        if let Some(v) = self.d.get_mut(&key) {
            *v = atof(value);
        } else if let Some(v) = self.i.get_mut(&key) {
            *v = atoi(value);
        } else if let Some(v) = self.s.get_mut(&key) {
            *v = value.to_string();
        } else {
            bail!("unknown parameter: {raw_key}");
        }
        Ok(())
    }

    pub fn get_d(&self, k: &str) -> f64 {
        *self.d.get(k).unwrap_or_else(|| panic!("unknown double parameter: {k}"))
    }
    pub fn get_i(&self, k: &str) -> i64 {
        *self
            .i
            .get(k)
            .unwrap_or_else(|| panic!("unknown integer parameter: {k}"))
    }
    pub fn get_s(&self, k: &str) -> &str {
        self.s.get(k).unwrap_or_else(|| panic!("unknown string parameter: {k}"))
    }
    pub fn try_d(&self, k: &str) -> Option<f64> {
        self.d.get(k).copied()
    }
    pub fn try_i(&self, k: &str) -> Option<i64> {
        self.i.get(k).copied()
    }
    pub fn set_d(&mut self, k: &str, v: f64) {
        self.d.insert(k.to_string(), v);
    }
    pub fn set_i(&mut self, k: &str, v: i64) {
        self.i.insert(k.to_string(), v);
    }
    pub fn set_s(&mut self, k: &str, v: &str) {
        self.s.insert(k.to_string(), v.to_string());
    }

    /// Per-surface key: `base` for surface 0, `base_{k+1}` otherwise.
    pub fn surface_key(base: &str, surface: usize) -> String {
        if surface == 0 {
            base.to_string()
        } else {
            format!("{base}_{}", surface + 1)
        }
    }

    /// All keys and values, sorted, as `.ini` text.
    pub fn to_ini(&self) -> String {
        let mut lines: Vec<String> = Vec::new();
        for (k, v) in &self.d {
            lines.push(format!("{k}={v}"));
        }
        for (k, v) in &self.i {
            lines.push(format!("{k}={v}"));
        }
        for (k, v) in &self.s {
            if !v.is_empty() {
                lines.push(format!("{k}={v}"));
            }
        }
        lines.sort();
        lines.join("\n") + "\n"
    }

    pub fn require_known(&self, key: &str) -> Result<()> {
        let k = self.resolve_key(key);
        if self.d.contains_key(&k) || self.i.contains_key(&k) || self.s.contains_key(&k) {
            Ok(())
        } else {
            Err(anyhow!("unknown parameter: {key}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn c_number_parsing() {
        assert_eq!(atoi("1.5"), 1);
        assert_eq!(atoi("-3x"), -3);
        assert_eq!(atoi("abc"), 0);
        assert_eq!(atof("0.25abc"), 0.25);
        assert_eq!(atof("1e-3"), 1e-3);
    }

    #[test]
    fn aliases_and_comments() {
        let p = Params::from_str_config("prob_catastrophe=0.01 # comment\nnucleation_per_step=0.5\nmax_steps=10\n")
            .unwrap();
        assert_eq!(p.get_d("proba_shrink"), 0.01);
        assert_eq!(p.get_d("proba_initialisation_par_step"), 0.5);
        assert_eq!(p.get_i("nb_max_steps"), 10);
    }

    #[test]
    fn unknown_key_is_an_error() {
        assert!(Params::from_str_config("proba_initialisation_par_step=1\nnot_a_key=1\n").is_err());
    }

    #[test]
    fn time_calibration() {
        let p = Params::from_str_config(
            "proba_initialisation_par_step=1\ngrowth_speed_um_per_s=0.08\nrate_catastrophe_per_s=0.003\n",
        )
        .unwrap();
        assert!((p.get_d("dt_seconds") - 0.1).abs() < 1e-12);
        assert!((p.get_d("proba_shrink") - 0.0003).abs() < 1e-12);
    }

    #[test]
    fn update_blocks() {
        let mut p = Params::from_str_config(
            "proba_initialisation_par_step=1\nUpdate_Time_Parameters=5\nUpdate\nproba_shrink=0.5\nUpdate\nproba_shrink=0.7\n",
        )
        .unwrap();
        assert_eq!(p.get_d("proba_shrink"), 0.0);
        p.update().unwrap();
        assert_eq!(p.get_d("proba_shrink"), 0.5);
        p.update().unwrap();
        assert_eq!(p.get_d("proba_shrink"), 0.7);
    }
}
