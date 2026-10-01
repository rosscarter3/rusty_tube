//! Interactive viewer: the simulation runs on its own thread and publishes
//! snapshots; the window renders the latest one every frame, with an egui
//! panel for the simulation, its parameters and the renderer.

use crate::render::camera::OrbitCamera;
use crate::render::offline::Offscreen;
use crate::render::scene::{self, MembraneGeometry, SceneStats, Snapshot};
use crate::render::{Look, Renderer, Settings};
use crate::setup::{self, SimSetup};
use anyhow::Result;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

/// Everything needed to (re)build a simulation.
#[derive(Clone, Debug)]
pub struct SimSpec {
    /// Save the window to this PNG after this many frames, then exit
    /// (for documentation and testing).
    pub capture: Option<(u32, PathBuf)>,
    /// Simulate this many steps before showing anything.
    pub warmup: i64,
    /// Initial renderer settings.
    pub settings: Settings,
    /// Initial camera (yaw, pitch) in radians.
    pub angles: Option<(f32, f32)>,
    pub config: Option<PathBuf>,
    pub outline: Option<String>,
    pub sets: Vec<String>,
    pub neck_stress: bool,
    pub shape_stress: bool,
    pub scale: f64,
}

enum Cmd {
    Reset(SimSpec),
    Set(String, String),
    Stride(usize),
}

#[derive(Default)]
struct Shared {
    cmds: Mutex<Vec<Cmd>>,
    latest: Mutex<Option<Arc<Snapshot>>>,
    version: AtomicU64,
    membrane: Mutex<Option<MembraneGeometry>>,
    status: Mutex<String>,
    running: AtomicBool,
    steps_per_frame: AtomicU32,
    quit: AtomicBool,
}

fn sim_thread(shared: Arc<Shared>, spec: SimSpec) {
    crate::qos::user_initiated();
    let build = |spec: &SimSpec| {
        setup::build(&SimSetup {
            config: spec.config.as_deref(),
            outline: spec.outline.as_deref(),
            sets: &spec.sets,
            neck_stress: spec.neck_stress,
            shape_stress: spec.shape_stress,
            scale: spec.scale,
        })
    };
    let mut stride = 3usize;
    let mut sim = None;
    let mut pending = Some(spec);
    let mut t = 0i64;
    let mut last_publish = Instant::now() - Duration::from_secs(1);
    while !shared.quit.load(Ordering::Relaxed) {
        let cmds: Vec<Cmd> = std::mem::take(&mut *shared.cmds.lock().unwrap());
        for c in cmds {
            match c {
                Cmd::Reset(s) => pending = Some(s),
                Cmd::Set(k, v) => {
                    if let Some(sim) = sim.as_mut() {
                        let sim: &mut crate::sim::Sim = sim;
                        match sim.params.fill(&k, &v) {
                            Ok(()) => sim.refresh_params(),
                            Err(e) => *shared.status.lock().unwrap() = format!("{e}"),
                        }
                    }
                }
                Cmd::Stride(s) => stride = s,
            }
        }
        if let Some(spec) = pending.take() {
            *shared.status.lock().unwrap() = "building cell...".into();
            match build(&spec) {
                Ok(s) => {
                    let frame = scene::Frame::from_sim(&s);
                    *shared.membrane.lock().unwrap() = Some(scene::membrane_geometry(&s, &frame));
                    *shared.status.lock().unwrap() = format!(
                        "{} um^2, seed {}",
                        (s.membrane.total_area() / (125.0 * 125.0)) as i64,
                        s.rng.seed()
                    );
                    let mut s = s;
                    t = 0;
                    while t < spec.warmup {
                        if s.tick(t, false).is_err() {
                            break;
                        }
                        t += 1;
                    }
                    sim = Some(s);
                    last_publish = Instant::now() - Duration::from_secs(1);
                }
                Err(e) => *shared.status.lock().unwrap() = format!("error: {e:#}"),
            }
        }
        let Some(s) = sim.as_mut() else {
            std::thread::sleep(Duration::from_millis(20));
            continue;
        };
        let running = shared.running.load(Ordering::Relaxed);
        let mut seconds_per_step = 0.0;
        if running {
            let n = shared.steps_per_frame.load(Ordering::Relaxed).max(1);
            let t0 = Instant::now();
            for _ in 0..n {
                if let Err(e) = s.tick(t, false) {
                    *shared.status.lock().unwrap() = format!("error: {e:#}");
                    shared.running.store(false, Ordering::Relaxed);
                    break;
                }
                t += 1;
            }
            seconds_per_step = t0.elapsed().as_secs_f64() / n as f64;
        }
        // At most ~20 snapshots a second while running: building one costs
        // milliseconds, and the window cannot show more anyway.
        let due = if running {
            Duration::from_millis(50)
        } else {
            Duration::from_millis(250)
        };
        if last_publish.elapsed() >= due {
            let frame = scene::Frame::from_sim(s);
            let t_snap = Instant::now();
            let mut snap = scene::snapshot(s, &frame, stride);
            snap.stats.seconds_per_step = seconds_per_step;
            snap.stats.snapshot_ms = t_snap.elapsed().as_secs_f64() * 1000.0;
            *shared.latest.lock().unwrap() = Some(Arc::new(snap));
            shared.version.fetch_add(1, Ordering::Relaxed);
            last_publish = Instant::now();
        }
        if !running {
            std::thread::sleep(Duration::from_millis(15));
        } else {
            // Leave the render thread room to breathe at small step counts.
            std::thread::sleep(Duration::from_millis(2));
        }
    }
}

struct Gfx {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    renderer: Renderer,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct Ui {
    settings: Settings,
    camera: OrbitCamera,
    framed: bool,
    stats: SceneStats,
    capsules: usize,
    show_panel: bool,
    cells: Vec<String>,
    cell: String,
    neck_stress: bool,
    shape_stress: bool,
    csc: bool,
    csc_tether: bool,
    seed: i64,
    stride: usize,
    steps_per_frame: u32,
    // Live parameters.
    p_shrink: f64,
    p_rescue: f64,
    p_cross_shrink: f64,
    p_cross_cut: f64,
    nucleation: f64,
    angle_mb: f64,
    influence: f64,
    shot_scale: u32,
    shot_spp: u32,
    last_shot: String,
}

pub struct App {
    spec: SimSpec,
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
    gfx: Option<Gfx>,
    ui: Ui,
    mouse: (f64, f64),
    buttons: (bool, bool),
    egui_ctx: egui::Context,
    have_membrane: bool,
    fps: f32,
    last_frame: Instant,
    seen_version: u64,
    frames: u32,
    exit_requested: bool,
    panel_px: f32,
}

fn list_cells() -> Vec<String> {
    let mut v = Vec::new();
    for dir in ["cells/real", "cells/sim", "cells"] {
        if let Ok(rd) = std::fs::read_dir(dir) {
            let mut files: Vec<String> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().map(|e| e == "csv" || e == "json").unwrap_or(false))
                .map(|p| p.to_string_lossy().to_string())
                .collect();
            files.sort();
            v.extend(files);
        }
    }
    for s in 1..=8 {
        v.push(format!("procedural:{s}"));
    }
    v
}

impl App {
    pub fn new(spec: SimSpec) -> Self {
        let shared = Arc::new(Shared::default());
        shared.steps_per_frame.store(20, Ordering::Relaxed);
        shared.running.store(true, Ordering::Relaxed);
        let s2 = shared.clone();
        let sp = spec.clone();
        let thread = std::thread::spawn(move || sim_thread(s2, sp));
        let mut d = match &spec.config {
            Some(c) => crate::params::Params::from_file(c).unwrap_or_default(),
            None => crate::params::Params::from_str_config(setup::DEFAULT_CONFIG).unwrap_or_default(),
        };
        // Show what the run actually uses: presets and -s overrides.
        for kv in &spec.sets {
            if let Some((k, v)) = kv.split_once('=') {
                let _ = d.fill(k, v);
            }
        }
        let ui = Ui {
            settings: spec.settings.clone(),
            camera: {
                let mut c = OrbitCamera::framing(glam::Vec3::splat(-10.0), glam::Vec3::splat(10.0));
                if let Some((yaw, pitch)) = spec.angles {
                    c.yaw = yaw;
                    c.pitch = pitch;
                }
                c
            },
            framed: false,
            stats: SceneStats::default(),
            capsules: 0,
            show_panel: true,
            cells: list_cells(),
            cell: spec.outline.clone().unwrap_or_else(|| "config".into()),
            neck_stress: spec.neck_stress,
            shape_stress: spec.shape_stress,
            csc: d.get_i("csc_enable") != 0,
            csc_tether: d.get_i("csc_tether") != 0,
            seed: 1,
            stride: 3,
            steps_per_frame: 20,
            p_shrink: d.get_d("proba_shrink"),
            p_rescue: d.get_d("proba_rescue"),
            p_cross_shrink: d.get_d("proba_crossmicro_shrink"),
            p_cross_cut: d.get_d("proba_crossmicro_cut"),
            nucleation: 0.0,
            angle_mb: d.get_d("Angle_mb_limite"),
            influence: d.get_d("part_influence_influence"),
            shot_scale: 2,
            shot_spp: 256,
            last_shot: String::new(),
        };
        App {
            spec,
            shared,
            thread: Some(thread),
            gfx: None,
            ui,
            mouse: (0.0, 0.0),
            buttons: (false, false),
            egui_ctx: egui::Context::default(),
            have_membrane: false,
            fps: 0.0,
            last_frame: Instant::now(),
            seen_version: 0,
            frames: 0,
            exit_requested: false,
            panel_px: 330.0,
        }
    }

    fn send(&self, c: Cmd) {
        self.shared.cmds.lock().unwrap().push(c);
    }

    fn reset(&mut self) {
        let mut spec = self.spec.clone();
        if self.ui.cell != "config" {
            spec.outline = Some(self.ui.cell.clone());
        }
        spec.neck_stress = self.ui.neck_stress;
        spec.shape_stress = self.ui.shape_stress;
        spec.sets.retain(|s| !s.starts_with("csc_enable="));
        spec.sets.push(format!("csc_enable={}", self.ui.csc as u8));
        spec.sets.retain(|s| !s.starts_with("seed="));
        spec.sets.push(format!("seed={}", self.ui.seed));
        self.spec = spec.clone();
        self.ui.framed = false;
        self.have_membrane = false;
        self.send(Cmd::Reset(spec));
    }

    fn screenshot(&mut self) {
        let Some(g) = self.gfx.as_ref() else { return };
        let (w, h) = (
            g.config.width * self.ui.shot_scale,
            g.config.height * self.ui.shot_scale,
        );
        let mut off = Offscreen::with_device(g.renderer.device.clone(), g.renderer.queue.clone(), w, h);
        if let Some(m) = self.shared.membrane.lock().unwrap().as_ref() {
            off.renderer.set_membrane(m);
        }
        if let Some(s) = self.shared.latest.lock().unwrap().clone() {
            off.renderer.set_segments(&s.segments);
        }
        std::fs::create_dir_all("out/screenshots").ok();
        let name = format!(
            "out/screenshots/rusty_tube_{}.png",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0)
        );
        let s = self.ui.settings.clone();
        match off.save(&self.ui.camera, &s, self.ui.shot_spp, std::path::Path::new(&name)) {
            Ok(()) => self.ui.last_shot = format!("saved {name}"),
            Err(e) => self.ui.last_shot = format!("screenshot failed: {e}"),
        }
    }

    fn draw_ui(&mut self, ctx: &egui::Context) -> Vec<Cmd> {
        let mut cmds = Vec::new();
        let ui_state = &mut self.ui;
        let shared = &self.shared;
        let fps = self.fps;
        let mut reset = false;
        let mut shot = false;
        let mut panel_w = self.panel_px;
        let raw = {
            let g = self.gfx.as_mut().unwrap();
            g.egui_state.take_egui_input(&g.window)
        };
        let full = ctx.run_ui(raw, |root| {
            if !ui_state.show_panel {
                return;
            }
            let panel = egui::Panel::right("controls")
                .default_size(330.0)
                .resizable(true)
                .show(root, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.heading("rusty_tube");
                        ui.label(egui::RichText::new("tubulaton in Rust: cortical microtubules in 3D cells").weak());
                        ui.separator();
                        let st = &ui_state.stats;
                        let minutes = st.step as f64 * 0.1 / 60.0;
                        egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                            ui.label("step");
                            ui.label(format!("{} ({:.1} min at 0.1 s/step)", st.step, minutes));
                            ui.end_row();
                            ui.label("microtubules");
                            ui.label(format!(
                                "{}  ({} growing, {} shrinking)",
                                st.microtubules, st.growing, st.shrinking
                            ));
                            ui.end_row();
                            ui.label("subunits");
                            ui.label(format!(
                                "{} ({:.1} um of polymer)",
                                st.subunits,
                                st.subunits as f64 / 125.0
                            ));
                            ui.end_row();
                            ui.label("mean length");
                            ui.label(format!("{:.2} um", st.mean_length_um));
                            ui.end_row();
                            ui.label("cortical");
                            ui.label(format!("{:.1} %", st.cortical_fraction * 100.0));
                            ui.end_row();
                            ui.label("order (xy)");
                            ui.label(format!("{:.2}", st.order_xy));
                            ui.end_row();
                            if st.cscs > 0 {
                                ui.label("cellulose synthase");
                                ui.label(format!(
                                    "{} CSCs ({:.2}/um2), {:.0} % on MTs",
                                    st.cscs,
                                    st.csc_density,
                                    st.csc_on_mt * 100.0
                                ));
                                ui.end_row();
                                ui.label("cellulose");
                                ui.label(format!(
                                    "{:.2} um/um2, crystallinity {:.2}",
                                    st.cellulose_um_per_um2, st.crystallinity
                                ));
                                ui.end_row();
                            }
                            ui.label("snapshot");
                            ui.label(format!("{:.1} ms, {} segments", st.snapshot_ms, ui_state.capsules));
                            ui.end_row();
                            ui.label("speed");
                            let sps = if st.seconds_per_step > 0.0 {
                                1.0 / st.seconds_per_step
                            } else {
                                0.0
                            };
                            ui.label(format!(
                                "{:.0} steps/s, {:.0} fps, {} capsules",
                                sps, fps, ui_state.capsules
                            ));
                            ui.end_row();
                        });
                        ui.label(
                            egui::RichText::new(shared.status.lock().unwrap().clone())
                                .small()
                                .weak(),
                        );
                        ui.separator();

                        egui::CollapsingHeader::new("Simulation")
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.horizontal(|ui| {
                                    let running = shared.running.load(Ordering::Relaxed);
                                    if ui.button(if running { "⏸ pause" } else { "▶ run" }).clicked() {
                                        shared.running.store(!running, Ordering::Relaxed);
                                    }
                                    if ui.button("⟲ restart").clicked() {
                                        reset = true;
                                    }
                                });
                                if ui
                                    .add(
                                        egui::Slider::new(&mut ui_state.steps_per_frame, 1..=500)
                                            .logarithmic(true)
                                            .text("steps / frame"),
                                    )
                                    .changed()
                                {
                                    shared
                                        .steps_per_frame
                                        .store(ui_state.steps_per_frame, Ordering::Relaxed);
                                }
                                egui::ComboBox::from_label("cell")
                                    .selected_text(short(&ui_state.cell))
                                    .width(220.0)
                                    .show_ui(ui, |ui| {
                                        for c in &ui_state.cells {
                                            if ui.selectable_label(*c == ui_state.cell, short(c)).clicked() {
                                                ui_state.cell = c.clone();
                                                reset = true;
                                            }
                                        }
                                    });
                                ui.horizontal(|ui| {
                                    ui.add(
                                        egui::DragValue::new(&mut ui_state.seed)
                                            .range(1..=1_000_000)
                                            .prefix("seed "),
                                    );
                                    if ui.checkbox(&mut ui_state.neck_stress, "neck stress").changed() {
                                        reset = true;
                                    }
                                    if ui.checkbox(&mut ui_state.shape_stress, "shape stress").changed() {
                                        reset = true;
                                    }
                                });
                                ui.horizontal(|ui| {
                                    if ui.checkbox(&mut ui_state.csc, "cellulose synthase").changed() {
                                        reset = true;
                                    }
                                    if ui_state.csc && ui.checkbox(&mut ui_state.csc_tether, "CSI1 tether").changed() {
                                        cmds.push(Cmd::Set(
                                            "csc_tether".into(),
                                            format!("{}", ui_state.csc_tether as u8),
                                        ));
                                    }
                                });
                                if ui
                                    .add(egui::Slider::new(&mut ui_state.stride, 1..=16).text("subunits / capsule"))
                                    .changed()
                                {
                                    cmds.push(Cmd::Stride(ui_state.stride));
                                }
                            });

                        egui::CollapsingHeader::new("Dynamics (per 0.1 s step)")
                            .default_open(true)
                            .show(ui, |ui| {
                                let mut set = |ui: &mut egui::Ui,
                                               v: &mut f64,
                                               range: std::ops::RangeInclusive<f64>,
                                               key: &str,
                                               label: &str,
                                               log: bool| {
                                    if ui
                                        .add(egui::Slider::new(v, range).logarithmic(log).text(label))
                                        .changed()
                                    {
                                        cmds.push(Cmd::Set(key.into(), format!("{v}")));
                                    }
                                };
                                set(
                                    ui,
                                    &mut ui_state.p_shrink,
                                    0.0..=0.01,
                                    "proba_shrink",
                                    "catastrophe",
                                    true,
                                );
                                set(ui, &mut ui_state.p_rescue, 0.0..=0.01, "proba_rescue", "rescue", true);
                                set(
                                    ui,
                                    &mut ui_state.p_cross_shrink,
                                    0.0..=1.0,
                                    "proba_crossmicro_shrink",
                                    "catastrophe on steep encounter",
                                    false,
                                );
                                set(
                                    ui,
                                    &mut ui_state.p_cross_cut,
                                    0.0..=0.01,
                                    "proba_crossmicro_cut",
                                    "katanin severing at crossovers",
                                    true,
                                );
                                set(
                                    ui,
                                    &mut ui_state.angle_mb,
                                    0.1..=1.6,
                                    "Angle_mb_limite",
                                    "max turn at membrane (rad)",
                                    false,
                                );
                                set(
                                    ui,
                                    &mut ui_state.influence,
                                    0.0..=2.0,
                                    "part_influence_influence",
                                    "stress steering",
                                    false,
                                );
                                if ui
                                    .add(
                                        egui::Slider::new(&mut ui_state.nucleation, 0.0..=4.0)
                                            .text("nucleation / step (0 = config)"),
                                    )
                                    .changed()
                                    && ui_state.nucleation > 0.0
                                {
                                    cmds.push(Cmd::Set(
                                        "proba_initialisation_par_step".into(),
                                        format!("{}", ui_state.nucleation),
                                    ));
                                }
                            });

                        egui::CollapsingHeader::new("Look").default_open(true).show(ui, |ui| {
                            let s = &mut ui_state.settings;
                            ui.horizontal(|ui| {
                                ui.selectable_value(&mut s.look, Look::Hero, "Hero");
                                ui.selectable_value(&mut s.look, Look::Fluorescence, "Confocal");
                            });
                            if s.look == Look::Hero {
                                egui::ComboBox::from_label("colour")
                                    .selected_text(
                                        [
                                            "orientation",
                                            "dynamics",
                                            "per filament",
                                            "age",
                                            "stress alignment",
                                            "plain",
                                        ][s.colour_mode as usize % 6],
                                    )
                                    .show_ui(ui, |ui| {
                                        for (i, n) in [
                                            "orientation",
                                            "dynamics",
                                            "per filament",
                                            "age",
                                            "stress alignment",
                                            "plain",
                                        ]
                                        .iter()
                                        .enumerate()
                                        {
                                            ui.selectable_value(&mut s.colour_mode, i as u32, *n);
                                        }
                                    });
                                ui.add(
                                    egui::Slider::new(&mut s.mt_radius, 0.0125..=0.1)
                                        .logarithmic(true)
                                        .text("tube radius (um)"),
                                );
                                ui.add(egui::Slider::new(&mut s.tip_glow, 0.0..=3.0).text("plus-end glow"));
                                ui.add(egui::Slider::new(&mut s.membrane_opacity, 0.0..=1.0).text("membrane"));
                                ui.checkbox(&mut s.curvature_tint, "tint membrane by curvature");
                                ui.checkbox(&mut s.stress_tint, "show stress field on membrane");
                                ui.add(egui::Slider::new(&mut s.ao_strength, 0.0..=1.0).text("ambient occlusion"));
                                ui.add(egui::Slider::new(&mut s.shadow_softness, 0.5..=8.0).text("shadow softness"));
                                ui.add(
                                    egui::Slider::new(&mut s.light_yaw, -std::f32::consts::PI..=std::f32::consts::PI)
                                        .text("light azimuth"),
                                );
                                ui.add(egui::Slider::new(&mut s.light_pitch, 0.1..=1.55).text("light elevation"));
                                egui::ComboBox::from_label("background")
                                    .selected_text(["night", "black", "studio"][s.background as usize % 3])
                                    .show_ui(ui, |ui| {
                                        for (i, n) in ["night", "black", "studio"].iter().enumerate() {
                                            ui.selectable_value(&mut s.background, i as u32, *n);
                                        }
                                    });
                            } else {
                                ui.add(egui::Slider::new(&mut s.focus_offset, -8.0..=8.0).text("focus (um)"));
                                ui.add(
                                    egui::Slider::new(&mut s.section, 0.2..=20.0)
                                        .logarithmic(true)
                                        .text("optical section (um)"),
                                );
                                ui.add(egui::Slider::new(&mut s.psf, 0.05..=0.4).text("PSF sigma (um)"));
                                ui.add(
                                    egui::Slider::new(&mut s.fluor_gain, 0.05..=10.0)
                                        .logarithmic(true)
                                        .text("gain"),
                                );
                                ui.add(egui::Slider::new(&mut s.membrane_opacity, 0.0..=2.0).text("membrane dye"));
                            }
                            ui.add(egui::Slider::new(&mut s.clip, 0.0..=1.0).text("cut away above"));
                            ui.add(
                                egui::Slider::new(&mut s.exposure, 0.1..=4.0)
                                    .logarithmic(true)
                                    .text("exposure"),
                            );
                            ui.add(egui::Slider::new(&mut s.bloom, 0.0..=0.4).text("bloom"));
                            ui.add(egui::Slider::new(&mut s.vignette, 0.0..=1.0).text("vignette"));
                            ui.checkbox(&mut s.taa, "temporal accumulation");
                            ui.checkbox(&mut s.show_membrane, "show membrane");
                            ui.horizontal(|ui| {
                                ui.checkbox(&mut s.show_mts, "microtubules");
                                ui.checkbox(&mut s.show_cscs, "CSCs");
                                ui.checkbox(&mut s.show_fibrils, "cellulose");
                            });
                        });

                        ui.collapsing("Camera", |ui| {
                            ui.horizontal(|ui| {
                                if ui.button("top").clicked() {
                                    ui_state.camera.pitch = 1.5;
                                    ui_state.camera.yaw = -std::f32::consts::FRAC_PI_2;
                                }
                                if ui.button("oblique").clicked() {
                                    ui_state.camera.pitch = 0.75;
                                    ui_state.camera.yaw = -0.9;
                                }
                                if ui.button("side").clicked() {
                                    ui_state.camera.pitch = 0.05;
                                }
                                if ui.button("frame").clicked() {
                                    ui_state.framed = false;
                                }
                            });
                            ui.add(egui::Slider::new(&mut ui_state.camera.fov, 0.15..=1.2).text("field of view"));
                            ui.label(
                                egui::RichText::new("drag: orbit · right-drag: pan · scroll: zoom")
                                    .small()
                                    .weak(),
                            );
                        });

                        ui.collapsing("Screenshot", |ui| {
                            ui.add(egui::Slider::new(&mut ui_state.shot_scale, 1..=4).text("x window size"));
                            ui.add(
                                egui::Slider::new(&mut ui_state.shot_spp, 16..=1024)
                                    .logarithmic(true)
                                    .text("samples"),
                            );
                            if ui.button("📷 save PNG").clicked() {
                                shot = true;
                            }
                            ui.label(egui::RichText::new(&ui_state.last_shot).small());
                        });
                        ui.separator();
                        ui.label(
                            egui::RichText::new("space: run/pause · h: hide panel · s: screenshot · c: look")
                                .small()
                                .weak(),
                        );
                    });
                });
            panel_w = panel.response.rect.width();
        });
        self.panel_px = panel_w;
        let g = self.gfx.as_mut().unwrap();
        g.egui_state
            .handle_platform_output(&g.window, full.platform_output.clone());
        let ppp = full.pixels_per_point;
        let prims = ctx.tessellate(full.shapes, ppp);
        self.paint(prims, full.textures_delta, ppp);
        if reset {
            self.reset();
        }
        if shot {
            self.screenshot();
        }
        cmds
    }

    fn paint(&mut self, prims: Vec<egui::ClippedPrimitive>, textures: egui::TexturesDelta, ppp: f32) {
        let g = self.gfx.as_mut().unwrap();
        // Upload egui textures first: a skipped frame must not drop the
        // initial full upload that later partial updates depend on.
        {
            let device = g.renderer.device.clone();
            let queue = g.renderer.queue.clone();
            for (id, deltas) in &textures.set {
                for delta in deltas.iter() {
                    g.egui_renderer.update_texture(&device, &queue, *id, delta);
                }
            }
        }
        let capturing = self.spec.capture.is_some();
        // In capture mode draw into an offscreen texture: an occluded
        // window hands out no surface frames.
        let (frame, offscreen) = if capturing {
            let tex = g.renderer.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("capture"),
                size: wgpu::Extent3d {
                    width: g.config.width,
                    height: g.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: g.config.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            (None, Some(tex))
        } else {
            match g.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(f) | wgpu::CurrentSurfaceTexture::Suboptimal(f) => (Some(f), None),
                _ => {
                    g.surface.configure(&g.renderer.device, &g.config);
                    return;
                }
            }
        };
        let target_tex: &wgpu::Texture = match (&frame, &offscreen) {
            (Some(f), _) => &f.texture,
            (_, Some(t)) => t,
            _ => unreachable!(),
        };
        let view = target_tex.create_view(&Default::default());
        let device = g.renderer.device.clone();
        let queue = g.renderer.queue.clone();
        let mut enc = device.create_command_encoder(&Default::default());
        let mut settings = self.ui.settings.clone();
        if self.ui.show_panel {
            settings.view_shift = (self.panel_px * ppp / g.config.width as f32).min(0.6);
        }
        g.renderer.render(&mut enc, &view, &self.ui.camera, &settings);
        let sd = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [g.config.width, g.config.height],
            pixels_per_point: ppp,
        };
        let extra = g.egui_renderer.update_buffers(&device, &queue, &mut enc, &prims, &sd);
        {
            let mut pass = enc
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                })
                .forget_lifetime();
            g.egui_renderer.render(&mut pass, &prims, &sd);
        }
        if self.seen_version > 0 {
            self.frames += 1;
        }
        let capture = match &self.spec.capture {
            Some((n, path)) if self.frames == *n => Some(path.clone()),
            _ => None,
        };
        let (w, h) = (g.config.width, g.config.height);
        let padded = (w * 4).div_ceil(256) * 256;
        let readback = capture.as_ref().map(|_| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (padded * h) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_texture_to_buffer(
                target_tex.as_image_copy(),
                wgpu::TexelCopyBufferInfo {
                    buffer: &buf,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(padded),
                        rows_per_image: Some(h),
                    },
                },
                wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
            );
            buf
        });
        queue.submit(extra.into_iter().chain([enc.finish()]));
        if let (Some(buf), Some(path)) = (readback, capture) {
            let slice = buf.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            if let Ok(data) = slice.get_mapped_range() {
                let bgra = matches!(
                    g.config.format,
                    wgpu::TextureFormat::Bgra8UnormSrgb | wgpu::TextureFormat::Bgra8Unorm
                );
                let mut img = image::RgbaImage::new(w, h);
                for y in 0..h {
                    for x in 0..w {
                        let i = (y * padded + x * 4) as usize;
                        let px = if bgra {
                            [data[i + 2], data[i + 1], data[i], 255]
                        } else {
                            [data[i], data[i + 1], data[i + 2], 255]
                        };
                        img.put_pixel(x, y, image::Rgba(px));
                    }
                }
                let _ = img.save(&path);
                println!("captured {}", path.display());
            }
            self.exit_requested = true;
        }
        if let Some(frame) = frame {
            g.window.pre_present_notify();
            queue.present(frame);
        }
        for id in &textures.free {
            g.egui_renderer.free_texture(id);
        }
    }

    fn redraw(&mut self) {
        let dt = self.last_frame.elapsed().as_secs_f32();
        self.last_frame = Instant::now();
        self.fps = self.fps * 0.9 + 0.1 / dt.max(1e-4);
        let new_membrane = self.shared.membrane.lock().unwrap().take();
        if let Some(m) = new_membrane {
            let g = self.gfx.as_mut().unwrap();
            g.renderer.set_membrane(&m);
            self.have_membrane = true;
            if !self.ui.framed {
                let (pitch, yaw) = (self.ui.camera.pitch, self.ui.camera.yaw);
                self.ui.camera = OrbitCamera::framing(m.bounds_min, m.bounds_max);
                self.ui.camera.pitch = pitch;
                self.ui.camera.yaw = yaw;
                self.ui.camera.distance *= 1.25;
                self.ui.framed = true;
            }
            *self.shared.membrane.lock().unwrap() = Some(m);
        }
        let v = self.shared.version.load(Ordering::Relaxed);
        if v != self.seen_version {
            if let Some(snap) = self.shared.latest.lock().unwrap().clone() {
                let g = self.gfx.as_mut().unwrap();
                g.renderer.set_segments(&snap.segments);
                self.ui.stats = snap.stats.clone();
                self.ui.capsules = snap.segments.len();
            }
            self.seen_version = v;
        }
        let ctx = self.egui_ctx.clone();
        let cmds = self.draw_ui(&ctx);
        for c in cmds {
            self.send(c);
        }
    }
}

fn short(path: &str) -> String {
    std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| path.to_string())
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gfx.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("rusty_tube — microtubules in pavement cells")
            .with_inner_size(winit::dpi::LogicalSize::new(1500.0, 950.0));
        let window = Arc::new(event_loop.create_window(attrs).expect("window"));
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).expect("surface");
        let (adapter, device, queue) = pollster::block_on(crate::render::gpu(Some(&surface), &instance)).expect("gpu");
        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        let usage = if caps.usages.contains(wgpu::TextureUsages::COPY_SRC) {
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC
        } else {
            wgpu::TextureUsages::RENDER_ATTACHMENT
        };
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            color_space: Default::default(),
        };
        surface.configure(&device, &config);
        let renderer = Renderer::new(device.clone(), queue, format, (config.width, config.height));
        let egui_state = egui_winit::State::new(
            self.egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            None,
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, format, egui_wgpu::RendererOptions::default());
        self.gfx = Some(Gfx {
            window,
            surface,
            config,
            renderer,
            egui_state,
            egui_renderer,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(g) = self.gfx.as_mut() else { return };
        let resp = g.egui_state.on_window_event(&g.window, &event);
        let wants_pointer = self.egui_ctx.egui_wants_pointer_input() || self.egui_ctx.is_pointer_over_egui();
        let wants_kb = self.egui_ctx.egui_wants_keyboard_input();
        match event {
            WindowEvent::CloseRequested => {
                self.shared.quit.store(true, Ordering::Relaxed);
                if let Some(t) = self.thread.take() {
                    let _ = t.join();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                g.config.width = size.width.max(1);
                g.config.height = size.height.max(1);
                g.surface.configure(&g.renderer.device, &g.config);
                g.renderer.resize((g.config.width, g.config.height));
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => {
                let (dx, dy) = (position.x - self.mouse.0, position.y - self.mouse.1);
                self.mouse = (position.x, position.y);
                let h = g.config.height as f32;
                if self.buttons.0 {
                    self.ui.camera.orbit(dx as f32, dy as f32);
                } else if self.buttons.1 {
                    self.ui.camera.pan(dx as f32, dy as f32, h);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                if down && wants_pointer {
                    return;
                }
                match button {
                    MouseButton::Left => self.buttons.0 = down,
                    MouseButton::Right | MouseButton::Middle => self.buttons.1 = down,
                    _ => {}
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if wants_pointer {
                    return;
                }
                let amount = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                self.ui.camera.zoom(amount);
            }
            WindowEvent::KeyboardInput { event, .. }
                if event.state == ElementState::Pressed && !wants_kb && !resp.consumed =>
            {
                match event.logical_key.as_ref() {
                    Key::Named(NamedKey::Space) => {
                        let r = self.shared.running.load(Ordering::Relaxed);
                        self.shared.running.store(!r, Ordering::Relaxed);
                    }
                    Key::Character("h") => self.ui.show_panel = !self.ui.show_panel,
                    Key::Character("s") => self.screenshot(),
                    Key::Character("c") => {
                        self.ui.settings.look = match self.ui.settings.look {
                            Look::Hero => Look::Fluorescence,
                            Look::Fluorescence => Look::Hero,
                        }
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.exit_requested {
            self.shared.quit.store(true, Ordering::Relaxed);
            if let Some(t) = self.thread.take() {
                let _ = t.join();
            }
            event_loop.exit();
            return;
        }
        if let Some(g) = self.gfx.as_ref() {
            g.window.request_redraw();
        }
    }
}

pub fn run(spec: SimSpec) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new(spec);
    event_loop.run_app(&mut app)?;
    Ok(())
}
