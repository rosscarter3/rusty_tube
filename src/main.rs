use anyhow::{bail, Result};
use clap::{Parser, Subcommand, ValueEnum};
use indicatif::{ProgressBar, ProgressStyle};
use rusty_tube::cellgen::{self, CellSpec};
use rusty_tube::render::camera::OrbitCamera;
use rusty_tube::render::offline::Offscreen;
use rusty_tube::render::{scene, Look, Settings};
use rusty_tube::setup::{self, SimSetup};
use std::path::PathBuf;

#[derive(Parser)]
#[command(version, about = "tubulaton in Rust: microtubule self-organisation in 3D cells")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum)]
enum LookArg {
    Hero,
    Fluor,
}

#[derive(clap::Args)]
struct SimArgs {
    /// tubulaton .ini config (default: the calibrated pavement-cell set)
    #[arg(long)]
    config: Option<PathBuf>,
    /// Generate the cell from an outline: a CSV of x,y (um), a pavesim JSON,
    /// or procedural:SEED. Overrides the config's input surface.
    #[arg(long)]
    outline: Option<String>,
    /// Override config keys, e.g. -s seed=3 -s proba_shrink=0.001
    #[arg(short = 's', long = "set")]
    set: Vec<String>,
    /// Impose a neck-high stress field (strain_0) on a generated cell.
    #[arg(long)]
    neck_stress: bool,
    /// Impose the shape-derived stress field (inflated-membrane proxy) whose
    /// magnitude is the local stress anisotropy.
    #[arg(long)]
    shape_stress: bool,
    /// Scale a generated outline.
    #[arg(long, default_value_t = 1.0)]
    scale: f64,
    /// Parameter preset: calibrated (default) or dense (illustrative).
    #[arg(long, default_value = "calibrated")]
    preset: String,
}

impl SimArgs {
    fn all_sets(&self) -> Result<Vec<String>> {
        let mut v = setup::preset(&self.preset)?;
        v.extend(self.set.iter().cloned());
        Ok(v)
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Run a tubulaton config headless, writing the same report and VTK
    /// files the C++ writes.
    Run {
        config: PathBuf,
        #[arg(short = 's', long = "set")]
        set: Vec<String>,
    },
    /// Build a closed 3D pavement-cell surface from an outline.
    Mesh {
        #[arg(long, default_value = "procedural:7")]
        outline: String,
        #[arg(long)]
        out: PathBuf,
        #[arg(long)]
        neck_stress: bool,
        #[arg(long, default_value_t = 0.3)]
        spacing_um: f64,
        #[arg(long, default_value_t = 5.0)]
        height_um: f64,
        #[arg(long, default_value_t = 1.0)]
        scale: f64,
    },
    /// Interactive viewer: simulate and render live.
    View {
        #[command(flatten)]
        sim: SimArgs,
        /// Save the window to a PNG after N frames and exit.
        #[arg(long, num_args = 2, value_names = ["FRAMES", "PNG"])]
        capture_after: Vec<String>,
        /// Simulate this many steps before showing the cell.
        #[arg(long, default_value_t = 0)]
        warmup: i64,
        #[arg(long, value_enum, default_value = "hero")]
        look: LookArg,
        /// 0 orientation, 1 dynamics, 2 per filament, 3 age, 4 stress, 5 plain
        #[arg(long, default_value_t = 0)]
        colour: u32,
        /// Renderer settings, as for `render`.
        #[arg(short = 'r', long = "render-set")]
        rset: Vec<String>,
        #[arg(long, allow_hyphen_values = true)]
        yaw: Option<f32>,
        #[arg(long)]
        pitch: Option<f32>,
    },
    /// Simulate, then render stills or a time-lapse to PNG.
    Render {
        #[command(flatten)]
        sim: SimArgs,
        /// Steps to simulate before the (first) frame.
        #[arg(long, default_value_t = 3000)]
        steps: i64,
        /// Write a frame every N steps after the first, this many times.
        #[arg(long, default_value_t = 1)]
        frames: u32,
        #[arg(long, default_value_t = 100)]
        every: i64,
        #[arg(long, default_value = "out/render.png")]
        out: PathBuf,
        #[arg(long, default_value_t = 1920)]
        width: u32,
        #[arg(long, default_value_t = 1080)]
        height: u32,
        /// Accumulated samples per pixel.
        #[arg(long, default_value_t = 64)]
        spp: u32,
        #[arg(long, value_enum, default_value = "hero")]
        look: LookArg,
        /// 0 orientation, 1 dynamics, 2 per filament, 3 age, 4 stress, 5 plain
        #[arg(long, default_value_t = 0)]
        colour: u32,
        #[arg(long)]
        yaw: Option<f32>,
        #[arg(long)]
        pitch: Option<f32>,
        /// Camera distance as a multiple of the framing distance.
        #[arg(long, default_value_t = 1.0)]
        zoom: f32,
        /// Cut the cell open above this fraction of its height.
        #[arg(long, default_value_t = 1.0)]
        clip: f32,
        /// Aim at the middle of the longest microtubule (at the first frame).
        #[arg(long)]
        look_at_mt: bool,
        /// Camera distance in um (overrides --zoom).
        #[arg(long)]
        distance: Option<f32>,
        /// Frame rate of an .mp4 output.
        #[arg(long, default_value_t = 30)]
        fps: u32,
        /// Turn the camera this many degrees per frame (time-lapses).
        #[arg(long, default_value_t = 0.0)]
        orbit: f32,
        /// Subunits per drawn capsule.
        #[arg(long, default_value_t = 3)]
        stride: usize,
        /// Renderer settings, e.g. -r exposure=1.3 -r mt_radius=0.02
        #[arg(short = 'r', long = "render-set")]
        rset: Vec<String>,
    },
}

fn apply_render_sets(s: &mut Settings, sets: &[String]) -> Result<()> {
    for kv in sets {
        let Some((k, v)) = kv.split_once('=') else {
            bail!("-r expects key=value")
        };
        let f: f32 = v.parse().unwrap_or(0.0);
        match k {
            "exposure" => s.exposure = f,
            "bloom" => s.bloom = f,
            "ao" => s.ao_strength = f,
            "ao_radius" => s.ao_radius = f,
            "mt_radius" => s.mt_radius = f,
            "membrane" => s.membrane_opacity = f,
            "show_membrane" => s.show_membrane = f != 0.0,
            "tip_glow" => s.tip_glow = f,
            "shadow" => s.shadow_softness = f,
            "vignette" => s.vignette = f,
            "grain" => s.grain = f,
            "gain" => s.fluor_gain = f,
            "focus" => s.focus_offset = f,
            "section" => s.section = f,
            "psf" => s.psf = f,
            "background" => s.background = f as u32,
            "curvature" => s.curvature_tint = f != 0.0,
            "stress_tint" => s.stress_tint = f != 0.0,
            "saturation" => s.saturation = f,
            "contrast" => s.contrast = f,
            "light_yaw" => s.light_yaw = f,
            "light_pitch" => s.light_pitch = f,
            "light" => s.light_intensity = f,
            "aperture" => s.aperture = f,
            "show_cscs" => s.show_cscs = f != 0.0,
            "csc_scale" => s.csc_scale = f,
            "show_fibrils" => s.show_fibrils = f != 0.0,
            "show_mts" => s.show_mts = f != 0.0,
            "focus_dist" => s.focus_dist = f,
            _ => bail!("unknown render setting {k}"),
        }
    }
    Ok(())
}

/// A real pavement cell if one has been exported, else a procedural one.
fn default_cell() -> String {
    let real = "cells/real/WT_FR_312h_cell61.csv";
    if std::path::Path::new(real).exists() {
        real.into()
    } else {
        "procedural:7".into()
    }
}

fn main() -> Result<()> {
    rusty_tube::qos::user_initiated();
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Run { config, set } => {
            let sim_setup = SimSetup {
                config: Some(&config),
                outline: None,
                sets: &set,
                neck_stress: false,
                shape_stress: false,
                scale: 1.0,
            };
            let mut sim = setup::build(&sim_setup)?;
            let out_dir = sim.params.get_s("nom_folder_output_vtk").to_string();
            if !out_dir.is_empty() {
                std::fs::create_dir_all(&out_dir).ok();
            }
            let n = sim.params.get_i("nb_max_steps");
            let pb = ProgressBar::new(n as u64 + 1);
            pb.set_style(
                ProgressStyle::default_bar()
                    .template("{spinner:.green} {elapsed_precise} {bar:40} {pos}/{len} ({eta}) {msg}")?,
            );
            let start = std::time::Instant::now();
            for t in 0..=n {
                if !sim.tick(t, true)? {
                    break;
                }
                if t % 50 == 0 {
                    pb.set_position(t as u64);
                    pb.set_message(format!("{} MTs, {} subunits", sim.mts.len(), sim.n_elements()));
                }
            }
            pb.finish();
            let lens = sim.lengths();
            let mean = lens.iter().sum::<usize>() as f64 / lens.len().max(1) as f64;
            println!(
                "done in {:.1} s: {} microtubules, {} subunits, mean length {:.1} subunits, cortical fraction {:.3}",
                start.elapsed().as_secs_f64(),
                sim.mts.len(),
                sim.n_elements(),
                mean,
                sim.cortical_fraction()
            );
        }
        Cmd::Mesh {
            outline,
            out,
            neck_stress,
            spacing_um,
            height_um,
            scale,
        } => {
            let mut o = setup::load_outline(&outline)?;
            for p in &mut o.pts {
                *p *= scale;
            }
            let spec = CellSpec {
                spacing_um,
                height_um,
                neck_stress,
                ..Default::default()
            };
            let cell = cellgen::build_cell(&o, &spec)?;
            let res = cell.mesh.vertex_areas();
            println!(
                "{} vertices, {} triangles, euler {}, area {:.1} um^2, covering radius {:.1} units",
                cell.mesh.positions.len(),
                cell.mesh.triangles.len(),
                cell.mesh.euler_characteristic(),
                cell.mesh.area() / cellgen::UNITS_PER_UM.powi(2),
                res.covering_radius,
            );
            rusty_tube::vtk::write_surface(&out, &cell.mesh, &cell.normals)?;
            println!("wrote {}", out.display());
        }
        Cmd::View {
            sim: a,
            capture_after,
            warmup,
            look,
            colour,
            rset,
            yaw,
            pitch,
        } => {
            let mut settings = Settings {
                colour_mode: colour,
                ..Default::default()
            };
            if let LookArg::Fluor = look {
                settings.look = Look::Fluorescence;
                settings.background = 1;
                settings.bloom = 0.15;
            }
            apply_render_sets(&mut settings, &rset)?;
            let capture = match capture_after.as_slice() {
                [n, p] => Some((n.parse()?, PathBuf::from(p))),
                _ => None,
            };
            let sets = a.all_sets()?;
            let outline = a
                .outline
                .or_else(|| if a.config.is_none() { Some(default_cell()) } else { None });
            rusty_tube::viewer::run(rusty_tube::viewer::SimSpec {
                capture,
                warmup,
                settings,
                angles: if yaw.is_some() || pitch.is_some() {
                    Some((yaw.unwrap_or(-0.9), pitch.unwrap_or(0.75)))
                } else {
                    None
                },
                config: a.config,
                outline,
                sets,
                neck_stress: a.neck_stress,
                shape_stress: a.shape_stress,
                scale: a.scale,
            })?;
        }
        Cmd::Render {
            sim: a,
            steps,
            frames,
            every,
            out,
            width,
            height,
            spp,
            look,
            colour,
            yaw,
            pitch,
            zoom,
            clip,
            look_at_mt,
            distance,
            orbit,
            fps,
            stride,
            rset,
        } => {
            let all_sets = a.all_sets()?;
            let sim_setup = SimSetup {
                config: a.config.as_deref(),
                outline: a.outline.as_deref(),
                sets: &all_sets,
                neck_stress: a.neck_stress,
                shape_stress: a.shape_stress,
                scale: a.scale,
            };
            let mut sim = setup::build(&sim_setup)?;
            let frame = scene::Frame::from_sim(&sim);
            let mut off = Offscreen::new(width, height)?;
            let mem = scene::membrane_geometry(&sim, &frame);
            off.renderer.set_membrane(&mem);
            let mut cam = OrbitCamera::framing(mem.bounds_min, mem.bounds_max);
            if let Some(y) = yaw {
                cam.yaw = y;
            }
            if let Some(p) = pitch {
                cam.pitch = p;
            }
            cam.distance *= zoom;
            let mut s = Settings {
                look: match look {
                    LookArg::Hero => Look::Hero,
                    LookArg::Fluor => Look::Fluorescence,
                },
                colour_mode: colour,
                clip,
                ..Default::default()
            };
            if s.look == Look::Fluorescence {
                s.background = 1;
                s.bloom = 0.15;
            }
            apply_render_sets(&mut s, &rset)?;
            if let Some(dir) = out.parent() {
                std::fs::create_dir_all(dir).ok();
            }
            // A .mp4 output renders frames into a scratch folder, encodes
            // them, and deletes them.
            let movie = out.extension().map(|e| e == "mp4").unwrap_or(false);
            let frame_dir = out.with_extension("frames");
            if movie {
                std::fs::create_dir_all(&frame_dir)?;
            }
            let t0 = std::time::Instant::now();
            let mut t = 0;
            let pb = ProgressBar::new(steps as u64);
            while t < steps {
                sim.tick(t, false)?;
                t += 1;
                if t % 100 == 0 {
                    pb.set_position(t as u64);
                }
            }
            pb.finish_and_clear();
            println!("simulated {steps} steps in {:.1} s", t0.elapsed().as_secs_f64());
            for f in 0..frames {
                if f > 0 {
                    for _ in 0..every {
                        sim.tick(t, false)?;
                        t += 1;
                    }
                }
                cam.yaw += orbit.to_radians() * if f > 0 { 1.0 } else { 0.0 };
                let snap = scene::snapshot(&sim, &frame, stride);
                if look_at_mt && f == 0 {
                    if let Some(mt) = sim.mts.values().max_by_key(|m| m.body.len()) {
                        let mid = mt.body[mt.body.len() / 2];
                        cam.target = frame.to_render(sim.pool.elems[mid as usize].pos);
                    }
                }
                if let Some(d) = distance {
                    cam.distance = d;
                }
                off.renderer.set_segments(&snap.segments);
                let path = if movie {
                    frame_dir.join(format!("f_{f:05}.png"))
                } else if frames > 1 {
                    let stem = out.file_stem().unwrap().to_string_lossy();
                    out.with_file_name(format!("{stem}_{f:04}.png"))
                } else {
                    out.clone()
                };
                let t1 = std::time::Instant::now();
                off.save(&cam, &s, spp, &path)?;
                let st = &snap.stats;
                println!(
                    "{} step {}: {} MTs, {} capsules, mean length {:.2} um, cortical {:.3}, order {:.2}, stress alignment {:.3} ({:.2} s)",
                    path.display(),
                    t,
                    st.microtubules,
                    snap.segments.len(),
                    st.mean_length_um,
                    st.cortical_fraction,
                    st.order_xy,
                    st.stress_alignment,
                    t1.elapsed().as_secs_f64()
                );
            }
            if movie {
                encode_movie(&frame_dir, &out, fps)?;
                std::fs::remove_dir_all(&frame_dir)?;
                println!("wrote {}", out.display());
            }
        }
    }
    Ok(())
}

/// Encode `dir/f_%05d.png` to H.264. Uses $FFMPEG if set, else `ffmpeg`.
fn encode_movie(dir: &std::path::Path, out: &std::path::Path, fps: u32) -> Result<()> {
    let ffmpeg = std::env::var("FFMPEG").unwrap_or_else(|_| "ffmpeg".into());
    let status = std::process::Command::new(&ffmpeg)
        .args(["-y", "-loglevel", "error", "-framerate", &fps.to_string(), "-i"])
        .arg(dir.join("f_%05d.png"))
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"])
        .arg(out)
        .status();
    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => bail!("{ffmpeg} failed ({s}); frames kept in {}", dir.display()),
        Err(e) => bail!(
            "could not run {ffmpeg} ({e}); set FFMPEG; frames kept in {}",
            dir.display()
        ),
    }
}
