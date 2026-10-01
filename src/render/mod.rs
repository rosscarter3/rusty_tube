//! Real-time renderer for microtubule arrays in cell geometries.
//!
//! Two looks share one frame graph:
//!
//! * **Hero**: microtubules as ray-traced capsule impostors in a G-buffer,
//!   lit by a soft-shadowed key light, a hemisphere fill with SSAO, GGX and a
//!   fibre rim; the cell membrane as a translucent glass shell; plus ends
//!   glowing as EB1-like comets.
//! * **Fluorescence**: what a confocal microscope would record of a
//!   GFP-tubulin line with a membrane dye: every subunit splatted through a
//!   point-spread function, weighted by an optical section that can be
//!   focused through the cell.
//!
//! Both are temporally accumulated (TAA while moving, progressive
//! supersampling while still), bloomed, and AgX tone mapped.

pub mod camera;
pub mod offline;
pub mod scene;

use bytemuck::{Pod, Zeroable};
use camera::OrbitCamera;
use glam::{Mat4, Vec2, Vec3, Vec4};
use scene::{MembraneGeometry, MeshVertex, Segment};
use wgpu::util::DeviceExt;

const HDR: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_SIZE: u32 = 2048;
const BLOOM_LEVELS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Look {
    Hero,
    Fluorescence,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    pub look: Look,
    /// 0 orientation, 1 dynamics, 2 per filament, 3 age, 4 stress alignment, 5 plain.
    pub colour_mode: u32,
    pub exposure: f32,
    pub bloom: f32,
    pub ao_strength: f32,
    pub ao_radius: f32,
    /// Drawn radius of a microtubule, um. The real one is 0.0125 um.
    pub mt_radius: f32,
    pub membrane_opacity: f32,
    pub show_membrane: bool,
    /// Cut the cell open above this fraction of its height (1 = closed).
    pub clip: f32,
    pub tip_glow: f32,
    pub shadow_softness: f32,
    pub vignette: f32,
    pub grain: f32,
    pub fluor_gain: f32,
    /// Focal plane offset from the camera target along the view, um.
    pub focus_offset: f32,
    /// Optical section half-thickness, um.
    pub section: f32,
    /// Point-spread function sigma, um.
    pub psf: f32,
    /// 0 dark gradient, 1 black, 2 studio white.
    pub background: u32,
    pub curvature_tint: bool,
    /// Show the stress field (anisotropy heat and field lines) on the membrane.
    pub stress_tint: bool,
    pub saturation: f32,
    pub contrast: f32,
    pub light_yaw: f32,
    pub light_pitch: f32,
    pub light_intensity: f32,
    pub taa: bool,
    /// Shift the image centre horizontally, as a fraction of the half-width
    /// (to centre the cell beside a side panel).
    pub view_shift: f32,
    /// Lens aperture radius (um) for depth of field; applied while the
    /// scene is still, so accumulation integrates over the lens.
    pub aperture: f32,
    /// Focus distance relative to the orbit target (um).
    pub focus_dist: f32,
    pub show_cscs: bool,
    /// Drawn size of CSCs relative to the default (wide shots need more).
    pub csc_scale: f32,
    pub show_fibrils: bool,
    pub show_mts: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            look: Look::Hero,
            colour_mode: 0,
            exposure: 1.1,
            bloom: 0.04,
            ao_strength: 0.9,
            ao_radius: 0.35,
            mt_radius: 0.03,
            membrane_opacity: 0.35,
            show_membrane: true,
            clip: 1.0,
            tip_glow: 1.0,
            shadow_softness: 2.5,
            vignette: 0.35,
            grain: 0.6,
            fluor_gain: 1.0,
            focus_offset: 0.0,
            section: 0.6,
            psf: 0.12,
            background: 0,
            curvature_tint: false,
            stress_tint: false,
            saturation: 1.1,
            contrast: 1.05,
            light_yaw: 0.6,
            light_pitch: 0.95,
            light_intensity: 3.2,
            taa: true,
            view_shift: 0.0,
            aperture: 0.0,
            focus_dist: 0.0,
            show_cscs: true,
            csc_scale: 1.0,
            show_fibrils: true,
            show_mts: true,
        }
    }
}

fn att(v: &wgpu::TextureView, ops: wgpu::Operations<wgpu::Color>) -> Option<wgpu::RenderPassColorAttachment<'_>> {
    Some(wgpu::RenderPassColorAttachment {
        view: v,
        depth_slice: None,
        resolve_target: None,
        ops,
    })
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    prev_view_proj: [[f32; 4]; 4],
    view: [[f32; 4]; 4],
    light_view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 4],
    light_dir: [f32; 4],
    resolution: [f32; 4],
    jitter: [f32; 4],
    p0: [f32; 4],
    p1: [f32; 4],
    p2: [f32; 4],
    p3: [f32; 4],
    p4: [f32; 4],
    p5: [f32; 4],
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
}

struct Target {
    tex: wgpu::Texture,
    view: wgpu::TextureView,
}

fn target(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat) -> Target {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: w.max(1),
            height: h.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = tex.create_view(&Default::default());
    Target { tex, view }
}

/// Size-dependent render targets and the bind groups that read them.
struct Targets {
    size: (u32, u32),
    depth: Target,
    normal: Target,
    albedo: Target,
    aux: Target,
    ao: Target,
    hdr: Target,
    history: [Target; 2],
    bloom: Vec<Target>,
    ssao_bg: wgpu::BindGroup,
    lighting_bg: wgpu::BindGroup,
    taa_bg: [wgpu::BindGroup; 2],
    bloom_first_bg: [wgpu::BindGroup; 2],
    bloom_bg: Vec<wgpu::BindGroup>,
    composite_bg: [wgpu::BindGroup; 2],
}

#[derive(Clone, Copy)]
enum B {
    Uniform,
    Storage,
    Tex,
    Depth,
    Sampler,
    Compare,
}

fn bgl(device: &wgpu::Device, label: &str, entries: &[B]) -> wgpu::BindGroupLayout {
    let e: Vec<wgpu::BindGroupLayoutEntry> = entries
        .iter()
        .enumerate()
        .map(|(i, b)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
            ty: match b {
                B::Uniform => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                B::Storage => wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                B::Tex => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                B::Depth => wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                B::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                B::Compare => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
            },
            count: None,
        })
        .collect();
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &e,
    })
}

enum R<'a> {
    View(&'a wgpu::TextureView),
    Samp(&'a wgpu::Sampler),
    Buf(&'a wgpu::Buffer),
}

fn bg(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, res: &[R]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = res
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: match r {
                R::View(v) => wgpu::BindingResource::TextureView(v),
                R::Samp(s) => wgpu::BindingResource::Sampler(s),
                R::Buf(b) => b.as_entire_binding(),
            },
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout,
        entries: &entries,
    })
}

struct Layouts {
    frame: wgpu::BindGroupLayout,
    segs: wgpu::BindGroupLayout,
    membrane: wgpu::BindGroupLayout,
    ssao: wgpu::BindGroupLayout,
    lighting: wgpu::BindGroupLayout,
    taa: wgpu::BindGroupLayout,
    tex_samp: wgpu::BindGroupLayout,
    composite: wgpu::BindGroupLayout,
}

struct Pipelines {
    capsule: wgpu::RenderPipeline,
    capsule_shadow: wgpu::RenderPipeline,
    membrane_back: wgpu::RenderPipeline,
    membrane_front: wgpu::RenderPipeline,
    membrane_fluor: wgpu::RenderPipeline,
    ssao: wgpu::RenderPipeline,
    lighting: wgpu::RenderPipeline,
    splat: wgpu::RenderPipeline,
    taa: wgpu::RenderPipeline,
    bloom_first: wgpu::RenderPipeline,
    bloom_down: wgpu::RenderPipeline,
    bloom_up: wgpu::RenderPipeline,
    composite: wgpu::RenderPipeline,
}

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    out_format: wgpu::TextureFormat,
    layouts: Layouts,
    pipes: Pipelines,
    frame_buf: wgpu::Buffer,
    frame_bg: wgpu::BindGroup,
    seg_buf: wgpu::Buffer,
    seg_cap: usize,
    seg_count: u32,
    seg_bg: wgpu::BindGroup,
    mem_vbuf: Option<wgpu::Buffer>,
    mem_ibuf: Option<wgpu::Buffer>,
    mem_count: u32,
    bounds: (Vec3, Vec3),
    shadow: Target,
    shadow_samp: wgpu::Sampler,
    linear: wgpu::Sampler,
    membrane_bg: wgpu::BindGroup,
    targets: Targets,
    frame_index: u32,
    history_cur: usize,
    accum: u32,
    last_state: Option<(OrbitCamera, Settings, (u32, u32))>,
    prev_view_proj: Mat4,
    scene_dirty: bool,
    last_segments: Vec<Segment>,
    glyphs: Vec<Segment>,
    glyphs_shown: bool,
}

fn shader(device: &wgpu::Device, label: &str, body: &str) -> wgpu::ShaderModule {
    let src = format!("{}\n{}", include_str!("shaders/common.wgsl"), body);
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(src.into()),
    })
}

#[allow(clippy::too_many_arguments)]
fn pipeline(
    device: &wgpu::Device,
    label: &str,
    module: &wgpu::ShaderModule,
    vs: &str,
    fs: Option<&str>,
    layouts: &[&wgpu::BindGroupLayout],
    buffers: &[Option<wgpu::VertexBufferLayout>],
    targets: &[Option<wgpu::ColorTargetState>],
    depth: Option<wgpu::DepthStencilState>,
    cull: Option<wgpu::Face>,
) -> wgpu::RenderPipeline {
    let bgls: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(|l| Some(*l)).collect();
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &bgls,
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vs),
            compilation_options: Default::default(),
            buffers,
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: cull,
            ..Default::default()
        },
        depth_stencil: depth,
        multisample: Default::default(),
        fragment: fs.map(|f| wgpu::FragmentState {
            module,
            entry_point: Some(f),
            compilation_options: Default::default(),
            targets,
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn color(format: wgpu::TextureFormat, blend: Option<wgpu::BlendState>) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState {
        format,
        blend,
        write_mask: wgpu::ColorWrites::ALL,
    })
}

const ADDITIVE: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> Option<wgpu::DepthStencilState> {
    Some(wgpu::DepthStencilState {
        format: DEPTH,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: Default::default(),
        bias: Default::default(),
    })
}

impl Renderer {
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, out_format: wgpu::TextureFormat, size: (u32, u32)) -> Self {
        let layouts = Layouts {
            frame: bgl(&device, "frame", &[B::Uniform]),
            segs: bgl(&device, "segs", &[B::Storage]),
            membrane: bgl(&device, "membrane", &[B::Depth, B::Compare]),
            ssao: bgl(&device, "ssao", &[B::Depth, B::Tex]),
            lighting: bgl(
                &device,
                "lighting",
                &[B::Depth, B::Tex, B::Tex, B::Tex, B::Tex, B::Depth, B::Compare],
            ),
            taa: bgl(&device, "taa", &[B::Tex, B::Tex, B::Depth, B::Sampler]),
            tex_samp: bgl(&device, "tex_samp", &[B::Tex, B::Sampler]),
            composite: bgl(&device, "composite", &[B::Tex, B::Tex, B::Sampler]),
        };
        let pipes = Self::build_pipelines(&device, &layouts, out_format);

        let frame_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("frame"),
            size: std::mem::size_of::<FrameUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let frame_bg = bg(&device, &layouts.frame, &[R::Buf(&frame_buf)]);
        let seg_cap = 1024;
        let seg_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("segments"),
            size: (seg_cap * std::mem::size_of::<Segment>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let seg_bg = bg(&device, &layouts.segs, &[R::Buf(&seg_buf)]);

        let shadow = {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("shadow"),
                size: wgpu::Extent3d {
                    width: SHADOW_SIZE,
                    height: SHADOW_SIZE,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            Target { tex, view }
        };
        let shadow_samp = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let linear = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let membrane_bg = bg(
            &device,
            &layouts.membrane,
            &[R::View(&shadow.view), R::Samp(&shadow_samp)],
        );
        let targets = Self::build_targets(&device, &layouts, size, &shadow, &shadow_samp, &linear);
        Renderer {
            device,
            queue,
            out_format,
            layouts,
            pipes,
            frame_buf,
            frame_bg,
            seg_buf,
            seg_cap,
            seg_count: 0,
            seg_bg,
            mem_vbuf: None,
            mem_ibuf: None,
            mem_count: 0,
            bounds: (Vec3::splat(-10.0), Vec3::splat(10.0)),
            shadow,
            shadow_samp,
            linear,
            membrane_bg,
            targets,
            frame_index: 0,
            history_cur: 0,
            accum: 0,
            last_state: None,
            prev_view_proj: Mat4::IDENTITY,
            scene_dirty: true,
            last_segments: Vec::new(),
            glyphs: Vec::new(),
            glyphs_shown: false,
        }
    }

    fn build_pipelines(device: &wgpu::Device, l: &Layouts, out_format: wgpu::TextureFormat) -> Pipelines {
        let capsule = shader(device, "capsule", include_str!("shaders/capsule.wgsl"));
        let membrane = shader(device, "membrane", include_str!("shaders/membrane.wgsl"));
        let ssao = shader(device, "ssao", include_str!("shaders/ssao.wgsl"));
        let lighting = shader(device, "lighting", include_str!("shaders/lighting.wgsl"));
        let fluor = shader(device, "fluor", include_str!("shaders/fluor.wgsl"));
        let taa = shader(device, "taa", include_str!("shaders/taa.wgsl"));
        let bloom = shader(device, "bloom", include_str!("shaders/bloom.wgsl"));
        let composite = shader(device, "composite", include_str!("shaders/composite.wgsl"));

        let mesh_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32, 2 => Float32x3, 3 => Float32, 4 => Float32x4],
        };
        let premul = wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING;
        let srgb_out = out_format.is_srgb();
        Pipelines {
            capsule: pipeline(
                device,
                "capsule",
                &capsule,
                "vs_capsule",
                Some("fs_capsule"),
                &[&l.frame, &l.segs],
                &[],
                &[color(HDR, None), color(HDR, None), color(HDR, None)],
                depth_state(true, wgpu::CompareFunction::Less),
                Some(wgpu::Face::Front),
            ),
            capsule_shadow: pipeline(
                device,
                "capsule_shadow",
                &capsule,
                "vs_capsule_shadow",
                Some("fs_capsule_shadow"),
                &[&l.frame, &l.segs],
                &[],
                &[],
                depth_state(true, wgpu::CompareFunction::Less),
                Some(wgpu::Face::Front),
            ),
            membrane_back: pipeline(
                device,
                "membrane_back",
                &membrane,
                "vs_membrane",
                Some("fs_membrane"),
                &[&l.frame, &l.membrane],
                &[Some(mesh_layout.clone())],
                &[color(HDR, Some(premul))],
                depth_state(false, wgpu::CompareFunction::LessEqual),
                Some(wgpu::Face::Front),
            ),
            membrane_front: pipeline(
                device,
                "membrane_front",
                &membrane,
                "vs_membrane",
                Some("fs_membrane"),
                &[&l.frame, &l.membrane],
                &[Some(mesh_layout.clone())],
                &[color(HDR, Some(premul))],
                depth_state(false, wgpu::CompareFunction::LessEqual),
                Some(wgpu::Face::Back),
            ),
            membrane_fluor: pipeline(
                device,
                "membrane_fluor",
                &membrane,
                "vs_membrane",
                Some("fs_membrane_fluor"),
                &[&l.frame, &l.membrane],
                &[Some(mesh_layout)],
                &[color(HDR, Some(ADDITIVE))],
                None,
                None,
            ),
            ssao: pipeline(
                device,
                "ssao",
                &ssao,
                "vs_fullscreen",
                Some("fs_ssao"),
                &[&l.frame, &l.ssao],
                &[],
                &[color(wgpu::TextureFormat::R16Float, None)],
                None,
                None,
            ),
            lighting: pipeline(
                device,
                "lighting",
                &lighting,
                "vs_fullscreen",
                Some("fs_lighting"),
                &[&l.frame, &l.lighting],
                &[],
                &[color(HDR, None)],
                None,
                None,
            ),
            splat: pipeline(
                device,
                "splat",
                &fluor,
                "vs_splat",
                Some("fs_splat"),
                &[&l.frame, &l.segs],
                &[],
                &[color(HDR, Some(ADDITIVE))],
                None,
                None,
            ),
            taa: pipeline(
                device,
                "taa",
                &taa,
                "vs_fullscreen",
                Some("fs_taa"),
                &[&l.frame, &l.taa],
                &[],
                &[color(HDR, None)],
                None,
                None,
            ),
            bloom_first: pipeline(
                device,
                "bloom_first",
                &bloom,
                "vs_fullscreen",
                Some("fs_down_first"),
                &[&l.frame, &l.tex_samp],
                &[],
                &[color(HDR, None)],
                None,
                None,
            ),
            bloom_down: pipeline(
                device,
                "bloom_down",
                &bloom,
                "vs_fullscreen",
                Some("fs_down"),
                &[&l.frame, &l.tex_samp],
                &[],
                &[color(HDR, None)],
                None,
                None,
            ),
            bloom_up: pipeline(
                device,
                "bloom_up",
                &bloom,
                "vs_fullscreen",
                Some("fs_up"),
                &[&l.frame, &l.tex_samp],
                &[],
                &[color(HDR, Some(ADDITIVE))],
                None,
                None,
            ),
            composite: pipeline(
                device,
                "composite",
                &composite,
                "vs_fullscreen",
                Some(if srgb_out { "fs_composite" } else { "fs_composite_gamma" }),
                &[&l.frame, &l.composite],
                &[],
                &[color(out_format, None)],
                None,
                None,
            ),
        }
    }

    fn build_targets(
        device: &wgpu::Device,
        l: &Layouts,
        size: (u32, u32),
        shadow: &Target,
        shadow_samp: &wgpu::Sampler,
        linear: &wgpu::Sampler,
    ) -> Targets {
        let (w, h) = (size.0.max(1), size.1.max(1));
        let depth = target(device, "depth", w, h, DEPTH);
        let normal = target(device, "normal", w, h, HDR);
        let albedo = target(device, "albedo", w, h, HDR);
        let aux = target(device, "aux", w, h, HDR);
        let ao = target(device, "ao", w, h, wgpu::TextureFormat::R16Float);
        let hdr = target(device, "hdr", w, h, HDR);
        let history = [
            target(device, "history0", w, h, HDR),
            target(device, "history1", w, h, HDR),
        ];
        let bloom: Vec<Target> = (0..BLOOM_LEVELS)
            .map(|i| target(device, "bloom", (w >> (i + 1)).max(1), (h >> (i + 1)).max(1), HDR))
            .collect();
        let ssao_bg = bg(device, &l.ssao, &[R::View(&depth.view), R::View(&normal.view)]);
        let lighting_bg = bg(
            device,
            &l.lighting,
            &[
                R::View(&depth.view),
                R::View(&normal.view),
                R::View(&albedo.view),
                R::View(&aux.view),
                R::View(&ao.view),
                R::View(&shadow.view),
                R::Samp(shadow_samp),
            ],
        );
        // taa_bg[k] writes history[k] reading history[1-k].
        let taa_bg = [0, 1].map(|k| {
            bg(
                device,
                &l.taa,
                &[
                    R::View(&hdr.view),
                    R::View(&history[1 - k].view),
                    R::View(&depth.view),
                    R::Samp(linear),
                ],
            )
        });
        let bloom_first_bg = [0, 1].map(|k| bg(device, &l.tex_samp, &[R::View(&history[k].view), R::Samp(linear)]));
        let bloom_bg = bloom
            .iter()
            .map(|t| bg(device, &l.tex_samp, &[R::View(&t.view), R::Samp(linear)]))
            .collect();
        let composite_bg = [0, 1].map(|k| {
            bg(
                device,
                &l.composite,
                &[R::View(&history[k].view), R::View(&bloom[0].view), R::Samp(linear)],
            )
        });
        Targets {
            size: (w, h),
            depth,
            normal,
            albedo,
            aux,
            ao,
            hdr,
            history,
            bloom,
            ssao_bg,
            lighting_bg,
            taa_bg,
            bloom_first_bg,
            bloom_bg,
            composite_bg,
        }
    }

    pub fn size(&self) -> (u32, u32) {
        self.targets.size
    }

    pub fn resize(&mut self, size: (u32, u32)) {
        if size != self.targets.size && size.0 > 0 && size.1 > 0 {
            self.targets = Self::build_targets(
                &self.device,
                &self.layouts,
                size,
                &self.shadow,
                &self.shadow_samp,
                &self.linear,
            );
            self.accum = 0;
            self.last_state = None;
        }
    }

    pub fn set_membrane(&mut self, g: &MembraneGeometry) {
        self.mem_vbuf = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("membrane vertices"),
            contents: bytemuck::cast_slice(&g.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));
        self.mem_ibuf = Some(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("membrane indices"),
            contents: bytemuck::cast_slice(&g.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));
        self.mem_count = g.indices.len() as u32;
        self.glyphs = g.glyphs.clone();
        self.upload_segments();
        self.bounds = (g.bounds_min, g.bounds_max);
        self.scene_dirty = true;
    }

    /// Microtubule capsules, plus the stress glyphs when they are shown.
    pub fn set_segments(&mut self, segs: &[Segment]) {
        self.last_segments.clear();
        self.last_segments.extend_from_slice(segs);
        self.upload_segments();
    }

    fn upload_segments(&mut self) {
        let mut all = std::mem::take(&mut self.last_segments);
        let n_mt = all.len();
        if self.glyphs_shown {
            all.extend_from_slice(&self.glyphs);
        }
        self.write_segments(&all);
        all.truncate(n_mt);
        self.last_segments = all;
    }

    fn write_segments(&mut self, segs: &[Segment]) {
        if segs.len() > self.seg_cap {
            self.seg_cap = segs.len().next_power_of_two();
            self.seg_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("segments"),
                size: (self.seg_cap * std::mem::size_of::<Segment>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.seg_bg = bg(&self.device, &self.layouts.segs, &[R::Buf(&self.seg_buf)]);
        }
        if !segs.is_empty() {
            self.queue.write_buffer(&self.seg_buf, 0, bytemuck::cast_slice(segs));
        }
        self.seg_count = segs.len() as u32;
        self.scene_dirty = true;
    }

    pub fn bounds(&self) -> (Vec3, Vec3) {
        self.bounds
    }

    /// Frames accumulated on a still scene so far.
    pub fn accumulated(&self) -> u32 {
        self.accum
    }

    fn light(&self, s: &Settings) -> (Vec3, Mat4) {
        let dir = Vec3::new(
            s.light_pitch.cos() * s.light_yaw.cos(),
            s.light_pitch.cos() * s.light_yaw.sin(),
            s.light_pitch.sin(),
        )
        .normalize();
        let (lo, hi) = self.bounds;
        let c = (lo + hi) * 0.5;
        let r = (hi - lo).length() * 0.5 + 0.5;
        let up = if dir.z.abs() > 0.95 { Vec3::Y } else { Vec3::Z };
        let view = glam::camera::rh::view::look_at_mat4(c + dir * r * 2.0, c, up);
        let proj = glam::camera::rh::proj::directx::orthographic(-r, r, -r, r, 0.01, r * 4.0);
        (dir, proj * view)
    }

    /// Render one frame into `out` (which must have the renderer's size and
    /// output format).
    pub fn render(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        out: &wgpu::TextureView,
        cam: &OrbitCamera,
        s: &Settings,
    ) {
        if s.stress_tint != self.glyphs_shown {
            self.glyphs_shown = s.stress_tint;
            self.upload_segments();
        }
        let (w, h) = self.targets.size;
        let state = (*cam, s.clone(), (w, h));
        let still = !self.scene_dirty && self.last_state.as_ref() == Some(&state);
        let camera_moved = self
            .last_state
            .as_ref()
            .map(|l| l.0 != *cam || l.2 != (w, h))
            .unwrap_or(true);
        self.accum = if still { self.accum + 1 } else { 1 };
        self.scene_dirty = false;
        self.last_state = Some(state);
        self.frame_index = self.frame_index.wrapping_add(1);

        let size = Vec2::new(w as f32, h as f32);
        let jitter = if s.taa {
            camera::halton_jitter(self.frame_index)
        } else {
            Vec2::ZERO
        };
        let (eye, view) = if s.aperture > 0.0 && still && self.accum > 1 {
            // Thin lens: move the eye across the aperture, keep the focal
            // point fixed.
            let base = cam.view().inverse();
            let (right, up) = (base.x_axis.truncate(), base.y_axis.truncate());
            let fwd = (cam.target - cam.eye()).normalize();
            let focus = cam.eye() + fwd * (cam.distance + s.focus_dist).max(0.01);
            let u = camera::halton_jitter(self.frame_index * 7 + 3) + Vec2::splat(0.5);
            let (r, phi) = (u.x.sqrt(), u.y * std::f32::consts::TAU);
            let e = cam.eye() + (right * phi.cos() + up * phi.sin()) * r * s.aperture;
            (e, glam::camera::rh::view::look_at_mat4(e, focus, up))
        } else {
            (cam.eye(), cam.view())
        };
        let shift = Vec2::new(s.view_shift * w as f32 * 0.5, 0.0);
        let proj = cam.proj(w as f32 / h as f32, jitter + shift, size);
        let view_proj = proj * view;
        let unjittered = cam.proj(w as f32 / h as f32, shift, size) * view;
        let prev = if camera_moved && self.accum == 1 && self.frame_index > 1 {
            self.prev_view_proj
        } else {
            unjittered
        };
        let (ldir, light_vp) = self.light(s);
        let (lo, hi) = self.bounds;
        let clip_z = if s.clip >= 0.999 {
            1e6
        } else {
            lo.z + (hi.z - lo.z) * s.clip
        };
        let (taa_alpha, clamp) = if !s.taa {
            (1.0, 0.0)
        } else if still {
            ((1.0 / self.accum as f32).max(1.0 / 4096.0), 0.0)
        } else {
            (0.15, 1.0)
        };
        let focal = (cam.eye() - cam.target).length() + s.focus_offset;
        let u = FrameUniform {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: prev.to_cols_array_2d(),
            view: view.to_cols_array_2d(),
            light_view_proj: light_vp.to_cols_array_2d(),
            cam_pos: eye.extend(cam.near_far().0).to_array(),
            light_dir: ldir.extend(s.light_intensity).to_array(),
            resolution: [w as f32, h as f32, 1.0 / w as f32, 1.0 / h as f32],
            jitter: [
                jitter.x,
                jitter.y,
                self.frame_index as f32,
                2.0 * (cam.fov * 0.5).tan() / h as f32,
            ],
            p0: [s.exposure, s.bloom, s.ao_strength, s.ao_radius],
            p1: [s.colour_mode as f32, s.mt_radius, s.membrane_opacity, clip_z],
            p2: [s.tip_glow, s.shadow_softness, s.vignette, s.grain],
            p3: [0.0, taa_alpha, s.fluor_gain, focal],
            p4: [s.section, s.psf, s.background as f32, clamp],
            p5: [
                if s.stress_tint {
                    2.0
                } else {
                    s.curvature_tint as u32 as f32
                },
                s.saturation,
                s.contrast,
                (!s.show_cscs as u32 | (!s.show_fibrils as u32) << 1 | (!s.show_mts as u32) << 2) as f32,
            ],
            bounds_min: lo.extend(s.csc_scale).to_array(),
            bounds_max: hi.extend(0.0).to_array(),
        };
        self.prev_view_proj = unjittered;
        self.queue.write_buffer(&self.frame_buf, 0, bytemuck::bytes_of(&u));

        let t = &self.targets;
        let clear = |c: wgpu::Color| wgpu::Operations {
            load: wgpu::LoadOp::Clear(c),
            store: wgpu::StoreOp::Store,
        };
        let fullscreen = |enc: &mut wgpu::CommandEncoder,
                          label: &str,
                          view: &wgpu::TextureView,
                          load: wgpu::LoadOp<wgpu::Color>,
                          pipe: &wgpu::RenderPipeline,
                          group: &wgpu::BindGroup,
                          frame_bg: &wgpu::BindGroup| {
            let mut p = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some(label),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            p.set_pipeline(pipe);
            p.set_bind_group(0, frame_bg, &[]);
            p.set_bind_group(1, group, &[]);
            p.draw(0..3, 0..1);
        };

        // Shadow map (used by both looks for the membrane).
        {
            let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            if s.look == Look::Hero && self.seg_count > 0 {
                p.set_pipeline(&self.pipes.capsule_shadow);
                p.set_bind_group(0, &self.frame_bg, &[]);
                p.set_bind_group(1, &self.seg_bg, &[]);
                p.draw(0..36, 0..self.seg_count);
            }
        }

        match s.look {
            Look::Hero => {
                {
                    let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("gbuffer"),
                        color_attachments: &[
                            att(&t.normal.view, clear(wgpu::Color::TRANSPARENT)),
                            att(&t.albedo.view, clear(wgpu::Color::TRANSPARENT)),
                            att(&t.aux.view, clear(wgpu::Color::TRANSPARENT)),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &t.depth.view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    if self.seg_count > 0 {
                        p.set_pipeline(&self.pipes.capsule);
                        p.set_bind_group(0, &self.frame_bg, &[]);
                        p.set_bind_group(1, &self.seg_bg, &[]);
                        p.draw(0..36, 0..self.seg_count);
                    }
                }
                fullscreen(
                    encoder,
                    "ssao",
                    &t.ao.view,
                    wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                    &self.pipes.ssao,
                    &t.ssao_bg,
                    &self.frame_bg,
                );
                fullscreen(
                    encoder,
                    "lighting",
                    &t.hdr.view,
                    wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    &self.pipes.lighting,
                    &t.lighting_bg,
                    &self.frame_bg,
                );
                if s.show_membrane && self.mem_count > 0 && s.membrane_opacity > 0.0 {
                    let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("membrane"),
                        color_attachments: &[att(
                            &t.hdr.view,
                            wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        )],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &t.depth.view,
                            depth_ops: None,
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                    p.set_bind_group(0, &self.frame_bg, &[]);
                    p.set_bind_group(1, &self.membrane_bg, &[]);
                    p.set_vertex_buffer(0, self.mem_vbuf.as_ref().unwrap().slice(..));
                    p.set_index_buffer(self.mem_ibuf.as_ref().unwrap().slice(..), wgpu::IndexFormat::Uint32);
                    p.set_pipeline(&self.pipes.membrane_back);
                    p.draw_indexed(0..self.mem_count, 0, 0..1);
                    p.set_pipeline(&self.pipes.membrane_front);
                    p.draw_indexed(0..self.mem_count, 0, 0..1);
                }
            }
            Look::Fluorescence => {
                {
                    // Depth is unused here but TAA reads it: clear to far.
                    let _p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("clear depth"),
                        color_attachments: &[],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: &t.depth.view,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Clear(1.0),
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        timestamp_writes: None,
                        occlusion_query_set: None,
                        multiview_mask: None,
                    });
                }
                let mut p = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("fluorescence"),
                    color_attachments: &[att(&t.hdr.view, clear(wgpu::Color::BLACK))],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                p.set_bind_group(0, &self.frame_bg, &[]);
                if self.seg_count > 0 {
                    p.set_pipeline(&self.pipes.splat);
                    p.set_bind_group(1, &self.seg_bg, &[]);
                    p.draw(0..6, 0..self.seg_count);
                }
                if s.show_membrane && self.mem_count > 0 {
                    p.set_pipeline(&self.pipes.membrane_fluor);
                    p.set_bind_group(1, &self.membrane_bg, &[]);
                    p.set_vertex_buffer(0, self.mem_vbuf.as_ref().unwrap().slice(..));
                    p.set_index_buffer(self.mem_ibuf.as_ref().unwrap().slice(..), wgpu::IndexFormat::Uint32);
                    p.draw_indexed(0..self.mem_count, 0, 0..1);
                }
            }
        }

        // Temporal accumulation into history[cur].
        let cur = 1 - self.history_cur;
        self.history_cur = cur;
        fullscreen(
            encoder,
            "taa",
            &t.history[cur].view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &self.pipes.taa,
            &t.taa_bg[cur],
            &self.frame_bg,
        );

        // Bloom.
        fullscreen(
            encoder,
            "bloom0",
            &t.bloom[0].view,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &self.pipes.bloom_first,
            &t.bloom_first_bg[cur],
            &self.frame_bg,
        );
        for i in 1..BLOOM_LEVELS {
            fullscreen(
                encoder,
                "bloom down",
                &t.bloom[i].view,
                wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                &self.pipes.bloom_down,
                &t.bloom_bg[i - 1],
                &self.frame_bg,
            );
        }
        for i in (1..BLOOM_LEVELS).rev() {
            fullscreen(
                encoder,
                "bloom up",
                &t.bloom[i - 1].view,
                wgpu::LoadOp::Load,
                &self.pipes.bloom_up,
                &t.bloom_bg[i],
                &self.frame_bg,
            );
        }

        fullscreen(
            encoder,
            "composite",
            out,
            wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            &self.pipes.composite,
            &t.composite_bg[cur],
            &self.frame_bg,
        );
    }

    pub fn out_format(&self) -> wgpu::TextureFormat {
        self.out_format
    }

    #[allow(dead_code)]
    fn unused(&self) -> (&wgpu::Texture, Vec4) {
        (&self.targets.depth.tex, Vec4::ZERO)
    }
}

/// Create a device and queue, optionally compatible with a window surface.
pub async fn gpu(
    compatible_surface: Option<&wgpu::Surface<'_>>,
    instance: &wgpu::Instance,
) -> anyhow::Result<(wgpu::Adapter, wgpu::Device, wgpu::Queue)> {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            force_fallback_adapter: false,
            apply_limit_buckets: false,
        })
        .await?;
    let mut limits = wgpu::Limits::default();
    let al = adapter.limits();
    limits.max_storage_buffer_binding_size = al.max_storage_buffer_binding_size;
    limits.max_buffer_size = al.max_buffer_size;
    limits.max_texture_dimension_2d = al.max_texture_dimension_2d;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("rusty_tube"),
            required_features: wgpu::Features::empty(),
            required_limits: limits,
            ..Default::default()
        })
        .await?;
    Ok((adapter, device, queue))
}
