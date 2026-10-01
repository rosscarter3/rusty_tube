// Shared by every pass.

struct Frame {
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    prev_view_proj: mat4x4<f32>,
    view: mat4x4<f32>,
    light_view_proj: mat4x4<f32>,
    cam_pos: vec4<f32>,      // xyz, near plane
    light_dir: vec4<f32>,    // unit vector towards the key light, intensity
    resolution: vec4<f32>,   // w, h, 1/w, 1/h
    jitter: vec4<f32>,       // jitter (ndc), frame index, radians per pixel
    p0: vec4<f32>,           // exposure, bloom strength, ao strength, ao radius (um)
    p1: vec4<f32>,           // colour mode, MT radius (um), membrane opacity, clip height (um)
    p2: vec4<f32>,           // tip glow, shadow softness, vignette, grain
    p3: vec4<f32>,           // sim step, TAA blend, fluorescence gain, focal depth (view, um)
    p4: vec4<f32>,           // section half-thickness (um), PSF sigma (um), background, TAA clamp
    p5: vec4<f32>,           // membrane tint mode, saturation, contrast, dof strength
    bounds_min: vec4<f32>,
    bounds_max: vec4<f32>,
}

@group(0) @binding(0) var<uniform> F: Frame;

const PI: f32 = 3.14159265359;

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let k = vec3<f32>(1.0, 2.0 / 3.0, 1.0 / 3.0);
    let p = abs(fract(vec3<f32>(h) + k) * 6.0 - 3.0);
    return v * mix(vec3<f32>(1.0), clamp(p - 1.0, vec3<f32>(0.0), vec3<f32>(1.0)), s);
}

fn srgb_to_linear(c: vec3<f32>) -> vec3<f32> {
    return pow(c, vec3<f32>(2.2));
}

// Interleaved gradient noise (Jimenez 2014), decorrelated per frame.
fn ign(px: vec2<f32>, frame: f32) -> f32 {
    let p = px + 5.588238 * (frame % 64.0);
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}

fn hash_u32(x0: u32) -> u32 {
    var x = x0;
    x = x ^ (x >> 16u);
    x = x * 0x7feb352du;
    x = x ^ (x >> 15u);
    x = x * 0x846ca68bu;
    return x ^ (x >> 16u);
}

fn world_from_depth(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let w = F.inv_view_proj * ndc;
    return w.xyz / w.w;
}

struct FsIn {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// A triangle covering the screen.
@vertex
fn vs_fullscreen(@builtin(vertex_index) i: u32) -> FsIn {
    var o: FsIn;
    let uv = vec2<f32>(f32((i << 1u) & 2u), f32(i & 2u));
    o.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    o.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return o;
}

// Microtubule capsules ------------------------------------------------------

struct Segment {
    a: vec3<f32>,
    tip_dist: f32,
    b: vec3<f32>,
    minus_dist: f32,
    hash: u32,
    flags: u32,
    age: f32,
    stress: f32,
}

const FLAG_GROWING: u32 = 1u;
const FLAG_SHRINKING: u32 = 2u;
const FLAG_CORTICAL: u32 = 4u;
const FLAG_TREADMILL: u32 = 8u;
const FLAG_BUNDLED: u32 = 16u;
const FLAG_PLUS_END: u32 = 32u;

// Inferno-like ramp.
fn ramp_inferno(t0: f32) -> vec3<f32> {
    let t = clamp(t0, 0.0, 1.0);
    let c0 = vec3<f32>(0.0002, 0.0016, -0.0194);
    let c1 = vec3<f32>(0.1065, 0.5639, 3.9327);
    let c2 = vec3<f32>(11.6024, -3.9728, -15.9423);
    let c3 = vec3<f32>(-41.7039, 17.4363, 44.3541);
    let c4 = vec3<f32>(77.1629, -33.4023, -81.8073);
    let c5 = vec3<f32>(-71.3194, 32.6260, 73.2095);
    let c6 = vec3<f32>(25.1311, -12.2426, -23.0703);
    return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Base colour of a microtubule segment, by colour mode, plus emission from
// dynamic plus ends (EB1-like comets on growing tips, a hot glow on
// shrinking ones). Returns rgb = albedo, a = emission intensity; the emission
// colour is returned through `emit`.
fn segment_colour(s: Segment, dir: vec3<f32>, emit: ptr<function, vec3<f32>>) -> vec3<f32> {
    let mode = i32(F.p1.x + 0.5);
    var c = vec3<f32>(0.8);
    if ((s.flags & 64u) != 0u) {
        // Stress glyph: pale, faintly self-lit, brighter where more anisotropic.
        *emit = vec3<f32>(0.55, 0.75, 1.0) * 0.25 * s.stress;
        return vec3<f32>(0.85, 0.9, 1.0);
    }
    let horiz = length(dir.xy);
    if (mode == 0) {
        // Nematic orientation in the leaf plane: a filament and its reverse
        // get the same hue; filaments running up the anticlinal wall fade to
        // white.
        let phi = atan2(dir.y, dir.x);
        let h = fract(phi / PI + 1.0);
        c = hsv2rgb(h, 0.85 * smoothstep(0.15, 0.85, horiz), 0.8);
    } else if (mode == 1) {
        if ((s.flags & FLAG_SHRINKING) != 0u) {
            c = vec3<f32>(0.95, 0.38, 0.26);
        } else if ((s.flags & FLAG_GROWING) != 0u) {
            c = vec3<f32>(0.62, 0.8, 0.95);
        } else {
            c = vec3<f32>(0.6, 0.6, 0.62);
        }
        if ((s.flags & FLAG_TREADMILL) != 0u) {
            let m = exp(-s.minus_dist / 0.6);
            c = mix(c, vec3<f32>(0.7, 0.45, 0.95), m);
        }
    } else if (mode == 2) {
        let h = f32(s.hash & 0xffffu) / 65535.0;
        c = hsv2rgb(h, 0.62, 0.95);
    } else if (mode == 3) {
        c = ramp_inferno(1.0 - clamp(s.age / 3000.0, 0.0, 1.0) * 0.85);
    } else if (mode == 4) {
        if (s.stress < 0.0) {
            c = vec3<f32>(0.45);
        } else {
            let t = smoothstep(0.2, 0.95, s.stress);
            c = mix(vec3<f32>(0.08, 0.3, 0.95), vec3<f32>(1.0, 0.42, 0.06), t);
        }
    } else {
        c = vec3<f32>(0.85, 0.87, 0.9);
    }
    // Plus-end comets.
    var e = vec3<f32>(0.0);
    let glow = F.p2.x;
    if ((s.flags & FLAG_GROWING) != 0u) {
        e = vec3<f32>(0.35, 1.0, 0.85) * exp(-s.tip_dist / 0.18) * glow * 2.0;
    } else if ((s.flags & FLAG_SHRINKING) != 0u) {
        e = vec3<f32>(1.0, 0.25, 0.12) * exp(-s.tip_dist / 0.15) * glow * 1.5;
    }
    *emit = e;
    return c;
}
