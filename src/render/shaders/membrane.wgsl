// The cell surface: a glassy, translucent shell in the hero render, and the
// optical section of a plasma-membrane marker in the fluorescence render.

struct VIn {
    @location(0) pos: vec3<f32>,
    @location(1) kappa: f32,
    @location(2) normal: vec3<f32>,
    @location(3) wall: f32,
    @location(4) stress: vec4<f32>,
}

struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) kappa: f32,
    @location(3) wall: f32,
    @location(4) view_z: f32,
    @location(5) stress: vec4<f32>,
}

@group(1) @binding(0) var shadow_map: texture_depth_2d;
@group(1) @binding(1) var shadow_samp: sampler_comparison;

@vertex
fn vs_membrane(v: VIn) -> VOut {
    var o: VOut;
    o.clip = F.view_proj * vec4<f32>(v.pos, 1.0);
    o.world = v.pos;
    o.normal = v.normal;
    o.kappa = v.kappa;
    o.wall = v.wall;
    o.view_z = -(F.view * vec4<f32>(v.pos, 1.0)).z;
    o.stress = v.stress;
    return o;
}

fn membrane_shadow(p: vec3<f32>, n: vec3<f32>) -> f32 {
    let lp = F.light_view_proj * vec4<f32>(p + n * 0.05, 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    var s = 0.0;
    let texel = 1.0 / 2048.0;
    for (var i = -1; i <= 1; i++) {
        for (var j = -1; j <= 1; j++) {
            s += textureSampleCompareLevel(shadow_map, shadow_samp, uv + vec2<f32>(f32(i), f32(j)) * texel * 2.0, lp.z - 0.0015);
        }
    }
    return s / 9.0;
}

@fragment
fn fs_membrane(in: VOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    if (F.p1.w < 1e5 && in.world.z > F.p1.w) {
        discard;
    }
    var n = normalize(in.normal);
    let v = normalize(F.cam_pos.xyz - in.world);
    if (!front) {
        n = -n;
    }
    let l = F.light_dir.xyz;
    let ndv = clamp(abs(dot(n, v)), 0.0, 1.0);
    let fres = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
    // Tint: neutral glass, or the outline curvature (lobes warm, necks cool).
    var tint = vec3<f32>(0.62, 0.8, 0.88);
    if (F.p5.x > 0.5) {
        let k = clamp(in.kappa * 1.5, -1.0, 1.0);
        tint = mix(tint, select(vec3<f32>(0.3, 0.55, 1.0), vec3<f32>(1.0, 0.55, 0.35), k > 0.0), abs(k) * in.wall);
    }
    // Stress field: anisotropy as a faint violet glow (the glyphs carry the
    // direction).
    var field_glow = vec3<f32>(0.0);
    var field_alpha = 0.0;
    if (F.p5.x > 1.5 && in.stress.w > 0.0) {
        let a = clamp(in.stress.w, 0.0, 1.0);
        field_glow = vec3<f32>(0.55, 0.35, 1.0) * 0.12 * a * a;
        field_alpha = 0.06 * a;
    }
    let shadow = membrane_shadow(in.world, n);
    let diff = max(dot(n, l), 0.0) * 0.6 + 0.25;
    let h = normalize(l + v);
    // Thin-film interference: a lipid bilayer is far too thin to show it,
    // but a faint angle-dependent sheen reads as a membrane rather than glass.
    let film = 0.5 + 0.5 * cos(2.0 * PI * (vec3<f32>(0.0, 0.33, 0.67) + ndv * 1.7 + 0.15));
    let sheen = mix(vec3<f32>(1.0), film, 0.45);
    let spec = pow(max(dot(n, h), 0.0), 128.0) * 3.0 * shadow;
    let rim = pow(1.0 - ndv, 3.0);
    var col = tint * diff * (0.55 + 0.45 * shadow) * F.light_dir.w * 0.25 + sheen * spec + tint * sheen * rim * 0.7;
    var alpha = F.p1.z * (0.05 + 0.95 * fres) + field_alpha;
    col += field_glow;
    if (!front) {
        alpha = alpha * 0.5;
        col = col * 0.55;
    }
    return vec4<f32>(col * alpha, alpha);
}

// Fluorescence: a membrane dye seen through an optical section. A surface
// crossing the focal plane obliquely contributes along a longer path, which
// is why a confocal section shows the cell as a bright outline.
@fragment
fn fs_membrane_fluor(in: VOut) -> @location(0) vec4<f32> {
    if (F.p1.w < 1e5 && in.world.z > F.p1.w) {
        discard;
    }
    let n = normalize(in.normal);
    let v = normalize(F.cam_pos.xyz - in.world);
    let dz = in.view_z - F.p3.w;
    let half = F.p4.x;
    let w = exp(-0.5 * dz * dz / (half * half));
    let path = 1.0 / max(abs(dot(n, v)), 0.08);
    let i = w * path * 0.035 * F.p3.z * F.p1.z;
    return vec4<f32>(vec3<f32>(1.0, 0.18, 0.72) * i, 0.0);
}
