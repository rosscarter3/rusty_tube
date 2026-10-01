// Deferred shading of the microtubules: a warm key light with soft
// shadows, a sky/ground fill modulated by ambient occlusion, GGX and an
// anisotropic fibre highlight along each filament, and emission from
// dynamic plus ends.

@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var albedo_tex: texture_2d<f32>;
@group(1) @binding(3) var aux_tex: texture_2d<f32>;
@group(1) @binding(4) var ao_tex: texture_2d<f32>;
@group(1) @binding(5) var shadow_map: texture_depth_2d;
@group(1) @binding(6) var shadow_samp: sampler_comparison;

fn background(uv: vec2<f32>) -> vec3<f32> {
    let mode = F.p4.z;
    let c = uv - vec2<f32>(0.5, 0.45);
    let r = length(c * vec2<f32>(F.resolution.x * F.resolution.w, 1.0));
    if (mode > 1.5) {
        // Studio white.
        return mix(vec3<f32>(0.92, 0.93, 0.95), vec3<f32>(0.62, 0.64, 0.68), smoothstep(0.1, 1.1, r));
    }
    if (mode > 0.5) {
        return vec3<f32>(0.0);
    }
    let top = vec3<f32>(0.035, 0.05, 0.075);
    let bottom = vec3<f32>(0.006, 0.008, 0.014);
    var col = mix(top, bottom, smoothstep(0.0, 1.0, uv.y));
    col += vec3<f32>(0.03, 0.045, 0.06) * exp(-r * r * 3.0);
    return col;
}

const POISSON = array<vec2<f32>, 12>(
    vec2<f32>(-0.326, -0.406), vec2<f32>(-0.840, -0.074), vec2<f32>(-0.696, 0.457),
    vec2<f32>(-0.203, 0.621), vec2<f32>(0.962, -0.195), vec2<f32>(0.473, -0.480),
    vec2<f32>(0.519, 0.767), vec2<f32>(0.185, -0.893), vec2<f32>(0.507, 0.064),
    vec2<f32>(0.896, 0.412), vec2<f32>(-0.322, -0.933), vec2<f32>(-0.792, -0.598));

fn shadow(p: vec3<f32>, n: vec3<f32>, px: vec2<f32>) -> f32 {
    let lp = F.light_view_proj * vec4<f32>(p + n * F.p1.y * 0.6, 1.0);
    let uv = vec2<f32>(lp.x * 0.5 + 0.5, 0.5 - lp.y * 0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
    }
    let a = ign(px, F.jitter.z) * 2.0 * PI;
    let rot = mat2x2<f32>(cos(a), sin(a), -sin(a), cos(a));
    let spread = F.p2.y / 2048.0;
    var s = 0.0;
    for (var i = 0; i < 12; i++) {
        let o = rot * POISSON[i] * spread;
        s += textureSampleCompareLevel(shadow_map, shadow_samp, uv + o, lp.z - 0.0004);
    }
    return s / 12.0;
}

fn ggx(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, rough: f32) -> f32 {
    let h = normalize(v + l);
    let a = rough * rough;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 1e-4);
    let d = a2 / (PI * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    let k = a * 0.5;
    let g = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    let f = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    return d * g * f / max(4.0 * ndl * ndv, 1e-4);
}

@fragment
fn fs_lighting(in: FsIn) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth_tex, px, 0);
    if (d >= 1.0) {
        return vec4<f32>(background(in.uv), 1.0);
    }
    let p = world_from_depth(in.uv, d);
    let nr = textureLoad(normal_tex, px, 0);
    let n = normalize(nr.xyz);
    let rough = nr.w;
    let albedo = textureLoad(albedo_tex, px, 0).rgb;
    let emit = textureLoad(aux_tex, px, 0).rgb;
    let ao_raw = textureLoad(ao_tex, px, 0).r;
    let ao = mix(1.0, ao_raw, F.p0.z);
    let v = normalize(F.cam_pos.xyz - p);
    let l = F.light_dir.xyz;
    let sh = shadow(p, n, in.pos.xy);
    let key = vec3<f32>(1.0, 0.94, 0.86) * F.light_dir.w;
    // Slight wrap: thin tubes scatter light round the terminator.
    let ndl = max((dot(n, l) + 0.25) / 1.25, 0.0);
    var col = albedo * key * ndl * sh;
    col += key * ggx(n, v, l, rough) * max(dot(n, l), 0.0) * sh * 1.6;
    // Fill from a cool sky above and a dim warm ground below.
    let sky = vec3<f32>(0.38, 0.46, 0.6);
    let ground = vec3<f32>(0.1, 0.08, 0.07);
    let hemi = mix(ground, sky, n.z * 0.5 + 0.5);
    col += albedo * hemi * ao * 0.55;
    // Rim light from behind-left.
    let rim_dir = normalize(vec3<f32>(-l.x, -l.y, 0.3));
    let rim = pow(1.0 - max(dot(n, v), 0.0), 4.0) * max(dot(n, rim_dir) + 0.3, 0.0);
    col += mix(albedo, vec3<f32>(0.75, 0.85, 1.0), 0.4) * rim * ao * 1.3;
    col += emit;
    // Aerial perspective: distant filaments sink toward the background.
    let dist = length(F.cam_pos.xyz - p);
    let target_dist = length(F.cam_pos.xyz - 0.5 * (F.bounds_min.xyz + F.bounds_max.xyz));
    let fog = 1.0 - exp(-max(dist - target_dist * 0.8, 0.0) * 0.06);
    col = mix(col, background(in.uv) * 1.2, clamp(fog, 0.0, 0.85));
    return vec4<f32>(col, 1.0);
}
