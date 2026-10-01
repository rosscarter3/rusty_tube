// Normal-oriented hemisphere ambient occlusion, in world space. The kernel
// is rotated per pixel and per frame; temporal accumulation integrates it.

@group(1) @binding(0) var depth_tex: texture_depth_2d;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;

fn view_depth(p: vec3<f32>) -> f32 {
    return -(F.view * vec4<f32>(p, 1.0)).z;
}

@fragment
fn fs_ssao(in: FsIn) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let d = textureLoad(depth_tex, px, 0);
    if (d >= 1.0) {
        return vec4<f32>(1.0);
    }
    let p = world_from_depth(in.uv, d);
    let n = normalize(textureLoad(normal_tex, px, 0).xyz);
    let radius = F.p0.w;
    let noise = ign(in.pos.xy, F.jitter.z);
    let noise2 = ign(in.pos.yx + 17.0, F.jitter.z);
    // Tangent frame.
    var helper = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(n.z) > 0.9) {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    let t = normalize(cross(helper, n));
    let b = cross(n, t);
    let size = F.resolution.xy;
    var occ = 0.0;
    let N = 12;
    for (var i = 0; i < N; i++) {
        let fi = f32(i);
        // Cosine-weighted direction on a spiral.
        let u1 = fract(fi / f32(N) + noise);
        let u2 = fract(fi * 0.618034 + noise2);
        let r = sqrt(u1);
        let phi = 2.0 * PI * u2;
        let local = vec3<f32>(r * cos(phi), r * sin(phi), sqrt(max(0.0, 1.0 - u1)));
        let scale = mix(0.15, 1.0, fract(fi * 0.37 + noise * 0.5));
        let q = p + (t * local.x + b * local.y + n * local.z) * radius * scale * scale;
        let c = F.view_proj * vec4<f32>(q, 1.0);
        let suv = vec2<f32>(c.x / c.w * 0.5 + 0.5, 0.5 - c.y / c.w * 0.5);
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) {
            continue;
        }
        let sd = textureLoad(depth_tex, vec2<i32>(suv * size), 0);
        let sp = world_from_depth(suv, sd);
        let scene_z = view_depth(sp);
        let q_z = view_depth(q);
        let range = smoothstep(0.0, 1.0, radius / max(abs(view_depth(p) - scene_z), 1e-4));
        if (scene_z < q_z - 0.004) {
            occ += range;
        }
    }
    let ao = 1.0 - occ / f32(N);
    return vec4<f32>(ao, ao, ao, 1.0);
}
