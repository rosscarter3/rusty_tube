// Temporal accumulation. With a moving camera or a running simulation the
// history is reprojected and clamped to the current neighbourhood; with a
// static scene it becomes a progressive average, i.e. supersampling that
// converges to a noise-free image.

@group(1) @binding(0) var cur_tex: texture_2d<f32>;
@group(1) @binding(1) var hist_tex: texture_2d<f32>;
@group(1) @binding(2) var depth_tex: texture_depth_2d;
@group(1) @binding(3) var lin: sampler;

fn rgb2ycocg(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}
fn ycocg2rgb(c: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

@fragment
fn fs_taa(in: FsIn) -> @location(0) vec4<f32> {
    let px = vec2<i32>(in.pos.xy);
    let cur = textureLoad(cur_tex, px, 0).rgb;
    let alpha = F.p3.y;
    if (alpha >= 0.999) {
        return vec4<f32>(cur, 1.0);
    }
    if (F.p4.w < 0.5) {
        // Static scene: progressive average at the same pixel.
        let h = textureLoad(hist_tex, px, 0).rgb;
        return vec4<f32>(mix(h, cur, alpha), 1.0);
    }
    let d = textureLoad(depth_tex, px, 0);
    let wp = world_from_depth(in.uv, min(d, 0.99999));
    let pc = F.prev_view_proj * vec4<f32>(wp, 1.0);
    let puv = vec2<f32>(pc.x / pc.w * 0.5 + 0.5, 0.5 - pc.y / pc.w * 0.5);
    if (puv.x < 0.0 || puv.x > 1.0 || puv.y < 0.0 || puv.y > 1.0) {
        return vec4<f32>(cur, 1.0);
    }
    var hist = textureSampleLevel(hist_tex, lin, puv, 0.0).rgb;
    {
        var mn = vec3<f32>(1e9);
        var mx = vec3<f32>(-1e9);
        for (var i = -1; i <= 1; i++) {
            for (var j = -1; j <= 1; j++) {
                let s = rgb2ycocg(textureLoad(cur_tex, px + vec2<i32>(i, j), 0).rgb);
                mn = min(mn, s);
                mx = max(mx, s);
            }
        }
        hist = ycocg2rgb(clamp(rgb2ycocg(hist), mn, mx));
    }
    return vec4<f32>(mix(hist, cur, alpha), 1.0);
}
