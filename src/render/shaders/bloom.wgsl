// Physically based bloom (Jimenez 2014): a 13-tap downsample chain, with a
// Karis average on the first level to stop fireflies, then a tent-filtered
// upsample accumulated back up the chain.

@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(1) var lin: sampler;

fn karis(c: vec3<f32>) -> f32 {
    return 1.0 / (1.0 + dot(c, vec3<f32>(0.2126, 0.7152, 0.0722)));
}

fn down13(uv: vec2<f32>, first: bool) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let a = textureSampleLevel(src, lin, uv + t * vec2<f32>(-2.0, -2.0), 0.0).rgb;
    let b = textureSampleLevel(src, lin, uv + t * vec2<f32>(0.0, -2.0), 0.0).rgb;
    let c = textureSampleLevel(src, lin, uv + t * vec2<f32>(2.0, -2.0), 0.0).rgb;
    let d = textureSampleLevel(src, lin, uv + t * vec2<f32>(-2.0, 0.0), 0.0).rgb;
    let e = textureSampleLevel(src, lin, uv, 0.0).rgb;
    let f = textureSampleLevel(src, lin, uv + t * vec2<f32>(2.0, 0.0), 0.0).rgb;
    let g = textureSampleLevel(src, lin, uv + t * vec2<f32>(-2.0, 2.0), 0.0).rgb;
    let h = textureSampleLevel(src, lin, uv + t * vec2<f32>(0.0, 2.0), 0.0).rgb;
    let i = textureSampleLevel(src, lin, uv + t * vec2<f32>(2.0, 2.0), 0.0).rgb;
    let j = textureSampleLevel(src, lin, uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    let k = textureSampleLevel(src, lin, uv + t * vec2<f32>(1.0, -1.0), 0.0).rgb;
    let l = textureSampleLevel(src, lin, uv + t * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    let m = textureSampleLevel(src, lin, uv + t * vec2<f32>(1.0, 1.0), 0.0).rgb;
    if (first) {
        let g0 = (a + b + d + e) * 0.25;
        let g1 = (b + c + e + f) * 0.25;
        let g2 = (d + e + g + h) * 0.25;
        let g3 = (e + f + h + i) * 0.25;
        let g4 = (j + k + l + m) * 0.25;
        let w0 = karis(g0) * 0.125;
        let w1 = karis(g1) * 0.125;
        let w2 = karis(g2) * 0.125;
        let w3 = karis(g3) * 0.125;
        let w4 = karis(g4) * 0.5;
        return (g0 * w0 + g1 * w1 + g2 * w2 + g3 * w3 + g4 * w4) / (w0 + w1 + w2 + w3 + w4);
    }
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

@fragment
fn fs_down_first(in: FsIn) -> @location(0) vec4<f32> {
    return vec4<f32>(down13(in.uv, true), 1.0);
}

@fragment
fn fs_down(in: FsIn) -> @location(0) vec4<f32> {
    return vec4<f32>(down13(in.uv, false), 1.0);
}

@fragment
fn fs_up(in: FsIn) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    var s = textureSampleLevel(src, lin, in.uv, 0.0).rgb * 4.0;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(-1.0, 0.0), 0.0).rgb * 2.0;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(1.0, 0.0), 0.0).rgb * 2.0;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(0.0, -1.0), 0.0).rgb * 2.0;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(0.0, 1.0), 0.0).rgb * 2.0;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(-1.0, -1.0), 0.0).rgb;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(1.0, -1.0), 0.0).rgb;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(-1.0, 1.0), 0.0).rgb;
    s += textureSampleLevel(src, lin, in.uv + t * vec2<f32>(1.0, 1.0), 0.0).rgb;
    return vec4<f32>(s / 16.0, 1.0);
}
