// Bloom, exposure, AgX tone mapping, grading, vignette and grain.

@group(1) @binding(0) var hdr_tex: texture_2d<f32>;
@group(1) @binding(1) var bloom_tex: texture_2d<f32>;
@group(1) @binding(2) var lin: sampler;

// AgX (Troy Sobotka), in Benjamin Wrensch's minimal fit.
fn agx_contrast(x: vec3<f32>) -> vec3<f32> {
    let x2 = x * x;
    let x4 = x2 * x2;
    return 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232;
}

fn agx(c: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = m * max(c, vec3<f32>(1e-10));
    v = clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev));
    v = (v - min_ev) / (max_ev - min_ev);
    return agx_contrast(v);
}

fn agx_look(v: vec3<f32>) -> vec3<f32> {
    let lw = vec3<f32>(0.2126, 0.7152, 0.0722);
    let luma = dot(v, lw);
    let slope = vec3<f32>(1.0);
    let power = vec3<f32>(F.p5.z);
    let sat = F.p5.y;
    let o = pow(max(v * slope, vec3<f32>(0.0)), power);
    let l2 = dot(o, lw);
    return l2 + sat * (o - l2);
}

fn agx_eotf(c: vec3<f32>) -> vec3<f32> {
    let m = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    return pow(max(m * c, vec3<f32>(0.0)), vec3<f32>(2.2));
}

fn shade(in: FsIn) -> vec3<f32> {
    let hdr = textureSampleLevel(hdr_tex, lin, in.uv, 0.0).rgb;
    let bloom = textureSampleLevel(bloom_tex, lin, in.uv, 0.0).rgb;
    var c = mix(hdr, bloom, F.p0.y) * F.p0.x;
    c = agx_eotf(agx_look(agx(c)));
    let q = in.uv - 0.5;
    let vig = 1.0 - F.p2.z * dot(q, q) * 1.6;
    c *= vig;
    // White film grain, scaled toward the shadows where it shows.
    let px = vec2<u32>(in.pos.xy);
    let hsh = hash_u32(px.x * 1973u + px.y * 9277u + u32(F.jitter.z) * 26699u);
    let n = f32(hsh & 0xffffu) / 65535.0 - 0.5;
    let lum = dot(c, vec3<f32>(0.2126, 0.7152, 0.0722));
    c += vec3<f32>(n * F.p2.w * 0.025 * (1.0 - 0.6 * sqrt(clamp(lum, 0.0, 1.0))));
    return max(c, vec3<f32>(0.0));
}

@fragment
fn fs_composite(in: FsIn) -> @location(0) vec4<f32> {
    return vec4<f32>(shade(in), 1.0);
}

@fragment
fn fs_composite_gamma(in: FsIn) -> @location(0) vec4<f32> {
    let c = shade(in);
    return vec4<f32>(pow(c, vec3<f32>(1.0 / 2.2)), 1.0);
}
