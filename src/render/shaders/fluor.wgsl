// Fluorescence microscopy: every subunit emits, the optics blur it into a
// point-spread function, and the optical section weights it by distance from
// the focal plane, with out-of-focus light spread wider and dimmer.

@group(1) @binding(0) var<storage, read> segs: array<Segment>;

struct SOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) q: vec2<f32>,
    @location(1) @interpolate(flat) a: vec2<f32>,
    @location(2) @interpolate(flat) b: vec2<f32>,
    @location(3) @interpolate(flat) sigma: f32,
    @location(4) @interpolate(flat) weight: f32,
    @location(5) @interpolate(flat) colour: vec3<f32>,
}

@vertex
fn vs_splat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SOut {
    let s = segs[ii];
    let va = (F.view * vec4<f32>(s.a, 1.0)).xyz;
    let vb = (F.view * vec4<f32>(s.b, 1.0)).xyz;
    let mid_z = -(va.z + vb.z) * 0.5;
    let dz = mid_z - F.p3.w;
    let half = F.p4.x;
    let psf = F.p4.y;
    // Defocus widens the spot; the total light is conserved.
    let sigma = sqrt(psf * psf + (0.12 * dz) * (0.12 * dz));
    let section = exp(-0.5 * dz * dz / (half * half));
    let len = length(vb - va);
    var w = section * len;
    let bits = u32(F.p5.w + 0.5);
    if ((s.flags & 128u) != 0u) {
        // A CESA rosette: a point source as bright as ~0.15 um of tubulin.
        w = section * 0.45;
        if ((bits & 1u) != 0u) { w = 0.0; }
    } else if ((s.flags & 256u) != 0u) {
        // Cellulose stained with a dye such as Pontamine Fast Scarlet S4B.
        w = section * len * 0.12;
        if ((bits & 2u) != 0u) { w = 0.0; }
    } else if ((s.flags & 64u) != 0u) {
        w = 0.0;
    } else if ((bits & 4u) != 0u) {
        w = 0.0;
    }
    var o: SOut;
    if (F.p1.w < 1e5 && max(s.a.z, s.b.z) > F.p1.w) {
        w = 0.0;
    }
    // Quad around the segment in the view plane, 3 sigma wide; collapsed
    // when the segment is outside the optical section.
    var ext = 3.0 * sigma;
    if (w < 1e-3) {
        ext = 0.0;
    }
    var dir = vb.xy - va.xy;
    let dl = length(dir);
    if (dl < 1e-6) {
        dir = vec2<f32>(1.0, 0.0);
    } else {
        dir = dir / dl;
    }
    let perp = vec2<f32>(-dir.y, dir.x);
    if (w < 1e-3) {
        o.pos = vec4<f32>(0.0, 0.0, -1.0, 1.0);
        o.weight = 0.0;
        return o;
    }
    let corners = array<vec2<f32>, 6>(
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, -1.0), vec2<f32>(1.0, 1.0), vec2<f32>(0.0, 1.0));
    let c = corners[vi];
    let along = mix(-ext, dl + ext, c.x);
    let q = va.xy + dir * along + perp * c.y * ext;
    let z = (va.z + vb.z) * 0.5;
    let world = (transpose_view_inv(vec3<f32>(q, z)));
    o.pos = F.view_proj * vec4<f32>(world, 1.0);
    o.q = q;
    o.a = va.xy;
    o.b = vb.xy;
    o.sigma = sigma;
    o.weight = w;
    o.colour = vec3<f32>(0.25, 1.0, 0.42);
    if ((s.flags & 128u) != 0u) {
        o.colour = vec3<f32>(1.0, 0.45, 0.12);
    } else if ((s.flags & 256u) != 0u) {
        o.colour = vec3<f32>(0.3, 0.5, 1.0);
    }
    return o;
}

// View space back to world: the view matrix is a rigid transform.
fn transpose_view_inv(v: vec3<f32>) -> vec3<f32> {
    let r = mat3x3<f32>(F.view[0].xyz, F.view[1].xyz, F.view[2].xyz);
    return transpose(r) * (v - F.view[3].xyz);
}

@fragment
fn fs_splat(in: SOut) -> @location(0) vec4<f32> {
    let ab = in.b - in.a;
    let h = clamp(dot(in.q - in.a, ab) / max(dot(ab, ab), 1e-12), 0.0, 1.0);
    let d = length(in.q - in.a - ab * h);
    let g = exp(-0.5 * d * d / (in.sigma * in.sigma)) / (2.0 * PI * in.sigma * in.sigma);
    let i = g * in.weight * F.p3.z * 0.02;
    return vec4<f32>(in.colour * i, 0.0);
}
