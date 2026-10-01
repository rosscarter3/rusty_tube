// Microtubules as ray-traced capsule impostors.
//
// Each segment is drawn as the back faces of a box around it; the fragment
// shader intersects the view ray with the exact capsule and writes the true
// depth, so tubes are perfectly round at any zoom, join seamlessly, and
// intersect each other and the membrane correctly. The same shader renders
// the shadow map with an orthographic light ray.

@group(1) @binding(0) var<storage, read> segs: array<Segment>;

const BOX_IDX = array<u32, 36>(
    1u, 3u, 7u, 1u, 7u, 5u,
    0u, 4u, 6u, 0u, 6u, 2u,
    2u, 6u, 7u, 2u, 7u, 3u,
    0u, 1u, 5u, 0u, 5u, 4u,
    4u, 5u, 7u, 4u, 7u, 6u,
    0u, 2u, 3u, 0u, 3u, 1u);

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) @interpolate(flat) inst: u32,
}

fn radius_of(s: Segment) -> f32 {
    if ((s.flags & 128u) != 0u) {
        return max(F.p1.y * 1.6, 0.02) * max(F.bounds_min.w, 0.1);
    }
    if ((s.flags & 256u) != 0u) {
        return max(F.p1.y * 0.17, 0.004);
    }
    if ((s.flags & 64u) != 0u) {
        return max(F.p1.y * 0.7, 0.012);
    }
    return F.p1.y;
}

fn box_corner(s: Segment, corner: u32) -> vec3<f32> {
    let r = radius_of(s) * 1.05;
    var axis = s.b - s.a;
    let len = length(axis);
    if (len < 1e-6) {
        axis = vec3<f32>(0.0, 0.0, 1.0);
    } else {
        axis = axis / len;
    }
    var helper = vec3<f32>(0.0, 0.0, 1.0);
    if (abs(axis.z) > 0.9) {
        helper = vec3<f32>(1.0, 0.0, 0.0);
    }
    let u = normalize(cross(axis, helper));
    let v = cross(axis, u);
    let c = vec3<f32>(f32(corner & 1u), f32((corner >> 1u) & 1u), f32((corner >> 2u) & 1u));
    let lx = mix(-r, r, c.x);
    let ly = mix(-r, r, c.y);
    let lz = mix(-r, len + r, c.z);
    return s.a + u * lx + v * ly + axis * lz;
}

// Visibility switches in F.p5.w: bit 0 hides CSCs, bit 1 fibrils, bit 2
// microtubules.
fn hidden(s: Segment) -> bool {
    let bits = u32(F.p5.w + 0.5);
    if ((s.flags & 128u) != 0u) {
        return (bits & 1u) != 0u;
    }
    if ((s.flags & 256u) != 0u) {
        return (bits & 2u) != 0u;
    }
    if ((s.flags & 64u) != 0u) {
        return false;
    }
    return (bits & 4u) != 0u;
}

@vertex
fn vs_capsule(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let s = segs[ii];
    let w = box_corner(s, BOX_IDX[vi]);
    var o: VsOut;
    o.pos = F.view_proj * vec4<f32>(w, 1.0);
    if (hidden(s)) {
        o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    }
    o.world = w;
    o.inst = ii;
    return o;
}

@vertex
fn vs_capsule_shadow(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VsOut {
    let s = segs[ii];
    let w = box_corner(s, BOX_IDX[vi]);
    var o: VsOut;
    o.pos = F.light_view_proj * vec4<f32>(w, 1.0);
    if (hidden(s)) {
        o.pos = vec4<f32>(0.0, 0.0, -2.0, 1.0);
    }
    o.world = w;
    o.inst = ii;
    return o;
}

// Ray-capsule intersection (after Inigo Quilez). Returns t or -1.
fn cap_intersect(ro: vec3<f32>, rd: vec3<f32>, pa: vec3<f32>, pb: vec3<f32>, r: f32) -> f32 {
    let ba = pb - pa;
    let oa = ro - pa;
    let baba = dot(ba, ba);
    let bard = dot(ba, rd);
    let baoa = dot(ba, oa);
    let rdoa = dot(rd, oa);
    let oaoa = dot(oa, oa);
    let a = baba - bard * bard;
    var b = baba * rdoa - baoa * bard;
    var c = baba * oaoa - baoa * baoa - r * r * baba;
    var h = b * b - a * c;
    var oc = oa;
    if (a > 1e-10 * baba && h >= 0.0) {
        let t = (-b - sqrt(h)) / a;
        let y = baoa + t * bard;
        if (y > 0.0 && y < baba) {
            return t;
        }
        if (y > 0.0) {
            oc = ro - pb;
        }
    } else if (bard < 0.0) {
        oc = ro - pb;
    }
    b = dot(rd, oc);
    c = dot(oc, oc) - r * r;
    h = b * b - c;
    if (h > 0.0) {
        return -b - sqrt(h);
    }
    return -1.0;
}

fn cap_normal(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> vec3<f32> {
    let ba = b - a;
    let pa = p - a;
    let h = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-12), 0.0, 1.0);
    return normalize(pa - h * ba);
}

struct GOut {
    @location(0) normal: vec4<f32>,
    @location(1) albedo: vec4<f32>,
    @location(2) aux: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}

@fragment
fn fs_capsule(in: VsOut) -> GOut {
    let s = segs[in.inst];
    let r = radius_of(s);
    let rd = normalize(in.world - F.cam_pos.xyz);
    // Start the ray just in front of the box, not at the camera: keeps the
    // quadratic well conditioned for thin, distant capsules.
    let back = length(s.b - s.a) + 4.0 * r;
    let ro = in.world - rd * back * 2.0;
    let t = cap_intersect(ro, rd, s.a, s.b, r);
    if (t < 0.0) {
        discard;
    }
    let p = ro + rd * t;
    let clip = F.view_proj * vec4<f32>(p, 1.0);
    if (F.p1.w < 1e5 && p.z > F.p1.w) {
        discard;
    }
    var n = cap_normal(p, s.a, s.b);
    let dir = normalize(s.b - s.a + vec3<f32>(1e-9));
    var emit: vec3<f32>;
    var base = segment_colour(s, dir, &emit);

    // Molecular surface detail, faded in once it is resolved on screen: 13
    // protofilaments around the tube, and along each the 8 nm alpha/beta
    // tubulin dimers, offset between neighbours by the 3-start helix (12 nm
    // of rise per turn). Scaled with the drawn radius so an exaggerated tube
    // keeps the real proportions.
    let scale = r / 0.0125;
    let px_um = length(p - F.cam_pos.xyz) * F.jitter.w;
    let lod = 1.0 - smoothstep(0.0025, 0.007, px_um / scale);
    if (lod > 0.0 && (s.flags & (64u | 128u | 256u)) == 0u) {
        var helper = vec3<f32>(0.0, 0.0, 1.0);
        if (abs(dir.z) > 0.9) {
            helper = vec3<f32>(1.0, 0.0, 0.0);
        }
        let u = normalize(cross(dir, helper));
        let v = cross(dir, u);
        let theta = atan2(dot(n, v), dot(n, u));
        let along = s.minus_dist + dot(p - s.a, dir);
        let pf = theta * 13.0 / (2.0 * PI);
        let ridge = sin(2.0 * PI * pf);
        let e_theta = cross(dir, n);
        // Lattice phase along the protofilament, in dimers.
        let dimer = 0.008 * scale;
        let phase = along / dimer + pf * (12.0 / 13.0) / 8.0;
        let mono = fract(phase * 2.0);
        let groove = smoothstep(0.0, 0.1, 1.0 - abs(fract(phase) - 0.5) * 2.0);
        n = normalize(n - e_theta * ridge * 0.55 * lod);
        let alpha_beta = select(0.92, 1.0, mono < 0.5);
        let seam = smoothstep(0.0, 0.35, abs(ridge));
        let shade = mix(1.0, alpha_beta * mix(0.86, 1.0, groove) * mix(0.62, 1.0, seam), lod);
        base = base * shade;
    }
    var o: GOut;
    o.normal = vec4<f32>(n, 0.32);
    o.albedo = vec4<f32>(base, 1.0);
    o.aux = vec4<f32>(emit, 0.0);
    o.depth = clip.z / clip.w;
    return o;
}

@fragment
fn fs_capsule_shadow(in: VsOut) -> @builtin(frag_depth) f32 {
    let s = segs[in.inst];
    let r = radius_of(s);
    // Orthographic light: every ray travels against the light direction.
    let rd = -F.light_dir.xyz;
    let back = length(s.b - s.a) + 4.0 * r;
    let ro = in.world - rd * back * 2.0;
    let t = cap_intersect(ro, rd, s.a, s.b, r);
    if (t < 0.0) {
        discard;
    }
    let p = ro + rd * t;
    if (F.p1.w < 1e5 && p.z > F.p1.w) {
        discard;
    }
    let clip = F.light_view_proj * vec4<f32>(p, 1.0);
    return clip.z / clip.w;
}
