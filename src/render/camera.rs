//! Orbit camera with sub-pixel jitter for temporal anti-aliasing.

use glam::{Mat4, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    /// Radians around +z.
    pub yaw: f32,
    /// Radians above the xy plane.
    pub pitch: f32,
    pub distance: f32,
    /// Vertical field of view, radians.
    pub fov: f32,
}

impl OrbitCamera {
    pub fn framing(min: Vec3, max: Vec3) -> Self {
        let size = (max - min).length();
        OrbitCamera {
            target: (min + max) * 0.5,
            yaw: -0.9,
            pitch: 0.75,
            distance: size * 1.05,
            fov: 30f32.to_radians(),
        }
    }

    pub fn eye(&self) -> Vec3 {
        let dir = Vec3::new(
            self.pitch.cos() * self.yaw.cos(),
            self.pitch.cos() * self.yaw.sin(),
            self.pitch.sin(),
        );
        self.target + dir * self.distance
    }

    pub fn view(&self) -> Mat4 {
        let up = if self.pitch.abs() > 1.55 { Vec3::Y } else { Vec3::Z };
        glam::camera::rh::view::look_at_mat4(self.eye(), self.target, up)
    }

    pub fn near_far(&self) -> (f32, f32) {
        (self.distance * 0.02, self.distance * 4.0 + 60.0)
    }

    /// Perspective projection with a sub-pixel offset (in pixels).
    pub fn proj(&self, aspect: f32, jitter_px: Vec2, size_px: Vec2) -> Mat4 {
        let (n, f) = self.near_far();
        let mut p = glam::camera::rh::proj::directx::perspective(self.fov, aspect, n, f);
        p.z_axis.x += 2.0 * jitter_px.x / size_px.x;
        p.z_axis.y += 2.0 * jitter_px.y / size_px.y;
        p
    }

    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * 0.008;
        self.pitch = (self.pitch + dy * 0.008).clamp(-1.5, 1.5);
    }

    pub fn pan(&mut self, dx: f32, dy: f32, viewport_h: f32) {
        let v = self.view().inverse();
        let right = v.x_axis.truncate();
        let up = v.y_axis.truncate();
        let scale = 2.0 * self.distance * (self.fov * 0.5).tan() / viewport_h;
        self.target += (-right * dx + up * dy) * scale;
    }

    pub fn zoom(&mut self, amount: f32) {
        self.distance = (self.distance * (1.0 - amount * 0.1)).clamp(0.03, 500.0);
    }
}

/// Halton(2,3) sequence in [-0.5, 0.5]^2, the usual TAA jitter.
pub fn halton_jitter(i: u32) -> Vec2 {
    fn halton(mut i: u32, b: u32) -> f32 {
        let mut f = 1.0;
        let mut r = 0.0;
        while i > 0 {
            f /= b as f32;
            r += f * (i % b) as f32;
            i /= b;
        }
        r
    }
    let k = (i % 64) + 1;
    Vec2::new(halton(k, 2) - 0.5, halton(k, 3) - 0.5)
}
