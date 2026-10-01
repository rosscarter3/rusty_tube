//! Headless rendering to PNG: stills and frame sequences.

use super::camera::OrbitCamera;
use super::{Renderer, Settings};
use anyhow::Result;
use std::path::Path;

pub const OUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

pub struct Offscreen {
    pub renderer: Renderer,
    tex: wgpu::Texture,
    view: wgpu::TextureView,
    readback: wgpu::Buffer,
    padded_row: u32,
}

impl Offscreen {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let (_adapter, device, queue) = pollster::block_on(super::gpu(None, &instance))?;
        Ok(Self::with_device(device, queue, width, height))
    }

    /// An offscreen target sharing an existing device (e.g. the viewer's).
    pub fn with_device(device: wgpu::Device, queue: wgpu::Queue, width: u32, height: u32) -> Self {
        let renderer = Renderer::new(device, queue, OUT_FORMAT, (width, height));
        let tex = renderer.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: OUT_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = tex.create_view(&Default::default());
        let padded_row = (width * 4).div_ceil(256) * 256;
        let readback = renderer.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Offscreen {
            renderer,
            tex,
            view,
            readback,
            padded_row,
        }
    }

    /// Render `spp` accumulated frames of a still scene and read back the
    /// last one.
    pub fn render(&mut self, cam: &OrbitCamera, s: &Settings, spp: u32) -> Result<image::RgbaImage> {
        let (w, h) = self.renderer.size();
        for i in 0..spp.max(1) {
            let mut enc = self.renderer.device.create_command_encoder(&Default::default());
            self.renderer.render(&mut enc, &self.view, cam, s);
            if i + 1 == spp.max(1) {
                enc.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: &self.readback,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(self.padded_row),
                            rows_per_image: Some(h),
                        },
                    },
                    wgpu::Extent3d {
                        width: w,
                        height: h,
                        depth_or_array_layers: 1,
                    },
                );
            }
            self.renderer.queue.submit([enc.finish()]);
            // Keep the GPU queue short.
            if i % 8 == 7 {
                self.renderer.device.poll(wgpu::PollType::wait_indefinitely())?;
            }
        }
        let slice = self.readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        self.renderer.device.poll(wgpu::PollType::wait_indefinitely())?;
        let data = slice.get_mapped_range().map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let mut img = image::RgbaImage::new(w, h);
        for y in 0..h {
            let row = &data[(y * self.padded_row) as usize..(y * self.padded_row + w * 4) as usize];
            img.as_mut()[(y * w * 4) as usize..((y + 1) * w * 4) as usize].copy_from_slice(row);
        }
        drop(data);
        self.readback.unmap();
        Ok(img)
    }

    pub fn save(&mut self, cam: &OrbitCamera, s: &Settings, spp: u32, path: &Path) -> Result<()> {
        let img = self.render(cam, s, spp)?;
        img.save(path)?;
        Ok(())
    }
}
