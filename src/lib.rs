//! rusty_tube: a Rust port of tubulaton, the 3D simulator of cortical
//! microtubule self-organisation (Mirabet et al. 2018), with pavement-cell
//! geometry generation and a real-time renderer.

pub mod cellgen;
pub mod mesh;
pub mod params;
pub mod qos;
pub mod render;
pub mod rng;
pub mod setup;
pub mod sim;
pub mod viewer;
pub mod vtk;
