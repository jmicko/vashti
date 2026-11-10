//! Pure-Rust model engine. It replaces stable-diffusion.cpp one model at a time; each model
//! is checked against the C++ engine for numbers and speed before it takes over.
pub mod audio_vae;
pub mod conv;
pub mod gpu;
pub mod safetensors;
pub mod video_vae;
