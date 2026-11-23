//! Pure-Rust model engine. It replaces stable-diffusion.cpp one model at a time; each model
//! is checked against the C++ engine for numbers and speed before it takes over.
pub mod audio_vae;
pub mod conv;
pub mod dit;
pub mod dit_gpu;
pub mod encoder_gpu;
pub mod gpu;
pub mod prompt;
pub mod qwen_dit;
pub mod qwen_gpu;
pub mod rng;
pub mod safetensors;
pub mod text_encoder;
pub mod text_gpu;
pub mod tokenizer;
pub mod video_encoder;
pub mod video_gpu;
pub mod video_tiles;
pub mod video_vae;
pub mod vision;
pub mod vision_gpu;
pub mod xe2;
