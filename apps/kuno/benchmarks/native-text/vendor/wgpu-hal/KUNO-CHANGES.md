# Kuno native laboratory change

Vendored wgpu-hal 29.0.4 retains its original MIT/Apache-2.0 licenses and source.
The only source change is a documented `unsafe raw_buffer_handle` accessor on
Vulkan Buffer. Kuno's serialized Rust native compute executor borrows the handle
while retaining its Wgpu owner; it must neither destroy the buffer nor submit
unsynchronized work. This patch is confined to the standalone laboratory workspace.
