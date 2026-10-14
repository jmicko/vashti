# Kuno native laboratory change

Vendored wgpu-hal 29.0.4 retains its original MIT/Apache-2.0 licenses and source.
The initial source change is a documented `unsafe raw_buffer_handle` accessor on
Vulkan Buffer. Kuno's serialized Rust native compute executor borrows the handle
while retaining its Wgpu owner; it must neither destroy the buffer nor submit
unsynchronized work. This patch is confined to the standalone laboratory workspace.

The coherent native core additionally enables supported buffer device addresses
and addressable Vulkan buffers when KUNO_STRATA_INTEL_CORE=public is explicitly
selected. Other runtime modes retain upstream feature and allocation behavior.
