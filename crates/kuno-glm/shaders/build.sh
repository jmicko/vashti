#!/bin/sh
# Compiles every shader (and the batched variants of the per-token ones) to SPIR-V.
set -e
cd "$(dirname "$0")"
for f in *.comp; do
    glslangValidator -V --target-env vulkan1.3 --quiet -o "${f%.comp}.spv" "$f"
done
for f in hcdot hcmix kda dsanorm idxstore idxscore idxselect route; do
    glslangValidator -V --target-env vulkan1.3 --quiet -DBATCHED -o "${f}_b.spv" "$f.comp"
done
