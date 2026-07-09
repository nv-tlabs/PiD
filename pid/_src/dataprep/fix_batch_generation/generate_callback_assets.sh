#!/bin/bash
# Generate PiD fix_batch assets for prompts_small_face_and_text.txt across
# flux / flux2 / sd3 / sdxl / zimage / qwenimage / qwenimage_2512 at
# 512 / 768 / 1024 / 1328 LQ resolution.
#
# This script now calls create_pid_fix_batch_asset.py directly. It no longer
# creates intermediate webdataset tar shards or runs prepare_pixeldit_sr_fix_batch.py.
# Output layout per backbone:
#   <OUTPUT_ROOT>/<backbone>/full_step/<4*resolution>/fix_batch_XXXX.pt
#   <OUTPUT_ROOT>/<backbone>/<K>step/<4*resolution>/fix_batch_XXXX.pt
#
# Usage:
#   bash pid/_src/dataprep/fix_batch_generation/generate_callback_assets.sh
#   bash pid/_src/dataprep/fix_batch_generation/generate_callback_assets.sh flux flux2
#   RESOLUTIONS="1024 1328" bash .../generate_callback_assets.sh qwenimage
#
# Override on the fly:
#   OUTPUT_ROOT=./debug NPROC=1 RESOLUTIONS=512 bash .../generate_callback_assets.sh flux
#   CPU_OFFLOAD=1 bash .../generate_callback_assets.sh flux2
#   CPU_OFFLOAD=1 RESOLUTIONS="1024 1328" bash .../generate_callback_assets.sh qwenimage

set -euo pipefail

DEFAULT_PROMPTS_FILE=pid/_src/dataprep/prompts/prompts_harder_cases.txt
PROMPTS_FILE=${PROMPTS_FILE:-$DEFAULT_PROMPTS_FILE}
OUTPUT_ROOT=${OUTPUT_ROOT:-assets/pid_callback_assets}
NUM_IMAGES_PER_PROMPT=${NUM_IMAGES_PER_PROMPT:-1}
RESOLUTIONS=(${RESOLUTIONS:-512})
SEED=${SEED:-42}
NPROC=${NPROC:-4}
MASTER_PORT=${MASTER_PORT:-12341}
TORCHRUN=${TORCHRUN:-torchrun}
DTYPE=${DTYPE:-bf16}

EXTRA_ARGS=()
if [ -n "${CPU_OFFLOAD:-}" ]; then
    EXTRA_ARGS+=(--cpu_offload)
fi

ALL_BACKBONES=(flux flux2 sd3 sdxl zimage qwenimage qwenimage_2512)

if [ $# -gt 0 ]; then
    BACKBONES=("$@")
else
    BACKBONES=("${ALL_BACKBONES[@]}")
fi

echo "=== Prompts:           $PROMPTS_FILE ==="
echo "=== Output root:       $OUTPUT_ROOT ==="
echo "=== Backbones:         ${BACKBONES[*]} ==="
echo "=== Resolutions:       ${RESOLUTIONS[*]} ==="
echo "=== NPROC:             $NPROC ==="
echo "=== Master port:       $MASTER_PORT ==="
echo "=== Extra args:        ${EXTRA_ARGS[*]:-(none)} ==="

run_backbone() {
    local backbone=$1
    local resolution=$2
    local num_inference_steps
    local guidance_scale
    local save_steps=()

    case $backbone in
        flux)
            num_inference_steps=28
            guidance_scale=4.0
            save_steps=(22 24 26)
            ;;
        flux2)
            num_inference_steps=50
            guidance_scale=4.0
            save_steps=(44 46 48)
            ;;
        sd3)
            num_inference_steps=28
            guidance_scale=7.0
            save_steps=(22 24 26)
            ;;
        sdxl)
            # create_pid_fix_batch_asset.py uses the same SDXL Euler-to-VP
            # latent/sigma conversion as create_dataset.py.
            num_inference_steps=30
            guidance_scale=5.0
            save_steps=(24 26 28)
            ;;
        zimage)
            num_inference_steps=50
            guidance_scale=5.0
            save_steps=(44 46 48)
            ;;
        qwenimage)
            num_inference_steps=50
            guidance_scale=4.0
            save_steps=(42 44 46 48)
            ;;
        qwenimage_2512)
            num_inference_steps=50
            guidance_scale=4.0
            save_steps=(42 44 46 48)
            ;;
        *)
            echo "ERROR: Unknown backbone '$backbone'. Available: ${ALL_BACKBONES[*]}"
            exit 1
            ;;
    esac

    PYTHONPATH=. "$TORCHRUN" --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
        -m pid._src.dataprep.fix_batch_generation.create_pid_fix_batch_asset \
        --backbone "$backbone" \
        --prompts_file "$PROMPTS_FILE" \
        --num_images_per_prompt "$NUM_IMAGES_PER_PROMPT" \
        --resolution "$resolution" \
        --num_inference_steps "$num_inference_steps" \
        --guidance_scale "$guidance_scale" \
        --save_xt_steps "${save_steps[@]}" \
        --output_dir "$OUTPUT_ROOT/$backbone" \
        --seed "$SEED" \
        --dtype "$DTYPE" \
        "${EXTRA_ARGS[@]}"
}

for resolution in "${RESOLUTIONS[@]}"; do
    for backbone in "${BACKBONES[@]}"; do
        echo ""
        echo "====== $backbone @ ${resolution}px ======"
        run_backbone "$backbone" "$resolution"
    done
done
