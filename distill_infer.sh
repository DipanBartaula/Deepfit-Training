#!/usr/bin/env bash
# run_inference.sh
# Usage: ./run_inference.sh
# Script to launch inference with distilled DeepFit model.

set -e

# Activate virtual environment if desired
# Uncomment if using the venv from setup_env.sh
# source venv/bin/activate

# ---------------------------
# User-editable parameters:
# ---------------------------

# Path to the distilled student checkpoint
STUDENT_CKPT="${STUDENT_CKPT:-checkpoints_distill/student_step_100.pth}"

# Input images
PERSON_PATH="${PERSON_PATH:-/path/to/person.png}"
MASK_PATH="${MASK_PATH:-/path/to/mask.png}"
CLOTHING_PATH="${CLOTHING_PATH:-/path/to/clothing.png}"

# Text prompt
PROMPT="${PROMPT:-A person wearing a red dress in studio lighting}"

# Generation parameters
HEIGHT="${HEIGHT:-1024}"
WIDTH="${WIDTH:-1024}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-7.0}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-28}"

# Output path
OUTPUT_PATH="${OUTPUT_PATH:-output.png}"

# Use EMA? "true" or "false"
USE_EMA="${USE_EMA:-false}"

# Debug flag: set to "true" to pass --debug
DEBUG="${DEBUG:-false}"

# Device: let script decide or override
DEVICE="${DEVICE:-cuda}"

# ---------------------------
# End of user parameters
# ---------------------------

# Check checkpoint exists
if [ ! -f "${STUDENT_CKPT}" ]; then
    echo "Error: student checkpoint not found at ${STUDENT_CKPT}"
    exit 1
fi

# Construct command
CMD=(python inference.py \
    --student_ckpt "${STUDENT_CKPT}" \
    --person_path "${PERSON_PATH}" \
    --mask_path "${MASK_PATH}" \
    --clothing_path "${CLOTHING_PATH}" \
    --prompt "${PROMPT}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --guidance_scale "${GUIDANCE_SCALE}" \
    --num_inference_steps "${NUM_INFERENCE_STEPS}" \
    --output_path "${OUTPUT_PATH}" \
    --device "${DEVICE}"
)

# Add --use_ema if requested
if [ "${USE_EMA}" = "true" ]; then
    CMD+=(--use_ema)
fi

# Add debug flag
if [ "${DEBUG}" = "true" ]; then
    CMD+=(--debug)
fi

echo "Running inference:"
echo "${CMD[@]}"
# Execute
"${CMD[@]}"
