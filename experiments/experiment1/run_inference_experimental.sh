#!/usr/bin/env bash
# Wrapper to run modified inference.py (image-only latents).

# -------- Configuration --------
CHECKPOINT_STEP=100                       # checkpoint step to load
CHECKPOINT_DIR="checkpoints"              # directory where checkpoints are saved
DEVICE="cuda"                             # or "cpu"

PERSON_PATH="/path/to/person_image.png"   # path to person image
MASK_PATH="/path/to/person_mask.png"      # path to mask image
CLOTHING_PATH="/path/to/clothing_image.png"  # path to clothing image
PROMPT="A person wearing a red dress in studio lighting"  # your text prompt

OUTPUT_PATH="output.png"                  # where to save generated image
HEIGHT=1024                               # inference resolution height
WIDTH=1024                                # inference resolution width
GUIDANCE_SCALE=7.0
NUM_INFERENCE_STEPS=28

# Enable debug prints? Set to "--debug" or leave empty.
DEBUG_FLAG="--debug"

# -------- Run inference --------
CMD="python inference_experimental.py \
  --checkpoint_step $CHECKPOINT_STEP \
  --checkpoint_dir \"$CHECKPOINT_DIR\" \
  --device $DEVICE \
  --person_path \"$PERSON_PATH\" \
  --mask_path \"$MASK_PATH\" \
  --clothing_path \"$CLOTHING_PATH\" \
  --prompt \"$PROMPT\" \
  --height $HEIGHT \
  --width $WIDTH \
  --guidance_scale $GUIDANCE_SCALE \
  --num_inference_steps $NUM_INFERENCE_STEPS $DEBUG_FLAG \
  --output_path \"$OUTPUT_PATH\""

echo "Running inference command:"
echo $CMD
eval $CMD
