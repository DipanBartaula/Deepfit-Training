#!/usr/bin/env bash
# run_train.sh
# Usage: ./run_train.sh
# Script to launch training with LADD distillation.

set -e

# Activate virtual environment if desired
# Uncomment if using the venv from setup_env.sh
# source venv/bin/activate

# ---------------------------
# User-editable parameters:
# ---------------------------

# Path to your dataset root; JointVirtualTryOnDataset should read from here
DATA_ROOT="${DATA_ROOT:-/path/to/your/data}"

# Batch size and epochs
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"

# Learning rates
LR_STUDENT="${LR_STUDENT:-1e-4}"
LR_DISCRIMINATOR="${LR_DISCRIMINATOR:-2e-4}"

# W&B configuration (optional). If not using W&B, leave project empty.
WANDB_PROJECT="${WANDB_PROJECT:-your_wandb_project}"
WANDB_ENTITY="${WANDB_ENTITY:-your_wandb_entity}"
WANDB_NAME="${WANDB_NAME:-distill_run}"

# Checkpoint directory
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_distill}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"

# Debug flag: set to "true" to pass --debug
DEBUG="${DEBUG:-false}"

# ---------------------------
# End of user parameters
# ---------------------------

# Create checkpoint directory
mkdir -p "${CHECKPOINT_DIR}"

# Construct command
CMD=(python distill_train.py \
    --data_root "${DATA_ROOT}" \
    --batch_size "${BATCH_SIZE}" \
    --num_epochs "${NUM_EPOCHS}" \
    --lr_student "${LR_STUDENT}" \
    --lr_discriminator "${LR_DISCRIMINATOR}" \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --save_every_steps "${SAVE_EVERY_STEPS}"
)

# Add W&B args if provided
if [ -n "${WANDB_PROJECT}" ]; then
    CMD+=(--wandb_project "${WANDB_PROJECT}")
    # Only add entity/name if non-empty
    if [ -n "${WANDB_ENTITY}" ]; then
        CMD+=(--wandb_entity "${WANDB_ENTITY}")
    fi
    if [ -n "${WANDB_NAME}" ]; then
        CMD+=(--wandb_name "${WANDB_NAME}")
    fi
fi

# Add debug flag
if [ "${DEBUG}" = "true" ]; then
    CMD+=(--debug)
fi

echo "Running training:"
echo "${CMD[@]}"
# Execute
"${CMD[@]}"
