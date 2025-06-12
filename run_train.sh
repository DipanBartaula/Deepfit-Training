#!/usr/bin/env bash
# Example wrapper to run train.py with appropriate arguments.
# Adjust DATA_ROOT, DEVICE, and W&B settings as needed.

# -------- Configuration --------
DATA_ROOT="/path/to/your/data"        # <-- change to your dataset root
DEVICE="cuda"                         # or "cpu"
BATCH_SIZE=1
NUM_EPOCHS=10
LR=1e-4
SEED=42

# If using W&B:
WANDB_PROJECT="your_wandb_project"    # or leave empty to disable W&B logging
WANDB_ENTITY="your_wandb_entity"      # or leave empty
WANDB_NAME="run_$(date +%Y%m%d_%H%M%S)"  # example run name with timestamp

CHECKPOINT_DIR="checkpoints"
SAVE_EVERY_STEPS=100

# Enable debug prints? Set to "--debug" or leave empty.
DEBUG_FLAG="--debug"

# -------- Run training --------
CMD="python train.py \
  --data_root \"$DATA_ROOT\" \
  --device $DEVICE \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --lr $LR \
  --seed $SEED \
  --checkpoint_dir \"$CHECKPOINT_DIR\" \
  --save_every_steps $SAVE_EVERY_STEPS $DEBUG_FLAG"

# Add W&B args if desired:
if [ -n \"$WANDB_PROJECT\" ]; then
  CMD=\"$CMD --wandb_project $WANDB_PROJECT\"
  if [ -n \"$WANDB_ENTITY\" ]; then
    CMD=\"$CMD --wandb_entity $WANDB_ENTITY\"
  fi
  if [ -n \"$WANDB_NAME\" ]; then
    CMD=\"$CMD --wandb_name $WANDB_NAME\"
  fi
fi

echo "Running training command:"
echo $CMD
eval $CMD
