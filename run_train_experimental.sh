#!/usr/bin/env bash
# run_train.sh
# A script to launch train_experimental.py with configurable arguments and environment variables.

# === Configuration Defaults ===
# You can override these by exporting environment variables or passing CLI flags.
DATA_ROOT="${DATA_ROOT:-/path/to/dataset}"
DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LR="${LR:-1e-4}"
SEED="${SEED:-42}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-checkpoints_experimental}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
RESUME_STEP="${RESUME_STEP:-}"  # if empty, no resume
DEBUG_FLAG="${DEBUG_FLAG:-}"     # set to "--debug" to enable debug logs
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_NAME="${WANDB_NAME:-}"

# === Usage Function ===
usage() {
  cat << EOF
Usage: $0 [options]

Options:
  --data_root PATH         Path to dataset root (required if not set via DATA_ROOT env)
  --device DEVICE          Device to use: cuda or cpu (default: $DEVICE)
  --batch_size N           Batch size (default: $BATCH_SIZE)
  --num_epochs N           Number of epochs (default: $NUM_EPOCHS)
  --lr LR                  Learning rate (default: $LR)
  --seed SEED              Random seed (default: $SEED)
  --val_fraction FRAC      Validation split fraction (default: $VAL_FRACTION)
  --checkpoint_dir DIR     Directory to save checkpoints (default: $CHECKPOINT_DIR)
  --save_every_steps N     Save checkpoint every N steps (default: $SAVE_EVERY_STEPS)
  --resume_step N          Resume from this step (default: none)
  --debug                  Enable debug logging
  --wandb_project NAME     W&B project name (optional)
  --wandb_entity NAME      W&B entity name (optional)
  --wandb_name NAME        W&B run name (optional)
  -h, --help               Show this help message

Environment Variables:
  You can pre-export any of: DATA_ROOT, DEVICE, BATCH_SIZE, NUM_EPOCHS,
  LR, SEED, VAL_FRACTION, CHECKPOINT_DIR, SAVE_EVERY_STEPS, RESUME_STEP,
  DEBUG_FLAG, WANDB_PROJECT, WANDB_ENTITY, WANDB_NAME
EOF
  exit 1
}

# === Parse CLI Arguments ===
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --data_root)
      DATA_ROOT="$2"; shift 2;;
    --device)
      DEVICE="$2"; shift 2;;
    --batch_size)
      BATCH_SIZE="$2"; shift 2;;
    --num_epochs)
      NUM_EPOCHS="$2"; shift 2;;
    --lr)
      LR="$2"; shift 2;;
    --seed)
      SEED="$2"; shift 2;;
    --val_fraction)
      VAL_FRACTION="$2"; shift 2;;
    --checkpoint_dir)
      CHECKPOINT_DIR="$2"; shift 2;;
    --save_every_steps)
      SAVE_EVERY_STEPS="$2"; shift 2;;
    --resume_step)
      RESUME_STEP="$2"; shift 2;;
    --debug)
      DEBUG_FLAG="--debug"; shift 1;;
    --wandb_project)
      WANDB_PROJECT="$2"; shift 2;;
    --wandb_entity)
      WANDB_ENTITY="$2"; shift 2;;
    --wandb_name)
      WANDB_NAME="$2"; shift 2;;
    -h|--help)
      usage;;
    *)
      echo "Unknown option: $1";
      usage;;
  esac
done

# Check required argument DATA_ROOT
if [[ -z "$DATA_ROOT" ]]; then
  echo "Error: data_root must be specified via --data_root or DATA_ROOT env variable."
  usage
fi

# Create checkpoint directory if not exists
mkdir -p "$CHECKPOINT_DIR"

# Optionally set CUDA_VISIBLE_DEVICES
# Uncomment and set GPU IDs as needed, e.g.: export CUDA_VISIBLE_DEVICES=0
# export CUDA_VISIBLE_DEVICES=0

# Construct the command
CMD=(python train_experimental.py \
  --data_root "$DATA_ROOT" \
  --device "$DEVICE" \
  --batch_size "$BATCH_SIZE" \
  --num_epochs "$NUM_EPOCHS" \
  --lr "$LR" \
  --seed "$SEED" \
  --val_fraction "$VAL_FRACTION" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --save_every_steps "$SAVE_EVERY_STEPS" \
)

if [[ -n "$RESUME_STEP" ]]; then
  CMD+=(--resume_step "$RESUME_STEP")
fi
if [[ -n "$DEBUG_FLAG" ]]; then
  CMD+=("$DEBUG_FLAG")
fi
if [[ -n "$WANDB_PROJECT" ]]; then
  CMD+=(--wandb_project "$WANDB_PROJECT")
fi
if [[ -n "$WANDB_ENTITY" ]]; then
  CMD+=(--wandb_entity "$WANDB_ENTITY")
fi
if [[ -n "$WANDB_NAME" ]]; then
  CMD+=(--wandb_name "$WANDB_NAME")
fi

# Print and execute
echo "Running training with command:"
echo "${CMD[@]}"

# Execute
"${CMD[@]}"
