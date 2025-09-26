#!/usr/bin/env bash
set -euo pipefail

EPOCHS=50
SAVE_INTERVAL=10

RESUME_FROM=""
RESUME_OPTIMIZER=false    # true to restore optimizer when resuming
FINETUNE=false            # true to finetune with smaller LR (weights only)
FINETUNE_LR=1e-5
FREEZE_ENCODER_EPOCHS=0   # e.g., 2~5 for stable finetune warmup

IMAGE_DIRS=("freespace_dataset/images")
MASK_DIRS=("freespace_dataset/masks")

OLD_IMAGE_DIRS=()
OLD_MASK_DIRS=()
NEW_IMAGE_DIRS=()
NEW_MASK_DIRS=()
NEW_RATIO=0.8

# Layer-wise learning rates
ENCODER_LR=1e-5
DECODER_LR=1e-4
WEIGHT_DECAY=1e-4

PYTHON_BIN="python3"

add_flag_if_true() {
  local cond="$1"; shift
  local flag="$1"; shift
  if [[ "$cond" == "true" ]]; then
    echo -n " ${flag}"
  fi
}

CMD="${PYTHON_BIN} train.py --epochs ${EPOCHS} --save_interval ${SAVE_INTERVAL}"

# Resume / finetune
if [[ -n "${RESUME_FROM}" ]]; then
  CMD+=" --resume_from ${RESUME_FROM}"
fi
CMD+="$(add_flag_if_true "${RESUME_OPTIMIZER}" "--resume_optimizer")"
CMD+="$(add_flag_if_true "${FINETUNE}" "--finetune")"
CMD+=" --finetune_lr ${FINETUNE_LR}"
CMD+=" --freeze_encoder_epochs ${FREEZE_ENCODER_EPOCHS}"

if (( ${#IMAGE_DIRS[@]} )); then
  CMD+=" --image_dirs"
  for d in "${IMAGE_DIRS[@]}"; do CMD+=" ${d}"; done
fi
if (( ${#MASK_DIRS[@]} )); then
  CMD+=" --mask_dirs"
  for d in "${MASK_DIRS[@]}"; do CMD+=" ${d}"; done
fi

# Domain-split training (old/new)
if (( ${#OLD_IMAGE_DIRS[@]} )); then
  CMD+=" --old_image_dirs"
  for d in "${OLD_IMAGE_DIRS[@]}"; do CMD+=" ${d}"; done
fi
if (( ${#OLD_MASK_DIRS[@]} )); then
  CMD+=" --old_mask_dirs"
  for d in "${OLD_MASK_DIRS[@]}"; do CMD+=" ${d}"; done
fi
if (( ${#NEW_IMAGE_DIRS[@]} )); then
  CMD+=" --new_image_dirs"
  for d in "${NEW_IMAGE_DIRS[@]}"; do CMD+=" ${d}"; done
fi
if (( ${#NEW_MASK_DIRS[@]} )); then
  CMD+=" --new_mask_dirs"
  for d in "${NEW_MASK_DIRS[@]}"; do CMD+=" ${d}"; done
fi
CMD+=" --new_ratio ${NEW_RATIO}"

# Layer-wise LRs
CMD+=" --encoder_lr ${ENCODER_LR} --decoder_lr ${DECODER_LR} --weight_decay ${WEIGHT_DECAY}"

# ==========================
# Execute
# ==========================
echo "Running: ${CMD}"
eval "${CMD}" 