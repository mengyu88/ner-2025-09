#!/usr/bin/env bash
set -euo pipefail

# Exploratory GENIA experiment. PGD is disabled and BHPC is enabled.  The
# highest test F1 selects checkpoints and controls early stopping by request;
# do not compare this number directly with dev-selected results.
PROJECT=/root/shared-nvme/projects/SPSR-Net
PY=/root/.conda/envs/difinet/bin/python
MODEL="$PROJECT/pretrained_models/biobert-v1.1"
STAMP=${1:-$(date +%Y%m%d_%H%M%S)}
RUN_DIR="$PROJECT/runs/genia_bhpc_test_selected_$STAMP"

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$PROJECT/runs/latest_genia_bhpc_test_selected_dir.txt"

ARGS=(
  -u train.py
  -n 30 -d genia
  --model_name "$MODEL"
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0
  --cnn_dim 400 --biaffine_size 200 --n_head 4
  -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2
  --ent_thres 0.48
  --head_type linear
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0
  --adv_type none
  --sad_topk 2 --use_snsa 1 --use_hsr 1 --sad_use_rel_bias 1 --sad_gate 1
  --bhpc_weight 0.05 --bhpc_dim 128 --bhpc_temperature 0.1 --bhpc_momentum 0.95 --bhpc_margin 0.1
  --fp16 --fp16_init_scale 1.0 --num_workers 4
  --checkpoint_monitor 'f#f#test'
  --early_stop_monitor 'f#f#test' --early_stop_patience 4
)

printf '%q ' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${ARGS[@]}" > "$RUN_DIR/command.txt"
printf '\n' >> "$RUN_DIR/command.txt"

(
  cd "$PROJECT"
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" "${ARGS[@]}"
) 2>&1 | tee "$RUN_DIR/train.log"
