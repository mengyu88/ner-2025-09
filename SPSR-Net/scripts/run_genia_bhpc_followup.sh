#!/usr/bin/env bash
# Follow-up stages for the overnight GENIA BHPC search. All runs use no PGD.
set -uo pipefail

PROJECT_ROOT="/root/shared-nvme/projects/SPSR-Net"
PYTHON_BIN="/root/.venvs/spsr-net/bin/python"
RUN_ROOT="$PROJECT_ROOT/runs/genia_bhpc_replacement_20260818_overnight"
MODEL_PATH="$PROJECT_ROOT/pretrained_models/biobert-v1.1"

COMMON=(
  -u train.py -n 30 -d genia --model_name "$MODEL_PATH"
  --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2 --ent_thres 0.48
  --head_type linear --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0
  --adv_type none --sad_topk 2 --use_snsa 1 --use_hsr 1 --sad_use_rel_bias 1 --sad_gate 1
  --fp16 --fp16_init_scale 1.0 --num_workers 4 --seed 0
  --checkpoint_monitor f#f#dev --early_stop_monitor f#f#dev --early_stop_patience 5
)

run_stage() {
  local name="$1"
  shift
  local stage_dir="$RUN_ROOT/$name"
  mkdir -p "$stage_dir"
  touch "$stage_dir/start.marker"
  printf '%q ' "$PYTHON_BIN" "${COMMON[@]}" "$@" > "$stage_dir/command.txt"
  printf '\n' >> "$stage_dir/command.txt"
  echo "[$(date -u +%FT%TZ)] START $name" | tee "$stage_dir/status.log"
  (
    cd "$PROJECT_ROOT" || exit 1
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PYTHON_BIN" "${COMMON[@]}" "$@"
  ) >> "$stage_dir/train.log" 2>&1
  local code=$?
  find "$PROJECT_ROOT/_saved_models" -mindepth 1 -maxdepth 1 -type d -newer "$stage_dir/start.marker" -print | sort > "$stage_dir/checkpoint_dirs.txt"
  echo "[$(date -u +%FT%TZ)] END $name exit_code=$code" | tee -a "$stage_dir/status.log"
  return "$code"
}

# Weight sweep around the mild setting. This distinguishes over-regularisation
# from an auxiliary signal that is simply too weak.
run_stage bhpc_005_warmup_balanced \
  --bhpc_weight 0.005 --bhpc_dim 128 --bhpc_temperature 0.2 \
  --bhpc_momentum 0.99 --bhpc_margin 0.05 --bhpc_warmup_steps 11268 \
  --bhpc_class_balance sqrt_inv || exit $?

run_stage bhpc_020_warmup_balanced \
  --bhpc_weight 0.02 --bhpc_dim 128 --bhpc_temperature 0.2 \
  --bhpc_momentum 0.99 --bhpc_margin 0.05 --bhpc_warmup_steps 11268 \
  --bhpc_class_balance sqrt_inv || exit $?

# GENIA is dominated by 1--2 token entities; downweighting only the boundary
# hinge tests whether its local-margin pressure is the source of degradation.
run_stage bhpc_boundary025_warmup_balanced \
  --bhpc_weight 0.01 --bhpc_dim 128 --bhpc_temperature 0.2 \
  --bhpc_momentum 0.99 --bhpc_margin 0.05 --bhpc_warmup_steps 11268 \
  --bhpc_class_balance sqrt_inv --bhpc_prototype_scale 1.0 --bhpc_boundary_scale 0.25 || exit $?
