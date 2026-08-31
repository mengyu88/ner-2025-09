#!/usr/bin/env bash
# Fair reproduction of the historical two-stage PGD recipe.
# Both stages select checkpoints by development F1 only; test is reported but
# never used for selection.
set -euo pipefail

project_dir="/root/shared-nvme/projects/SPSR-Net"
python_bin="/root/.conda/envs/difinet/bin/python"
model_dir="/root/.cache/difinet_backbones/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"
stage1_root="$project_dir/_saved_models"
marker="$project_dir/runs/genia_pgd_replication_seed0_20260820/stage1_model.txt"

common=(
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 -d genia
  --model_name "$model_dir" --ent_thres 0.48 --logit_drop 0.15
  --cnn_depth 1 --n_layer 2 --use_snsa 1 --use_hsr 1
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.1
  --adv_every_n_steps 1 --adv_random_start --fp16 --num_workers 4
  --accumulation_steps 2 --seed 0 --checkpoint_monitor 'f#f#dev'
)

cd "$project_dir"
mkdir -p "$(dirname "$marker")"
before_file="$(mktemp)"
find "$stage1_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort > "$before_file"

# Stage 1 matches the historical PGD base-training regime.
env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u train.py -n 30 --lr 7e-6 --encoder_lr 7e-6 --warmup 0.1 \
  --loss_type bce --early_stop_patience 5 "${common[@]}"

stage1_run="$(find "$stage1_root" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort | comm -13 "$before_file" - | tail -1)"
if [ -z "$stage1_run" ]; then
  echo 'Could not locate stage-1 checkpoint directory.' >&2
  exit 1
fi
stage1_model="$(find "$stage1_root/$stage1_run" -type f -name fastnlp_model.pkl.tar | sort | tail -1)"
if [ -z "$stage1_model" ]; then
  echo 'Stage-1 produced no checkpoint.' >&2
  exit 1
fi
printf '%s\n' "$stage1_model" | tee "$marker"

# Stage 2 matches the historical mild ASL PGD refinement, but uses dev F1.
env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u train.py -n 8 --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0 \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --init_from_checkpoint "$stage1_model" --early_stop_patience 3 "${common[@]}"
