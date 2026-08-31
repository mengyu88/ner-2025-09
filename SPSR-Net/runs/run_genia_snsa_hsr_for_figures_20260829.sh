#!/usr/bin/env bash
# Reproduce the 2026-08-20 GENIA full-model setting, changing exactly one
# module switch per run.  Outputs are kept separately for figure provenance.
set -euo pipefail

project_dir="/root/shared-nvme/projects/SPSR-Net"
python_bin="/root/.conda/envs/difinet/bin/python"
model_dir="/root/.cache/difinet_backbones/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"
run_root="$project_dir/runs/genia_snsa_hsr_for_figures_20260829"

mkdir -p "$run_root"

common=(
  -u train.py -n 30 -d genia
  --model_name "$model_dir"
  --lr 7e-6 --encoder_lr 7e-6 --warmup 0.1
  --cnn_dim 400 --biaffine_size 200 --n_head 4
  -b 4 --accumulation_steps 2
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2
  --ent_thres 0.48 --seed 0
  --loss_type bce
  --adv_type pgd --adv_epsilon 1.0 --adv_alpha 0.3 --adv_k 3
  --adv_emb_name word_embeddings --adv_loss_weight 0.8 --adv_warmup_ratio 0.1
  --adv_every_n_steps 1 --adv_random_start
  --fp16 --num_workers 4
  --early_stop_patience 5 --early_stop_monitor 'f#f#dev'
  --checkpoint_monitor 'f#f#dev'
  --sad_topk 2 --sad_use_rel_bias 1 --sad_gate 1
)

run_variant() {
  local name="$1"
  shift
  local output_dir="$run_root/$name"
  mkdir -p "$output_dir"
  date -Is > "$output_dir/start_time.txt"
  printf '%q ' env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    "$python_bin" "${common[@]}" "$@" > "$output_dir/command.txt"
  printf '\n' >> "$output_dir/command.txt"
  (
    cd "$project_dir"
    env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      "$python_bin" "${common[@]}" "$@"
  ) > "$output_dir/train.log" 2>&1
  date -Is > "$output_dir/end_time.txt"
}

run_variant w_o_snsa --use_snsa 0 --use_hsr 1
run_variant w_o_hsr --use_snsa 1 --use_hsr 0
