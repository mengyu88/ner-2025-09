#!/usr/bin/env bash
# GENIA: HSR-oriented, training-only span quality supervision.
set -euo pipefail

project_dir="/root/shared-nvme/projects/SPSR-Net"
python_bin="/root/.conda/envs/difinet/bin/python"
model_dir="$project_dir/pretrained_models/biomedbert-base-uncased-abstract-fulltext"

cd "$project_dir"
exec env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u train.py \
  -n 30 --lr 1e-6 --encoder_lr 1e-6 --warmup 0.0 \
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 \
  -d genia --model_name "$model_dir" --ent_thres 0.48 \
  --logit_drop 0.15 --cnn_depth 1 --n_layer 2 \
  --use_snsa 1 --use_hsr 1 --head_type linear \
  --loss_type asl --asl_gamma_pos 0.0 --asl_gamma_neg 1.0 --asl_clip 0.0 \
  --quality_aux_weight 0.1 --quality_min_iou 0.1 \
  --adv_type none --bhpc_weight 0.0 \
  --fp16 --accumulation_steps 2 --num_workers 4 \
  --early_stop_patience 5 --early_stop_monitor 'f#f#dev' \
  --checkpoint_monitor 'f#f#dev'
