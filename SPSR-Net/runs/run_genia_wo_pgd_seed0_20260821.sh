#!/usr/bin/env bash
# GENIA ablation matching the PGD stage-1 setup, with adversarial training off.
set -euo pipefail

project_dir="/root/shared-nvme/projects/SPSR-Net"
python_bin="/root/.conda/envs/difinet/bin/python"
model_dir="/root/.cache/difinet_backbones/models--dmis-lab--biobert-v1.1/snapshots/551ca18efd7f052c8dfa0b01c94c2a8e68bc5488"

cd "$project_dir"
env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  "$python_bin" -u train.py -n 30 --lr 7e-6 --encoder_lr 7e-6 --warmup 0.1 \
  --loss_type bce --early_stop_patience 5 \
  --cnn_dim 400 --biaffine_size 200 --n_head 4 -b 4 -d genia \
  --model_name "$model_dir" --ent_thres 0.48 --logit_drop 0.15 \
  --cnn_depth 1 --n_layer 2 --use_snsa 1 --use_hsr 1 \
  --adv_type none \
  --fp16 --num_workers 4 --accumulation_steps 2 --seed 0 \
  --checkpoint_monitor 'f#f#dev'
