#!/usr/bin/env bash
# From-scratch Pangaea finetune: 16 consecutive LMDB minutes, stride 2, +15 min
# target. Does not pass --finetune (random init, no UOH pretrain weights).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${ROOT}/my_env/bin/torchrun"
DATA="${DATA:-/home/ubuntu/PVOutputPrediction/preprocessing/data/pangaea_finetune}"
LMDB="${LMDB:-/home/ubuntu/PVOutputPrediction/preprocessing/data/pangaea_lmdb_224_png/frames.lmdb}"
OUT="${OUT:-${ROOT}/checkpoints_pangaea_scratch}"
NPROC="${NPROC:-1}"

mkdir -p "$OUT"
export OMP_NUM_THREADS=1
export MASTER_PORT="${MASTER_PORT:-$((12000 + RANDOM % 20000))}"

exec "$PY" --standalone --nproc_per_node="$NPROC" \
  run_class_finetuning.py \
  --model vit_base_patch16_224 \
  --data_set PVOutputPrediction \
  --data_path "$DATA" \
  --lmdb_path "$LMDB" \
  --num_frames 16 \
  --sampling_rate 1 \
  --batch_size "${BATCH_SIZE:-48}" \
  --num_workers "${NUM_WORKERS:-8}" \
  --epochs "${EPOCHS:-30}" \
  --lr 5e-4 \
  --warmup_epochs 1 \
  --layer_decay 0.9 \
  --weight_decay 0.1 \
  --mixup 0 --cutmix 0 \
  --clip_grad 1.0 \
  --use_residual \
  --dist_eval \
  --save_ckpt --save_ckpt_freq 3 \
  --output_dir "$OUT" \
  --log_dir "$OUT" \
  --device cuda \
  "$@"
