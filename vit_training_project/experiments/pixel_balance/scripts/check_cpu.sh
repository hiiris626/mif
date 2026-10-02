#!/usr/bin/env bash
# Software verification only: tiny synthetic arrays, no GPU, no real training.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-/home/weiyh/.conda/envs/virchow_env/bin/python}"
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
"$PYTHON" -B -m unittest discover -s tests -p 'test_*.py'
"$PYTHON" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${TEST_PORT:-29743}" tests/ddp_cpu_check.py
"$PYTHON" -B -m torch.distributed.run --nnodes=1 --nproc_per_node=4 \
    --master_addr=127.0.0.1 --master_port="${TEST_PORT:-29743}" tests/production_cpu_check.py
