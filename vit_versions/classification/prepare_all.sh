#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
OUT="${1:-$PROJECT_ROOT/vit_versions/classification/data_prepared}"
PYTHON="${PYTHON:-/home/weiyh/.conda/envs/virchow_env/bin/python}"
SHARDS="${SHARDS:-8}"
[[ "$SHARDS" =~ ^[1-9][0-9]*$ ]] || exit 2
mkdir -p "$OUT/logs"
PIDS=()
stop_children() { for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap 'stop_children; exit 130' INT TERM
for ((i=0; i<SHARDS; i++)); do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" -B -m vit_seg.prepare inventory \
        --out "$OUT" --shards "$SHARDS" --shard "$i" \
        --batch-size 32 --io-workers 2 \
        --split-policy patient_70_15_15 --seed 42 > "$OUT/logs/shard_${i}.log" 2>&1 &
    PIDS+=("$!")
done
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then FAILED=1; fi
done
if [[ "$FAILED" != 0 ]]; then
    echo "Statistics incomplete; inspect $OUT/logs; rerun safely to resume" >&2
    exit 1
fi
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON" -B -m vit_seg.prepare finalize --out "$OUT" --shards "$SHARDS"
