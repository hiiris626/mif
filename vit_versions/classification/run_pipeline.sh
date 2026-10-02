#!/usr/bin/env bash
# Statistics -> mild online augmentation and multilabel training -> held-out test.
set -euo pipefail
echo 'This old automatic pipeline is retired. Review vit_training_project/README.md before execution.' >&2
exit 2
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON:-/home/weiyh/.conda/envs/virchow_env/bin/python}"
DATA="$PROJECT_ROOT/vit_versions/classification/data_prepared"
RESULTS="$PROJECT_ROOT/vit_versions/classification/results"
DEVICE="${DEVICE:-cuda:2}"
mkdir -p "$RESULTS"
exec 9>"$RESULTS/pipeline.lock"
flock -n 9 || { echo 'Another classification pipeline is active' >&2; exit 1; }
state() { printf '{"stage":"%s","updated_at":"%s"}\n' "$1" "$(date -Is)" > "$RESULTS/pipeline_state.json"; }
trap 'state failed' ERR
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "${1:-}" == "--wait-for-statistics" ]]; then
    state waiting_for_statistics
    # Attach to an already running preparation without writing its shard files.
    deadline=$((SECONDS+86400))
    until [[ -f "$DATA/COMPLETE.json" ]]; do
        if (( SECONDS > deadline )); then echo 'Statistics did not finish within 24h' >&2; exit 1; fi
        sleep 20
    done
else
    state statistics
    bash vit_versions/classification/prepare_all.sh "$DATA"
fi
RESUME=()
state data_figures
"$PYTHON" -B -m vit_seg.report --data "$DATA" --results "$RESULTS" --data-only
if [[ -f "$RESULTS/model/last.pt" ]]; then RESUME=(--resume "$RESULTS/model/last.pt"); fi
state training
"$PYTHON" -B -m vit_seg.train --data "$DATA" --out "$RESULTS/model" \
    --config vit_versions/classification/config.json --device "$DEVICE" "${RESUME[@]}"
state test_evaluation
"$PYTHON" -B -m vit_seg.evaluate --checkpoint "$RESULTS/model/best.pt" --data "$DATA" \
    --split test --out "$RESULTS/test_metrics.json" --device "$DEVICE"
state final_figures
"$PYTHON" -B -m vit_seg.report --data "$DATA" --results "$RESULTS" --device "$DEVICE"
state complete
echo 'Training, held-out evaluation and final figures complete'
