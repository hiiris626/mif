#!/usr/bin/env bash
set -euo pipefail
VERSION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$VERSION_ROOT/code"
PYTHON_BIN="${PYTHON_BIN:-/home/weiyh/.conda/envs/virchow_env/bin/python}"
exec "$PYTHON_BIN" -B -m vit_matte.sample --ckpt "/data/weiyh/weights/vit_matte/virchow2_vitmatte_v3_256_best.pt" "$@"
