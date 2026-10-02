"""七模型统一重评流水线（2026-09-20，可续跑）。

流程：
  1) 预测清单校验：每个模型必须恰好 10,952 个测试 TIFF、文件大小完整、命名与
     test_dataframe.csv 完全一致（防漏/防重/防截断）；校验结果写 inventory.json；
  2) 三类评估任务（--tasks 可分别开关）：
     - linear：eval.metrics_gpu（线性域 PSNR/SSIM/Pearson，可选 RGB 复合 FID；
       --only-psnr 仅修正 PSNR 而保留其余已有指标）；
     - log   ：同脚本加 --domain log（论文 log 口径）；
     - cell  ：eval.cell_classify（细胞级 AUC/F1，共享特征缓存 + 分类器缓存）；
  3) 并发策略：GPU 任务先跑（并发 = GPU 数，模型轮流分卡），cell 串行（CPU 密集）；
  4) 汇总图：调用 scripts.plot_all_results 生成总表/逐 marker/细胞/耗时图；
  5) 可续跑：输出文件已存在则跳过（--force 强制重跑）；任务状态写 task_status.json。

典型用法：
    python -m scripts.run_full_benchmark \
        --model vitmatte_512=/data/weiyh/results/vit_matte/virchow2_vitmatte_v2_512_epoch15 \
        --model vitmatte_256=/data/weiyh/results/vit_matte/virchow2_vitmatte_v3_256_best \
        --out /data1/weiyh/results/full_benchmark_20260920
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
import time
import threading
from contextlib import nullcontext
_GPU_LOCKS = {}
from pathlib import Path


ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"
EXPECTED_TEST_TILES = 10952


def parse_models(values):
    models = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"model must be NAME=DIR, got {value}")
        name, path = value.split("=", 1)
        if not name or name in models:
            raise ValueError(f"invalid/duplicate model name: {name}")
        models[name] = os.path.abspath(path)
    return models


def prediction_inventory(path):
    """统计一个预测目录：文件数、截断文件（< 16×256×256 字节）、basename 集合。"""
    files = [p for p in Path(path).iterdir() if p.suffix.lower() in (".tif", ".tiff")]
    invalid = [str(p) for p in files if p.stat().st_size < 16 * 256 * 256]
    return len(files), invalid, {p.stem for p in files}


def run_task(task):
    """执行单个评估任务：输出已存在且非 force 则跳过；日志重定向到 log_path。"""
    name, kind, cmd, log_path, out_path, force = task
    if os.path.exists(out_path) and not force:
        return dict(model=name, task=kind, status="skipped", output=out_path, seconds=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    started = time.time()
    with open(log_path, "w") as log:
        gpu = cmd[cmd.index("--gpu")+1] if "--gpu" in cmd else None
        lock = _GPU_LOCKS.setdefault(gpu, threading.Lock()) if gpu is not None else nullcontext()
        with lock:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return dict(model=name, task=kind, status="ok" if result.returncode == 0 else "failed",
                output=out_path, log=log_path, returncode=result.returncode,
                seconds=round(time.time() - started, 2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", action="append", default=[], help="NAME=PREDICTION_DIR")
    parser.add_argument("--out", default="/data1/weiyh/results/full_benchmark_20260920")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip_cell", action="store_true")
    parser.add_argument("--tasks", nargs="+", choices=("linear", "log", "cell"),
                        default=["linear", "log", "cell"])
    parser.add_argument("--fid", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute RGB-composite FID with linear metrics (default: true)")
    parser.add_argument("--only-psnr", action="store_true",
                        help="仅修正已有线性结果的 PSNR，保留其余指标")
    args = parser.parse_args()
    models = parse_models(args.model)
    if not models:
        parser.error("at least one --model NAME=DIR is required")
    gpus = [x.strip() for x in args.gpus.split(",") if x.strip()]
    if not gpus:
        parser.error("--gpus cannot be empty")
    out = os.path.abspath(args.out)
    logs = os.path.join(out, "logs")
    os.makedirs(logs, exist_ok=True)

    # ---- 预测清单校验（防漏/防重/防截断；不完整直接报错终止）----
    with open(os.path.join(ROOT, "test_dataframe.csv"), newline="") as stream:
        rows = list(csv.DictReader(stream))
    he_col = "image_path" if "image_path" in rows[0] else "he_path"
    expected_stems = {Path(row[he_col]).stem for row in rows}
    inventory = {}
    for name, path in models.items():
        if not os.path.isdir(path):
            raise FileNotFoundError(path)
        count, invalid, stems = prediction_inventory(path)
        missing = sorted(expected_stems - stems)
        extra = sorted(stems - expected_stems)
        inventory[name] = dict(path=path, n_predictions=count,
                               invalid_files=invalid[:20],
                               missing_stems=missing[:20], extra_stems=extra[:20],
                               complete=count == EXPECTED_TEST_TILES and not invalid
                               and not missing and not extra)
        if count != EXPECTED_TEST_TILES:
            raise ValueError(f"{name}: expected {EXPECTED_TEST_TILES} TIFFs, found {count}: {path}")
        if invalid:
            raise ValueError(f"{name}: found {len(invalid)} truncated prediction TIFFs: {invalid[:3]}")
        if missing or extra:
            raise ValueError(f"{name}: prediction names mismatch split: "
                             f"missing={missing[:3]} extra={extra[:3]}")
    with open(os.path.join(out, "inventory.json"), "w") as stream:
        json.dump(inventory, stream, indent=2, ensure_ascii=False)

    python = sys.executable
    # ---- 构造三类任务命令（按模型轮流分配 GPU：i % len(gpus)）----
    task_groups = {"linear": [], "log": [], "cell": []}
    for i, (name, path) in enumerate(models.items()):
        model_out = os.path.join(out, "metrics", name)
        linear = os.path.join(model_out, "metrics_linear.json")
        log = os.path.join(model_out, "metrics_log.json")
        cell = os.path.join(model_out, "cell_metrics.json")
        linear_cmd = [python, "-m", "eval.metrics_gpu", "--pred_dir", path,
                      "--data_root", ROOT, "--split", "test", "--gpu", gpus[i % len(gpus)],
                      "--out", linear]
        if args.fid:
            linear_cmd.append("--fid")
        if args.only_psnr:
            linear_cmd.append("--only_psnr")
        if "linear" in args.tasks:
            task_groups["linear"].append((name, "linear", linear_cmd,
                                          os.path.join(logs, f"{name}_linear.log"), linear, args.force))
        if "log" in args.tasks:
            task_groups["log"].append((name, "log", [python, "-m", "eval.metrics_gpu", "--pred_dir", path,
                          "--data_root", ROOT, "--split", "test", "--domain", "log",
                          "--gpu", gpus[i % len(gpus)], "--out", log], os.path.join(logs, f"{name}_log.log"),
                          log, args.force))
        if "cell" in args.tasks and not args.skip_cell:
            task_groups["cell"].append((name, "cell", [python, "-m", "eval.cell_classify", "--pred_dir", path,
                          "--data_root", ROOT, "--train_split", "val", "--test_split", "test",
                          "--feature_cache_dir", os.path.join(out, "cell_feature_cache"),
                          "--out", cell], os.path.join(logs, f"{name}_cell.log"), cell, args.force))

    results = []
    # 先跑 GPU 指标、后跑 CPU 指标：避免同一模型的多任务同时争抢磁盘带宽。
    # GPU 任务并发 = GPU 数；cell 任务限 1 个（CPU 密集）。
    for kind in ("linear", "log", "cell"):
        group = task_groups[kind]
        if not group:
            continue
        group_workers = min(args.workers, len(gpus)) if kind in ("linear", "log") else min(2, args.workers)
        if kind == "cell":
            group_workers = 1
        print(f"[stage] {kind}: {len(group)} task(s), workers={group_workers}", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, group_workers)) as pool:
            futures = {pool.submit(run_task, task): task[:2] for task in group}
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                print(f"[{result['status']}] {result['model']} {result['task']} "
                      f"{result.get('seconds', 0):.1f}s", flush=True)
                with open(os.path.join(out, "task_status.json"), "w") as stream:
                    json.dump(results, stream, indent=2, ensure_ascii=False)
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        raise SystemExit(f"{len(failed)} benchmark task(s) failed; inspect {logs}")
    subprocess.run([python, "-m", "scripts.plot_all_results", "--out", os.path.join(out, "figures"),
                    "--metric_roots", os.path.join(out, "metrics"), "/data1/weiyh/results/unetpp_s1",
                    "--log_roots", "/data/weiyh/logs", "/data1/weiyh/logs",
                    os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")], check=True)
    print(f"[done] {len(results)} tasks -> {out}")


if __name__ == "__main__":
    main()
