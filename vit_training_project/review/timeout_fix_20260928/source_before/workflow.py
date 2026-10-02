"""Explicit review gate and ordered preprocessing -> DDP -> held-out reporting."""
import argparse
from datetime import datetime
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from .artifacts import file_hash

PROJECT = Path(__file__).resolve().parents[1]


def validate_config(cfg):
    if cfg.get('use_data_cache', False) and (not isinstance(cfg.get('cache_workers'), int) or cfg['cache_workers'] < 1):
        raise ValueError('cache_workers must be a positive integer')
    for key in ("lora_lr", "decoder_lr", "min_delta", "grad_clip", "memory_target_fraction", "memory_reserve_gib",
                "mse_weight", "overlap_weight", "weight_decay", "eps", "min_lr_ratio"):
        if not math.isfinite(cfg[key]): raise ValueError(f"Nonfinite config value: {key}")
    if cfg["task"] != "pixel_multilabel" or cfg["label_policy"] != "multilabel":
        raise ValueError("Expected multilabel expression classification")
    if cfg["world_size"] != 4 or cfg["precision"] != "bf16" or cfg["optimizer"] != "AdamW":
        raise ValueError("This project configures four GPUs, BF16 and AdamW")
    if not 0 <= cfg["warmup_epochs"] < cfg["epochs"] or not 0 <= cfg["early_stop_warmup"] < cfg["epochs"]:
        raise ValueError("Warmup must be shorter than training")
    if cfg["patience"] < 1 or cfg["min_delta"] < 0 or cfg["grad_clip"] <= 0:
        raise ValueError("Invalid early stopping or gradient clipping")
    if not 0 < cfg["memory_target_fraction"] < 1 or cfg["memory_reserve_gib"] <= 0:
        raise ValueError("Invalid GPU capacity budget")
    if min(cfg["lora_lr"], cfg["decoder_lr"], cfg["target_global_batch"]) <= 0:
        raise ValueError("Learning rates and batch target must be positive")
    if cfg["overlap_loss"] not in ("dice", "lovasz_hinge") or min(cfg["mse_weight"], cfg["overlap_weight"]) < 0 or cfg["mse_weight"]+cfg["overlap_weight"] <= 0:
        raise ValueError("Invalid multilabel loss")
    if cfg["split_policy"] != "patient_70_15_15" or not cfg["balanced_training"]:
        raise ValueError("Expected patient 70/15/15 split and channel-balanced training")
    if cfg["batch_size"] != "auto" or cfg["grad_accum"] != "auto":
        raise ValueError("Submit auto batch/accumulation; capacity search writes the resolved config")
    if not isinstance(cfg["snapshot_every"], int) or cfg["snapshot_every"] < 0:
        raise ValueError("snapshot_every must be a non-negative integer number of epochs")
    if cfg["batch_min"] < 2 or cfg["batch_max"] < cfg["batch_min"] or cfg["batch_min"] % 2 or cfg["batch_max"] % 2:
        raise ValueError("Invalid even-batch search bounds")
    if cfg["vit_size"] % 14 or cfg["tile_size"] < 16 or cfg["tile_size"] % 16:
        raise ValueError("ViT size must be a multiple of 14; decoder size a multiple of 16")
    if cfg["probability_threshold"] != .5:
        raise ValueError("Training model selection uses fixed 0.5; post-training thresholds use val only")
    selection = cfg["threshold_selection"]
    if not 0 < selection["min"] <= .5 <= selection["max"] < 1 or min(selection["min_positive_pixels"], selection["min_negative_pixels"], selection["min_patients_per_label"]) < 1:
        raise ValueError("Invalid validation threshold selection settings")


def gpu_preflight(cfg):
    import torch
    if torch.cuda.device_count() != 4:
        raise ValueError("Exactly four visible GPUs are required")
    for i in range(4):
        with torch.cuda.device(i):
            if not torch.cuda.is_bf16_supported():
                raise ValueError(f"GPU {i} lacks BF16 support")
            free, total = torch.cuda.mem_get_info()
            if total-free > cfg["max_external_memory_gib"]*2**30:
                raise RuntimeError(f"GPU {i} is occupied. Wait for other tasks to finish; no processes will be killed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--run-dir", default="runs/ddp_baseline")
    parser.add_argument("--approved", action="store_true", help="Use only after project review is approved")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stage", choices=("all", "prepare", "cache", "capacity", "train", "calibrate", "test", "report"), default="all")
    args = parser.parse_args()
    def interrupted(signum, frame):
        raise InterruptedError(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    os.chdir(PROJECT)
    cfg_path = Path(args.config).resolve(); cfg = json.loads(cfg_path.read_text())
    validate_config(cfg)
    if not args.approved:
        print("REVIEW ONLY: no data processing, GPU calibration or training executed.")
        print(json.dumps(cfg, ensure_ascii=False, indent=2))
        return
    run = Path(args.run_dir).resolve()
    if run == PROJECT or PROJECT.is_relative_to(run):
        raise ValueError("Run directory must not contain the project")
    if run.exists() and any(run.iterdir()) and not args.resume:
        raise ValueError("Run directory exists; choose a new name or use --resume after review")
    run.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (run/"run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        copied_cfg = run/"submitted_config.json"
        if copied_cfg.exists() and json.loads(copied_cfg.read_text()) != cfg:
            raise ValueError("Submitted config changed; start a new run")
        copied_cfg.write_text(json.dumps(cfg, indent=2))
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        os.environ["MPLCONFIGDIR"] = str(run/".mpl_cache")
        if not Path(cfg["data_root"]).is_dir() or not Path(cfg["weights_path"]).is_file():
            raise FileNotFoundError("Set existing data_root and Virchow2 weights_path in the reviewed config")
        code_hashes = {str(p.relative_to(PROJECT)):file_hash(p) for folder in ("vit_seg", "vit_matte", "datacore")
                       for p in (PROJECT/folder).glob("*.py")}
        architecture = PROJECT/"configs/virchow2_config.json"
        if not architecture.is_file(): raise FileNotFoundError(architecture)
        code_hashes["configs/virchow2_config.json"] = file_hash(architecture)
        provenance_path = run/"provenance.json"
        provenance = dict(config_sha256=file_hash(cfg_path), code_sha256=code_hashes,
                          weights_sha256=file_hash(cfg["weights_path"]))
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise ValueError("Code/config/pretrained weights changed; use a new run")
        provenance_path.write_text(json.dumps(provenance, indent=2))
        packages = {name:importlib.metadata.version(name) for name in
                    ("torch", "torchvision", "timm", "numpy", "pandas", "tifffile", "safetensors", "opencv-python", "imagecodecs", "matplotlib", "Pillow")}
        environment_path = run/"environment.json"
        if environment_path.exists() and json.loads(environment_path.read_text()) != packages:
            raise ValueError("Runtime package versions changed; use the recorded environment or a new run")
        environment_path.write_text(json.dumps(packages, indent=2))
        (run/"approval.json").write_text(json.dumps(dict(explicit_flag=True, time=datetime.now().astimezone().isoformat()), indent=2))
        data, results = run/"data", run/"results"
        results.mkdir(exist_ok=True)
        def state(stage):
            (run/"status.json").write_text(json.dumps(dict(stage=stage, time=datetime.now().astimezone().isoformat()), indent=2))
        def command(argv, logfile):
            with (run/logfile).open("a") as stream:
                process = subprocess.Popen([sys.executable, "-B", *argv], stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = process.wait()
                    if code:
                        raise subprocess.CalledProcessError(code, argv)
                except BaseException:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                        try: process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL); process.wait()
                    raise
        chosen = ("prepare", "cache", "capacity", "train", "calibrate", "test", "report") if args.stage == "all" else (args.stage,)
        try:
            for stage in chosen:
                state(stage)
                if stage == "prepare":
                    if not (data/"COMPLETE.json").exists():
                        from .prepare import source_table
                        data.mkdir(exist_ok=True)
                        source_table(cfg["data_root"], cfg["split_policy"], cfg["split_seed"]).to_csv(data/"source_manifest.csv", index=False)
                        jobs, logs = [], []
                        try:
                            for shard in range(8):
                                log = (run/f"inventory_{shard}.log").open("a"); logs.append(log)
                                jobs.append(subprocess.Popen([sys.executable, "-B", "-m", "vit_seg.prepare", "inventory",
                                    "--root", cfg["data_root"], "--out", str(data), "--shards", "8", "--shard", str(shard),
                                    "--seed", str(cfg["split_seed"])], stdout=log, stderr=subprocess.STDOUT))
                            if any(job.wait() for job in jobs):
                                raise RuntimeError("Inventory failed; inspect inventory logs")
                        finally:
                            for job in jobs:
                                if job.poll() is None: job.terminate()
                            for log in logs: log.close()
                        command(["-m", "vit_seg.prepare", "finalize", "--out", str(data), "--shards", "8", "--seed", str(cfg["split_seed"])], "prepare.log")
                    command(["-m", "vit_seg.validate", "--data", str(data)], "validation.log")
                    command(["-m", "vit_seg.report", "--data", str(data), "--results", str(results), "--data-only"], "data_figures.log")
                elif stage == "cache":
                    if cfg.get('use_data_cache', False):
                        command(['-m','vit_seg.cache','--data',str(data),'--out',str(run/'cache'),
                                 '--workers',str(cfg['cache_workers'])], 'cache.log')
                elif stage == "capacity":
                    if not (run/"capacity/resolved_config.json").exists():
                        gpu_preflight(cfg)
                        command(["-m", "vit_seg.capacity", "search", "--config", str(copied_cfg), "--out", str(run/"capacity")], "capacity.log")
                elif stage in ("train", "calibrate", "test"):
                    gpu_preflight(cfg)
                    resolved = run/"capacity/resolved_config.json"
                    if not resolved.exists(): raise FileNotFoundError("Complete capacity calibration first")
                    options = ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4", "-m", "vit_seg.train_ddp",
                               "--config", str(resolved), "--data", str(data)]
                    if stage == "train":
                        options += ["--out", str(results/"model")]
                        if args.resume and (results/"model/last.pt").exists():
                            options += ["--resume", str(results/"model/last.pt")]
                    elif stage == "calibrate":
                        options += ["--out", str(results), "--evaluate-only", "val", "--checkpoint", str(results/"model/best.pt"), "--calibrate"]
                    else:
                        if not (results/"thresholds.json").exists():
                            raise FileNotFoundError("Select validation thresholds before test evaluation")
                        options += ["--out", str(results), "--evaluate-only", "test", "--checkpoint", str(results/"model/best.pt"),
                                    "--thresholds", str(results/"thresholds.json")]
                    command(options, f"{stage}.log")
                else:
                    command(["-m", "vit_seg.report", "--data", str(data), "--results", str(results), "--device", "cuda:0"], "report.log")
            state("complete" if args.stage == "all" else f"{args.stage}_complete")
        except BaseException:
            state("failed_or_interrupted")
            raise


if __name__ == "__main__":
    main()
