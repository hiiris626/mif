"""Approved-run only: search one common microbatch on all four physical GPUs."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .distributed import DistributedPixelLoss, optimizer_for, resolve_batch
from .model import build_model


def probe_worker(config, output, batch):
    cfg = json.loads(Path(config).read_text())
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=3))
    result = dict(rank=rank, batch_size=batch, gpu_name=torch.cuda.get_device_name(device),
                  visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version())
    try:
        free, total = torch.cuda.mem_get_info(device)
        budget = min(total*cfg["memory_target_fraction"], free-cfg["memory_reserve_gib"]*2**30)
        torch.cuda.reset_peak_memory_stats(device)
        model = build_model(cfg, device)
        if cfg["sync_batchnorm"]:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True)
        optimizer = optimizer_for(model.module, cfg)
        from .bce_balance import load_pixel_weights
        wp,wn=load_pixel_weights(cfg)
        loss_fn = DistributedPixelLoss([1.]*16, cfg["overlap_loss"], cfg["bce_weight"], cfg["overlap_weight"], cfg.get('dice_scope','all_valid'),wp,wn).to(device)
        image = torch.randn(batch, 3, cfg["tile_size"], cfg["tile_size"], device=device)
        label = torch.randint(0, 2, (batch, 16, cfg["tile_size"], cfg["tile_size"]), device=device)
        for _ in range(cfg["probe_steps"]):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(image), label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
            optimizer.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_reserved(device)
        result.update(status="ok", fits=peak <= budget, peak_reserved_bytes=peak,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      budget_bytes=budget, total_bytes=total, loss=float(loss))
    except torch.cuda.OutOfMemoryError:
        result.update(status="oom", fits=False)
        raise
    finally:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/f"rank_{rank}.json").write_text(json.dumps(result, indent=2))
        if result.get("status") == "ok":
            dist.destroy_process_group()


def calibrate(config, output):
    cfg = json.loads(Path(config).read_text())
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    def trial(batch):
        folder = output/f"batch_{batch}"
        attempt = 1
        while folder.exists():
            attempt += 1
            folder = output/f"batch_{batch}_attempt_{attempt}"
        folder.mkdir()
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=4",
                   "-m", "vit_seg.capacity", "worker", "--config", str(config), "--out", str(folder), "--batch", str(batch)]
        with (folder/"probe.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=cfg["probe_timeout_seconds"])
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise RuntimeError("Capacity probe timed out; inspect probe.log")
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise
        rows = [json.loads(p.read_text()) for p in folder.glob("rank_*.json")]
        if code == 0 and sorted(row.get("rank", -1) for row in rows) != [0,1,2,3]:
            raise RuntimeError(f"Incomplete four-rank capacity report: {folder}")
        if code and not any(row.get("status") == "oom" for row in rows):
            raise RuntimeError(f"Probe failed for a non-OOM reason: {folder/'probe.log'}")
        fits = code == 0 and len(rows) == 4 and all(row.get("fits") for row in rows)
        print(json.dumps(dict(batch=batch, fits=fits)), flush=True)
        return fits
    smallest, largest = cfg["batch_min"], cfg["batch_max"]
    if smallest < 2 or smallest % 2 or largest % 2:
        raise ValueError("Capacity bounds must be positive even batches >=2")
    if not trial(smallest):
        raise RuntimeError("Even the minimum batch does not fit; inspect GPU occupancy")
    low, candidate, upper = smallest, smallest*2, largest+2
    while candidate <= largest:
        if trial(candidate):
            low = candidate; candidate *= 2
        else:
            upper = candidate; break
    while upper-low > 2:
        middle = ((low+upper)//4)*2
        if trial(middle): low = middle
        else: upper = middle
    cfg.update(resolve_batch(cfg, low))
    cfg["batch_resolution"] = "largest tested common even batch within configured memory budget"
    (output/"resolved_config.json").write_text(json.dumps(cfg, indent=2))
    return output/"resolved_config.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("worker", "search"))
    p.add_argument("--config", required=True); p.add_argument("--out", required=True)
    p.add_argument("--batch", type=int)
    args = p.parse_args()
    if args.mode == "search":
        def interrupted(signum, frame):
            raise InterruptedError(f"Capacity search interrupted: {signum}")
        signal.signal(signal.SIGTERM, interrupted)
    if args.mode == "worker": probe_worker(args.config, args.out, args.batch)
    else: calibrate(args.config, args.out)


if __name__ == "__main__":
    main()
