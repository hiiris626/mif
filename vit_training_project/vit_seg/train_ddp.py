"""Four-GPU BF16 training; global loss/metrics, synchronized stopping and resume."""
import argparse
from contextlib import nullcontext
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .data import CHANNELS, IGNORE, IMAGENET_STATS, PixelDataset, read_mif
from .distributed import DistributedPixelLoss, EvaluationLossAccumulator, EvaluationSampler, optimizer_for, lr_factor, reduce_metrics
from .metrics import MultilabelMetrics, DeviceMultilabelMetrics
from .model import build_model, audit_lora
from datacore.training_monitor import EarlyStopper, save_training_artifacts
from .artifacts import file_hash, data_signature
from .thresholds import select_thresholds, load_thresholds
from .display import MARKER_COLORS, PALETTE, argmax_display


def atomic_save(value, path):
    tmp = path.with_suffix(".tmp.pt")
    torch.save(value, tmp)
    tmp.replace(path)


# Display palette for the per-epoch virtual mIF preview, in datacore channel order.

LOG_FIELDS = ("epoch", "step", "train_loss", "val_loss", "val_macro_iou", "val_macro_f1",
              "lr", "train_seconds", "val_seconds", "epoch_seconds", "elapsed_seconds", "snapshot",
              "train_mse_loss", "train_dice_loss", "val_mse_loss", "val_dice_loss")


def mif_composite(stack, colors=MARKER_COLORS):
    """(C,H,W) intensities in [0,1] -> additive RGB composite, same rule for GT and prediction."""
    image = np.zeros((*stack.shape[1:], 3), np.float32)
    for channel in range(stack.shape[0]):
        image += np.clip(stack[channel], 0, 1)[..., None]*colors[channel]
    return np.clip(image, 0, 1)


def display_stretch(stack, percentile=99.5):
    """Per-channel percentile stretch; raw mIF is too dim to inspect unscaled."""
    scaled = np.zeros(stack.shape, np.float32)
    for channel in range(stack.shape[0]):
        high = float(np.percentile(stack[channel], percentile))
        if high > 0:
            scaled[channel] = np.clip(stack[channel]/high, 0, 1)
    return scaled


def save_mif_snapshot(out, epoch, patch_id, he, gt_mif, gt_label, probabilities, q=None):
    """Write one fixed validation patch as H&E | GT mIF | GT labels | virtual mIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    folder = Path(out)/"snapshots"
    folder.mkdir(parents=True, exist_ok=True)
    target = np.where(gt_label == IGNORE, 0, gt_label).astype(np.float32)
    panels = (("H&E", he), ("GT mIF (16-channel TIFF)", mif_composite(display_stretch(gt_mif))),
              ("GT expression labels", mif_composite(target)),
              ("virtual mIF (predicted probability)", mif_composite(probabilities)))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.3*len(panels), 4.6))
    for axis, (title, image) in zip(axes, panels):
        axis.imshow(image); axis.set_title(title); axis.axis("off")
    fig.suptitle(f"epoch {epoch:03d} · patch {patch_id} · fixed validation patch")
    fig.tight_layout()
    figure = folder/f"epoch_{epoch:03d}_patch_{patch_id}.png"
    fig.savefig(figure, dpi=170, bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(4, 4, figsize=(16, 9))
    for channel, axis in enumerate(axes.flat):
        axis.imshow(np.hstack((target[channel], probabilities[channel])), cmap="magma", vmin=0, vmax=1)
        axis.set_title(f"{CHANNELS[channel]}: GT | probability"); axis.axis("off")
    fig.tight_layout(); fig.savefig(folder/f"epoch_{epoch:03d}_patch_{patch_id}_channels.png", dpi=170); plt.close(fig)
    np.savez_compressed(folder/f"epoch_{epoch:03d}_patch_{patch_id}.npz",
                        probabilities=probabilities, labels=gt_label, channels=np.asarray(CHANNELS),
                        patch_id=patch_id, epoch=epoch)
    import cv2
    from matplotlib.patches import Patch
    shape=probabilities.shape[1:][::-1]
    raw=np.stack([cv2.resize(a.astype(np.float32),shape,interpolation=cv2.INTER_NEAREST) for a in gt_mif])
    scale=np.ones(16) if q is None else np.asarray(q)
    tissue=probabilities.max(0)>0
    gt_id,gt_rgb=argmax_display(raw/scale[:,None,None],tissue,(gt_label!=IGNORE).any((1,2)))
    pred_id,pred_rgb=argmax_display(probabilities,tissue)
    fig,axes=plt.subplots(1,3,figsize=(12,4.8))
    for ax,title,values in zip(axes,('H&E','GT dominant intensity / train q','Prediction probability argmax'),(he,gt_rgb,pred_rgb)):
        ax.imshow(values);ax.set_title(title);ax.axis('off')
    fig.legend(handles=[Patch(color=MARKER_COLORS[c],label=name) for c,name in enumerate(CHANNELS)],loc='lower center',ncol=8,fontsize=8)
    fig.suptitle(f'Epoch {epoch} / patch {patch_id}; display only, coexpression retained in metrics')
    fig.tight_layout(rect=(0,.11,1,.94))
    fig.savefig(folder/f'epoch_{epoch:03d}_patch_{patch_id}_argmax.png',dpi=160);plt.close(fig)
    np.savez_compressed(folder/f'epoch_{epoch:03d}_patch_{patch_id}_argmax.npz',gt=gt_id,prediction=pred_id,palette=PALETTE)
    return figure.name


def write_epoch_log(out, history):
    """Persistent per-epoch log: train/val loss, metrics and wall-clock time."""
    path = Path(out)/"train_log.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(LOG_FIELDS), extrasaction="ignore")
        writer.writeheader(); writer.writerows(history)
    return path


def initialize_worker(worker_id):
    # OpenCV has its own thread pool; avoid nested pools in every DDP worker.
    import cv2
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def loader_for(dataset, batch, sampler, cfg, training=False):
    options = dict(batch_size=batch, sampler=sampler, num_workers=cfg["num_workers"],
                   pin_memory=True, drop_last=training)
    if cfg["num_workers"]:
        options["prefetch_factor"] = cfg["prefetch_factor"]
        options['persistent_workers'] = True
        options['worker_init_fn'] = initialize_worker
        options['timeout'] = 180
    # Epoch is shared with persistent workers; augmentation still changes each epoch.
    return DataLoader(dataset, **options)


@torch.no_grad()
def evaluate_distributed(model, loader, device, cfg, thresholds=None, patient_indices=None,
                         criterion=None, loss_out=None, observer=None, progress_dir=None):
    # CPU rendezvous/reductions must not occupy a GPU waiting for a slower rank.
    own_group = dist.get_backend() != 'gloo'
    group = dist.new_group(backend='gloo', timeout=timedelta(minutes=10)) if own_group else dist.group.WORLD
    try:
        return _evaluate_distributed(model,loader,device,cfg,thresholds,patient_indices,
                                     criterion,loss_out,observer,progress_dir,group)
    finally:
        if own_group: dist.destroy_process_group(group)


@torch.no_grad()
def _evaluate_distributed(model, loader, device, cfg, thresholds, patient_indices,
                          criterion, loss_out, observer, progress_dir, group):
    # Use the unwrapped module: uneven validation lengths cannot issue DDP
    # buffer-broadcast collectives on each forward.
    model.eval()
    for value in model.buffers():
        dist.broadcast(value, src=0)
    metrics = DeviceMultilabelMetrics(device, threshold=cfg["probability_threshold"])
    selected = DeviceMultilabelMetrics(device, threshold=thresholds) if thresholds is not None else None
    support = np.zeros((2, int(np.max(patient_indices))+1, 16), dtype=np.int64) if patient_indices is not None else None
    loss_accumulator = EvaluationLossAccumulator(criterion, device) if criterion is not None else None
    rank = dist.get_rank()
    maximum = torch.tensor([len(loader)], dtype=torch.int64)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    total_batches = int(maximum.item())
    iterator = iter(loader); began = time.monotonic(); processed = 0
    folder = Path(progress_dir) if progress_dir is not None else None
    if folder is not None: folder.mkdir(parents=True,exist_ok=True)
    def progress(phase, batch):
        if folder is not None:
            path = folder/f'validation_rank_{rank}.json'
            tmp = path.with_suffix('.tmp.json')
            tmp.write_text(json.dumps(dict(rank=rank,phase=phase,batch=batch,
                local_batches=len(loader),total_batches=total_batches,processed_patches=processed,
                elapsed_seconds=round(time.monotonic()-began,3),unix_time=time.time()),indent=2))
            tmp.replace(path)
    for position in range(total_batches):
        # All ranks take the same heartbeat path, even an empty validation rank.
        if position >= len(loader):
            if (position+1)%8 == 0 or position+1 == total_batches:
                progress('waiting_for_ranks',position+1)
                dist.monitored_barrier(group=group,timeout=timedelta(minutes=5),wait_all_ranks=True)
            continue
        progress('loading_batch',position+1)
        batch = next(iterator)
        progress('forward',position+1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(batch["image"].to(device, non_blocking=True))
        labels = batch['label'].to(device,non_blocking=True)
        if loss_accumulator is not None:
            loss_accumulator.update(logits, labels)
        probabilities = logits.float().sigmoid()
        if observer is not None and position == 0:
            observer(batch, probabilities)
        progress('metrics',position+1)
        metrics.update(probabilities, labels)
        if selected is not None:
            selected.update(probabilities, labels)
        if support is not None:
            for index, label in zip(batch["index"].tolist(), batch["label"]):
                patient = patient_indices[index]
                support[0,patient] |= (label == 1).any((1,2)).numpy()
                support[1,patient] |= (label == 0).any((1,2)).numpy()
        processed += len(batch['index'])
        if (position+1)%8 == 0 or position+1 == total_batches:
            if device.type == 'cuda': torch.cuda.synchronize(device)
            progress('waiting_for_ranks',position+1)
            dist.monitored_barrier(group=group,timeout=timedelta(minutes=5),wait_all_ranks=True)
        if position == 0 or (position+1)%20 == 0 or position+1 == total_batches:
            print(json.dumps(dict(validation_rank=rank,batch=position+1,batches=len(loader),
                patches=processed,elapsed_seconds=round(time.monotonic()-began,2))),flush=True)
    progress('exporting_statistics',total_batches)
    metrics = metrics.as_numpy()
    if selected is not None: selected = selected.as_numpy()
    # Synchronize outstanding device work before any rank enters the final sum.
    if device.type == 'cuda': torch.cuda.synchronize(device)
    dist.monitored_barrier(group=group,timeout=timedelta(minutes=5),wait_all_ranks=True)
    progress('reducing_statistics_cpu',total_batches)
    if loss_accumulator is not None:
        value = loss_accumulator.result(group=group)
        if loss_out is not None:
            loss_out["val_loss"] = value
            loss_out["val_mse_loss"] = loss_accumulator.components['mse']
            loss_out["val_dice_loss"] = loss_accumulator.components['overlap']
            loss_out["loss_components"] = loss_accumulator.components
    reduce_metrics(metrics, 'cpu', group=group)
    if support is not None:
        tensor = torch.as_tensor(support)
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX, group=group)
        support = tensor.cpu().numpy().sum(1)
        progress('complete',total_batches)
        return metrics, support
    if selected is not None:
        reduce_metrics(selected, 'cpu', group=group)
        progress('complete',total_batches)
        return selected.result(CHANNELS), metrics.result(CHANNELS)
    progress('complete',total_batches)
    return metrics.result(CHANNELS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True); parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True); parser.add_argument("--resume")
    parser.add_argument("--evaluate-only", choices=("val", "test")); parser.add_argument("--checkpoint")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--thresholds")
    args = parser.parse_args()
    if args.calibrate and (args.evaluate_only != "val" or args.thresholds):
        parser.error("Threshold selection requires --evaluate-only val and no existing thresholds")
    cfg = json.loads(Path(args.config).read_text())
    if int(os.environ.get("WORLD_SIZE", 0)) != cfg["world_size"] or cfg["world_size"] != 4:
        raise ValueError("Launch with torchrun --nproc_per_node=4")
    if not isinstance(cfg["batch_size"], int) or not isinstance(cfg["grad_accum"], int):
        raise ValueError("Run approved capacity calibration first; use resolved_config.json")
    rank, local = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group("nccl", timeout=timedelta(minutes=45))
    try:
        run(args, cfg, rank, device)
    finally:
        dist.destroy_process_group()


def run(args, cfg, rank, device):
    data, out = Path(args.data), Path(args.out)
    complete = json.loads((data/"COMPLETE.json").read_text())
    if not complete["complete_dataset"] or complete["n_source"] != complete["n_retained"]:
        raise ValueError("Expected complete published data with no patch filtering")
    if not (data/"VALIDATED.json").exists():
        raise ValueError("Run data validation before training")
    signature_box = [data_signature(data) if rank == 0 else None]
    dist.broadcast_object_list(signature_box, src=0)
    signature = signature_box[0]
    receipt = json.loads((data/"VALIDATED.json").read_text())
    if receipt["signature"] != signature:
        raise ValueError("Data changed after validation")
    stats = json.loads((data/"statistics.json").read_text())
    cache_dir = data.parent/'cache' if cfg.get('use_data_cache', False) else None
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    torch.backends.cudnn.benchmark = False
    model = build_model(cfg, device)
    if cfg["sync_batchnorm"]:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    optimizer = optimizer_for(model, cfg)
    criterion = DistributedPixelLoss(stats["channel_weights"], cfg["overlap_loss"],
                                      cfg["mse_weight"], cfg["overlap_weight"], cfg.get('dice_scope','all_valid')).to(device)
    stopper = EarlyStopper(cfg["patience"], cfg["min_delta"], "max", cfg["early_stop_warmup"])
    history, best, start, step = [], -math.inf, 0, 0
    checkpoint = args.checkpoint if args.evaluate_only else args.resume
    if checkpoint:
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if ck["config"] != cfg or ck["data_signature"] != signature:
            raise ValueError("Checkpoint config or data provenance mismatch")
        model.load_state_dict(ck["model"])
        if not args.evaluate_only:
            optimizer.load_state_dict(ck["optimizer"]); stopper.load_state_dict(ck["stopper"])
            history, best, start, step = ck["history"], ck["best"], ck["epoch"]+1, ck["step"]
            rng = ck["rank_rng"][rank]
            random.setstate(rng["python"]); np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"]); torch.cuda.set_rng_state(rng["cuda"], device)
    if args.evaluate_only:
        if not checkpoint:
            raise ValueError("Evaluation requires --checkpoint")
        ds = PixelDataset(data/f"{args.evaluate_only}.csv", stats, cfg["data_root"], cfg["tile_size"], cache_dir=cache_dir)
        loader = loader_for(ds, cfg["batch_size"], EvaluationSampler(ds, rank, 4), cfg)
        if args.calibrate:
            import pandas as pd
            patients = pd.read_csv(data/"val.csv", usecols=["orion_slide_id"]).orion_slide_id
            patient_indices = pd.Categorical(patients).codes
            metrics, support = evaluate_distributed(model, loader, device, cfg, patient_indices=patient_indices, progress_dir=out/'validation_progress')
            if rank == 0:
                result = select_thresholds(metrics, support[0], support[1], cfg["threshold_selection"])
                result.update(checkpoint_sha256=file_hash(checkpoint), data_signature=signature)
                out.mkdir(parents=True, exist_ok=True)
                (out/"thresholds.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            return
        thresholds = None
        if args.thresholds:
            box = [load_thresholds(args.thresholds, checkpoint, signature) if rank == 0 else None]
            dist.broadcast_object_list(box, src=0); thresholds = box[0]
        evaluation_loss = {}
        evaluated = evaluate_distributed(model, loader, device, cfg, thresholds=thresholds,
            criterion=criterion if args.evaluate_only == 'val' else None, loss_out=evaluation_loss,
            progress_dir=out/'validation_progress')
        result, baseline = evaluated if thresholds is not None else (evaluated, None)
        if rank == 0:
            result.update(split=args.evaluate_only, n_patches=len(ds))
            if evaluation_loss: result.update(evaluation_loss)
            out.mkdir(parents=True, exist_ok=True)
            (out/f"{args.evaluate_only}_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            if baseline is not None:
                baseline.update(split=args.evaluate_only, n_patches=len(ds))
                (out/f"{args.evaluate_only}_metrics_default05.json").write_text(json.dumps(baseline, indent=2, allow_nan=False))
        return
    if args.resume and (start >= cfg["epochs"] or
                        (stopper.checks > stopper.warmup and stopper.bad_checks >= stopper.patience)):
        if rank == 0:
            print("Training already completed or early-stopped; preserving checkpoint", flush=True)
        return
    ddp = DDP(model, device_ids=[device.index], broadcast_buffers=False,
              find_unused_parameters=False, gradient_as_bucket_view=True)
    tr = PixelDataset(data/"train_balanced.csv", stats, cfg["data_root"], cfg["tile_size"], True, seed=cfg["seed"], cache_dir=cache_dir)
    va = PixelDataset(data/"val.csv", stats, cfg["data_root"], cfg["tile_size"], cache_dir=cache_dir)
    sampler = DistributedSampler(tr, num_replicas=4, rank=rank, shuffle=True, seed=cfg["seed"], drop_last=True)
    loader = loader_for(tr, cfg["batch_size"], sampler, cfg, training=True)
    val_loader = loader_for(va, cfg["batch_size"], EvaluationSampler(va, rank, 4), cfg)
    snapshot_every = int(cfg.get("snapshot_every", 0))
    snapshot_name = {}

    def snapshot_observer(batch, probabilities):
        """Rank 0 only: one deterministic validation patch as a virtual mIF preview."""
        try:
            label = batch["label"]
            positive = ((label == 1) & (label != IGNORE)).sum((1, 2, 3))
            choice = int(torch.argmax(positive))
            row = va.df.iloc[int(batch["index"][choice])]
            image = batch["image"][choice].float()
            image = (image*torch.as_tensor(IMAGENET_STATS["std"]).view(3, 1, 1)
                     + torch.as_tensor(IMAGENET_STATS["mean"]).view(3, 1, 1))
            image = (image.clamp(0, 1)*255).to(torch.uint8).permute(1, 2, 0).numpy()
            image[~batch["tissue"][choice].numpy()] = 0
            probability = probabilities[choice].float().cpu().numpy().copy()
            probability[:, ~batch["tissue"][choice].numpy()] = 0
            (out/"snapshot_patch.json").write_text(json.dumps(dict(
                patch_id=int(row["patch_id"]), image_path=row["image_path"], target_path=row["target_path"],
                split="val", selection="most positive labels in fixed rank-0 first validation batch"), indent=2))
            snapshot_name["value"] = save_mif_snapshot(
                out, epoch+1, int(row["patch_id"]), image,
                read_mif(Path(cfg["data_root"])/row["target_path"]), label[choice].numpy(),
                probability, q=stats['q'])
        except Exception as error:  # a preview must never abort a long training run
            print(f"snapshot skipped: {type(error).__name__}: {error}", flush=True)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out/"lora_audit.json").write_text(json.dumps(audit_lora(model), indent=2))
        (out/"config.json").write_text(json.dumps(cfg, indent=2))
    dist.barrier()
    for epoch in range(start, cfg["epochs"]):
        sampler.set_epoch(epoch); tr.epoch = epoch
        if not len(loader):
            raise ValueError("Too few balanced samples for the selected batch")
        steps_per_epoch = math.ceil(len(loader)/cfg["grad_accum"])
        ddp.train(); total, n = 0., 0
        component_total = torch.zeros(2, device=device, dtype=torch.float64)
        epoch_started = started = time.monotonic()
        batch_finished = started
        data_wait_seconds = model_step_seconds = 0.
        if rank == 0:
            print(f"Epoch [{epoch+1}/{cfg['epochs']}] lr={optimizer.param_groups[0]['lr']:.6g} "
                  f"batches={len(loader)} steps={steps_per_epoch}", flush=True)
        for i, batch in enumerate(loader):
            ready = time.monotonic()
            data_wait_seconds += ready-batch_finished
            ga = cfg["grad_accum"]
            group_size = min(ga, len(loader)-(i//ga)*ga)
            if i % ga == 0:
                optimizer.zero_grad(set_to_none=True)
            update = (i+1) % ga == 0 or i+1 == len(loader)
            context = nullcontext() if update else ddp.no_sync()
            with context:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = ddp(batch["image"].to(device, non_blocking=True))
                    loss = criterion(logits, batch["label"].to(device, non_blocking=True))
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite global loss")
                (loss/group_size).backward()
            if update:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
                step += 1
                factor = lr_factor(step, steps_per_epoch*cfg["epochs"],
                                   steps_per_epoch*cfg["warmup_epochs"], cfg["min_lr_ratio"])
                for group in optimizer.param_groups:
                    group["lr"] = group["initial_lr"]*factor
                optimizer.step()
            total += loss.item(); n += 1
            component_total += criterion.last_components
            batch_finished = time.monotonic()
            model_step_seconds += batch_finished-ready
            if i == 0 or (i+1) % cfg["log_every"] == 0:
                timing = torch.tensor([data_wait_seconds, model_step_seconds], device=device)
                dist.all_reduce(timing, op=dist.ReduceOp.MAX)
            if rank == 0 and (i == 0 or (i+1) % cfg["log_every"] == 0):
                progress = dict(epoch=epoch+1, batch=i+1, batches=len(loader), step=step,
                                loss=loss.item(), mean_loss=total/n,
                                patches_per_second=(i+1)*cfg["batch_size"]*4/(time.monotonic()-started))
                progress.update(train_loss=total/n, elapsed_seconds=round(time.monotonic()-started, 2),
                                train_remaining_seconds=round((time.monotonic()-started)/(i+1)*(len(loader)-i-1), 2))
                progress.update(train_mse_loss=float(component_total[0]/n), train_dice_loss=float(component_total[1]/n),
                                dice_scope=criterion.dice_scope)
                progress.update(max_rank_data_wait_seconds=round(timing[0].item(),3),
                                max_rank_model_step_including_sync_seconds=round(timing[1].item(),3),
                                native_cache_enabled=cache_dir is not None)
                (out/"progress.json").write_text(json.dumps(progress, indent=2))
                print(json.dumps(progress), flush=True)
        train_seconds = time.monotonic()-started
        validation_started = time.monotonic()
        if rank == 0:
            print(f"Epoch {epoch+1}: validation started ({len(va)} patches)", flush=True)
        snapshot_name.pop("value", None)
        loss_out = {}
        observer = snapshot_observer if (rank == 0 and snapshot_every > 0
                                        and (epoch+1) % snapshot_every == 0) else None
        result = evaluate_distributed(model, val_loader, device, cfg, criterion=criterion,
                                      loss_out=loss_out, observer=observer, progress_dir=out/'validation_progress')
        score = result["macro"]["iou"]
        if score is None:
            raise ValueError("Validation has no usable positive support")
        val_loss = loss_out.get("val_loss", float("nan"))
        val_seconds = time.monotonic()-validation_started
        epoch_seconds = time.monotonic()-epoch_started
        elapsed_seconds = sum(row.get("epoch_seconds", 0.) for row in history)+epoch_seconds
        control = [None]
        if rank == 0:
            _, stop = stopper.update(score)
            improved = score > best
            best = max(best, score)
            history.append(dict(epoch=epoch+1, step=step, train_loss=total/n, val_loss=val_loss,
                                train_mse_loss=float(component_total[0]/n), train_dice_loss=float(component_total[1]/n),
                                val_mse_loss=loss_out['val_mse_loss'], val_dice_loss=loss_out['val_dice_loss'],
                                val_macro_iou=score, val_macro_f1=result["macro"]["f1"],
                                lr=optimizer.param_groups[0]["lr"], epoch_seconds=round(epoch_seconds, 3),
                                train_seconds=round(train_seconds, 3), val_seconds=round(val_seconds, 3),
                                elapsed_seconds=round(elapsed_seconds, 3),
                                snapshot=snapshot_name.get("value", "")))
            write_epoch_log(out, history)
            metrics_folder = out/"epoch_metrics"
            metrics_folder.mkdir(exist_ok=True)
            (metrics_folder/f"epoch_{epoch+1:03d}.json").write_text(json.dumps(
                dict(log=history[-1], validation=result, loss_components=loss_out['loss_components']), indent=2, allow_nan=False))
            control[0] = dict(stop=stop, improved=improved)
        dist.broadcast_object_list(control, src=0)
        rng = dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                   cuda=torch.cuda.get_rng_state(device))
        rank_rng = [None]*4 if rank == 0 else None
        dist.gather_object(rng, rank_rng, dst=0)
        if rank == 0:
            ck = dict(task="pixel_multilabel", model=model.state_dict(), optimizer=optimizer.state_dict(),
                      config=cfg, statistics=stats, data_signature=signature, epoch=epoch, step=step,
                      history=history, best=best, stopper=stopper.state_dict(), rank_rng=rank_rng)
            atomic_save(ck, out/"last.pt")
            if control[0]["improved"]:
                atomic_save(ck, out/"best.pt")
                (out/"best_validation_metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
            save_training_artifacts(history, str(out), "classification", x_key="epoch")
            print(json.dumps(history[-1]), flush=True)
            print(f"[epoch {epoch+1}/{cfg['epochs']}] train_loss={total/n:.4f} val_loss={val_loss:.4f} "
                  f"val_macro_iou={score:.4f} val_macro_f1={result['macro']['f1']:.4f} "
                  f"epoch_seconds={epoch_seconds:.1f} elapsed_seconds={elapsed_seconds:.1f}", flush=True)
        dist.barrier()
        if control[0]["stop"]:
            break


if __name__ == "__main__":
    main()
