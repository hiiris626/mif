"""Patient-disjoint splits and channel statistics. No StarDist or second cleaning."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sqlite3
import time
import cv2
import numpy as np
import pandas as pd
from .data import CHANNELS, read_he, read_mif, tissue_mask
ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"


def patient_id(slide_id):
    """CRC33_01/02 are two sections of the same ORION CRC33 case."""
    return {"CRC33_01": "CRC33", "CRC33_02": "CRC33"}.get(slide_id, slide_id)


def inventory(args):
    """Read native pairs once; preserve every published patch, no nuclear model."""
    cv2.setNumThreads(1)
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = source_table(root, args.split_policy, args.seed)
    if args.limit:
        df = pd.concat([d.sample(min(args.limit, len(d)), random_state=args.seed) for _, d in df.groupby("split")])
    df = df[df.patch_id % args.shards == args.shard]
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in root.glob("*_dataframe.csv")}
    spec = dict(signature=hashlib.sha256(df.to_csv(index=False).encode()).hexdigest(),
                root=str(root.resolve()), shards=args.shards, limit=args.limit,
                split_policy=args.split_policy, split_seed=args.seed, stage="inventory",
                source_qc="published_MIPHEI_QC_no_rescreening", tissue="HE_v1",
                source_csv_sha256=source_hashes)
    db = sqlite3.connect(out/f"shard_{args.shard}.sqlite")
    db.execute("CREATE TABLE IF NOT EXISTS patches (id INTEGER PRIMARY KEY, record TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB)")
    previous = db.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()
    if previous and json.loads(previous[0]) != spec:
        raise ValueError("Source/config changed: start a new run directory")
    db.execute("INSERT OR IGNORE INTO metadata VALUES ('spec', ?)", (json.dumps(spec),))
    stored = db.execute("SELECT value FROM metadata WHERE key='hist'").fetchone()
    hist = np.frombuffer(stored[0], np.int64).reshape(16, 256).copy() if stored else np.zeros((16, 256), np.int64)
    completed = {row[0] for row in db.execute("SELECT id FROM patches")}
    pending = df[~df.patch_id.isin(completed)].to_dict("records")
    db.commit()
    start = time.time()
    def load(row):
        he, mif = read_he(root/row["image_path"]), read_mif(root/row["target_path"])
        if he.shape[:2] != mif.shape[1:]:
            raise ValueError(f"Unregistered dimensions: {row['patch_id']}")
        mask = tissue_mask(he)
        foreground = (mif > 0) & mask[None]
        counts = foreground.sum((1, 2))
        row.update(patch_retained=True, retention_reason="published_QC_no_rescreening",
                   nuclear_qc_status="not_run", tissue_pixels=int(mask.sum()),
                   general_qc_status="usable" if mask.sum() >= 64 else "review_low_tissue")
        local_hist = np.zeros((16, 256), np.int64)
        for c, name in enumerate(CHANNELS):
            row[f"{name}_pixels"] = int(counts[c])
            row[f"{name}_coverage"] = float(counts[c]/max(mask.sum(), 1))
            row[f"{name}_structurally_present"] = True
            if row["split"] == "train":
                local_hist[c] = np.bincount(mif[c][foreground[c]], minlength=256)
        return row, local_hist
    with ThreadPoolExecutor(args.io_workers) as pool:
        for offset in range(0, len(pending), args.batch_size):
            batch = list(pool.map(load, pending[offset:offset+args.batch_size]))
            with db:
                for row, local_hist in batch:
                    hist += local_hist
                    db.execute("INSERT INTO patches VALUES (?, ?)", (row["patch_id"], json.dumps(row)))
                db.execute("INSERT OR REPLACE INTO metadata VALUES ('hist', ?)", (hist.tobytes(),))
            n = min(offset+args.batch_size, len(pending))
            if offset == 0 or n % 640 == 0 or n == len(pending):
                print(json.dumps(dict(shard=args.shard, processed=len(completed)+n, total=len(df),
                                      tiles_per_second=n/max(time.time()-start, .001))), flush=True)
    with db:
        db.execute("INSERT OR REPLACE INTO metadata VALUES ('complete', ?)", (str(len(df)),))
    db.close()

def patient_split(df, seed=42):
    """Assign whole ORION cases, rounding 70/15/15 by largest remainder."""
    if "orion_slide_id" not in df or df.orion_slide_id.isna().any():
        raise ValueError("Patient splitting requires complete ORION case identifiers")
    patients = np.array(sorted(df.orion_slide_id.unique()))
    if len(patients) < 3:
        raise ValueError("At least three patients are needed")
    np.random.default_rng(seed).shuffle(patients)
    raw = np.array([.70, .15, .15])*len(patients)
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw-counts), kind="stable")[:len(patients)-counts.sum()]:
        counts[i] += 1
    if not counts.min():
        raise ValueError("Too few patients for nonempty 70/15/15 splits")
    mapping = {case: split for split, group in zip(("train","val","test"), np.split(patients, np.cumsum(counts)[:-1])) for case in group}
    # Preserve seeded historical case assignments except a multi-section case
    # split across sets: keep the whole patient in the most held-out split.
    priority = {'train': 0, 'val': 1, 'test': 2}
    grouped = {}
    for case, split in mapping.items():
        identity = patient_id(case)
        grouped[identity] = max(grouped.get(identity, 'train'), split, key=priority.get)
    mapping = {case: grouped[patient_id(case)] for case in mapping}
    raw_counts = np.array([.70, .15, .15])*len(grouped)
    required = np.floor(raw_counts).astype(int)
    for i in np.argsort(-(raw_counts-required), kind='stable')[:len(grouped)-required.sum()]:
        required[i] += 1
    actual = np.array([list(grouped.values()).count(s) for s in ('train','val','test')])
    if not np.array_equal(actual, required):
        raise ValueError('Canonical patient grouping needs an explicit 70/15/15 assignment; do not split sections')
    result = df.copy()
    result["original_split"] = result["split"]
    result["split"] = result.orion_slide_id.map(mapping)
    result["patient_id"] = result.orion_slide_id.map(patient_id)
    return result

def source_table(root, split_policy="patient_70_15_15", seed=42):
    frames = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(Path(root)/f"{split}_dataframe.csv", usecols=["in_slide_name", "image_path", "target_path", "nuclei_path"])
        df = df[["in_slide_name", "image_path", "target_path", "nuclei_path"]].copy()
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if df.image_path.duplicated().any():
        raise ValueError("Duplicate patch names across input splits")
    if (df.groupby("in_slide_name").split.nunique() > 1).any():
        raise ValueError("Slide leakage across train/val/test")
    df["patch_id"] = np.arange(len(df))
    slide_meta = Path(root)/"slide_dataframe.csv"
    if slide_meta.exists():
        info = pd.read_csv(slide_meta, usecols=["in_slide_name", "orion_slide_id"])
        if info.in_slide_name.duplicated().any():
            raise ValueError("Ambiguous slide-to-case mapping")
        df = df.merge(info, on="in_slide_name", how="left", validate="many_to_one")
        if df.orion_slide_id.isna().any() or (df.groupby("orion_slide_id").split.nunique()>1).any():
            raise ValueError("Missing case identifier or ORION case leakage across splits")
    if split_policy == "patient_70_15_15":
        df = patient_split(df, seed)
    elif split_policy != "original":
        raise ValueError("Unknown split policy")
    else:
        df["original_split"] = df["split"]
        df['patient_id'] = df.orion_slide_id.map(patient_id)
    return df

def balanced_view(frame, channel, coverage_mean, seed):
    valid = frame[f"{channel}_valid"].astype(bool)
    positive = valid & (frame[f"{channel}_pixels"] > 0)
    # A blank/nearly blank H&E tile is not a trustworthy negative even when
    # the assay channel is known to exist on this slide.
    usable_tissue = (frame["general_qc_status"] == "usable") if "general_qc_status" in frame else pd.Series(True, index=frame.index)
    negative = valid & ~positive & usable_tissue
    high = positive & (frame[f"{channel}_coverage"] > coverage_mean)
    low = positive & ~high
    n = min(int(negative.sum())//2, int(high.sum()), int(low.sum()))
    counts = dict(unavailable=int((~valid).sum()), untrusted_zero=int((valid & ~positive & ~usable_tissue).sum()),
                  false=int(negative.sum()), true=int(positive.sum()), true1=int(high.sum()), true2=int(low.sum()),
                  balanced_false=2*n, balanced_true1=n, balanced_true2=n,
                  feasible=bool(n), downsampled=True)
    parts = []
    for mask, count, tag in ((negative, 2*n, "false"), (high, n, "true1"), (low, n, "true2")):
        part = frame[mask].sample(n=count, random_state=seed).copy()
        part["stratum"] = tag
        part["balance_channel"] = channel
        parts.append(part)
    return pd.concat(parts), counts

def finalize(args):
    out = Path(args.out)
    records, hist, specs = [], np.zeros((16, 256), np.int64), []
    for shard in range(args.shards):
        path = out/f"shard_{shard}.sqlite"
        if not path.is_file():
            raise ValueError(f"Missing inventory shard: {path}")
        db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        if not db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone():
            raise ValueError(f"inventory shard not complete: {shard}")
        specs.append(json.loads(db.execute("SELECT value FROM metadata WHERE key='spec'").fetchone()[0]))
        records.extend(json.loads(row[0]) for row in db.execute("SELECT record FROM patches ORDER BY id"))
        hist += np.frombuffer(db.execute("SELECT value FROM metadata WHERE key='hist'").fetchone()[0], np.int64).reshape(16, 256)
        db.close()
    if len({json.dumps({k:v for k,v in s.items() if k != 'signature'}, sort_keys=True) for s in specs}) != 1:
        raise ValueError("Incompatible shard configurations")
    df = pd.DataFrame(records).sort_values("patch_id")
    if df.patch_id.duplicated().any():
        raise ValueError("Duplicate patch IDs across shards")
    expected = source_table(specs[0]["root"], specs[0]["split_policy"], specs[0]["split_seed"])
    check_source = expected
    if specs[0]["limit"]:
        check_source = pd.concat([d.sample(min(specs[0]["limit"],len(d)),random_state=specs[0]["split_seed"]) for _,d in expected.groupby("split")])
    for shard, spec in enumerate(specs):
        rows = check_source[check_source.patch_id % args.shards == shard]
        if hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest() != spec["signature"]:
            raise ValueError("Source manifest or patient assignment changed since scan")
    if set(df.patch_id) != set(check_source.patch_id):
        raise ValueError("Inventory records do not match the selected source")
    if not specs[0]["limit"] and set(df.patch_id) != set(expected.patch_id):
        raise ValueError("Full source coverage is incomplete")
    out.mkdir(exist_ok=True, parents=True)
    if 'patient_id' not in expected:
        expected['patient_id'] = expected.orion_slide_id.map(patient_id)
    expected[["patient_id", "orion_slide_id", "in_slide_name", "original_split", "split"]].drop_duplicates().to_csv(out/"patient_split.csv", index=False)
    # Nuclear agreement never deletes a patch. A channel is trusted only when
    # the file contains it and the same slide shows signal for that channel at
    # least once, distinguishing observed negatives from assay-wide absence.
    kept = df[df.patch_retained].copy()
    slide_health_rows = []
    for slide, frame in kept.groupby("in_slide_name"):
        for name in CHANNELS:
            healthy = bool(frame[f"{name}_structurally_present"].all()
                           and (frame[f"{name}_pixels"] > 0).any())
            kept.loc[frame.index, f"{name}_valid"] = healthy
            slide_health_rows.append(dict(in_slide_name=slide, channel=name,
                                          channel_valid=healthy,
                                          availability_basis="signal_observed_on_slide" if healthy else "unknown_all_zero_or_missing; not_confirmed_assay_failure",
                                          positive_patches=int((frame[f"{name}_pixels"] > 0).sum()),
                                          total_patches=len(frame)))
    for name in CHANNELS:
        kept[f"{name}_valid"] = kept[f"{name}_valid"].astype(bool)
    pd.DataFrame(slide_health_rows).to_csv(out/"slide_channel_health.csv", index=False)
    kept.to_csv(out/"patch_manifest.csv", index=False)
    tr = kept[kept.split == "train"]
    if tr.empty:
        raise ValueError("No retained training patches")
    # Histogram was accumulated before slide-health was known. Recompute only
    # if an entire train slide/channel is unavailable; this is expected to be rare.
    count = hist.sum(1)
    expected_count = np.array([tr.loc[tr[f"{c}_valid"], f"{c}_pixels"].sum() for c in CHANNELS])
    if not np.array_equal(count, expected_count):
        raise ValueError("Unavailable train slide/channel detected; rerun with explicit channel-valid histogram support")
    mean = (hist*np.arange(256)).sum(1)/np.maximum(count, 1)
    std = np.sqrt(np.maximum(0, (hist*np.arange(256)**2).sum(1)/np.maximum(count, 1)-mean**2))
    q = [max(1, int(np.searchsorted(h.cumsum(), h.sum()*.999))) for h in hist]
    coverage_mean = [float(tr.loc[tr[f"{c}_valid"] & (tr[f"{c}_pixels"]>0), f"{c}_coverage"].mean())
                     if (tr[f"{c}_valid"] & (tr[f"{c}_pixels"]>0)).any() else 0 for c in CHANNELS]
    from .losses import channel_weights
    # Scale std to [0,1] for numerical stability; normalized inverse weights are scale invariant.
    weights = channel_weights(np.where(count > 0, std/255, np.nan)).tolist()
    stats = dict(channels=CHANNELS, q=q, positive_pixel_mean=mean.tolist(), positive_pixel_std=std.tolist(),
                 positive_pixel_count=count.tolist(), coverage_mean_train_true=coverage_mean,
                 channel_weights=weights, weights_rule="inverse positive-pixel std; normalize, clip [.25,4], renormalize",
                 fitted_split="train", pixel_scope="positive pixels only within HE tissue in channel-valid true patches",
                 equal_coverage_rule="true2", split_policy=specs[0]["split_policy"], source_spec=specs[0],
                 patient_counts=expected.groupby("split").patient_id.nunique().to_dict(),
                 patient_identity_rule='CRC33_01 and CRC33_02 are sections of CRC33; keep whole patient held out',
                 nuclear_qc_counts={g:int((df.nuclear_qc_status==g).sum()) for g in ("not_evaluable", "rejected", "retained", "gold")},
                 nuclear_qc_counts_by_split={s:{g:int(((df.split==s)&(df.nuclear_qc_status==g)).sum()) for g in ("not_evaluable","rejected","retained","gold")} for s in ("train","val","test")},
                 nuclear_qc_performed=specs[0].get("stage") != "inventory",
                 split_patch_counts=kept.split.value_counts().to_dict(),
                 general_qc_counts=df.general_qc_status.value_counts().to_dict(),
                 retention_policy="all readable patches retained; nuclear QC is diagnostic; invalid channels ignored",
                 n_source=len(df), n_retained=len(kept), complete_dataset=not bool(specs[0]["limit"]))
    if not stats["nuclear_qc_performed"]:
        stats["nuclear_qc_counts"] = None
        stats["nuclear_qc_counts_by_split"] = None
        stats["retention_policy"] = "reuse published MIPHEI QC; no rescreening; all source patches retained"
    (out/"statistics.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    counts, sampling = [], []
    for split in ("train", "val", "test"):
        frame = kept[kept.split == split]
        frame.to_csv(out/f"{split}.csv", index=False)
        if stats["nuclear_qc_performed"]:
            frame[frame.nuclear_qc_status=="gold"].to_csv(out/f"{split}_nuclear_qc_gold.csv", index=False)
            frame[frame.nuclear_qc_status=="not_evaluable"].to_csv(out/f"{split}_nuclear_qc_review.csv", index=False)
        folder = out/"balanced"/split
        folder.mkdir(parents=True, exist_ok=True)
        for c, name in enumerate(CHANNELS):
            view, summary = balanced_view(frame, name, coverage_mean[c], args.seed+c)
            view = view[["patch_id", "orion_slide_id", "image_path", "target_path", "stratum", "balance_channel",
                         *(f"{c}_valid" for c in CHANNELS)]]
            view.to_csv(folder/f"{name}.csv", index=False)
            counts.append(dict(split=split, channel=name, coverage_threshold_train=coverage_mean[c], **summary))
            if split == "train" and len(view):
                sampling.append(view)
    count_frame = pd.DataFrame(counts)
    count_frame.to_csv(out/"channel_patch_counts.csv", index=False)
    count_frame.groupby("channel", sort=False)[["true", "false", "true1", "true2", "unavailable", "untrusted_zero"]].sum().to_csv(out/"channel_patch_totals.csv")
    if sampling:
        # Retain duplicates with the channel/stratum provenance: each view is exactly 2:1:1.
        pd.concat(sampling).to_csv(out/"train_balanced.csv", index=False)
    (out/"COMPLETE.json").write_text(json.dumps(dict(n_source=len(df), n_retained=len(kept),
                                complete_dataset=stats["complete_dataset"], shards=args.shards), indent=2))
    print(json.dumps(dict(n_source=len(df), n_retained=len(kept),
                          nuclear_qc_performed=stats["nuclear_qc_performed"],
                          split_patch_counts=stats["split_patch_counts"]), ensure_ascii=False), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("inventory", "finalize"))
    parser.add_argument("--root", default=ROOT); parser.add_argument("--out", required=True)
    parser.add_argument("--shards", type=int, default=8); parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32); parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-policy", default="patient_70_15_15")
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard < args.shards or args.batch_size < 1:
        parser.error("Invalid shard or batch configuration")
    (inventory if args.stage == "inventory" else finalize)(args)


if __name__ == "__main__":
    main()
