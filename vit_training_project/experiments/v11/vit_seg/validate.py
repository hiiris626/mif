"""Fail closed on patient leakage, missing patches, invalid balance or provenance."""
import argparse
import json
from pathlib import Path
import pandas as pd
from .data import CHANNELS
from .prepare import source_table, patient_id
from .artifacts import data_signature


def validate(data):
    data = Path(data)
    stats = json.loads((data/"statistics.json").read_text())
    complete = json.loads((data/"COMPLETE.json").read_text())
    policy=json.loads((data/'training_policy.json').read_text()) if (data/'training_policy.json').exists() else {}
    natural=policy.get('sampling_policy') in ('shuffle_patches','pixel_balanced_patches')
    if not complete["complete_dataset"] or stats["nuclear_qc_performed"]:
        raise ValueError("Only full inventory without rescreening is accepted")
    expected = source_table(stats["source_spec"]["root"], stats["split_policy"], stats["source_spec"]["split_seed"])
    patients, ids, totals, train_channels = {}, set(), {}, set()
    for split in ("train", "val", "test"):
        frame = pd.read_csv(data/f"{split}.csv", usecols=["patch_id", "orion_slide_id"])
        patients[split] = set(frame.orion_slide_id.map(patient_id))
        if frame.patch_id.duplicated().any() or ids & set(frame.patch_id):
            raise ValueError("Duplicate/leaked natural patch IDs")
        desired = expected[expected.split == split]
        if set(frame.patch_id) != set(desired.patch_id) or patients[split] != set(desired.orion_slide_id.map(patient_id)):
            raise ValueError("Patient assignment or source coverage mismatch")
        ids.update(frame.patch_id); totals[split] = len(frame)
        for channel in ([] if natural else CHANNELS):
            view = pd.read_csv(data/"balanced"/split/f"{channel}.csv", usecols=["patch_id", "stratum"], dtype={"stratum": str})
            if not len(view):
                continue
            if split == "train": train_channels.add(channel)
            count = view.stratum.value_counts().to_dict()
            if set(count) != {"false", "true1", "true2"} or count["false"] != 2*count["true1"] or count["true1"] != count["true2"]:
                raise ValueError(f"Invalid 2:1:1 balance: {split}/{channel}")
            if view.patch_id.duplicated().any() or not set(view.patch_id) <= set(frame.patch_id):
                raise ValueError("Balanced view duplicates/leaks samples")
    if any(patients[a] & patients[b] for a,b in (("train","val"),("train","test"),("val","test"))):
        raise ValueError("Patient leakage")
    if len(ids) != len(expected) or stats["fitted_split"] != "train":
        raise ValueError("Incomplete inventory or leaked fitted statistics")
    combined = pd.read_csv(data/('train.csv' if natural else 'train_balanced.csv'),
        usecols=['patch_id'] if natural else ['patch_id','balance_channel','stratum'],dtype={'stratum':str})
    if natural and (combined.patch_id.duplicated().any() or len(combined)%4):
        raise ValueError('Natural training requires unique patches and equal four-rank counts')
    if not natural and set(combined.balance_channel) != train_channels:
        raise ValueError("Mixed training manifest omitted or invented a channel view")
    for channel, frame in ([] if natural else combined.groupby("balance_channel")):
        individual = pd.read_csv(data/"balanced/train"/f"{channel}.csv", usecols=["patch_id", "stratum"], dtype={"stratum": str})
        if set(map(tuple, frame[["patch_id", "stratum"]].to_numpy())) != set(map(tuple, individual.to_numpy())) or len(frame) != len(individual):
            raise ValueError("Mixed training manifest differs from channel views")
    if not len(combined):
        raise ValueError("No feasible balanced training samples")
    receipt = dict(n_source=len(expected), n_retained=len(ids), patient_counts={s:len(p) for s,p in patients.items()},
                   patch_counts=totals, balanced_train_rows=0 if natural else len(combined),
                   balanced_unique_patches=0 if natural else int(combined.patch_id.nunique()),
                   training_rows=len(combined),training_unique_patches=int(combined.patch_id.nunique()),
                   sampling_policy=policy.get('sampling_policy','channel_balanced'),
                   signature=data_signature(data), patient_leakage=False, nuclear_qc_performed=False)
    (data/"VALIDATED.json").write_text(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--data", required=True)
    print(json.dumps(validate(parser.parse_args().data), indent=2))
