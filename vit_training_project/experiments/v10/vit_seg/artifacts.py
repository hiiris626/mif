"""Lightweight content identity for data, models and post-training decisions."""
import hashlib


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def data_signature(data):
    names=["statistics.json", "patient_split.csv", "train_balanced.csv", "val.csv", "test.csv"]
    if (data/'binary_targets.json').exists():names.append('binary_targets.json')
    for name in ('train.csv','training_policy.json','bce_pixel_weights.json','patch_sampling.npz'):
        if (data/name).exists():names.append(name)
    return {name:file_hash(data/name) for name in names}
