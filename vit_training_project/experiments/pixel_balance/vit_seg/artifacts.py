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
    names += [name for name in ('training_pixels.npz','pixel_balance.json') if (data/name).exists()]
    return {name:file_hash(data/name) for name in names}
