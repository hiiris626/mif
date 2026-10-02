"""Read-only analysis of existing run outputs; never imports/changes live training."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parents[2]
RUN = PROJECT / 'runs/ddp_baseline'
OUT = Path(__file__).resolve().parent
stats = json.loads((RUN/'data/statistics.json').read_text())
channels = stats['channels']
weights = np.array(stats['channel_weights'])
history = pd.read_csv(RUN/'results/model/train_log.csv')
rows = []
for row in history.to_dict('records'):
    metrics = json.loads((RUN/f"results/model/epoch_metrics/epoch_{row['epoch']:03d}.json").read_text())
    classes = metrics['validation']['per_class']
    # Brier is masked per-channel probability MSE. Float64 vs float32
    # subtraction/squaring introduces only rounding differences here.
    mse = np.average([classes[c]['brier'] for c in channels], weights=weights)
    rows.append(dict(**row, reconstructed_val_mse=mse,
                     reconstructed_val_dice=row['val_loss']-mse))
curves = pd.DataFrame(rows)
curves.to_csv(OUT/'loss_components.csv', index=False)

columns = ['patch_id', 'split']+[f'{c}_{suffix}' for c in channels for suffix in ['pixels', 'valid']]
manifest = pd.read_csv(RUN/'data/patch_manifest.csv', usecols=columns)
valid = manifest[[c+'_valid' for c in channels]].to_numpy(bool)
pixels = manifest[[c+'_pixels' for c in channels]].to_numpy()
eligible = ((pixels > 0) & valid).any(1)
balance_ids = pd.read_csv(RUN/'data/train_balanced.csv', usecols=['patch_id'])
balance_index = pd.Index(manifest.patch_id).get_indexer(balance_ids.patch_id)
assert (balance_index >= 0).all()
empty_rows = []
for split, index in [('natural_train', np.flatnonzero(manifest.split.eq('train'))),
                     ('natural_val', np.flatnonzero(manifest.split.eq('val'))),
                     ('balanced_train', balance_index)]:
    index = index[eligible[index]]
    numerator = ((pixels[index] == 0) & valid[index]).sum(0)
    denominator = valid[index].sum(0)
    for c, a, b in zip(channels, numerator, denominator):
        empty_rows.append(dict(split=split, channel=c, empty_images=int(a),
                               available_images=int(b), native_empty_fraction=float(a/b)))
empty = pd.DataFrame(empty_rows)
empty.to_csv(OUT/'native_empty_channel_rates.csv', index=False)

latest = int(history.epoch.max())
latest_metrics = json.loads((RUN/f'results/model/epoch_metrics/epoch_{latest:03d}.json').read_text())
channel_rows = []
for c, w in zip(channels, weights):
    m = latest_metrics['validation']['per_class'][c]
    channel_rows.append(dict(channel=c, weight=w, precision=m['precision'], recall=m['recall'],
        f1=m['f1'], iou=m['iou'], mse=m['brier'], auroc=m['auroc_histogram'],
        average_precision=m['average_precision_histogram'],
        positive_fraction=m['support']/(m['support']+m['negative_support'])))
pd.DataFrame(channel_rows).to_csv(OUT/'latest_channel_metrics.csv', index=False)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
axes[0].plot(curves.epoch, curves.train_loss, label='Train: balanced + augmentation')
axes[0].plot(curves.epoch, curves.val_loss, label='Validation: natural distribution')
axes[0].set(title='Composite loss', ylabel='MSE + image/channel soft Dice')
axes[1].plot(curves.epoch, curves.reconstructed_val_dice, label='Validation Dice loss')
axes[1].plot(curves.epoch, curves.reconstructed_val_mse, label='Validation MSE')
axes[1].set(title='Validation loss components', ylabel='Weighted component')
axes[2].plot(curves.epoch, curves.val_macro_f1, label='Validation macro F1')
axes[2].plot(curves.epoch, curves.val_macro_iou, label='Validation macro IoU')
axes[2].set(title='Classification metrics (threshold 0.5)', ylabel='Score')
for ax in axes:
    ax.set_xlabel('Completed epoch'); ax.grid(alpha=.2); ax.legend(fontsize=8)
fig.savefig(OUT/'loss_diagnosis.png', dpi=160)
plt.close(fig)

summary = dict(last_completed_epoch=latest,
    first=rows[0], last=rows[-1], best_iou_epoch=int(history.loc[history.val_macro_iou.idxmax(), 'epoch']),
    native_empty_weighted={split: float(np.average(
        empty.loc[empty.split.eq(split)].set_index('channel').loc[channels].native_empty_fraction,
        weights=weights)) for split in empty.split.unique()},
    empty_dice_example={str(p): 1-1e-6/(10000*p+1e-6) for p in [.1, .01, .001]},
    scope='No GPU use, checkpoint writes, training changes or test-set evaluation. '
          'Empty fractions are native-resolution manifest statistics, before resizing/augmentation; '
          'not measured per-epoch empty-Dice contributions or mathematical lower bounds.')
(OUT/'summary.json').write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2), flush=True)
