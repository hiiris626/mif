"""Independent-channel classification metrics; no regression or argmax scoring."""
import numpy as np
import torch
from .data import IGNORE

class MultilabelMetrics:
    """Independent binary confusion matrices; coexpression is never argmaxed."""
    def __init__(self, classes=16, bins=256, threshold=.5):
        self.classes, self.bins = classes, bins
        self.threshold = np.broadcast_to(np.asarray(threshold, float), (classes,)).copy()
        if np.any((self.threshold <= 0) | (self.threshold >= 1)):
            raise ValueError("Probability thresholds must be in (0,1)")
        self.confusion = np.zeros((classes, 2, 2), np.int64)
        self.pos = np.zeros((classes, bins), np.int64)
        self.neg = self.pos.copy()
        self.cal_n = np.zeros((classes, 15), np.int64)
        self.cal_y = np.zeros((classes, 15))
        self.cal_p = np.zeros((classes, 15))
        self.brier = np.zeros(classes)
        self.n_pixels = self.exact = 0

    def update(self, probabilities, target):
        p = probabilities.detach().float().cpu().numpy() if torch.is_tensor(probabilities) else np.asarray(probabilities)
        t = target.detach().cpu().numpy() if torch.is_tensor(target) else np.asarray(target)
        if p.shape != t.shape or p.ndim != 4 or p.shape[1] != self.classes:
            raise ValueError("Expected matching N,C,H,W probabilities and multilabel targets")
        valid = t != IGNORE
        pred = p >= self.threshold[None, :, None, None]
        joint = valid.all(1)
        self.n_pixels += int(joint.sum())
        self.exact += int(((pred == t).all(1) & joint).sum())
        for c in range(self.classes):
            pc, yc = p[:, c][valid[:, c]], t[:, c][valid[:, c]].astype(int)
            if not len(yc):
                continue
            if np.any((yc != 0) & (yc != 1)):
                raise ValueError("Multilabel targets must be 0, 1 or IGNORE")
            prediction = pc >= self.threshold[c]
            self.confusion[c] += np.bincount(2*yc+prediction, minlength=4).reshape(2, 2)
            bucket = np.minimum((np.clip(pc, 0, 1)*self.bins).astype(int), self.bins-1)
            self.pos[c] += np.bincount(bucket[yc == 1], minlength=self.bins)
            self.neg[c] += np.bincount(bucket[yc == 0], minlength=self.bins)
            cb = np.minimum((np.clip(pc, 0, 1)*15).astype(int), 14)
            self.cal_n[c] += np.bincount(cb, minlength=15)
            self.cal_y[c] += np.bincount(cb, weights=yc, minlength=15)
            self.cal_p[c] += np.bincount(cb, weights=pc, minlength=15)
            self.brier[c] += float(np.square(pc-yc).sum())

    def result(self, names):
        ratio = lambda a, b: float(a/b) if b else None
        output = {}
        for c, name in enumerate(names):
            tn, fp, fn, tp = self.confusion[c].ravel().tolist()
            pos, neg = self.pos[c][::-1], self.neg[c][::-1]
            roc = ap = None
            if pos.sum():
                ap = float(np.sum(pos/pos.sum()*pos.cumsum()/np.maximum((pos+neg).cumsum(), 1)))
                if neg.sum():
                    roc = float(np.trapezoid(np.r_[0, pos.cumsum()/pos.sum()], np.r_[0, neg.cumsum()/neg.sum()]))
            n = tn+fp+fn+tp
            output[name] = dict(precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn),
                f1=ratio(2*tp,2*tp+fp+fn), dice=ratio(2*tp,2*tp+fp+fn), iou=ratio(tp,tp+fp+fn),
                specificity=ratio(tn,tn+fp), support=tp+fn, negative_support=tn+fp,
                auroc_histogram=roc, average_precision_histogram=ap,
                ece=ratio(np.abs(self.cal_y[c]-self.cal_p[c]).sum(), n), brier=ratio(self.brier[c], n),
                threshold=float(self.threshold[c]), confusion_matrix=[[tn,fp],[fn,tp]])
        macro = {}
        for key in ("precision", "recall", "f1", "dice", "iou", "auroc_histogram", "average_precision_histogram"):
            values = [r[key] for r in output.values() if r[key] is not None and r["support"] > 0]
            macro[key] = float(np.mean(values)) if values else None
        tn, fp, fn, tp = self.confusion.sum(0).ravel().tolist()
        n = tn+fp+fn+tp
        return dict(task="pixel_multilabel", n_valid_pixels=self.n_pixels, n_valid_channel_pixels=n,
            exact_match_accuracy=ratio(self.exact,self.n_pixels), hamming_accuracy=ratio(tp+tn,n),
            micro_f1=ratio(2*tp,2*tp+fp+fn), macro=macro, per_class=output,
            ece=ratio(np.abs(self.cal_y-self.cal_p).sum(),n), brier=ratio(self.brier.sum(),n),
            roc_ap_bins=self.bins, background_included=False, ignored_label=IGNORE, argmax_used=False)
