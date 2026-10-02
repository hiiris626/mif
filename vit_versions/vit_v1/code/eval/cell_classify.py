"""细胞级评估：per-marker 二分类 AUC / F1（MIPHEI 论文核心指标）。

流程（对齐 MIPHEI）：
  1. 用数据自带 nuclei 掩码定位细胞（无需 Cellpose/StarDist 实例分割）；
  2. 每个细胞提取 mIF 16 通道在核区域的平均强度作为特征；
  3. ground truth：csv_nuclei_pos 的逐 marker 阳性布尔（15 个非 Hoechst marker）；
  4. 逻辑回归（StandardScaler + class_weight=balanced）：用 train（默认 val split，
     1,528,069 细胞）的「真实 mIF 特征」训练；
  5. 测试（test split，1,780,545 细胞）：
     - model ：特征来自模型预测的 mIF → 报 auc_model / f1_model；
     - oracle：特征来自真实 mIF（同一分类器）→ 无关模型的上界 auc_oracle / f1_oracle；
     F1 阈值取 0.5；AUC 用阳性概率。
  汇总 = 15 个 marker 的算术平均（与 v1/v2/v3 各报告的细胞指标一致）。

历史输出：v1/v2/v3 分别在
    /data/weiyh/results/vit_matte/vitmatte_512_cellf1.json
    /data/weiyh/results/vit_matte/virchow2_vitmatte_v2_512_epoch15/cell_f1.json
    /data/weiyh/results/vit_matte/virchow2_vitmatte_v3_256_best/cell_f1.json

用法：
    python -m eval.cell_classify \
        --pred_dir /data/weiyh/results/vit_matte/vitmatte_virchow2_epoch15 \
        --data_root /data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x \
        --train_split val --test_split test \
        --out cell_f1.json
"""
import os
import hashlib
import json
import glob
import argparse
import numpy as np
import pandas as pd
import tifffile
import cv2
from collections import OrderedDict

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score

from datacore.orioncrc_dataset import CHANNELS, MIF_SELECT

DEFAULT_ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"
CSV_DIR = "csv_nuclei_pos"

# csv 中 marker 列名（去掉 _pos）→ 用于规则判断的标准名
MARKER_NAMES = [
    "CD31", "CD45", "CD68", "CD4", "FOXP3", "CD8a", "CD45RO", "CD20",
    "PD-L1", "CD3e", "CD163", "E-cadherin", "Ki67", "Pan-CK", "SMA",
]


def phenotype(m):
    """从 marker 阳性 dict 推断细胞类型（优先级从高到低，首个匹配生效）。

    参考 MIPHEI 的细胞分型：肿瘤/上皮、T 亚群、B、巨噬、内皮、基质。
    注：当前评估主输出为 per-marker 二分类（不经过本函数）；此函数为保留的
    细胞类型聚合辅助（早期版本/其他分析可能引用）。
    """
    if m["Pan-CK"] or m["E-cadherin"]:
        return "Tumor"
    if m["CD20"]:
        return "B_cell"
    if m["CD68"] or m["CD163"]:
        return "Macrophage"
    if m["CD31"]:
        return "Endothelial"
    if m["CD4"] and m["FOXP3"]:
        return "Treg"
    if m["CD8a"]:
        return "CD8_T"
    if m["CD4"]:
        return "CD4_T"
    if m["CD3e"]:
        return "Other_T"
    if m["SMA"]:
        return "Stromal"
    if m["CD45"]:
        return "Other_immune"
    return "Other"


def slide_prefix_from_path(path):
    """从 tile 路径提取 slide 名（去掉 _x_y_0_512_512 坐标段）。"""
    base = os.path.basename(path)
    # 去掉扩展名，再去掉最后 5 个 '_' 分隔的坐标段
    stem = os.path.splitext(base)[0]
    return stem.rsplit("_", 5)[0]


def load_slide_csv(data_root, slide_prefix):
    """读取 slide 级细胞表型 csv -> (marker_bool [N,15], index_map [max_label+1])。

    index_map[label] = 1-based row index（0 表示无记录），用 numpy 索引避免逐细胞查 dict。
    """
    csv_path = os.path.join(data_root, CSV_DIR, slide_prefix + ".csv")
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    marker_cols = [f"{n}_pos" for n in MARKER_NAMES]  # 15 个 marker，去掉 PD-1，顺序与 MARKER_NAMES 一致
    marker_bool = df[marker_cols].to_numpy(dtype=bool)  # [N, 15]
    labels = df["label"].to_numpy(dtype=np.int64)
    max_label = int(labels.max())
    index_map = np.zeros(max_label + 1, dtype=np.int64)  # 0 = 未找到
    index_map[labels] = np.arange(1, len(labels) + 1)  # 1-based row index
    return marker_bool, index_map


def per_label_mean(nuclei, mif):
    """向量化：每个 label 在 mif 各通道的平均强度。

    nuclei: [H,W] int label (0=背景)；mif: [16,H,W] float。
    返回 (labels[K], feats[K,16])。
    """
    flat = nuclei.ravel()
    valid = flat > 0
    if not valid.any():
        return np.array([], dtype=np.int64), np.zeros((0, mif.shape[0]), dtype=np.float32)
    uniq, inverse = np.unique(flat[valid], return_inverse=True)
    counts = np.bincount(inverse).astype(np.float32)
    C = mif.shape[0]
    feats = np.zeros((len(uniq), C), dtype=np.float32)
    mif_flat = mif.reshape(C, -1)[:, valid]
    for c in range(C):
        sums = np.bincount(inverse, weights=mif_flat[c])
        feats[:, c] = sums / counts
    return uniq, feats


def load_mif(path, target_size=256):
    """读 mIF TIFF -> [16,H,W] float32(0-255)，resize 到 target_size。"""
    a = tifffile.imread(path, maxworkers=8)
    if a.ndim == 2:
        a = a[None]
    if a.ndim == 3 and a.shape[0] not in (16, 17):
        a = a.transpose(2, 0, 1)
    a = a.astype(np.float32)
    if a.shape[0] == 17:
        a = a[MIF_SELECT]
    if a.shape[0] != 16:
        raise ValueError(f"{path}: 通道数 {a.shape[0]} != 16")
    if a.shape[1] != target_size or a.shape[2] != target_size:
        out = np.empty((16, target_size, target_size), dtype=np.float32)
        for c in range(16):
            out[c] = cv2.resize(a[c], (target_size, target_size), interpolation=cv2.INTER_AREA)
        a = out
    return a


def load_nuclei(path, target_size=256):
    """读 nuclei label TIFF -> [H,W] int，最近邻 resize（保留 label）。"""
    n = tifffile.imread(path)
    n = n.astype(np.int64)
    if n.ndim == 3:
        n = n[..., 0]
    if n.shape[0] != target_size or n.shape[1] != target_size:
        n = cv2.resize(n, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
    return n


def collect_cells(data_root, split, pred_dir=None, max_tiles=None, load_real=True):
    """遍历 split 的 tile，收集 (per-cell 特征, 类型, 来源)。

    返回 dict: {"X_real": [N,16], "X_pred": [N,16], "y": [N], "marker_feats": [N,15]}
    其中 X_real 来自真实 mIF，X_pred 来自预测 mIF（pred_dir 提供时）。
    """
    df = pd.read_csv(os.path.join(data_root, f"{split}_dataframe.csv"))
    he_col = "image_path" if "image_path" in df.columns else "he_path"
    if_col = "target_path" if "target_path" in df.columns else "if_path"
    nuc_col = "nuclei_path"

    # 缓存 slide csv，避免重复读
    csv_cache = {}

    X_real, X_pred, marker_feats, cell_ids = [], [], [], []
    n_tiles = 0
    for i, row in df.iterrows():
        if max_tiles is not None and n_tiles >= max_tiles:
            break
        n_tiles += 1
        he_path = row[he_col]
        if_path = row[if_col]
        nuc_path = row[nuc_col]

        # 解析绝对路径
        for p in (he_path, if_path, nuc_path):
            if not os.path.isabs(p):
                pass
        abs_if = if_path if os.path.isabs(if_path) else os.path.join(data_root, if_path)
        abs_nuc = nuc_path if os.path.isabs(nuc_path) else os.path.join(data_root, nuc_path)

        # 加载 nuclei；真实特征可从共享缓存读取，避免每个模型重复解码 GT。
        try:
            mif_real = load_mif(abs_if) if load_real else None
            nuclei = load_nuclei(abs_nuc)
        except Exception as e:
            raise RuntimeError(f"Cannot evaluate unreadable tile {if_path}") from e

        # 预测 mIF（可选）
        mif_pred = None
        if pred_dir is not None:
            he_stem = os.path.splitext(os.path.basename(he_path))[0]
            for ext in (".tiff", ".tif"):
                cand = os.path.join(pred_dir, he_stem + ext)
                if os.path.exists(cand):
                    mif_pred = load_mif(cand)
                    break
            if mif_pred is None:
                raise FileNotFoundError(f"Missing prediction: {he_stem}")

        # slide csv -> 细胞 marker ground truth（marker_bool + label->row 索引）
        sp = slide_prefix_from_path(if_path)
        if sp not in csv_cache:
            csv_cache[sp] = load_slide_csv(data_root, sp)
        csv_data = csv_cache[sp]
        if csv_data is None:
            raise FileNotFoundError(f"Missing phenotype CSV: {sp}")
        marker_bool_all, index_map = csv_data

        # per-label 特征
        labels, feats_primary = per_label_mean(nuclei, mif_real if mif_real is not None else mif_pred)
        if labels.size == 0:
            continue
        # label -> csv row（越界 label 映射到 0，row_idx=-1 被过滤）
        safe_labels = np.where(labels < len(index_map), labels, 0)
        row_idx = index_map[safe_labels] - 1
        keep = row_idx >= 0
        labels = labels[keep]
        feats_primary = feats_primary[keep]
        row_idx = row_idx[keep]
        if labels.size == 0:
            continue

        # marker 特征 [K,15]（每个 marker 的阳性/阴性，作为 per-marker 二分类的 ground truth）
        marker = marker_bool_all[row_idx].astype(np.float32)  # [K, 15]

        if load_real:
            X_real.append(feats_primary)
        marker_feats.append(marker)
        cell_ids.append(np.column_stack((np.full(len(labels), i, np.int64), labels.astype(np.int64))))

        if mif_pred is not None:
            if load_real:
                _, feats_pred = per_label_mean(nuclei, mif_pred)
                X_pred.append(feats_pred[keep])
            else:
                X_pred.append(feats_primary)


    X_real = np.concatenate(X_real, axis=0) if X_real else np.zeros((0, 16), dtype=np.float32)
    marker_feats = np.concatenate(marker_feats, axis=0) if marker_feats else np.zeros((0, len(MARKER_NAMES)), dtype=np.float32)
    X_pred = np.concatenate(X_pred, axis=0) if X_pred else None
    n_cells = len(marker_feats)
    print(f"[{split}] tiles={n_tiles}, cells={n_cells}")
    return {"X_real": X_real, "X_pred": X_pred, "marker_feats": marker_feats,
            "cell_ids": np.concatenate(cell_ids) if cell_ids else np.zeros((0,2), np.int64), "n_tiles": n_tiles}


def load_or_build_real_cache(data_root, split, cache_dir, max_tiles=None):
    """共享缓存：真实 mIF 的 per-cell 特征 + marker 标签（与模型无关，多模型可复用）。

    缓存文件 <cache_dir>/<split>_real_cells.npz；命中则直接加载，避免多模型评估
    重复解码 GT mIF。返回 dict {X_real, marker_feats, X_pred: None}。
    """
    manifest_path = os.path.join(data_root, f"{split}_dataframe.csv")
    digest = hashlib.sha256(("cell-cache-v2" + os.path.realpath(data_root) + str(max_tiles)).encode())
    with open(manifest_path, "rb") as stream:
        digest.update(stream.read())
    source = pd.read_csv(manifest_path)
    if max_tiles is not None:
        source = source.head(max_tiles)
    # Detect replacement of source tiles/labels even if CSV text stays identical.
    cols = [c for c in ("target_path", "if_path", "nuclei_path") if c in source]
    paths = {os.path.join(data_root, str(v)) for c in cols for v in source[c]}
    paths.update(glob.glob(os.path.join(data_root, "csv_nuclei_pos", "*.csv")))
    for filename in sorted(paths):
        info = os.stat(filename)
        digest.update(f"{filename}:{info.st_size}:{info.st_mtime_ns}".encode())
    fingerprint = digest.hexdigest()[:16]
    cache_path = os.path.join(cache_dir, f"{split}_{fingerprint}_real_cells.npz")
    if os.path.exists(cache_path):
        with np.load(cache_path) as data:
            result = {"X_real": data["X_real"], "marker_feats": data["marker_feats"],
                      "X_pred": None, "cell_ids": data["cell_ids"], "n_tiles": int(data["n_tiles"])}
        print(f"[{split}] loaded shared real-cell cache: {cache_path}")
        return result
    result = collect_cells(data_root, split, pred_dir=None, max_tiles=max_tiles)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(cache_path, X_real=result["X_real"], marker_feats=result["marker_feats"], cell_ids=result["cell_ids"], n_tiles=result["n_tiles"])
    print(f"[{split}] saved shared real-cell cache: {cache_path}")
    return result


def evaluate_per_marker(train_data, test_data, max_iter=1000, classifier_cache=""):
    """per-marker 二分类：对每个 marker，用 mIF 16 通道特征预测「阳性/阴性」。

    训练用真实 mIF 特征（train split），测试用预测 mIF 特征（model）
    或真实 mIF 特征（oracle 上界）。输出每个 marker 的 AUC + F1 + 跨 marker 平均。

    这直接评估「虚拟染色能否还原每个 marker 的表达」，是虚拟免疫染色的核心指标。
    """
    classifiers = None
    if classifier_cache and os.path.exists(classifier_cache):
        import joblib
        cached = joblib.load(classifier_cache)
        scaler, classifiers = cached["scaler"], cached["classifiers"]
        print(f"[cell] loaded shared classifiers: {classifier_cache}")
    else:
        scaler = StandardScaler()  # 各通道强度尺度差异大，先标准化
        X_tr = scaler.fit_transform(train_data["X_real"])
        classifiers = {}
        for k, name in enumerate(MARKER_NAMES):
            y_tr = train_data["marker_feats"][:, k].astype(bool)
            if y_tr.sum() == 0 or y_tr.sum() == len(y_tr):
                continue  # 全阳/全阴的 marker 无法训练（如某些 split 的极稀疏 marker）
            clf = LogisticRegression(max_iter=max_iter, class_weight="balanced", n_jobs=-1)
            clf.fit(X_tr, y_tr)
            classifiers[name] = clf
        if classifier_cache:
            import joblib
            os.makedirs(os.path.dirname(os.path.abspath(classifier_cache)), exist_ok=True)
            joblib.dump({"scaler": scaler, "classifiers": classifiers}, classifier_cache)
            print(f"[cell] saved shared classifiers: {classifier_cache}")

    X_te_real = scaler.transform(test_data["X_real"])
    X_te_pred = scaler.transform(test_data["X_pred"]) if test_data["X_pred"] is not None else None

    per_marker = {}
    for k, name in enumerate(MARKER_NAMES):
        y_tr = train_data["marker_feats"][:, k].astype(bool)
        y_te = test_data["marker_feats"][:, k].astype(bool)
        n_pos_tr, n_pos_te = int(y_tr.sum()), int(y_te.sum())
        entry = {"n_pos_train": n_pos_tr, "n_pos_test": n_pos_te}
        clf = classifiers.get(name)
        if clf is None:
            per_marker[name] = entry
            continue
        # oracle：用真实 mIF 特征测试（与模型无关的上界）
        prob = clf.predict_proba(X_te_real)[:, 1]
        valid_auc = 0 < n_pos_te < len(y_te)
        entry["auc_oracle"] = float(roc_auc_score(y_te, prob)) if valid_auc else None
        entry["f1_oracle"] = float(f1_score(y_te, prob > 0.5, zero_division=0))

        if test_data["X_pred"] is not None:
            # model：用模型预测的 mIF 特征测试（本脚本关注的最终指标）
            prob_m = clf.predict_proba(X_te_pred)[:, 1]
            entry["auc_model"] = float(roc_auc_score(y_te, prob_m)) if valid_auc else None
            entry["f1_model"] = float(f1_score(y_te, prob_m > 0.5, zero_division=0))

        per_marker[name] = entry

    def _mean(key):
        v = [m[key] for m in per_marker.values() if m.get(key) is not None]
        return float(np.mean(v)) if v else None

    return {
        "markers": per_marker,
        "mean_auc_oracle": _mean("auc_oracle"),
        "mean_f1_oracle": _mean("f1_oracle"),
        "mean_auc_model": _mean("auc_model"),
        "mean_f1_model": _mean("f1_model"),
        "n_train_cells": int(train_data["X_real"].shape[0]),
        "n_test_cells": int(test_data["X_real"].shape[0]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_dir", default=None, help="预测 mIF 目录（16 通道 tiff，按 H&E basename 命名）")
    ap.add_argument("--data_root", default=DEFAULT_ROOT)
    ap.add_argument("--train_split", default="val", help="逻辑回归训练 split")
    ap.add_argument("--test_split", default="test")
    ap.add_argument("--max_train_tiles", type=int, default=None)
    ap.add_argument("--max_test_tiles", type=int, default=None)
    ap.add_argument("--feature_cache_dir", default="",
                    help="共享真实细胞特征缓存目录；多模型评估时显著减少重复 I/O")
    ap.add_argument("--out", default="cell_f1.json")
    args = ap.parse_args()

    print(f"收集训练集 ({args.train_split}) 细胞...")
    if args.feature_cache_dir and args.max_train_tiles is None:
        train_data = load_or_build_real_cache(args.data_root, args.train_split,
                                              args.feature_cache_dir)
    else:
        train_data = collect_cells(args.data_root, args.train_split, pred_dir=None,
                                   max_tiles=args.max_train_tiles)
    print(f"收集测试集 ({args.test_split}) 细胞...")
    if args.feature_cache_dir and args.max_test_tiles is None and args.pred_dir is not None:
        test_data = load_or_build_real_cache(args.data_root, args.test_split,
                                             args.feature_cache_dir)
        pred_data = collect_cells(args.data_root, args.test_split, pred_dir=args.pred_dir,
                                  max_tiles=None, load_real=False)
        # 一致性校验：预测特征必须与共享真实特征逐细胞对齐（数量/顺序/标签全同）
        if (pred_data["X_pred"] is None or pred_data["X_pred"].shape[0] != test_data["X_real"].shape[0]
                or not np.array_equal(pred_data["marker_feats"], test_data["marker_feats"])
                or not np.array_equal(pred_data["cell_ids"], test_data["cell_ids"])):
            raise ValueError("Prediction cell order/count does not match shared real-cell cache")
        test_data["X_pred"] = pred_data["X_pred"]
    else:
        test_data = collect_cells(args.data_root, args.test_split, pred_dir=args.pred_dir,
                                  max_tiles=args.max_test_tiles)

    fingerprint = hashlib.sha256(train_data["X_real"].tobytes() + train_data["marker_feats"].tobytes()).hexdigest()[:16]
    classifier_cache = (os.path.join(args.feature_cache_dir, f"marker_classifiers_{fingerprint}.joblib")
                        if args.feature_cache_dir else "")
    result = evaluate_per_marker(train_data, test_data, classifier_cache=classifier_cache)

    def _f(x):
        return f"{x:.3f}" if x is not None else "   -"

    print("\n===== 虚拟免疫染色 per-marker 二分类 =====")
    print(f"训练细胞数: {result['n_train_cells']}, 测试细胞数: {result['n_test_cells']}")
    print(f"{'marker':<12} {'AUC_oracle':>10} {'F1_oracle':>10} {'AUC_model':>10} {'F1_model':>10} {'n_pos_test':>10}")
    for name, m in result["markers"].items():
        print(f"{name:<12} {_f(m.get('auc_oracle')):>10} {_f(m.get('f1_oracle')):>10} "
              f"{_f(m.get('auc_model')):>10} {_f(m.get('f1_model')):>10} {m['n_pos_test']:>10}")
    print("\n跨 marker 平均:")
    print(f"  oracle: AUC={_f(result['mean_auc_oracle'])}  F1={_f(result['mean_f1_oracle'])}")
    print(f"  model : AUC={_f(result['mean_auc_model'])}  F1={_f(result['mean_f1_model'])}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"结果已写 {args.out}")


if __name__ == "__main__":
    main()
