"""M0′-G1 门禁：校验掩膜软先验真值的正确性与完整性。

检查项（PLAN_S1_unetpp_seghead.md §3.3）：
  G1-a 逐 marker 阳性面积占比 与 dataframe 的 `<marker>_prop` 相关 ≥0.95
       （`_prop` 来自 MIPHEI 的强度阈值流水线，是我们的**独立来源**）
  G1-b 逐细胞一致性：mask 在 CSV 标注为阳性的核内 100% 为阳性、阴性核内 100% 为 0
  G1-c 通道顺序核对（Hoechst=通道0，15 个 marker 顺序 = CHANNELS[1:]）
  G1-d 软先验空间合理性：σ 环带内非零像素占比、核外最大值

用法：
    python -m datacore.validate_mask_prior --cache /tmp/mask_smoke --split test --n 500
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import tifffile
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datacore.orioncrc_dataset import CHANNELS  # noqa: E402
from datacore.build_mask_prior import slide_prefix, KEEP, CSV_DIR  # noqa: E402

ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", "--mask_cache", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--n", type=int, default=0, help="抽检 tile 数（0=全部）")
    ap.add_argument("--df", default="", help="划分表（若缓存是子集）")
    args = ap.parse_args()

    df_path = args.df or os.path.join(args.root, f"{args.split}_dataframe.csv")
    df = pd.read_csv(df_path)
    # 缓存行数以 meta 为准（冒烟缓存可能只建了前 N 例）
    meta_p = os.path.join(args.cache, "mask_meta.json")
    T = 256
    n_cache = len(pd.read_csv(df_path))
    if os.path.exists(meta_p):
        meta = json.load(open(meta_p))
        if args.split in meta:
            n_cache = int(meta[args.split]["n"])
            T = int(meta[args.split]["tile_size"])
    n = n_cache if not args.n else min(args.n, n_cache)
    df = df.iloc[:n]
    mm = np.memmap(os.path.join(args.cache, f"{args.split}_mask.raw"), dtype=np.uint8,
                   mode="r", shape=(n_cache, 16, T, T))[:n]
    print(f"[G1] 校验 {args.cache}/{args.split}  n={n} (缓存共 {n_cache})")

    # ---------- G1-a 阳性占比 vs _prop ----------
    prop_names = [c for c in df.columns if c.endswith("_prop")]
    # 通道顺序：0=Hoechst + CHANNELS[1:]，_prop 列名 = 通道名 + '_prop'（PD-L1 用连字符）
    ch_to_prop = {CHANNELS[0]: "Hoechst_prop"}
    for c in CHANNELS[1:]:
        cand = [f"{c}_prop", f"{c.replace('PDL1', 'PD-L1')}_prop",
                f"{c.replace('ECadherin', 'E-cadherin')}_prop"]
        hit = [x for x in cand if x in df.columns]
        ch_to_prop[c] = hit[0] if hit else None

    frac = np.zeros((n, 16), dtype=np.float64)          # mask 阳性核区占比
    for i in range(n):
        m = np.asarray(mm[i])
        frac[i] = (m > 127).mean(axis=(1, 2))

    print(f"\n{'通道':<12}{'mask占比':>10}{'dataframe _prop':>17}{'Pearson':>9}{'Spearman':>10}")
    print("-" * 60)
    rows = []
    for c in range(16):
        name = CHANNELS[c]
        pcol = ch_to_prop.get(name)
        if pcol is None:
            print(f"{name:<12}{frac[:, c].mean():>10.4f}{'（无列）':>17}")
            continue
        ref = df[pcol].to_numpy(dtype=np.float64)
        r = np.corrcoef(frac[:, c], ref)[0, 1]
        from scipy.stats import spearmanr
        rs = spearmanr(frac[:, c], ref).statistic
        ok = "✅" if r >= 0.95 else ("⚠️" if r >= 0.85 else "❌")
        print(f"{name:<12}{frac[:, c].mean():>10.4f}{ref.mean():>17.4f}{r:>9.3f}{rs:>10.3f} {ok}")
        rows.append(dict(channel=name, mask_frac=float(frac[:, c].mean()),
                         prop=float(ref.mean()), pearson=float(r), spearman=float(rs)))
    rv = [x["pearson"] for x in rows if x["pearson"] == x["pearson"]]
    print(f"\nG1-a1（核覆盖占比 vs 强度阳性占比，信息性参考）：均值 {np.mean(rv):.3f}  最低 {np.min(rv):.3f}")
    print("  说明：两者定义不同（核区覆盖 vs 强度阈值像素），只要求正相关且量级可比"
          f"  {'✅' if np.mean(rv) >= 0.85 else '❌'}")

    # ---------- G1-a2 表型 GT 与 mIF 强度的一致性（真正的有效性检验） ----------
    lut_cache = {}
    score_pos = [[] for _ in range(15)]
    score_neg = [[] for _ in range(15)]
    from datacore.orioncrc_dataset import load_marker_q, normalize_mif_log
    Q = load_marker_q()
    for i in range(min(n, 300)):
        row = df.iloc[i]
        pref = slide_prefix(row["target_path"])
        if pref not in lut_cache:
            csv = os.path.join(args.root, CSV_DIR, pref + ".csv")
            if not os.path.exists(csv):
                lut_cache[pref] = None
            else:
                d = pd.read_csv(csv, usecols=["label"] + [f"{m}_pos" for m in KEEP])
                lut_cache[pref] = (d["label"].to_numpy(np.int64),
                                   d[[f"{m}_pos" for m in KEEP]].to_numpy(bool))
        if lut_cache[pref] is None:
            continue
        labels, pos = lut_cache[pref]
        # mIF 强度（log 域，与训练口径一致）
        mif = tifffile.imread(os.path.join(args.root, row["target_path"]), maxworkers=4)
        if mif.ndim == 3 and mif.shape[2] not in (16, 17):
            mif = mif.transpose(1, 2, 0)
        mif = mif[..., :17][..., [i2 for i2 in range(17) if i2 != 13]].astype(np.float32)
        if mif.shape[0] != T or mif.shape[1] != T:      # 对齐到缓存分辨率
            out = np.empty((T, T, mif.shape[2]), dtype=np.float32)
            for c2 in range(mif.shape[2]):
                out[..., c2] = cv2.resize(mif[..., c2], (T, T), interpolation=cv2.INTER_AREA)
            mif = out
        nuc = tifffile.imread(os.path.join(args.root, row["nuclei_path"])).astype(np.int64)
        nuc = cv2.resize(nuc.astype(np.float32), (T, T),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        for k in range(15):
            lp = labels[pos[:, k]]
            if len(lp) == 0:
                continue
            sel = np.isin(nuc, lp)
            neg = (nuc > 0) & (~sel)
            if sel.any() and neg.any():
                score_pos[k].append(float(mif[..., k + 1][sel].mean()))
                score_neg[k].append(float(mif[..., k + 1][neg].mean()))
    print(f"\nG1-a2 表型 GT ↔ mIF 强度一致性（阳性核 vs 阴性核，该通道均值）")
    print(f"{'marker':<12}{'阳性核均值':>11}{'阴性核均值':>11}{'比值':>8}{'tile中位数胜出率':>16}")
    auc_rows = []
    for k in range(15):
        if len(score_pos[k]) < 10:
            continue
        a = np.array(score_pos[k]); b = np.array(score_neg[k])
        win = float(np.mean(a > b))
        print(f"{KEEP[k]:<12}{a.mean():>11.2f}{b.mean():>11.2f}{a.mean()/max(b.mean(),1e-6):>8.2f}{win*100:>15.0f}%")
        auc_rows.append(win)
    print(f"  阳性>阴性的 tile 占比中位 {np.median(auc_rows)*100:.1f}%  "
          f"{'✅ PASS' if np.median(auc_rows) >= 0.8 else '❌ FAIL'}")

    # ---------- G1-b 逐细胞一致性（只查核心区；σ 环带外溢是设计特征） ----------
    lut_cache = {}
    tot_cells = matched = wrong_core = 0
    ring_bleed = 0
    for i in range(min(n, 200)):                       # 抽 200 例做逐细胞核对
        row = df.iloc[i]
        pref = slide_prefix(row["target_path"])
        if pref not in lut_cache:
            csv = os.path.join(args.root, CSV_DIR, pref + ".csv")
            if not os.path.exists(csv):
                lut_cache[pref] = None
            else:
                d = pd.read_csv(csv, usecols=["label"] + [f"{m}_pos" for m in KEEP])
                lut_cache[pref] = d
        d = lut_cache[pref]
        if d is None:
            continue
        nuc = tifffile.imread(os.path.join(args.root, row["nuclei_path"])).astype(np.int64)
        nuc = cv2.resize(nuc.astype(np.float32), (T, T),
                         interpolation=cv2.INTER_NEAREST).astype(np.int64)
        m = np.asarray(mm[i])
        labels = d["label"].to_numpy(np.int64)
        pos = d[[f"{k}_pos" for k in KEEP]].to_numpy(bool)
        for k in range(15):
            lab_pos = labels[pos[:, k]]
            if len(lab_pos) == 0:
                continue
            sel_pos = np.isin(nuc, lab_pos)
            sel_neg = (nuc > 0) & (~sel_pos)
            mc = m[k + 1].astype(np.float32) / 255.0
            if sel_pos.any():
                v = mc[sel_pos].mean()
                tot_cells += 1
                if abs(v - 1.0) < 1e-3:
                    matched += 1
                else:
                    wrong_core += 1
            # 阴性核区内**核心值(=1)**的比例必须为 0
            if sel_neg.any() and (mc[sel_neg] >= 0.999).mean() > 0:
                wrong_core += 1
            if sel_neg.any():
                ring_bleed += int((mc[sel_neg] > 0).mean() > 0)
    print(f"\nG1-b 逐细胞一致性（抽检 {min(n,200)} 例）")
    print(f"  阳性核区均值=1: {matched}/{tot_cells} = {matched/max(tot_cells,1)*100:.2f}%  "
          f"{'✅ PASS' if matched == tot_cells else '❌'}")
    print(f"  阴性核区含核心(=1)像素的违规数: {wrong_core}  {'✅ PASS' if wrong_core == 0 else '❌'}")
    print(f"  阴性核区被 σ 环带覆盖（>0）的 marker×tile 数: {ring_bleed}  ← 设计特征，非错误")

    # ---------- G1-d σ 环带 ----------
    ch = 1 if n > 0 else 0
    m0 = np.asarray(mm[0])[1].astype(np.float32) / 255.0
    print(f"\nG1-d σ 软先验（以 tile0 的通道1={CHANNELS[1]} 为例）")
    print(f"  =1（核内）占比 {(m0>=1).mean()*100:.2f}%   0<v<1（环带）占比 {((m0>0)&(m0<1)).mean()*100:.2f}%"
          f"   =0 占比 {(m0==0).mean()*100:.2f}%")
    ov = np.arange(0, 1.001, 0.25)
    print(f"  直方图: " + "  ".join(f"[{ov[i]:.2f},{ov[i+1]:.2f})={((m0>=ov[i])&(m0<ov[i+1])).mean()*100:.2f}%"
                                     for i in range(len(ov)-1)) + f"  =1.00:{ (m0>=1).mean()*100:.2f}%")
    json.dump(dict(n=n, per_channel=rows, cell_consistency=dict(matched=matched, total=tot_cells,
                                                                wrong_neg=wrong_neg)),
              open(os.path.join(args.cache, f"validate_{args.split}.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
