"""批量生成多 patch 的 4 模型虚拟 mIF 对比图。

控制变量设计：
- 同一 tile：不同模型对同一个 patch 的虚拟染色对比
- 同一配色（MARKER_COLORS）+ 同一信号增益（GAIN）
- 每个 marker 对应一种颜色（多色荧光复合）
- 每张图：H&E 原图 | GT 真实 mIF | vit512 | pix2pixHD | pix2pix | I2SB

用法：
    python scripts/make_patch_comparisons.py --n 8 --out /data/weiyh/results/visualization/patch_comparisons

说明：复用 visualize_virtual_stain 的默认模型清单与渲染函数；
--model NAME=DIR 可重复传入覆盖模型（如换成 v1/v2/v3 的预测目录）。
"""
import os
import sys
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from scripts.visualize_virtual_stain import MODELS, make_patch_panel

DATA_ROOT = "/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x"

# 各免疫 marker 的 tile 选择（覆盖不同免疫微环境）
IMMUNE_MARKERS = {
    "CD4_T": "CD4_count",        # CD4+ T 辅助细胞
    "CD8a_T": "CD8a_count",      # CD8+ T 杀伤细胞
    "Treg": "FOXP3_count",       # Treg
    "Bcell": "CD20_count",       # B 细胞
    "Macro_CD68": "CD68_count",  # 巨噬细胞
    "Macro_CD163": "CD163_count",# M2 巨噬细胞
}


def select_tiles(n_per_marker=1, n_total=None):
    """按不同免疫 marker 选 top tiles（去重）。"""
    df = pd.read_csv(os.path.join(DATA_ROOT, "test_dataframe.csv"))
    selected = {}
    for label, col in IMMUNE_MARKERS.items():
        sub = df.sort_values(col, ascending=False)
        cnt = 0
        for _, r in sub.iterrows():
            tile = os.path.basename(r["image_path"])
            if tile in selected:
                continue
            selected[tile] = label
            cnt += 1
            if cnt >= n_per_marker:
                break
    # 总免疫信号 top（若 n_total 更大则补充）
    if n_total is not None and len(selected) >= n_total:
        return dict(list(selected.items())[:n_total])
    if n_total:
        df["pos_total"] = sum(df[c].fillna(0) for c in IMMUNE_MARKERS.values())
        for _, r in df.sort_values("pos_total", ascending=False).iterrows():
            tile = os.path.basename(r["image_path"])
            if tile in selected:
                continue
            selected[tile] = "Total_immune"
            if len(selected) >= n_total:
                break
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="生成 patch 数")
    ap.add_argument("--size", type=int, default=384, help="每格边长(px)")
    ap.add_argument("--out", default="/data/weiyh/results/visualization/patch_comparisons")
    ap.add_argument("--model", action="append", default=[], help="NAME=DIR，可重复")
    args = ap.parse_args()

    if args.model:
        MODELS.clear()
        for spec in args.model:
            if "=" not in spec:
                ap.error(f"--model 需要 NAME=DIR，收到 {spec}")
            name, path = spec.split("=", 1)
            MODELS[name] = path

    os.makedirs(args.out, exist_ok=True)
    selected = select_tiles(n_total=args.n)
    print(f"[patch] 选中 {len(selected)} 个 patch:")

    for i, (tile, label) in enumerate(selected.items()):
        panel = make_patch_panel(tile, size=args.size)
        if panel is None:
            print(f"[warn] {label}: 无预测，跳过")
            continue
        # 文件名：编号_免疫类型_tile短名
        short = tile[:45].replace("/", "_")
        fname = os.path.join(args.out, f"patch_{i:02d}_{label}_{short}.png")
        from PIL import Image
        Image.fromarray(panel).save(fname)
        print(f"  [{i:02d}] {label}: {tile[:60]} -> {fname}")

    print(f"[patch] 完成，输出目录: {args.out}")


if __name__ == "__main__":
    main()
