"""Re-render the fixed 24 previews with the original additive fluorescence code.

Reads existing scores only: no training, inference or metric recalculation.
Regression intensity and classification probability are explicitly distinguished.
"""
import base64
import hashlib
import html
import json
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
REPO = PROJECT.parent
sys.path.insert(0, str(REPO))
from scripts.visualize_virtual_stain import multicolor_composite, MARKER_COLORS, GAIN
sys.path.insert(0, str(PROJECT))
from vit_seg.data import CHANNELS, read_he, read_mif

OUT = HERE / 'legacy_style'
MODELS = ['v1', 'v2', 'v3', 'baseline']
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def probability_composite(values):
    """Same historical colors; unit gain for probability/GT-label comparison."""
    rgb = np.zeros((*values.shape[1:], 3), np.float32)
    for name, color in MARKER_COLORS.items():
        rgb += values[CHANNELS.index(name), ..., None] * np.array(color, np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def panel(images, titles, heading, note, size=384, gain_legend=True):
    width = len(images) * size
    canvas = Image.new('RGB', (width, size + 178), '#101216')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(FONT, 17)
    small = ImageFont.truetype(FONT, 14)
    draw.text((12, 8), heading, fill='white', font=font)
    draw.text((12, 34), note, fill='#dfc386', font=small)
    for i, (rgb, title) in enumerate(zip(images, titles)):
        # Enlargement is display-only; saved predictions stay on the 256 grid.
        im = Image.fromarray(rgb).resize((size, size), Image.Resampling.BILINEAR)
        canvas.paste(im, (i * size, 62))
        draw.text((i * size + 10, size + 72), title, fill='white', font=font)
    gap = width // 5
    for i, (name, color) in enumerate(MARKER_COLORS.items()):
        x, y = 10 + (i % 5) * gap, size + 107 + (i // 5) * 25
        draw.rectangle((x, y, x + 13, y + 13), fill=color)
        label = f'{name} x{GAIN[name]:g}' if gain_legend else name
        draw.text((x + 20, y - 2), label, fill='white', font=small)
    return canvas


def main():
    cv2.setNumThreads(1)
    for folder in ['patches', 'probability_pairs', 'rgb']:
        (OUT / folder).mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(HERE / 'preview_manifest.csv')
    if len(frame) != 24 or not frame.groupby('orion_slide_id').size().eq(4).all():
        raise ValueError('Expected the same 24 patches, four per original test section')
    root = Path(json.loads((PROJECT / 'runs/ddp_baseline/data/statistics.json').read_text())['source_spec']['root'])
    source_hashes, cards, pdf_pages, manifest = {}, [], [], []
    for row in frame.itertuples():
        pid = int(row.patch_id)
        arrays = {}
        for name in MODELS:
            path = HERE / name / 'previews' / f'patch_{pid}.npz'
            source_hashes[str(path.relative_to(HERE))] = digest(path)
            with np.load(path) as data:
                arrays[name] = {key: data[key] for key in data.files}
            score = arrays[name]['scores']
            if score.shape != (16, 256, 256) or not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
                raise ValueError(f'Invalid saved scores: {path}')
        mask = arrays['baseline']['tissue'].astype(bool)
        labels = arrays['baseline']['labels']
        available = (labels != 255).any((1, 2))
        he = cv2.resize(read_he(root / row.image_path), (256, 256), interpolation=cv2.INTER_AREA)
        gt = read_mif(root / row.target_path).astype(np.float32)
        gt[~available] = 0
        # Match original renderer: compose native GT first, then resize RGB.
        gt_rgb = cv2.resize(multicolor_composite(gt), (256, 256), interpolation=cv2.INTER_AREA)
        gt_rgb[~mask] = 0
        images, titles = [he, gt_rgb], ['H&E', 'GT mIF intensity']
        rgb_maps = {'he': he, 'gt': gt_rgb}
        for name in MODELS:
            rgb = multicolor_composite(arrays[name]['scores'] * 255.)
            rgb[~mask] = 0
            rgb_maps[name] = rgb
            images.append(rgb)
            titles.append(f'{name}: intensity' if name != 'baseline' else 'baseline: probability x gain')
        note = ('CRC02: shared unseen test patient' if row.orion_slide_id == 'CRC02' else
                'Exploratory: legacy models trained on this patient')
        if row.orion_slide_id.startswith('CRC33'):
            note += '; baseline trained on another CRC33 section'
        note += ' | Additive fluorescence; classifier brightness is NOT predicted mIF intensity'
        heading = f'{row.orion_slide_id} / patch {pid} | Original 10-marker colors and gains | fixed case'
        rendered = panel(images, titles, heading, note)
        name = f'{row.orion_slide_id}_patch_{pid}.png'
        rendered.save(OUT / 'patches' / name)
        pdf_pages.append(rendered)
        # A separate matched probability/label display avoids implying p*255 is intensity.
        gt_prob = probability_composite((labels == 1).astype(np.float32))
        pred_prob = probability_composite(arrays['baseline']['scores'])
        gt_prob[~mask] = 0
        pred_prob[~mask] = 0
        paired = panel([he, gt_prob, pred_prob], ['H&E', 'GT binary expression', 'baseline: sigmoid probability'],
                       f'{row.orion_slide_id} / patch {pid} | Classifier probability view',
                       'Same 10 colors; unit gain; no threshold or argmax; overlapping signals mix colors', gain_legend=False)
        paired.save(OUT / 'probability_pairs' / name)
        for key, rgb in rgb_maps.items():
            if key != 'he' and np.any(rgb[~mask]):
                raise AssertionError('Nonzero glass background')
            Image.fromarray(rgb).save(OUT / 'rgb' / f'patch_{pid}_{key}.png')
        encoded = base64.b64encode((OUT / 'patches' / name).read_bytes()).decode('ascii')
        probability_encoded = base64.b64encode((OUT / 'probability_pairs' / name).read_bytes()).decode('ascii')
        cards.append(f'<article><h2>{html.escape(row.orion_slide_id)} / patch {pid}</h2>'
                     f'<a href="patches/{name}">下载原尺寸对照PNG</a><img src="data:image/png;base64,{encoded}" alt="多色荧光对照">'
                     f'<details><summary>分类基线：GT标签与概率对照（无增益）</summary>'
                     f'<img src="data:image/png;base64,{probability_encoded}" alt="标签与概率对照"></details></article>')
        manifest.append(dict(patch_id=pid,section=row.orion_slide_id,image_path=row.image_path,target_path=row.target_path,
                             comparison=f'patches/{name}',probability_pair=f'probability_pairs/{name}'))
        print(f'Rendered {row.orion_slide_id} / {pid}', flush=True)
    pdf_pages[0].save(OUT / 'patch_comparisons.pdf', save_all=True, append_images=pdf_pages[1:], resolution=150.)
    pd.DataFrame(manifest).to_csv(OUT / 'patch_manifest.csv', index=False)
    protocol = dict(patches=24,sections=frame.groupby('orion_slide_id').size().to_dict(),models=MODELS,
        rendering='exact original multicolor_composite: sum RGB_color * clip(intensity/255 * gain,0,1), RGB clipped at 255',
        colors=MARKER_COLORS,gains=GAIN,omitted_channels=[c for c in CHANNELS if c not in MARKER_COLORS],
        argmax=False,probability_threshold=None,per_patch_or_model_auto_contrast=False,
        regression_input='stored inverse-transformed intensity scores * 255',
        classifier_input='stored sigmoid probability * 255 ONLY for legacy display; not reconstructed intensity',
        probability_pairs='GT binary labels and sigmoid probabilities, same historical colors, gains=1',
        background='same H&E tissue mask for GT/predictions; no DAPI or GT positivity gating of predictions',
        gt_geometry='native GT intensity -> legacy RGB composite -> area resize to 256',
        score_geometry='saved common 256 grid, RGB enlarged to 384 for display only',
        metrics_or_training_changed=False,new_inference=False,
        selected_checkpoint='baseline ddp_baseline best, not ongoing ddp_positive_dice',
        renderer_source_sha256=digest(REPO / 'scripts/visualize_virtual_stain.py'),
        preview_manifest_sha256=digest(HERE / 'preview_manifest.csv'),source_score_sha256=source_hashes)
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    readme = '''# 旧版多色荧光patch可视化

已按v1/v2/v3原始multicolor_composite函数重新生成：固定10通道颜色、固定GAIN、逐通道相加并裁剪RGB。没有argmax、概率阈值或逐图自动增强；共表达会混色。

沿用之前固定的6张测试切片、每张4个patch，共24张。主图为H&E | GT真实强度 | v1 | v2 | v3 | 分类基线。前三版为还原后的强度，分类基线为概率，后者的亮度不能解释为mIF强度。为便于核对，另附GT二值标签与分类概率的无增益配色图。所有图使用相同的H&E玻片背景掩膜，不用DAPI裁掉核外信号。

旧版函数仅展示10个marker，其余CD31、CD45、CD45RO、PDL1、ECadherin、Ki67没有加入合成图。这是忠实复用历史显示规则，16通道预测数据仍保留在原评估目录。

CRC02是四版本共同未训练患者；其他切片仅作定性查看，旧模型见过这些患者，CRC33还涉及旧基线另一切片的训练暴露。这里的分类基线是已完成的ddp_baseline，未替换成正在训练的阳性Dice模型。

文件：index.html为独立可浏览图册；patches/为24张六栏对照PNG；rgb/为各面板256像素PNG；probability_pairs/为24张标签/概率对照；patch_comparisons.pdf为24页图册。patch_manifest.csv记录全部patch身份，protocol.json记录配色、参数与预测输入哈希。

只重新渲染已有预测，没有重新推理，没有改变指标或正在进行的训练。
'''
    (OUT / 'README.md').write_text(readme)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原版多色荧光patch对照</title>
<style>body{font:16px/1.7 sans-serif;margin:24px;background:#11141a;color:#eee}a{color:#86c9ff}img{width:100%;height:auto}article{padding:18px;margin:24px 0;background:#202530}details{margin-top:16px}p{max-width:1200px}.note{color:#efc98a}</style>
<h1>原版多色荧光：24张固定patch对照</h1>
<p>每测试切片4张。复用v1/v2/v3原函数、10通道配色和固定增益，保留共表达混色。每张主图依次为H&amp;E、GT、v1、v2、v3、已完成的分类基线。</p>
<p class="note">分类基线输出概率，主图中的亮度不能与回归强度直接比较。展开各图下方可查看同配色、无增益的GT标签/分类概率对照。CRC02为共同未训练患者，其他切片仅供定性展示。未改变训练及评估指标。</p>
<p><a href="patch_comparisons.pdf">下载24页PDF</a> · <a href="patch_images.zip">下载PNG图片包</a> · <a href="protocol.json">配色与生成记录</a></p>'''
    (OUT / 'index.html').write_text(page + ''.join(cards) + '</html>')
    with zipfile.ZipFile(OUT / 'patch_images.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for folder in ['patches', 'probability_pairs', 'rgb']:
            for p in sorted((OUT / folder).glob('*.png')):
                z.write(p, p.relative_to(OUT))
        for name in ['README.md', 'protocol.json', 'patch_manifest.csv']:
            z.write(OUT / name, name)
    for relative, expected in source_hashes.items():
        if digest(HERE / relative) != expected:
            raise AssertionError(f'Source scores changed during rendering: {relative}')
    (OUT / 'COMPLETE.json').write_text(json.dumps(dict(patches=24,comparison_pngs=24,probability_pairs=24,
        standalone_rgb_pngs=144,source_scores_unchanged=True,gpu_used=False,training_changed=False), indent=2))
    print(f'COMPLETE: {OUT}', flush=True)


if __name__ == '__main__':
    main()
