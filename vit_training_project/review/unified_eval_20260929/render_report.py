"""Publish reproducible tables and fixed-case argmax panels from saved scores."""
import base64
import html
import json
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
from vit_seg.data import CHANNELS, read_he, read_mif
from vit_seg.display import PALETTE, MARKER_COLORS, argmax_display
OUT=Path(__file__).resolve().parent
BASE=PROJECT/'runs/ddp_baseline'
MODELS=['v1','v2','v3','baseline']


def main():
    stats=json.loads((BASE/'data/statistics.json').read_text())
    root=Path(stats['source_spec']['root']);q=np.array(stats['q'])[:,None,None]
    previews=pd.read_csv(OUT/'preview_manifest.csv');(OUT/'figures').mkdir(exist_ok=True)
    summary=[];channels=[]
    for name in MODELS:
        if not (OUT/name/'COMPLETE.json').is_file():raise RuntimeError(f'{name} evaluation incomplete')
        metrics=json.loads((OUT/name/'common_test_metrics.json').read_text())
        summary.append(dict(model=name,patches=6638,patients=1,**metrics['macro'],micro_f1=metrics['micro_f1'],
                            hamming_accuracy=metrics['hamming_accuracy']))
        for channel,values in metrics['per_class'].items():
            channels.append(dict(model=name,channel=channel,**{k:v for k,v in values.items() if k!='confusion_matrix'}))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'common_test_summary.csv',index=False)
    pd.DataFrame(channels).to_csv(OUT/'common_test_per_channel.csv',index=False)
    baseline_rows=[]
    for scope,patches in [('all_test',35153),('clean_test',31747),('common_test',6638)]:
        m=json.loads((OUT/'baseline'/f'{scope}_metrics.json').read_text())
        baseline_rows.append(dict(scope=scope,patches=patches,**m['macro'],micro_f1=m['micro_f1'],hamming_accuracy=m['hamming_accuracy']))
    baseline=pd.DataFrame(baseline_rows);baseline.to_csv(OUT/'baseline_test_scopes.csv',index=False)
    full=json.loads((OUT/'baseline/all_test_metrics.json').read_text())
    pd.DataFrame(full['per_class']).T.to_csv(OUT/'baseline_all_test_per_channel.csv')
    clean=json.loads((OUT/'baseline/clean_test_metrics.json').read_text())
    pd.DataFrame(clean['per_class']).T.to_csv(OUT/'baseline_clean_test_per_channel.csv')
    pd.DataFrame([dict(patient=k,**v['macro']) for k,v in json.loads((OUT/'baseline/per_patient.json').read_text()).items()]).to_csv(OUT/'baseline_per_patient.csv',index=False)
    thresholds=[]
    for name in MODELS:
        th=json.loads((OUT/name/'thresholds.json').read_text())
        for row in th['per_channel']:
            thresholds.append(dict(model=name,**{k:v for k,v in row.items() if k not in ['grid','val_f1']}))
    pd.DataFrame(thresholds).to_csv(OUT/'validation_selected_thresholds.csv',index=False)
    palette={str(i):dict(channel='excluded/no signal' if i==0 else CHANNELS[i-1],rgb=rgb.tolist()) for i,rgb in enumerate(PALETTE)}
    (OUT/'palette.json').write_text(json.dumps(palette,indent=2))
    legend=[Patch(color=MARKER_COLORS[c],label=name) for c,name in enumerate(CHANNELS)]
    for row in previews.itertuples():
        pid=int(row.patch_id);he=cv2.resize(read_he(root/row.image_path),(256,256),interpolation=cv2.INTER_AREA)
        mif=read_mif(root/row.target_path)
        mif=np.stack([cv2.resize(a.astype(np.float32),(256,256),interpolation=cv2.INTER_NEAREST) for a in mif])
        predictions={name:np.load(OUT/name/f'previews/patch_{pid}.npz') for name in MODELS}
        tissue=predictions['baseline']['tissue'].astype(bool)
        available=(predictions['baseline']['labels']!=255).any((1,2))
        gt,gt_rgb=argmax_display(mif/q,tissue,available)
        panels=[('H&E',he),('GT: dominant normalized intensity',gt_rgb)]
        masks={'gt':gt}
        for name in MODELS:
            scores=predictions[name]['scores']
            # Regression intensities are normalized by the SAME train-only q
            # as GT. The classifier argmax is over its requested probabilities.
            values=scores if name=='baseline' else scores*255/q
            label,rgb=argmax_display(values,tissue)
            masks[name]=label
            panels.append((name+(': probability argmax' if name=='baseline' else ': intensity argmax'),rgb))
            Image.fromarray(rgb).save(OUT/f'figures/patch_{pid}_{name}_argmax.png')
        Image.fromarray(gt_rgb).save(OUT/f'figures/patch_{pid}_gt_argmax.png')
        np.savez_compressed(OUT/f'figures/patch_{pid}_argmax_labels.npz',**masks,palette=PALETTE)
        fig,axes=plt.subplots(2,3,figsize=(12,9.8))
        for ax,(title,image) in zip(axes.flat,panels):ax.imshow(image);ax.set_title(title,fontsize=11);ax.axis('off')
        note='COMMON UNSEEN TEST CASE' if row.orion_slide_id=='CRC02' else 'EXPLORATORY: legacy models saw this case during training'
        if row.orion_slide_id.startswith('CRC33'):note+='; baseline also saw another CRC33 section'
        fig.suptitle(f'{row.orion_slide_id} / patch {pid}\n{note}',fontsize=11)
        fig.legend(handles=legend,loc='lower center',ncol=8,fontsize=8)
        fig.subplots_adjust(left=.025,right=.975,bottom=.10,top=.88,hspace=.20,wspace=.08)
        fig.savefig(OUT/f'figures/patch_{pid}_four_versions.png',dpi=160);plt.close(fig)
        for value in predictions.values():value.close()
        print('rendered',pid,flush=True)
    ax=summary.set_index('model')[['f1','iou','average_precision_histogram','auroc_histogram']].plot.bar(figsize=(9,4.5),ylim=(0,1))
    ax.set_title('Shared unseen CRC02: 6,638 patches; own-validation thresholds')
    ax.set_ylabel('Score');plt.xticks(rotation=0);plt.tight_layout();plt.savefig(OUT/'figures/common_test_metrics.png',dpi=170);plt.close()
    cards=[]
    for row in previews.itertuples():
        image_path=OUT/f'figures/patch_{row.patch_id}_four_versions.png'
        # Embed the preview so index.html also works in viewers that do not expose
        # sibling files (for example, sandboxed editor and artifact previews).
        image_data=base64.b64encode(image_path.read_bytes()).decode('ascii')
        cards.append(
            f'<article><h3>{html.escape(row.orion_slide_id)} / patch {row.patch_id}</h3>'
            f'<a href="figures/patch_{row.patch_id}_four_versions.png">'
            f'<img loading="lazy" width="1920" height="1568" '
            f'alt="{html.escape(row.orion_slide_id)} / patch {row.patch_id}" '
            f'src="data:image/png;base64,{image_data}"></a></article>'
        )
    page='''<!doctype html><meta charset="utf-8"><title>ViT统一评估与同patch对照</title>
<style>body{max-width:1280px;margin:32px auto;font:16px/1.7 sans-serif;color:#222;background:#fafafa}table{border-collapse:collapse;width:100%;background:white}td,th{padding:8px;border:1px solid #ddd}img{max-width:100%}article{background:white;padding:16px;margin:24px 0}.notice{padding:18px;background:#fff0cc}a{color:#145b96}</style>
<h1>ViT统一评估与同patch对照</h1>
<p class="notice">正式四版对比只使用共同未训练的CRC02（6,638张patch），只有1例，不能据此推断总体泛化排名。其余切片可视化仅供定性查看：旧版曾训练这些病例。CRC33两切片还存在旧基线内部的患者泄漏，当前基线严格测试需排除CRC33。</p>
<h2>共同测试集：统一多标签分类指标</h2><p>相同16通道、GT&gt;0标签、256像素网格、背景/未知忽略、逐通道聚合；阈值均仅在模型自身未训练过的验证集选择，验证病例不同。旧版强度不是概率；ROC/AP为256-bin近似。PSNR/SSIM不参与比较。</p>'''
    page+=summary.round(5).to_html(index=False,border=0)
    page+='<h2>当前基线：完整测试与去泄漏测试</h2>'+baseline.round(5).to_html(index=False,border=0)
    page+='''<h2>颜色与argmax定义</h2><p>每个通道固定一种颜色；每像素只显示最高分通道。GT用训练集q归一化后的强度argmax；v1/v2/v3用还原强度并除以同一q；当前模型用独立sigmoid概率argmax，不加0.5筛选。GT二值共表达标签不能直接argmax，否则并列值会偏向排在前面的通道。颜色图是有损展示，分类指标继续使用完整多标签。图中预测只按H&amp;E组织掩膜排除玻片背景，不利用GT裁切；全通道GT为零的像素不参与指标。</p>
<p><a href="common_test_summary.csv">四版本汇总CSV</a> · <a href="common_test_per_channel.csv">四版本逐通道CSV</a> · <a href="baseline_clean_test_per_channel.csv">当前模型去泄漏逐通道CSV</a> · <a href="baseline_per_patient.csv">逐切片指标</a> · <a href="validation_selected_thresholds.csv">验证阈值</a> · <a href="palette.json">颜色表</a> · <a href="protocol.json">评估口径</a></p>'''
    page+=''.join(cards)+'</body></html>'
    (OUT/'index.html').write_text(page)
    (OUT/'VISUALIZATION_COMPLETE.json').write_text(json.dumps(dict(patches=len(previews),models=MODELS,
        selection='4 evenly spaced patch IDs per baseline test section; fixed before predictions',
        score_metrics_use_argmax=False),indent=2))
    print(summary.to_string(index=False));print(baseline.to_string(index=False))


if __name__=='__main__':main()
