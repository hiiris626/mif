"""Render matched GT/model panels for the diverse CRC02 patch set."""
from pathlib import Path
import json,sys,html
import cv2
import numpy as np
import pandas as pd
import tifffile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

P=Path(__file__).resolve().parent;PROJECT=P.parents[1]
sys.path.insert(0,str(PROJECT/'experiments/v6_v7_v8_emptydice'))
from vit_seg.data import CHANNELS,read_he,read_mif
from vit_seg.display import PALETTE,MARKER_COLORS,argmax_display
MODELS=['v1','v2','v3','v4','v5','v6','v9','positive_dice_partial']
DISPLAY={'positive_dice_partial':'positive-Dice partial'}
ROOT=Path('/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x')

def gt_argmax(binary,mif,tissue,available):
 positive=(binary==1)&available[:,None,None]
 labels=(np.where(positive,mif.astype(np.float32),-1).argmax(0)+1).astype(np.uint8)
 labels[~tissue|~positive.any(0)]=0
 return labels,PALETTE[labels]

def composite(values,tissue):
 image=np.zeros((*tissue.shape,3),np.float32)
 for c in range(16):image+=np.clip(values[c],0,1)[...,None]*MARKER_COLORS[c]
 image=np.clip(image,0,1);image[~tissue]=0
 return image

def thresholds(model,policy):
 if policy=='fixed_05':return [.5]*16
 complete=json.loads((P/model/'COMPLETE.json').read_text());path=complete['threshold_path']
 return json.loads(Path(path).read_text())['thresholds'] if path else [.5]*16

def main():
 manifest=pd.read_csv(P/'preview_manifest.csv');figures=P/'figures';figures.mkdir(exist_ok=True)
 legend=[Patch(color=MARKER_COLORS[i],label=name) for i,name in enumerate(CHANNELS)]
 inventory=[]
 for row in manifest.itertuples():
  pid=int(row.patch_id);arrays={m:np.load(P/m/'previews'/f'patch_{pid}.npz') for m in MODELS}
  ref=arrays['v6'];tissue=ref['tissue'].astype(bool);available=ref['available'].astype(bool);binary=ref['expanded']
  he=cv2.resize(read_he(ROOT/row.image_path),(256,256),interpolation=cv2.INTER_AREA)
  mif=read_mif(ROOT/row.target_path);mif=np.stack([cv2.resize(x,(256,256),interpolation=cv2.INTER_NEAREST) for x in mif])
  gt_label,gt_rgb=gt_argmax(binary,mif,tissue,available)
  Image.fromarray(gt_rgb).save(figures/f'patch_{pid}_gt_argmax.png')
  for policy in ['fixed_05','own_validation']:
   panels=[('H&E',he),('GT dominant positive',gt_rgb)];saved={'gt':gt_label}
   for model in MODELS:
    label,rgb=argmax_display(arrays[model]['scores'],tissue,available,thresholds(model,policy))
    panels.append((DISPLAY.get(model,model),rgb));saved[model]=label
    Image.fromarray(rgb).save(figures/f'patch_{pid}_{model}_{policy}_argmax.png')
   np.savez_compressed(figures/f'patch_{pid}_{policy}_argmax_labels.npz',**saved,palette=PALETTE)
   fig,axes=plt.subplots(2,5,figsize=(19,8.3))
   for ax,(title,image) in zip(axes.flat,panels):ax.imshow(image);ax.set_title(title,fontsize=10);ax.axis('off')
   fig.suptitle(f'CRC02 patch {pid} · {row.selection} · '+('fixed threshold 0.5' if policy=='fixed_05' else 'each model validation threshold'),fontsize=12)
   fig.legend(handles=legend,loc='lower center',ncol=8,fontsize=8)
   fig.subplots_adjust(left=.015,right=.985,bottom=.105,top=.91,hspace=.12,wspace=.035)
   fig.savefig(figures/f'patch_{pid}_all_models_{policy}.png',dpi=155);plt.close(fig)
  # Complete multi-label/probability view: every marker retains its own plane.
  fig,axes=plt.subplots(4,4,figsize=(24,13))
  for c,ax in enumerate(axes.flat):
   strips=[(binary[c]==1).astype(float)]+[arrays[m]['scores'][c] for m in MODELS[:-1]]
   ax.imshow(np.hstack(strips),cmap='magma',vmin=0,vmax=1,interpolation='nearest')
   ax.set_title(CHANNELS[c]+' · GT | v1 | v2 | v3 | v4 | v5 | v6 | v9',fontsize=9);ax.axis('off')
  fig.suptitle(f'CRC02 patch {pid} · binary GT and score maps · fixed scale 0–1',fontsize=12)
  fig.tight_layout();fig.savefig(figures/f'patch_{pid}_all_channels_scores.png',dpi=155);plt.close(fig)
  # Multi-label additive composite (does not discard coexpression).
  panels=[('GT multilabel',composite((binary==1).astype(float),tissue))]+[(DISPLAY.get(m,m),composite(arrays[m]['scores'],tissue)) for m in MODELS]
  fig,axes=plt.subplots(3,3,figsize=(12,12))
  for ax,(title,image) in zip(axes.flat,panels):ax.imshow(image);ax.set_title(title);ax.axis('off')
  fig.suptitle(f'CRC02 patch {pid} · additive multi-label composite · scores 0–1')
  fig.tight_layout();fig.savefig(figures/f'patch_{pid}_multilabel_composite.png',dpi=155);plt.close(fig)
  for value in arrays.values():value.close()
  inventory.append(dict(patch_id=pid,selection=row.selection,fixed_argmax=f'figures/patch_{pid}_all_models_fixed_05.png',
    calibrated_argmax=f'figures/patch_{pid}_all_models_own_validation.png',channels=f'figures/patch_{pid}_all_channels_scores.png',
    multilabel=f'figures/patch_{pid}_multilabel_composite.png'))
  print('rendered',pid,flush=True)
 pd.DataFrame(inventory).to_csv(P/'visualization_inventory.csv',index=False)
 cards=[]
 for row in inventory:
  cards.append(f'''<article><h2>{html.escape(row['selection'])} · patch {row['patch_id']}</h2>
  <h3>固定0.5阈值argmax</h3><a href="{row['fixed_argmax']}"><img loading="lazy" src="{row['fixed_argmax']}"></a>
  <h3>各模型验证集阈值argmax</h3><a href="{row['calibrated_argmax']}"><img loading="lazy" src="{row['calibrated_argmax']}"></a>
  <p><a href="{row['channels']}">查看16通道 GT/概率图</a> · <a href="{row['multilabel']}">查看保留共表达的多标签合成图</a></p></article>''')
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>所有模型同patch可视化</title><style>body{max-width:1500px;margin:30px auto;font:15px/1.6 system-ui;color:#172234}article{padding:16px;margin:25px 0;background:#f7f9fc}img{max-width:100%}.note{background:#eef4fb;padding:15px}a{color:#1769aa}</style><h1>所有模型同patch可视化</h1><p class="note">12张图均来自正式共同测试患者CRC02。每个通道固定颜色；argmax仅用于展示。固定0.5图便于同操作点观察，各模型验证阈值图反映各自可用分类点。GT在多个通道同时阳性时，用原始mIF强度选主通道。16通道图和多标签合成图保留共表达信息。positive-Dice partial只训练4轮后中断。</p>'''+''.join(cards)+'</html>'
 (P/'visualizations.html').write_text(page)
 (P/'VISUALIZATION_COMPLETE.json').write_text(json.dumps(dict(patches=len(inventory),models=MODELS,patient='CRC02',argmax_is_display_only=True,score_scale=[0,1],selection_manifest='preview_manifest.csv'),indent=2))
if __name__=='__main__':main()
