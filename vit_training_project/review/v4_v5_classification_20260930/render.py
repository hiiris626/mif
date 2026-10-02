"""Binary GT vs probability/segmentation; raw intensity is diagnostic only."""
from pathlib import Path
import sys,json,html,hashlib
import numpy as np
import pandas as pd
import cv2,tifffile
from PIL import Image,ImageDraw,ImageFont
from matplotlib import colormaps
P=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(P/'experiments/binary_bce'))
from vit_seg.data import read_he,read_mif,CHANNELS
from vit_seg.cache import NativeCache
from vit_seg.display import argmax_display,PALETTE
OLD=P/'review/unified_eval_20260929';NEW=P/'review/metric_audit_20260930/binary_bce'
ROOT=Path('/data/weiyh/orioncrc_miphei/ORIONCRC_dataset_tile_20x')
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
font=ImageFont.truetype(FONT,15);small=ImageFont.truetype(FONT,12)
LUT=(colormaps['magma'](np.linspace(0,1,256))[:,:3]*255).astype(np.uint8)
cache=NativeCache(P/'runs/ddp_baseline/cache')
threshold_files={'v4':OLD/'baseline/thresholds.json','v5':P/'experiments/binary_bce/runs/prepared/results/thresholds.json'}
thresholds={k:np.array(json.loads(v.read_text())['thresholds']) for k,v in threshold_files.items()}

def bw(a):return np.repeat((a.astype(np.uint8)*255)[...,None],3,-1)
def heat(a,t):
 rgb=LUT[np.rint(a.clip(0,1)*255).astype(np.uint8)].copy();rgb[~t]=0;return rgb

def strip(images,titles,heading,note):
 w=256*len(images);out=Image.new('RGB',(w,345),'#101216');d=ImageDraw.Draw(out)
 d.text((8,5),heading,fill='white',font=font)
 for i,(im,title) in enumerate(zip(images,titles)):
  d.text((i*256+6,30),title,fill='white',font=small);out.paste(Image.fromarray(im),(i*256,50))
 d.text((8,314),note,fill='#d6d6d6',font=small)
 return out

def main():
 cv2.setNumThreads(1)
 for folder in ['channels','argmax','labels']: (OUT/folder).mkdir(exist_ok=True)
 frame=pd.read_csv(OLD/'preview_manifest.csv');frame.to_csv(OUT/'patch_manifest.csv',index=False)
 audits=[];gate_counts=[];records=[]
 for row in frame.itertuples():
  pid=int(row.patch_id);loaded={}
  for k,folder in [('v4',OLD/'baseline/previews'),('v5',NEW/'previews')]:
   with np.load(folder/f'patch_{pid}.npz') as z:loaded[k]={n:z[n] for n in z.files}
  y=loaded['v5']['labels'];t=loaded['v5']['tissue'].astype(bool)
  np.testing.assert_array_equal(y,loaded['v4']['labels']);np.testing.assert_array_equal(t,loaded['v4']['tissue'])
  gt=(y==1);available=(y!=255).any((1,2))
  he=cv2.resize(read_he(ROOT/row.image_path),(256,256),interpolation=cv2.INTER_AREA)
  raw=read_mif(ROOT/row.target_path);_,native_y,nt=cache.read(pid)
  raw256=np.stack([cv2.resize(c,(256,256),interpolation=cv2.INTER_NEAREST) for c in raw])
  binary={};argmax={};rgb={}
  for k in ['v4','v5']:
   prob=loaded[k]['scores'];binary[k]=(prob>=thresholds[k][:,None,None])&t[None]
   argmax[k],rgb[k]=argmax_display(prob,t,thresholds=thresholds[k])
   assert (argmax[k][~binary[k].any(0)]==0).all()
   yy,xx=np.where(argmax[k]>0);assert binary[k][argmax[k][yy,xx]-1,yy,xx].all()
   gate_counts.append(dict(model=k,patch_id=pid,tissue_pixels=int(t.sum()),no_positive_pixels=int((t&~binary[k].any(0)).sum())))
   tifffile.imwrite(OUT/'labels'/f'patch_{pid}_{k}_multilabel.tiff',binary[k].astype(np.uint8),metadata={'axes':'CYX'})
   tifffile.imwrite(OUT/'labels'/f'patch_{pid}_{k}_threshold_argmax.tiff',argmax[k])
  tifffile.imwrite(OUT/'labels'/f'patch_{pid}_gt_binary.tiff',gt.astype(np.uint8),metadata={'axes':'CYX'})
  tifffile.imwrite(OUT/'labels'/f'patch_{pid}_original_valid.tiff',(y!=255).astype(np.uint8),metadata={'axes':'CYX'})
  note='GT zeros inside tissue may be ignored in training. Probabilities: fixed 0-1, no intensity gain. V4 CRC33 has training exposure.'
  argim=strip([he,rgb['v4'],rgb['v5']],['H&E','v4: thresholded argmax','v5: thresholded argmax'],f'{row.orion_slide_id} / {pid}', 'Black = outside tissue OR no channel passes its threshold.')
  argim.save(OUT/'argmax'/f'patch_{pid}.png')
  for c,name in enumerate(CHANNELS):
   panels=[he,bw(gt[c]),heat(loaded['v4']['scores'][c],t),bw(binary['v4'][c]),heat(loaded['v5']['scores'][c],t),bw(binary['v5'][c])]
   titles=['H&E','GT binary: 0=black, 1=white','v4 probability [0,1]',f'v4 segmentation t={thresholds["v4"][c]:.4f}','v5 probability [0,1]',f'v5 segmentation t={thresholds["v5"][c]:.4f}']
   im=strip(panels,titles,f'{row.orion_slide_id} / patch {pid} / {name}',note)
   # Fixed probability scale; same palette for every channel/model/patch.
   d=ImageDraw.Draw(im);im.paste(Image.fromarray(np.repeat(LUT[None],9,axis=0)),(1280,332));d.text((1120,331),'Probability 0 -> 1',fill='white',font=small)
   filename=f'patch_{pid}_{name}.png';im.save(OUT/'channels'/filename)
   records.append(dict(patch_id=pid,slide=row.orion_slide_id,channel=name,file='channels/'+filename))
   valid=y[c]!=255;pos=gt[c];weak=pos&(raw256[c]<=5);strong=pos&(raw256[c]>5)
   npos=(native_y[c]==1).astype(np.uint8)
   count,cc,stats,_=cv2.connectedComponentsWithStats(npos,8)
   sampled=cv2.resize(cc.astype(np.float32),(256,256),interpolation=cv2.INTER_NEAREST).astype(np.int32)
   seen=np.bincount(sampled.ravel(),minlength=count)>0
   small_ids=np.flatnonzero((stats[:,cv2.CC_STAT_AREA]<=4)&(np.arange(count)>0))
   item=dict(patch_id=pid,slide=row.orion_slide_id,channel=name,native_h=raw.shape[1],native_w=raw.shape[2],native_positive=int(npos.sum()),grid_positive=int(pos.sum()),weak_1_to_5_pixels=int(weak.sum()),strong_gt5_pixels=int(strong.sum()),native_components=count-1,small_components_le4=len(small_ids),small_components_lost=int((~seen[small_ids]).sum()),native_positive_fraction=float(npos.mean()),grid_positive_fraction=float(pos.mean()))
   for k in ['v4','v5']:
    item[k+'_weak_tp']=int((binary[k][c]&weak).sum());item[k+'_strong_tp']=int((binary[k][c]&strong).sum())
   audits.append(item)
  print('Rendered',pid,flush=True)
 pd.DataFrame(records).to_csv(OUT/'image_manifest.csv',index=False);pd.DataFrame(gate_counts).to_csv(OUT/'argmax_gate_counts.csv',index=False)
 a=pd.DataFrame(audits);a.to_csv(OUT/'weak_signal_resize_patch_audit.csv',index=False)
 sums=a.groupby('channel').sum(numeric_only=True)
 for k in ['v4','v5']:
  sums[k+'_weak_recall']=sums[k+'_weak_tp']/sums.weak_1_to_5_pixels.replace(0,np.nan)
  sums[k+'_strong_recall']=sums[k+'_strong_tp']/sums.strong_gt5_pixels.replace(0,np.nan)
 sums['weak_share']=sums.weak_1_to_5_pixels/(sums.weak_1_to_5_pixels+sums.strong_gt5_pixels)
 sums['small_component_loss_fraction']=sums.small_components_lost/sums.small_components_le4.replace(0,np.nan)
 keep=['native_positive','grid_positive','weak_1_to_5_pixels','strong_gt5_pixels','weak_share','v4_weak_recall','v4_strong_recall','v5_weak_recall','v5_strong_recall','small_components_le4','small_components_lost','small_component_loss_fraction']
 sums[keep].to_csv(OUT/'weak_signal_resize_summary.csv')
 weights=pd.read_csv(P/'experiments/binary_bce/review/bce_pixel_weights.csv')
 stats=json.loads((P/'experiments/binary_bce/runs/prepared/data/statistics.json').read_text())
 weights['intensity_sigma_outer_gamma']=stats['channel_weights'];weights['v5_val_threshold']=thresholds['v5']
 weights.to_csv(OUT/'weights_and_thresholds.csv',index=False)
 protocol=dict(v4='ddp_baseline best: MSE on binary expression labels',v5='binary_bce best epoch18: weighted BCE on binary labels',patches=len(frame),channel_panels=len(records),channels=CHANNELS,gt='GT>0 binary 0/1, coexpression retained, original valid mask exported separately',probability_range=[0,1],intensity_gain=False,intensity_argmax=False,prediction_gate='HE tissue AND each channel >= its own frozen validation threshold, then largest passing probability; zero when none',thresholds={k:v.tolist() for k,v in thresholds.items()},raw_intensity_used_for='diagnostic weak-signal bins only; not rendering, label thresholds or inference',weak_signal_cutoff_5='exploratory diagnostic bin on 24 previews only, not a recommended label threshold',missing_signal='Zeros may reflect biological absence or preprocessing; cannot infer staining failure from prediction',heldout_limitation='CRC33_02 was exposed via another section in v4; other five patients independent for both',resize='native333 -> model256 labels nearest, input area; ViT branch224 bilinear',old_checkpoints_unchanged=True)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 payload=json.dumps([dict(patch_id=int(row.patch_id),slide=row.orion_slide_id) for row in frame.itertuples()])
 options=''.join(f'<option value="{c}">{html.escape(c)}</option>' for c in CHANNELS)
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v4/v5逐通道概率与二值GT</title><style>body{font:16px/1.65 sans-serif;margin:22px;color:#17212e}img{max-width:100%}select,button{font:inherit;padding:7px}table{border-collapse:collapse}td,th{border-bottom:1px solid #ddd;padding:6px}.note{background:#fff4d5;padding:12px}</style><h1>v4 / v5：只比较表达区域</h1><p>v4＝旧MSE分类基线；v5＝新加权BCE。24张固定patch，每切片4张，16通道，共384张逐通道对照。GT为0/1；概率固定0～1色标；不做强度增益，不做逐图亮度拉伸。</p><div class="note">模型均为逐通道二分类，不互斥。阈值来自各自验证集。GT为0不代表染色一定可靠；原监督忽略掩膜单独导出。CRC33_02对v4存在其他切片训练暴露，不作独立泛化比较。概率色标只表示模型分数，不是免疫强度，也未保证概率校准。</div><p>patch <select id="patch"></select> 通道 <select id="channel">'''+options+'''</select></p><img id="panel"><h2>带阈值的argmax</h2><p>先逐通道判断是否超过阈值，再从通过者中选择最高概率；没有通道通过则为黑色。GT保留多标签，不强行从同为1的标签中选一个。</p><img id="argmax"><p><a href="weights_and_thresholds.csv">通道权重与阈值</a> · <a href="weak_signal_resize_summary.csv">弱信号与缩放抽样分析</a> · <a href="image_manifest.csv">384张图片索引</a> · <a href="protocol.json">完整口径</a></p><script>const patches='''+payload+''';const p=document.querySelector('#patch'),c=document.querySelector('#channel');for(const row of patches){const o=document.createElement('option');o.value=row.patch_id;o.textContent=row.slide+' / '+row.patch_id;p.appendChild(o)}p.value='168299';c.value='CD4';function show(){document.querySelector('#panel').src='channels/patch_'+p.value+'_'+c.value+'.png';document.querySelector('#argmax').src='argmax/patch_'+p.value+'.png'}p.onchange=c.onchange=show;show();</script></html>'''
 if (OUT/'analysis_fragment.html').exists():
  page=page.replace('<script>const patches=',(OUT/'analysis_fragment.html').read_text()+'<script>const patches=')
 (OUT/'index.html').write_text(page)
 (OUT/'COMPLETE.json').write_text(json.dumps(dict(patches=len(frame),channel_panels=len(records),argmax_panels=len(frame),threshold_invariants_passed=True,labels_binary=True),indent=2))
 print(sums[keep].to_string(),flush=True)
if __name__=='__main__':main()
