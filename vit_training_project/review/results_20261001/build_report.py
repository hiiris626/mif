from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent
ROOT=P.parents[1]
RUNS={'v6':ROOT/'experiments/v6_v7_v8/runs/v6','v9':ROOT/'experiments/v6_v7_v8_emptydice/runs/v9'}
hist={v:pd.read_csv(r/'results/model/train_log.csv') for v,r in RUNS.items()}
fig,axes=plt.subplots(2,3,figsize=(15,8))
for v,h in hist.items():
 for ax,key,title in zip(axes.flat,['train_loss','val_loss','val_macro_iou','train_bce_loss','val_bce_loss','val_macro_f1'],['Train total loss (definitions differ)','Validation total loss (definitions differ)','Validation macro IoU','Train BCE','Validation BCE','Validation macro F1 / Dice']):
  ax.plot(h.epoch,h[key],label=v);ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(P/'training_comparison.png',dpi=160);plt.close(fig)
npzs={}
training=[]
for v,h in hist.items():
 row=h.loc[h.val_macro_iou.idxmax()];ep=int(row.epoch);run=RUNS[v]
 training.append(dict(version=v,completed_epochs=len(h),best_epoch=ep,best_val_iou=row.val_macro_iou,best_val_f1=row.val_macro_f1,best_train_loss=row.train_loss,best_val_loss=row.val_loss,training_hours=h.epoch_seconds.sum()/3600))
 npzs[v]=np.load(run/f'results/model/snapshots/epoch_{ep:03d}_patch_35422.npz')
assert np.array_equal(npzs['v6']['labels'],npzs['v9']['labels'])
fig,axes=plt.subplots(4,4,figsize=(19,13))
for c,ax in enumerate(axes.flat):
 gt=npzs['v6']['labels'][c];valid=gt!=255
 img=np.hstack([(gt==1).astype(float),np.where(valid,npzs['v6']['probabilities'][c],0),np.where(valid,npzs['v9']['probabilities'][c],0)])
 ax.imshow(img,cmap='magma',vmin=0,vmax=1);ax.set_title(str(npzs['v6']['channels'][c])+' : GT | v6 | v9');ax.axis('off')
fig.suptitle('Fixed validation patch 35422 — best checkpoints; fixed probability scale 0–1; not a test-set summary')
fig.tight_layout();fig.savefig(P/'validation_patch_35422_channels.png',dpi=170);plt.close(fig)
pd.DataFrame(training).to_csv(P/'training_summary.csv',index=False)
# Finish once both held-out evaluations are available.
if not all((r/'results/test_metrics.json').exists() for r in RUNS.values()):
 print('Training curves and matched validation patch prepared; waiting for v6 test results.');raise SystemExit(0)
metrics={};rows=[];channel_rows=[]
for v,run in RUNS.items():
 for policy,file in [('fixed_05','test_metrics_default05.json'),('validation_selected','test_metrics.json')]:
  m=json.loads((run/'results'/file).read_text());metrics[v,policy]=m
  rows.append(dict(version=v,threshold_policy=policy,n_patches=m['n_patches'],n_valid_channel_pixels=m['n_valid_channel_pixels'],micro_f1=m['micro_f1'],**m['macro']))
  for c,r in m['per_class'].items():
   tn,fp=r['confusion_matrix'][0];fn,tp=r['confusion_matrix'][1]
   channel_rows.append(dict(version=v,threshold_policy=policy,channel=c,precision=r['precision'],recall=r['recall'],dice=r['dice'],iou=r['iou'],threshold=r['threshold'],false_positive_rate=fp/(fp+tn),fp=fp,fn=fn,tp=tp,tn=tn))
for pol in ['fixed_05','validation_selected']:
 a,b=metrics['v6',pol],metrics['v9',pol]
 assert a['n_patches']==b['n_patches'] and a['n_valid_channel_pixels']==b['n_valid_channel_pixels']
 for c in a['per_class']:
  assert a['per_class'][c]['support']==b['per_class'][c]['support']
  assert a['per_class'][c]['negative_support']==b['per_class'][c]['negative_support']
assert hashlib.sha256((RUNS['v6']/'data/test.csv').read_bytes()).digest()==hashlib.sha256((RUNS['v9']/'data/test.csv').read_bytes()).digest()
summary=pd.DataFrame(rows);channels=pd.DataFrame(channel_rows)
summary.to_csv(P/'test_summary.csv',index=False);channels.to_csv(P/'per_channel_comparison.csv',index=False)
fig,axes=plt.subplots(2,2,figsize=(16,10))
for row,pol in enumerate(['fixed_05','validation_selected']):
 for col,key in enumerate(['dice','false_positive_rate']):
  a=channels[channels.threshold_policy==pol].pivot(index='channel',columns='version',values=key)
  a.plot.bar(ax=axes[row,col],rot=60);axes[row,col].set_title(pol+' : '+key);axes[row,col].set_ylim(0,1 if key=='dice' else max(.01,float(a.max().max())*1.15));axes[row,col].grid(axis='y',alpha=.25)
fig.tight_layout();fig.savefig(P/'test_channel_comparison.png',dpi=160);plt.close(fig)
historical=pd.read_csv(ROOT/'review/metric_audit_20260930/summary.csv');historical=historical[(historical.scope=='common')&(historical.policy=='expanded')].copy();historical['model']=historical['model'].replace({'baseline':'v4','binary_bce':'v5'});historical.to_csv(P/'historical_common_subset.csv',index=False)
links=''.join(f'<li>{v}: <a href="{run}/results">全部结果</a> · <a href="{run}/results/model/snapshots">逐轮快照</a> · <a href="{run}/results/predictions">测试patch概率图</a> · <a href="{run}/results/model/best.pt">最佳模型</a></li>' for v,run in RUNS.items())
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v6 / v9 结果核查</title><style>body{font:16px system-ui;margin:32px auto;max-width:1400px;padding:20px;color:#192334}table{border-collapse:collapse;margin:20px 0}th,td{border:1px solid #ccd4df;padding:8px;text-align:right}img{max-width:100%}a{color:#1766ae}.note{background:#eef3fa;padding:16px}h2{margin-top:36px}</style><h1>训练与测试结果：2026-10-01</h1><p>v6：原 BCE + 前景 Dice。v9：BCE + 分段重叠损失，纯阴性通道改为 mean(p)。v7、v8 未启动。</p><p class="note">两组使用相同测试清单：38,218 patch、6 位患者。组织内可靠通道的阴性参与评价；组织外及缺失通道忽略。逐通道独立分类，不使用 argmax 计算指标。阈值仅由验证集选择。模型按验证 IoU 选择。不同损失定义的总 loss 不能直接用于判定优劣。</p><h2>训练结果</h2>'''+pd.DataFrame(training).to_html(index=False,float_format=lambda x:f'{x:.4f}')+'<img src="training_comparison.png"><h2>完整测试集</h2>'+summary.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'<img src="test_channel_comparison.png"><h2>逐通道全部指标</h2>'+channels.to_html(index=False,float_format=lambda x:f'{x:.4f}')+'<h2>同一验证 patch 的 GT / v6 / v9</h2><p>固定 patch35422，仅用于观察，不代表测试集整体。</p><img src="validation_patch_35422_channels.png"><h2>历史版本参考</h2><p>下表为此前核算的共同测试子集：6,638 patch、1位患者，expanded组织掩膜；各模型使用历史阈值，不可与上面的6患者全测试集直接排名。</p>'+historical[['model','patches','patients','precision','recall','f1','iou']].to_html(index=False,float_format=lambda x:f'{x:.4f}')+'<h2>文件入口</h2><ul>'+links+'</ul><p><a href="test_summary.csv">总指标CSV</a> · <a href="per_channel_comparison.csv">逐通道CSV</a> · <a href="training_summary.csv">训练摘要CSV</a></p></html>'
(P/'index.html').write_text(html)
(P/'verification.json').write_text(json.dumps({'test_manifest_identical':True,'per_channel_positive_negative_support_identical':True,'test_patches':38218,'no_new_training':True,'v7_v8_not_started':True},indent=2))
print(summary.to_string(index=False));print('REPORT',P/'index.html')
