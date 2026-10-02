"""Recompute metric formulas from integer confusion matrices; export comparison."""
import json,html
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parents[1]
NAMES=['v1','v2','v3','baseline','binary_bce']
LABELS={'v1':'v1','v2':'v2','v3':'v3','baseline':'Old MSE baseline','binary_bce':'New weighted BCE'}

def read(p):return json.loads(p.read_text())
summary=[];channels=[];ignored=[];checks=[]
for name in NAMES:
 assert (ROOT/name/'COMPLETE.json').exists(),name
 for scope in (['common','full'] if name=='binary_bce' else ['common']):
  for policy in ['original','expanded','original_05']:
   result=read(ROOT/name/f'{scope}_{policy}.json')
   stats=np.load(ROOT/name/f'{scope}_{policy}_sufficient_stats.npz')
   for idx,(c,row) in enumerate(result['per_class'].items()):
    cm=np.array(row['confusion_matrix']);np.testing.assert_array_equal(cm,stats['confusion'][idx])
    tn,fp,fn,tp=map(int,cm.ravel());assert int(stats['pos'][idx].sum())==tp+fn;assert int(stats['neg'][idx].sum())==tn+fp
    for key,num,den in [('precision',tp,tp+fp),('recall',tp,tp+fn),('f1',2*tp,2*tp+fp+fn),('iou',tp,tp+fp+fn)]:
     expected=num/den if den else None
     assert (expected is None and row[key] is None) or abs(expected-row[key])<1e-12,(name,c,key)
    channels.append(dict(model=name,scope=scope,policy=policy,channel=c,tn=tn,fp=fp,fn=fn,tp=tp,threshold=row['threshold'],**{k:row[k] for k in ['precision','recall','f1','iou','auroc_histogram','average_precision_histogram']}))
   for key in ['precision','recall','f1','iou']:
    vals=[r[key] for r in result['per_class'].values() if r[key] is not None and r['support']>0]
    assert abs(np.mean(vals)-result['macro'][key])<1e-12
   summary.append(dict(model=name,scope=scope,policy=policy,patches=6638 if scope=='common' else 38218,patients=1 if scope=='common' else 6,n_valid_pixels=result['n_valid_pixels'],**result['macro'],micro_f1=result['micro_f1']))
  audit=read(ROOT/name/f'{scope}_ignored_tissue.json')
  original=read(ROOT/name/f'{scope}_original.json')['per_class'];expanded_rows=read(ROOT/name/f'{scope}_expanded.json')['per_class']
  for channel,row in original.items():
   aa=np.asarray(row['confusion_matrix']);bb=np.asarray(expanded_rows[channel]['confusion_matrix'])
   np.testing.assert_array_equal(aa[1],bb[1])
   assert bb[0,1]-aa[0,1]==audit['per_channel'][channel]['positive_pixels']
  for c,v in audit['per_channel'].items():ignored.append(dict(model=name,scope=scope,channel=c,tissue_pixels=audit['tissue_pixels'],ignored_tissue_pixels=audit['ignored_tissue_pixels'],any_marker_positive_fraction=audit['any_marker_positive_fraction'],**v))
 # Compare rerun original mask to previous independently stored evaluation.
 reference=(PROJECT/'experiments/binary_bce/runs/prepared/results/test_metrics.json') if name=='binary_bce' else PROJECT/f'review/unified_eval_20260929/{name}/common_test_metrics.json'
 a=read(reference);b=read(ROOT/name/('full_original.json' if name=='binary_bce' else 'common_original.json'))
 assert a['n_valid_pixels']==b['n_valid_pixels']
 for c in a['per_class']:assert a['per_class'][c]['support']==b['per_class'][c]['support'] and a['per_class'][c]['negative_support']==b['per_class'][c]['negative_support']
 delta={k:b['macro'][k]-a['macro'][k] for k in ['precision','recall','f1','iou','auroc_histogram','average_precision_histogram']}
 checks.append(dict(model=name,reference=str(reference),pixel_and_label_counts_match=True,metric_deltas=delta))
s=pd.DataFrame(summary);c=pd.DataFrame(channels);z=pd.DataFrame(ignored)
for policy in ['original','expanded']:
 common=s[(s.scope=='common')&(s.policy==policy)];assert common.n_valid_pixels.nunique()==1
 cc=c[(c.scope=='common')&(c.policy==policy)]
 for _,g in cc.groupby('channel'):assert (g.tp+g.fn).nunique()==1 and (g.tn+g.fp).nunique()==1
patient_rows=[]
for name in NAMES:
 for patient,result in read(ROOT/name/'per_patient.json').items():
  patient_rows.append(dict(model=name,patient=patient,**result['macro']))
pd.DataFrame(patient_rows).to_csv(ROOT/'per_patient.csv',index=False)
s.to_csv(ROOT/'summary.csv',index=False);c.to_csv(ROOT/'per_channel.csv',index=False);z.to_csv(ROOT/'ignored_tissue_predictions.csv',index=False)
(ROOT/'verification.json').write_text(json.dumps(dict(all_confusion_formulas_checked=True,all_histogram_counts_checked=True,common_population_and_labels_equal=True,expanded_scope_preserves_tp_fn_and_adds_exact_audited_fp=True,reference_reproduction=checks),indent=2))
base=s[(s.scope=='common')&(s.policy=='original')].set_index('model').loc[NAMES]
expanded=s[(s.scope=='common')&(s.policy=='expanded')].set_index('model').loc[NAMES]
fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(5)
for ax,key in zip(axes,['f1','iou']):
 ax.bar(x-.19,base[key],.38,label='Original supervision mask');ax.bar(x+.19,expanded[key],.38,label='All tissue zeros = negative (sensitivity)');ax.set_xticks(x,[LABELS[n] for n in NAMES],rotation=18,ha='right');ax.set_ylabel('Macro '+key.upper());ax.set_ylim(0,1);ax.grid(axis='y',alpha=.2)
axes[0].legend(fontsize=8);fig.suptitle('Matched unseen CRC02: 6,638 patches, frozen validation thresholds');fig.tight_layout();fig.savefig(ROOT/'common_comparison.png',dpi=170);plt.close(fig)
focus=c[(c.scope=='common')&(c.policy=='original')].pivot(index='channel',columns='model',values='f1')[NAMES]
fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(focus.values,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(5),[LABELS[n] for n in NAMES],rotation=20,ha='right');ax.set_yticks(range(len(focus)),focus.index)
for i in range(len(focus)):
 for j in range(5):ax.text(j,i,f'{focus.iloc[i,j]:.3f}',ha='center',va='center',color='white' if focus.iloc[i,j]<.6 else 'black',fontsize=9)
ax.set_title('Per-channel F1: common unseen CRC02, original mask');fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(ROOT/'common_channel_f1.png',dpi=170);plt.close(fig)
old=c[(c.model=='v3')&(c.scope=='common')&(c.policy=='original')].set_index('channel')
new=c[(c.model=='binary_bce')&(c.scope=='common')&(c.policy=='original')].set_index('channel')
delta=pd.DataFrame({k:new[k]-old[k] for k in ['precision','recall','f1','iou','average_precision_histogram']}).sort_values('f1');delta.to_csv(ROOT/'new_vs_v3_channel_delta.csv')
full=s[(s.model=='binary_bce')&(s.scope=='full')].set_index('policy')
show=['precision','recall','f1','iou','auroc_histogram','average_precision_histogram']
def table(df):return df.to_html(float_format=lambda x:f'{x:.4f}',border=0)
patch_audit=read(ROOT/'patch168299_comparison.json')
patch_rows=[]
for model,values in patch_audit.items():
 for channel,v in values['channels'].items():patch_rows.append(dict(model=model,channel=channel,**v))
patch_table=pd.DataFrame(patch_rows);patch_table.to_csv(ROOT/'patch168299_comparison.csv',index=False)
body='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>指标核算与对比</title><style>body{font:16px/1.65 sans-serif;max-width:1180px;margin:35px auto;padding:20px;color:#17212e}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:7px;border-bottom:1px solid #d8dee8;text-align:right}img{max-width:100%}.note{background:#fff5d9;padding:16px}h2{margin-top:32px}</style><h1>指标核算与对比 · 2026-09-30</h1>
<p>五个检查点重新推理；未训练、未修改模型。独立核算整数混淆矩阵、直方图计数及宏平均公式，详见verification.json。</p>
<div class="note">主比较限定所有版本共同未见的CRC02：6638张patch、1名患者。仅凭这一名患者不能证明总体泛化优劣。阈值冻结于各版本各自的验证集；验证人群和搜索范围并非完全相同，AP/AUROC同时列出。v1–v3为归一化强度分数，新旧分类模型为sigmoid分数，不能直接比较概率校准或统一采用0.5强度阈值。</div>
<h2>共同测试集：原监督掩膜</h2>'''+table(base[show])+'''<h2>共同测试集：全阴性组织敏感性分析</h2><p>保持同一检查点和同一阈值，只将组织内可用通道的处理后GT=0计为负例。组织外仍忽略；未知通道仍忽略。这是假设性负标签口径，不代表已证明生物学真阴性。不会使用GT修改或裁切预测。</p>'''+table(expanded[show])+'''<img src="common_comparison.png"><h2>逐通道F1</h2><img src="common_channel_f1.png"><h2>新模型相对v3的通道差值</h2>'''+table(delta)+'''<h2>新BCE完整测试集</h2><p>38218张patch、6名患者；不与旧模型不同的完整测试人群直接排名。original_05=固定0.5；original=验证集阈值；expanded=同阈值、扩展负标签的敏感性分析。</p>'''+table(full[show])+'''<h2>原忽略组织区域的预测阳性率</h2><p>分母为H&E组织内所有GT通道被忽略的像素；任一通道超过冻结阈值即算“任一通道阳性”。它是审计指标，不自动等于生物学假阳性率。</p>'''+table(z.drop_duplicates(['model','scope'])[['model','scope','ignored_tissue_pixels','any_marker_positive_fraction']])+'''<h2>文件</h2><ul><li><a href="summary.csv">汇总CSV</a></li><li><a href="per_channel.csv">所有通道混淆矩阵与指标</a></li><li><a href="ignored_tissue_predictions.csv">全阴性组织逐通道染色率</a></li><li><a href="verification.json">核算与历史结果复现检查</a></li></ul></html>'''
body=body.replace('<h2>文件</h2>','<h2>CRC36 / patch168299复核</h2><p>以下预测计数仅限原监督忽略的38881个组织像素，并不包含玻片空白。GT阳性数则为原有效监督区域内的计数。</p>'+table(patch_table)+'<h2>文件</h2>')
(ROOT/'index.html').write_text(body)
print(base[show].to_string());print('EXPANDED');print(expanded[show].to_string());print('NEW FULL');print(full[show].to_string());print('IGNORED');print(z.drop_duplicates(['model','scope'])[['model','scope','ignored_tissue_pixels','any_marker_positive_fraction']].to_string(index=False));print('CHANNEL DELTAS');print(delta.to_string());print('VERIFICATION',checks)
