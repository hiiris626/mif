"""Verify matched populations and render the unified result tables/figures."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
MODELS=['v1','v2','v3','v4','v5','v6','v9','positive_dice_partial']
FORMAL=[m for m in MODELS if m!='positive_dice_partial']
SCOPES=['expanded','original'];POLICIES=['fixed_05','own_validation']
rows=[];channels=[];supports={};manifest_hashes={};completes={}
for model in MODELS:
 complete=json.loads((P/model/'COMPLETE.json').read_text());completes[model]=complete
 manifest_hashes[model]=complete['test_manifest_sha256']
 for scope in SCOPES:
  for policy in POLICIES:
   path=P/model/f'{scope}_{policy}.json'
   if not path.exists():continue
   result=json.loads(path.read_text());macro=result['macro']
   rows.append(dict(model=model,status=complete['status'],scope=scope,threshold_policy=policy,
                    patches=result['n_patches'],valid_channel_pixels=result['n_valid_channel_pixels'],
                    micro_f1=result['micro_f1'],exact_match_accuracy=result['exact_match_accuracy'],
                    hamming_accuracy=result['hamming_accuracy'],**macro))
   support=[]
   for channel,r in result['per_class'].items():
    tn,fp=r['confusion_matrix'][0];fn,tp=r['confusion_matrix'][1]
    support.append((channel,r['support'],r['negative_support']))
    channels.append(dict(model=model,status=complete['status'],scope=scope,threshold_policy=policy,channel=channel,
                         precision=r['precision'],recall=r['recall'],f1=r['f1'],dice=r['dice'],iou=r['iou'],
                         specificity=r['specificity'],false_positive_rate=fp/(fp+tn),support=r['support'],
                         negative_support=r['negative_support'],tp=tp,fp=fp,fn=fn,tn=tn,
                         auroc_histogram=r['auroc_histogram'],average_precision_histogram=r['average_precision_histogram']))
   supports[scope,policy,model]=support
summary=pd.DataFrame(rows);per_channel=pd.DataFrame(channels)
assert len(set(manifest_hashes.values()))==1,manifest_hashes
for scope in SCOPES:
 for policy in POLICIES:
  available=[m for m in MODELS if (scope,policy,m) in supports]
  reference=supports[scope,policy,available[0]]
  assert all(supports[scope,policy,m]==reference for m in available)
assert (summary.patches==6638).all()
summary.to_csv(P/'summary.csv',index=False);per_channel.to_csv(P/'per_channel.csv',index=False)

primary=summary[(summary.scope=='expanded')&(summary.threshold_policy=='fixed_05')&(summary.model.isin(FORMAL))].copy()
primary['rank_macro_ap']=primary.average_precision_histogram.rank(method='min',ascending=False).astype(int)
primary['rank_macro_auroc']=primary.auroc_histogram.rank(method='min',ascending=False).astype(int)
primary['rank_macro_f1']=primary.f1.rank(method='min',ascending=False).astype(int)
primary=primary.sort_values(['rank_macro_ap','rank_macro_auroc'])
primary.to_csv(P/'primary_ranking.csv',index=False)

selected=summary[(summary.scope=='expanded')&(summary.threshold_policy=='own_validation')&(summary.model.isin(FORMAL))].copy()
selected=selected.sort_values('f1',ascending=False);selected.to_csv(P/'validation_threshold_ranking.csv',index=False)

fig,axes=plt.subplots(2,2,figsize=(14,9))
plot=primary.set_index('model')
for ax,metric,title in zip(axes.flat,['average_precision_histogram','auroc_histogram','f1','precision'],
                           ['Macro AP (threshold-free)','Macro AUROC (threshold-free)','Macro F1 at fixed 0.5','Macro precision at fixed 0.5']):
 plot[metric].sort_values().plot.barh(ax=ax,color='#3978b8');ax.set_title(title);ax.grid(axis='x',alpha=.25)
fig.tight_layout();fig.savefig(P/'overall_comparison.png',dpi=170);plt.close(fig)

pc=per_channel[(per_channel.scope=='expanded')&(per_channel.threshold_policy=='fixed_05')&(per_channel.model.isin(FORMAL))]
for metric,title,file in [('f1','Per-channel F1 at fixed 0.5','channel_f1.png'),('average_precision_histogram','Per-channel AP','channel_ap.png')]:
 table=pc.pivot(index='channel',columns='model',values=metric)
 fig,ax=plt.subplots(figsize=(14,7));im=ax.imshow(table.T.values,aspect='auto',vmin=0,vmax=1,cmap='viridis')
 ax.set_xticks(range(len(table.index)),table.index,rotation=55,ha='right');ax.set_yticks(range(len(table.columns)),table.columns)
 for y in range(len(table.columns)):
  for x in range(len(table.index)):ax.text(x,y,f'{table.iloc[x,y]:.2f}',ha='center',va='center',fontsize=7,color='white' if table.iloc[x,y]<.35 else 'black')
 ax.set_title(title);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(P/file,dpi=170);plt.close(fig)

inventory=[]
for model in MODELS:
 c=completes[model];inventory.append(dict(model=model,status=c['status'],included_in_formal_ranking=model in FORMAL,
    checkpoint_epoch=c['checkpoint_epoch_one_based'],checkpoint=c['checkpoint'],thresholds='available' if c['threshold_path'] else 'none'))
inventory += [dict(model='v7',status='prepared_not_started',included_in_formal_ranking=False,checkpoint_epoch=None,checkpoint='',thresholds='none'),
              dict(model='v8',status='prepared_not_started',included_in_formal_ranking=False,checkpoint_epoch=None,checkpoint='',thresholds='none'),
              dict(model='emptydice_v6_trial',status='stopped_during_epoch_1_no_checkpoint',included_in_formal_ranking=False,checkpoint_epoch=None,checkpoint='',thresholds='none')]
pd.DataFrame(inventory).to_csv(P/'experiment_inventory.csv',index=False)

best_ap=primary.iloc[0];best_fixed_f1=primary.sort_values('f1',ascending=False).iloc[0];best_selected=selected.iloc[0]
verification=dict(models_inferred=MODELS,formal_completed_models=FORMAL,common_patient='CRC02',common_patches=6638,
 test_manifest_identical=True,per_channel_support_identical=True,primary_scope='available channel AND H&E tissue; all GT-zero tissue is negative',
 primary_threshold='fixed 0.5',threshold_free_metrics=['AUROC','average precision'],own_validation_thresholds_secondary=True,
 incomplete_experiment_excluded='positive_dice_partial',best_macro_ap=best_ap.model,best_fixed05_macro_f1=best_fixed_f1.model,
 best_own_validation_macro_f1=best_selected.model,training_changed=False)
reference_root=P.parent/'metric_audit_20260930'
reference_names={'v1':'v1','v2':'v2','v3':'v3','v4':'baseline','v5':'binary_bce'}
reference_deltas={}
for model,old_name in reference_names.items():
 old=json.loads((reference_root/old_name/'common_expanded.json').read_text())['macro']
 new=json.loads((P/model/'expanded_own_validation.json').read_text())['macro']
 reference_deltas[model]={key:abs(new[key]-old[key]) for key in ['precision','recall','f1','iou','auroc_histogram','average_precision_histogram']}
verification['reference_reproduction_max_abs_delta']=max(value for model in reference_deltas.values() for value in model.values())
verification['reference_reproduction_deltas']=reference_deltas
(P/'verification.json').write_text(json.dumps(verification,indent=2))

def html_table(frame,cols):return frame[cols].to_html(index=False,float_format=lambda x:f'{x:.4f}')
html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>所有ViT实验统一评估</title><style>body{font:15px system-ui;margin:30px auto;max-width:1500px;padding:20px;color:#172234}table{border-collapse:collapse;margin:18px 0;font-size:13px}th,td{border:1px solid #ccd4df;padding:6px;text-align:right}img{max-width:100%}.note{background:#eef4fb;padding:15px}a{color:#1769aa}</style><h1>所有已跑ViT实验统一评估</h1><p><a href="visualizations.html">打开12张patch的所有模型同图对照</a> · <a href="v5_v6_change_analysis.md">查看v5→v6改动与提升分析</a></p><p class="note">共同测试集：CRC02，6,638 patch。主口径：H&amp;E组织内且通道可用的所有像素；GT全零作为阴性，组织外和缺失通道忽略。主阈值固定0.5；AUROC/AP不依赖阈值。每个模型自身验证集阈值另列，未用测试集调参。旧v1–v3输出是归一化染色强度，并非严格概率，因此0.5分类阈值仅是统一操作点，跨任务比较应优先看AUROC/AP。</p><h2>正式完成模型：主排名</h2>'''
html+=html_table(primary,['model','average_precision_histogram','auroc_histogram','f1','precision','recall','iou','micro_f1','rank_macro_ap','rank_macro_auroc','rank_macro_f1'])
html+='<img src="overall_comparison.png"><h2>各模型自身验证集阈值（次要视角）</h2>'+html_table(selected,['model','f1','precision','recall','iou','micro_f1'])
html+='<p>由于旧模型的验证患者、训练目标和输出标度不同，这一表不是完全受控的阈值消融；主结论以固定0.5和阈值无关指标为准。</p><h2>逐通道</h2><img src="channel_f1.png"><img src="channel_ap.png"><p><a href="per_channel.csv">全部逐通道数值</a></p><h2>实验清单</h2>'+pd.DataFrame(inventory).to_html(index=False)
html+='<h2>限制</h2><p>真正共同且未被旧模型训练使用的集合只有1位患者，因此不能计算可靠的患者级置信区间，也不能声称统计显著优于。positive_dice_partial在第4轮后中断，仅作探索结果；v7/v8未训练。不同版本训练数据、任务和采样不同，这是一致测试口径的模型比较，不是严格单变量消融。</p><p><a href="summary.csv">全部汇总CSV</a> · <a href="primary_ranking.csv">主排名CSV</a> · <a href="experiment_inventory.csv">实验清单CSV</a> · <a href="verification.json">一致性验证</a></p></html>'
(P/'index.html').write_text(html)
print(primary[['model','average_precision_histogram','auroc_histogram','f1','precision','recall','iou','micro_f1']].to_string(index=False))
print('REPORT',P/'index.html')
