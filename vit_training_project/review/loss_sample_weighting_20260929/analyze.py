"""Read-only loss/sampling diagnosis; no changes to training or configurations."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
PROJECT = OUT.parents[1]
RUN = PROJECT / 'runs/ddp_positive_dice'
data = RUN / 'data'
model = RUN / 'results/model'
proposal = PROJECT / 'experiments/pixel_balance'
history = pd.read_csv(model / 'train_log.csv')
history.to_csv(OUT / 'completed_epochs.csv', index=False)
progress = json.loads((model / 'progress.json').read_text())
epochs = [json.loads((model / f'epoch_metrics/epoch_{int(e):03d}.json').read_text()) for e in history.epoch]
channels = list(epochs[0]['validation']['per_class'])
first, last = epochs[0], epochs[-1]
rows = []
for i, c in enumerate(channels):
    a = first['validation']['per_class'][c]
    b = last['validation']['per_class'][c]
    tn, fp, fn, tp = np.array(b['confusion_matrix']).ravel()
    rows.append(dict(channel=c,precision=b['precision'],recall=b['recall'],f1=b['f1'],
        delta_f1=b['f1']-a['f1'],false_positive=int(fp),false_negative=int(fn),
        predicted_positive_to_gt=(tp+fp)/(tp+fn),fpr=fp/(tn+fp),
        gt_positive_fraction=(tp+fn)/(tp+fp+tn+fn),
        delta_mse=last['loss_components']['per_channel_mse'][i]-first['loss_components']['per_channel_mse'][i],
        delta_dice=last['loss_components']['per_channel_overlap'][i]-first['loss_components']['per_channel_overlap'][i]))
metrics = pd.DataFrame(rows)
metrics.to_csv(OUT / 'channel_diagnosis.csv', index=False)
train = pd.read_csv(data / 'train.csv', usecols=['patch_id','patient_id'])
draws = pd.read_csv(data / 'train_balanced.csv', usecols=['patch_id'])
repeats = draws.patch_id.value_counts()
train['draws'] = train.patch_id.map(repeats).fillna(0).astype(np.int64)
with np.load(proposal / 'runs/prepared/data/training_pixels.npz') as z:
    assert np.array_equal(train.patch_id.to_numpy(),z['patch_ids'])
    # One channel's supervised area; current data have identical valid masks across channels.
    valid = z['valid'].astype(np.int64)
    train['supervised_channel_pixels'] = valid.sum(1)
    valid_counts = valid.sum(0)
    positive_counts = z['positive'].sum(0)
    positive_patches = (z['positive'] > 0).sum(0)
train['exposure_channel_pixels'] = train.draws * train.supervised_channel_pixels
patients = train.groupby('patient_id').agg(unique_patches=('patch_id','size'),
    sampled_unique=('draws',lambda x:int((x>0).sum())),draws=('draws','sum'),
    channel_pixels=('exposure_channel_pixels','sum'))
patients['draw_share'] = patients.draws / patients.draws.sum()
patients['pixel_share'] = patients.channel_pixels / patients.channel_pixels.sum()
# Hypothetical mild patient weighting, calculated only; never applied to live training.
w = np.clip(np.sqrt(patients.draws.median()/patients.draws),.5,2.)
w /= np.average(w,weights=patients.draws)
patients['hypothetical_patient_weight'] = w
patients['weighted_draw_share'] = w*patients.draws / (w*patients.draws).sum()
patients['weighted_pixel_share'] = w*patients.channel_pixels / (w*patients.channel_pixels).sum()
patients.to_csv(OUT / 'patient_sample_counts.csv')
channel_counts = pd.DataFrame(dict(channel=channels,positive_patches=positive_patches,
    positive_patch_fraction=positive_patches/len(train),positive_pixel_fraction=positive_counts/valid_counts))
cw = 1/np.sqrt(positive_patches);cw /= cw.mean()
channel_counts['inverse_sqrt_positive_patch_weight'] = cw
channel_counts.to_csv(OUT / 'channel_sample_counts.csv',index=False)
with np.load(proposal / 'runs/prepared/data/channel_cohorts.npz') as z:
    assert np.array_equal(train.patch_id.to_numpy(),z['patch_ids'])
    cohorts=z['stratum']
stratum_rows=[]
for c,name in enumerate(channels):
    n=np.array([train.draws.to_numpy()[cohorts[:,c]==s].sum() for s in range(3)])
    q=n/n.sum()
    target=np.array([2/3,1/6,1/6])
    ideal=np.divide(target,q,out=np.full(3,np.nan),where=q>0)
    # Diagnostic candidate only; clipping sacrifices exact 4:1:1 composition.
    bounded=np.clip(ideal,.5,2)
    bounded/=np.sum(bounded*q)
    stratum_rows.append(dict(channel=name,false_draws=n[0],true1_draws=n[1],true2_draws=n[2],
        false_fraction=q[0],true1_fraction=q[1],true2_fraction=q[2],
        ideal_false_weight=ideal[0],ideal_true1_weight=ideal[1],ideal_true2_weight=ideal[2],
        capped_normalized_false_weight=bounded[0],capped_normalized_true1_weight=bounded[1],
        capped_normalized_true2_weight=bounded[2],
        capped_effective_false_fraction=q[0]*bounded[0]))
stratum_weights=pd.DataFrame(stratum_rows)
stratum_weights.to_csv(OUT/'channel_stratum_weights.csv',index=False)

plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax, title, train_key, val_key in [
    (axes[0,0],'Total loss','train_loss','val_loss'),
    (axes[0,1],'MSE component','train_mse_loss','val_mse_loss'),
    (axes[1,0],'Positive-image Dice loss','train_dice_loss','val_dice_loss')]:
    ax.plot(history.epoch,history[train_key],'o-',label='Train')
    ax.plot(history.epoch,history[val_key],'s-',label='Validation')
    ax.set(title=title,xlabel='Completed epoch',xticks=history.epoch)
    ax.grid(alpha=.2);ax.legend()
axes[1,1].plot(history.epoch,history.val_macro_f1,'o-',label='Macro F1')
axes[1,1].plot(history.epoch,history.val_macro_iou,'s-',label='Macro IoU')
axes[1,1].set(title='Validation classification at threshold 0.5',xlabel='Completed epoch',xticks=history.epoch)
axes[1,1].grid(alpha=.2);axes[1,1].legend()
fig.suptitle('Current positive-only Dice experiment | completed epochs only\nTrain and validation use different sampling/augmentation; component magnitude is not gradient magnitude')
fig.savefig(OUT/'loss_diagnosis.png',dpi=170)
fig.savefig(OUT/'loss_diagnosis.pdf')
plt.close(fig)

fig, axes = plt.subplots(1,2,figsize=(14,7),layout='constrained')
p = patients.sort_values('draws')
y = np.arange(len(p))
axes[0].barh(y-.18,p.draw_share*100,height=.35,label='Current sampling')
axes[0].barh(y+.18,p.weighted_draw_share*100,height=.35,label='Hypothetical sqrt weighting')
axes[0].axvline(100/len(p),ls='--',color='gray',label='Equal patient share')
axes[0].set(yticks=y,yticklabels=p.index,xlabel='Share of patch draws (%)',title='Patients: mild weighting changes exposure, not the split')
axes[0].legend(fontsize=8)
x = np.arange(len(channel_counts))
axes[1].plot(x,channel_counts.positive_patch_fraction*100,'o-',label='Positive patch fraction')
axes[1].plot(x,channel_counts.positive_pixel_fraction*100,'s-',label='Positive pixel fraction')
axes[1].set(xticks=x,xticklabels=channel_counts.channel,ylim=(0,105),ylabel='Natural training data (%)',title='A common positive patch can contain very few positive pixels')
axes[1].tick_params(axis='x',rotation=75);axes[1].legend()
fig.savefig(OUT/'sample_balance.png',dpi=170);plt.close(fig)

summary = dict(completed_epochs=history.epoch.tolist(),current_partial_progress=progress,
    first_completed=history.iloc[0].to_dict(),last_completed=history.iloc[-1].to_dict(),
    validation_mse_increased_channels=int((metrics.delta_mse>0).sum()),
    validation_dice_decreased_channels=int((metrics.delta_dice<0).sum()),
    validation_f1_improved_channels=int((metrics.delta_f1>0).sum()),
    training_unique=len(train),sampled_unique=len(repeats),sampling_rows=len(draws),
    max_repeat=int(repeats.max()),repeat_weight_kish_ess=float(repeats.sum()**2/(repeats**2).sum()),
    ess_interpretation='draw-concentration statistic only; NOT independent biological sample size',
    patient_draw_max_min=float(patients.draws.max()/patients.draws.min()),
    patient_pixel_max_min=float(patients.channel_pixels.max()/patients.channel_pixels.min()),
    hypothetical_patient_weight_min=float(w.min()),hypothetical_patient_weight_max=float(w.max()),
    proposed_weighted_patient_draw_max_min=float((w*patients.draws).max()/(w*patients.draws).min()),
    training_changed=False,
    input_sha256={str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [model/'train_log.csv',data/'train.csv',data/'train_balanced.csv']})
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))

lines = ['# 当前loss与按样本数加权分析', '',
'本次仅分析，没有修改或中断训练。统计取已完成的前两轮；第3轮部分训练均值不与完整轮次直接比较。', '',
'## 曲线结论', '',
'|指标|第1轮|第2轮|', '|---|---:|---:|']
for key in ['train_loss','val_loss','train_mse_loss','val_mse_loss','train_dice_loss','val_dice_loss','val_macro_f1','val_macro_iou']:
    lines.append(f'|{key}|{history.iloc[0][key]:.6f}|{history.iloc[-1][key]:.6f}|')
lines += ['', '![曲线](loss_diagnosis.png)', '',
'验证总loss仍下降，不能称为持续恶化。验证MSE上升约2.57%，阳性Dice loss下降约3.15%，当前总loss的改善来自Dice。16通道中11个MSE上升、16个soft Dice loss下降、13个阈值0.5下F1改善，说明软损失、阈值分类和概率误差不是同一个目标。', '',
'训练/验证在第2轮出现差距，但只有两轮，且训练存在重采样和增强，不能认定已过拟合。前两轮还处于学习率warmup，学习率翻倍；这可能影响变化幅度，但不能由曲线单独确定原因。当前loss定义及训练患者组成不同于旧baseline，不能用二者总loss绝对值判定模型优劣。Dice数值较大也不证明其梯度一定主导，尚未测量各项梯度范数。', '',
'## 通道需要区别处理', '',
'|通道|Precision|Recall|F1|预测阳性像素/GT阳性像素|F1变化|', '|---|---:|---:|---:|---:|---:|']
for r in metrics.itertuples():
    lines.append(f'|{r.channel}|{r.precision:.3f}|{r.recall:.3f}|{r.f1:.3f}|{r.predicted_positive_to_gt:.3f}|{r.delta_f1:+.3f}|')
lines += ['',
'Pan-CK预测阳性面积约为GT的1.41倍，Recall高而Precision较低，进一步强推阳性不合适。PDL1仅约0.34倍，Recall约0.185，主要表现为漏检；不能与Pan-CK套用同一方向的校正。FOXP3约1.01倍，但Precision/Recall都约0.17，说明总阳性量接近并不代表空间定位正确；仅改变阳性数量权重不足以解决。CD20 Precision约0.786、Recall约0.671，也不能简单归为稀疏通道过预测。', '',
'上述判断仅基于当前固定0.5阈值的验证统计。当前日志没有正、负像素分开的MSE，不能由总MSE确定分别是哪一部分贡献了全部上升。已有候选代码补充了这类日志，但未在当前训练中启用。', '',
'## 样本数到底指什么', '',
'|加权层级|能解决什么|本项目的局限|', '|---|---|---|',
'|按通道总有效patch数|减少可用监督量差异|当前通道有效像素支持数相同，主要差异不是有没有监督|',
'|按通道阳性patch数取倒数或平方根倒数|提高少见阳性patch通道的贡献|FOXP3在很多patch存在，但像素极稀疏；patch数无法反映像素稀疏程度|',
'|按正负像素数|调整正负误差的相对贡献|无上限逆频数会强推稀疏阳性，并改变概率校准；不能保证减少假阳性|',
'|按患者patch数量|缓解大切片/多patch患者主导|更贴近跨患者泛化目标，但不能消除染色、组织及标签差异|',
'|按重复patch的出现次数|降低重复数据支配|应先限制重复次数；直接再乘1/重复次数可能抵消原本设计的分层采样|', '',
'当前loss本来已按通道分别归一化：MSE除以该通道有效像素数，阳性Dice除以该通道阳性图像数，再对通道加权。它不是将所有通道误差不归一化地直接相加，因此少量通道不会仅因数量少而天然没有贡献。再做逆样本数通道加权属于第二次加强少量通道。', '',
'训练自然数据的实际数量：', '',
'|通道|阳性patch数|阳性patch占比|阳性像素占比|归一化1/√阳性patch数权重|', '|---|---:|---:|---:|---:|']
for r in channel_counts.itertuples():
    lines.append(f'|{r.channel}|{r.positive_patches}|{r.positive_patch_fraction:.1%}|{r.positive_pixel_fraction:.2%}|{r.inverse_sqrt_positive_patch_weight:.3f}|')
lines += ['', '例如FOXP3约69.2%的训练patch有阳性，而阳性像素仅约0.81%。按阳性patch计数并不极端稀缺，按像素计数却很稀缺，不能把二者混为一谈。这里的占比来自训练集256网格、增强前有效监督区域。', '',
'## 如果“样本数加权”指false/true1/true2数量', '',
'这比所有通道统一加强阳性更直接对应你希望的false:true=2:1。对每个通道的可评价图像/通道对，目标比例t=(2/3,1/6,1/6)，实际抽样比例q=(q_false,q_true1,q_true2)，理论校正权重为w=t/q。它是调整三组贡献，不等于正像素都乘上更大权重。应使用最终采样后的q，而不是一边重采样、一边再按原始分布乘逆频数。', '',
'下面用当前正在训练的抽样清单，按已准备的256网格队列重新计数；仅展示理想权重，不用于训练：', '',
'|通道|当前false抽样占比|理想false权重|理想true1权重|理想true2权重|', '|---|---:|---:|---:|---:|']
for name in ['Hoechst','FOXP3','CD31','PDL1','Pan-CK']:
    r=stratum_weights.set_index('channel').loc[name]
    lines.append(f'|{name}|{r.false_fraction:.2%}|{r.ideal_false_weight:.2f}|{r.ideal_true1_weight:.2f}|{r.ideal_true2_weight:.2f}|')
lines += ['',
'极少数可信false可能得到很大权重，此时权重虽不重复读取图像，仍然会让少数patch主导梯度。更稳妥的候选是温和化或限制最大/最小权重比，例如先限制到[0.5,2]再按q归一化；归一化后绝对值不再严格位于[0.5,2]，但组间比不超过4。封顶后的目标也不再严格等于4:1:1。完整数值在channel_stratum_weights.csv。', '',
'如果每图像/通道先求平均loss再乘w，t/q校正对应图像/通道对分布；若直接给全部像素乘w、最后按有效像素数归一化，结果还受patch有效面积影响，不能声称已严格匹配patch比例。阳性Dice本来不包含false图像，因此false组权重主要作用于MSE的阴性约束，不能给一个本来被排除的Dice项凭空增加惩罚。', '',
'建议将这种“有上限的分组loss加权”作为分层重采样的替代消融，而不是不加区分地叠加在同目标的重采样之上。能否提高准确率仍需固定验证集比较，特别关注PDL1等漏检较多通道是否被进一步压低。', '',
'## 更值得验证：温和的患者数量平衡', '',
f'当前固定训练清单共{len(draws):,}行，覆盖{len(repeats):,}/{len(train):,}个独立patch；单patch最多重复{repeats.max()}次。CRC04抽样42652次，CRC10为7771次，相差{summary["patient_draw_max_min"]:.2f}倍。患者样本最多者也不等于原始patch最多者，因为通道抽样会改变患者分布。', '',
'已准备的像素平衡方案保证自然数据覆盖并限制每patch最多4次，有助于降低重复集中；其中“打乱患者”只改变顺序，并不自动令患者贡献相等。', '',
'如果后续做数量加权消融，建议先试训练患者级的温和权重：`u_p=clip(sqrt(median(m)/m_p),0.5,2)`，再按实际抽样次数归一化到均值1。m_p取训练采样器预计每轮给该患者的抽样次数，只在训练集拟合，不使用验证患者表现来分配权重。数值应在最终采样方案确定后重新拟合；这里仅做现有采样清单的敏感性分析。', '',
f'对当前清单，归一化权重范围{w.min():.3f}～{w.max():.3f}，患者“加权抽样次数”的最大/最小比从{summary["patient_draw_max_min"]:.2f}降到{summary["proposed_weighted_patient_draw_max_min"]:.2f}。这不等于精度会提高，也不等于按有效像素计的贡献已经严格相等；对应像素占比另存patient_sample_counts.csv。', '',
'实现时MSE应同时加权分子和分母：Σ[u_p×有效掩膜×误差²]/Σ[u_p×有效掩膜]；Dice也在符合阳性条件的图像/通道对上做Σ[u_p×DiceLoss]/Σ[u_p]。四卡需汇总加权分子、分母，不能简单平均各卡局部均值。若改用患者平衡采样来达到同一目的，不应同时再完整叠加逆患者数loss权重。', '',
'这只是值得测试的方向，不是已实现或已验证的新训练配置。若目标是每患者等重要，应同时报告每患者指标均值与离散程度，不能仅看汇总像素指标。', '',
'![样本与像素比例](sample_balance.png)', '',
'## 有效样本数方法与本项目的关系', '',
'Class-Balanced Loss采用有效样本数E=(1−β^n)/(1−β)而不是机械按1/n放大少数类，反映重复增加数据的边际收益递减。[CVPR 2019原论文](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)。该研究不构成本项目病理像素任务会提升的证据。', '',
'多标签还存在共表达：为了FOXP3抽一张patch，也会改变Hoechst、CD4等通道的分布。Distribution-Balanced Loss专门讨论标签共现和阴性标签占优，但它围绕BCE构建，不能直接视作当前MSE+Dice的等价替代。[ECCV 2020原论文](https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/1631_ECCV_2020_paper.php)。', '',
'β并非无关紧要：若按十几万阳性patch计数而使用β=0.9999，β^n已接近0，通道间权重几乎相同。应以去重训练patch或独立患者支持量解释n，不能把重复抽样记录或高度相关的数十亿像素直接当成独立生物样本数。', '',
'## 建议的判断顺序', '',
'1. 原实验继续，先观察更多完整轮次；目前验证F1和总loss在改善，没有足够依据中途改变目标。',
'2. 下一次消融先观察自然覆盖、重复上限和患者温和平衡的作用；正负MSE先保持同权，避免同时更换太多因素。',
'3. 在同一采样条件下单独比较Dice系数1与0.5；再单独比较是否需要按像素比例轻度加权。当前候选三份配置可支持部分对照，但不是完整的患者平衡消融。',
'4. 固定验证患者、固定0.5阈值，查看宏F1/IoU、逐通道Precision/Recall/FPR、预测面积比及患者间差异。阈值校准另做验证内检查，测试集不参与权重或阈值选择。', '',
'全部表格和输入哈希在本目录；没有改动运行中的代码、配置或患者划分。', '']
(OUT/'分析报告.md').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))
