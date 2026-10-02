"""Render the prepared, non-training experiment's measured audit results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
review = ROOT / 'review'
weights = pd.read_csv(review / 'pixel_weight_table.csv')
cohorts = pd.read_csv(review / 'channel_cohort_counts.csv')
thresholds = pd.read_csv(review / 'threshold_stability/stability_summary.csv')
pooled = thresholds[thresholds.objective == 'pooled_pixels'].copy()
folds = pd.read_csv(review / 'threshold_stability/leave_one_patient_out.csv')
folds = folds[folds.objective == 'pooled_pixels']

fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout='constrained')
y = np.arange(len(pooled))
axes[0].hlines(y, pooled.threshold_min, pooled.threshold_max, color='#34699a', lw=3)
for i, name in enumerate(pooled.channel):
    rows = folds[folds.channel == name]
    axes[0].scatter(rows.threshold, np.full(len(rows), i), s=18, color='#34699a', alpha=.65)
axes[0].axvline(.5, color='#999999', ls='--')
axes[0].set(yticks=y, yticklabels=pooled.channel, xlim=(0, 1), xlabel='Threshold fitted on other 5 patients', title='Leave-one-patient-out threshold range')
axes[0].invert_yaxis()
delta = pooled.mean_heldout_f1.to_numpy() - pooled.mean_heldout_f1_at_05.to_numpy()
axes[1].barh(y, delta, color=np.where(delta >= 0, '#268b6a', '#bb514b'))
axes[1].axvline(0, color='#999999', lw=1)
axes[1].set(yticks=y, yticklabels=pooled.channel, xlabel='Mean held-out F1 change vs threshold 0.5', title='Threshold fitting does not always improve F1')
axes[1].invert_yaxis()
fig.suptitle('Old baseline, epoch 14 | 6 validation patients x 64 patches | pilot only', fontsize=12)
fig.savefig(review / 'threshold_stability.png', dpi=180)
fig.savefig(review / 'threshold_stability.pdf')
plt.close(fig)

schedule = json.loads((review / 'sampling_epoch_001.json').read_text())
lines = ['# 像素平衡方案与跨患者阈值检查（2026-09-29）', '',
'状态：仅准备与检查，没有启动候选方案。原 `ddp_positive_dice` 四卡训练继续，原模型代码哈希与运行记录一致。', '',
'## 1. 当前问题能确认到什么程度', '',
'现有通道权重为0.862～1.103，FOXP3约1.010、CD31约1.012、PDL1约1.055。没有证据表明当前 `1/σ` 把稀疏通道放大了很多倍。它是训练集阳性强度的标准差权重，不是阳性像素比例的倒数。', '',
'当前MSE中正、负像素原本同权；阳性图像/通道才计算Dice。这个Dice仍包含该图像内有效阴性像素，假阳性会增大分母；完全阴性的图像/通道只由MSE约束。当前第1轮未完成，不能据此认定阳性Dice实验已经失败。', '',
'Dice数值比MSE大，不等于梯度影响也按相同比例更大。候选Dice系数0.5是待验证的消融参数，不是由损失数值推导的最优值。', '',
'## 2. 原1/σ如何作用', '',
'对训练集阳性强度拟合σ，先除255，再取 `1/max(σ,0.001)`，均值归一化、裁剪到[0.25,4]后再次均值归一化。得到每通道γ，整体损失为各通道 `γ × (MSE + λDice)` 的加权平均。', '',
'它同时影响该通道的MSE和Dice，不单独区分正负像素；也不是以二值标签的标准差计算。对于强度回归有尺度平衡动机，改成二值分类后缺乏直接对应关系，因此新方案将γ设为1，避免叠加难解释的权重。', '',
'## 3. 固定患者集合，重新组织训练采样', '',
'保持train/val/test患者28/6/6不变，patch数223400/56832/38218；只改变训练抽样顺序和重复次数。每轮打乱患者顺序、患者内patch顺序并交错混合，验证与测试保持自然分布。没有重新清洗或排除DAPI阴性patch。', '',
'按模型256×256标签统计：覆盖率=该通道阳性像素数/有效监督像素数；true1高于训练阳性patch平均覆盖率，true2大于0且不高于平均值。false要求通道有效、有可监督像素、原始分辨率与256网格都无该通道阳性。原始微小信号缩小后消失的patch不直接视为可信false。未知通道不参与监督。', '',
'额外的每通道定向抽样采用false:true1:true2=4:1:1，即false:true=2:1；同时保证每个训练patch每轮至少出现一次，总重复不超过4次。定向额度不足时以自然抽样补齐，稀缺阴性优先保留额度。', '',
f'第1轮预生成：{schedule["epoch_size"]:,}次抽样、{schedule["unique_patches"]:,}张独立patch；自然抽样{schedule["natural_draws"]:,}次，定向抽样{schedule["targeted_draws"]:,}次；4卡每卡180736次，batch64，2824个全局更新，与现行训练更新数一致。', '',
'**4:1:1仅保证每通道额外抽样子清单。** 同一patch同时监督全部有效通道，因此最终混合后的各通道边际比例不可能据此声称严格2:1。Hoechst仅309张可信false，若强行达到全局2:1会严重重复少数阴性patch。数据相关性、自然覆盖与重复上限使严格全通道比例通常不可同时满足。', '',
'所有通道的原始队列数量：', '',
'|通道|false|true1|true2|不作该通道定向抽样|true1覆盖率分界|',
'|---|---:|---:|---:|---:|---:|']
for r in cohorts.itertuples():
    lines.append(f'|{r.channel}|{r.false}|{r.true1}|{r.true2}|{r.excluded_anchor}|{r.threshold:.2%}|')
lines += ['', '“不作定向抽样”包括不可评价或缩放丢失微小阳性等情况，仍保留原patch及其他有效通道监督。实际总阴性patch比例在sampling_epoch_*.json的actual_negative_patch_fraction中；额外抽样额度及不足原因见anchor_counts_epoch_*.csv。', '',
'## 4. 正负像素权重候选及其风险', '',
'α=clip[1+0.25×ln(N负/N正), 0.5, 2]；负像素权重1。正/负任一支持量不足100像素时回退α=1。统计仅使用训练集前三轮预生成采样曝光量，不使用验证/测试标签，且在随机增强前计算；后续各轮会有小幅偏差。', '',
'每通道MSE = Σ有效像素[(αy+1−y)(p−y)²] / Σ有效像素(αy+1−y)，p=sigmoid(logit)。分母也加权，避免仅增加样本权重却任意改变损失尺度。阳性Dice不额外叠加α。背景和缺失标签在二者中均忽略。', '',
'这不是强制正负总贡献各50%。例如阳性占0.77%、α=2时，阳性总权重仍仅约1.53%，避免1/π式极强加权。α>1依然可能增加假阳性，不能把它描述为必然抑制假阳性的方法。仅考虑加权MSE时，最优预测是αη/(1−η+αη)，不再等于原始概率η；α=2时预测阈值0.5对应η=1/3。Dice也可能改变概率校准，因此需同时检查precision、recall、FPR、预测阳性面积及验证阈值稳定性。', '',
'优先与“不额外加强阳性像素”对照比较，再判断温和加权是否必要。下面的比例来自实际模型网格，既不是patch数量比例，也不是原始图像强度比例。', '',
'|通道|原γ(1/σ)|旧采样阳性像素占比|新采样占比|候选α|',
'|---|---:|---:|---:|---:|']
for r in weights.itertuples():
    lines.append(f'|{r.channel}|{r.old_inverse_sigma_weight:.3f}|{r.previous_sampling_positive_fraction:.2%}|{r.positive_fraction:.2%}|{r.positive_pixel_weight:.3f}|')
lines += ['', '新采样降低了多数通道的阳性像素占比，但CD31、ECadherin和SMA略升，不能宣称所有通道都已减少阳性曝光。患者打乱只改变优化顺序，不消除患者间染色与表达差异。', '',
'已准备的三份配置（均未启动）：', '',
'|配置|像素α|MSE:Dice|作用|', '|---|---|---|---|',
'|train_pixel_balance_no_positive_boost.json|全部1|1:0.5|优先比较，不额外加强阳性MSE|',
'|train_pixel_balance.json|按比例、上限2|1:0.5|测试是否需要温和像素平衡|',
'|train_pixel_balance_dice1_control.json|同上|1:1|与上一份仅差Dice系数|', '',
'三份都采用相同新采样、均匀通道γ与训练步数。与当前训练比较会同时改变多个因素，不能据此单独归因；这三份之间可分别观察α或Dice系数的作用，但还不是完整的采样/通道γ消融矩阵。损失定义不同，不能横向仅看trainloss/valloss大小。主比较使用同一患者、同一阈值0.5下的分类指标及按患者指标，再单独比较验证校准后的指标。', '',
'## 5. 跨患者阈值检查', '',
'旧已完成baseline的best epoch14；6位验证患者每位固定随机64张，共384张；逐通道在其他5位患者上选择F1阈值，评价留出的1位患者。使用CPU BF16推理，不占用正在训练的GPU，CPU/GPU数值可能稍有差异。未访问测试集。', '',
'这是验证子集的阈值迁移诊断：模型本身曾用同一验证集合选择checkpoint，因此不是完整嵌套交叉验证，也不是新训练模型的结果。全量验证和新模型尚需复核；没有改写正式阈值文件。', '',
'|通道|六折拟合阈值范围|留出患者平均F1(调阈值)|固定0.5平均F1|', '|---|---|---:|---:|']
for r in pooled.itertuples():
    lines.append(f'|{r.channel}|{r.threshold_min:.3f}–{r.threshold_max:.3f}|{r.mean_heldout_f1:.4f}|{r.mean_heldout_f1_at_05:.4f}|')
lines += ['',
'PDL1五折阈值接近搜索下限0.05，留出CRC16时升至0.447。CRC16抽样有效像素中PDL1阳性约52.3%，其他患者差异很大；这表明汇总F1最优阈值对患者组成敏感。PDL1平均F1由固定0.5的0.3500降到0.3055，不能机械套用汇总最优阈值。', '',
'CD31阈值范围很宽，但留出平均F1与0.5相近；宽阈值范围本身不能证明性能同样大幅波动，可能涉及平坦最优区或样本噪声。FOXP3、CD8a调阈值改善F1，同时仍需检查precision代价。另算了患者等权的混淆统计目标，未解决PDL1/CD31不稳定。该目标并非直接最大化各患者F1的算术均值。', '',
'![阈值范围与留出F1变化](threshold_stability.png)', '',
'## 6. 检查范围与文件', '',
'24项CPU单元测试通过；四进程CPU DDP全局损失/梯度检查通过；包含新采样及像素权重的四进程微型模型生产流程通过（训练、早停、恢复、校准、测试、快照、输出）。真实缓存4张patch的加载、增强、损失、背景零梯度通过。训练像素统计另抽64条与标准缓存读取及最近邻缩放结果逐值核对。', '',
'候选方案尚未做真实模型GPU训练或四卡显存容量验证；CPU微型模型通过不等于候选精度已验证。未来启动前仍走现有容量校验，保持AdamW、余弦退火、40轮上限、patience8及batch64候选。', '',
'详细数据：pixel_weight_table.csv、channel_cohort_counts.csv、anchor_counts_epoch_*.csv、sampling_epoch_*.json；阈值逐折数据及抽样名单在threshold_stability/。原始缓存与固定划分在runs/prepared/复用，仅新增统计与配置；status.json记录prepared_not_started。', '']
(review / '方案与检查结果.md').write_text('\n'.join(lines))
print(review / '方案与检查结果.md')
