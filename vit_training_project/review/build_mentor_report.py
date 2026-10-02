"""Build a source-linked Chinese PDF report and tabular evidence, without training."""
from pathlib import Path
import json, hashlib, shutil, csv, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from xml.sax.saxutils import escape

P=Path(__file__).resolve().parents[1];ROOT=P.parent
OUT=ROOT/'reports/mentor_report_20261002';OUT.mkdir(parents=True,exist_ok=True)
(OUT/'tables').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
COMMON=P/'review/unified_all_experiments_20261001'
VIS=P/'review/v11_visualizations_20261002'
MODELS=['v1','v2','v3','v4','v5','v6','v7','v9','v10','v11']
RUNS={'v4':P/'runs/ddp_baseline','v5':P/'experiments/binary_bce/runs/prepared',
      'v6':P/'experiments/v6_v7_v8/runs/v6','v7':P/'experiments/v6_v7_v8/runs/v7',
      'v9':P/'experiments/v6_v7_v8_emptydice/runs/v9','v10':P/'experiments/v10/runs/v10','v11':P/'experiments/v11/runs/v11'}
def read(p):return json.loads(Path(p).read_text())
CHANNELS=list(read(RUNS['v11']/'results/test_metrics.json')['per_class'])
records=[];summaries=[];sources=[]
for m in MODELS:
    for policy in ['fixed_05','own_validation']:
        path=COMMON/m/f'expanded_{policy}.json'
        d=read(path);sources.append(path)
        summaries.append(dict(model=m,test='CRC02_6638',policy=policy,**d['macro']))
        for c,v in d['per_class'].items():
            row=dict(model=m,test='CRC02_6638',policy=policy,channel=c,**{k:x for k,x in v.items() if k!='confusion_matrix'})
            if 'confusion_matrix' in v:
                (tn,fp),(fn,tp)=v['confusion_matrix'];row.update(tn=tn,fp=fp,fn=fn,tp=tp)
            records.append(row)
for m in ['v6','v7','v9','v10','v11']:
    for policy,filename in [('fixed_05','test_metrics_default05.json'),('own_validation','test_metrics.json')]:
        path=RUNS[m]/'results'/filename;d=read(path);sources.append(path)
        assert d['n_patches']==38218
        summaries.append(dict(model=m,test='full_38218',policy=policy,**d['macro']))
        for c,v in d['per_class'].items():
            row=dict(model=m,test='full_38218',policy=policy,channel=c,**{k:x for k,x in v.items() if k!='confusion_matrix'})
            (tn,fp),(fn,tp)=v['confusion_matrix'];row.update(tn=tn,fp=fp,fn=fn,tp=tp);records.append(row)
df=pd.DataFrame(records);summary=pd.DataFrame(summaries)
partial_path=COMMON/'positive_dice_partial/expanded_fixed_05.json'
partial=read(partial_path);sources.append(partial_path)
partial_table=pd.DataFrame([dict(channel=c,**{k:x for k,x in v.items() if k!='confusion_matrix'}) for c,v in partial['per_class'].items()])
partial_table.to_csv(OUT/'tables/interrupted_positive_dice_channels.csv',index=False)
df['false_positive_rate']=df.fp/(df.fp+df.tn)
df.to_csv(OUT/'tables/all_channel_metrics.csv',index=False)
summary.to_csv(OUT/'tables/overall_metrics.csv',index=False)
history={};trainrows=[]
for m,run in RUNS.items():
    hp=run/'results/model/classification_history.json';rows=read(hp);history[m]=pd.DataFrame(rows);sources.append(hp)
    best=max(rows,key=lambda x:x['val_macro_iou'])
    trainrows.append(dict(model=m,completed_epochs=len(rows),best_epoch=best['epoch'],best_val_iou=best['val_macro_iou'],
        best_val_f1=best['val_macro_f1'],last_train_loss=rows[-1]['train_loss'],last_val_loss=rows[-1]['val_loss'],
        recorded_epoch_hours=sum(r.get('epoch_seconds',0) for r in rows)/3600))
training=pd.DataFrame(trainrows);training.to_csv(OUT/'tables/training_summary.csv',index=False)
with pd.ExcelWriter(OUT/'tables/导师汇报_完整指标.xlsx') as writer:
    summary.to_excel(writer,sheet_name='overall',index=False);df.to_excel(writer,sheet_name='all_channel_metrics',index=False)
    training.to_excel(writer,sheet_name='training',index=False)
    partial_table.to_excel(writer,sheet_name='interrupted_positive_dice',index=False)
    for scope in ['CRC02_6638','full_38218']:
        for metric in ['precision','recall','f1','iou','auroc_histogram','average_precision_histogram']:
            t=df[(df.test==scope)&(df.policy=='own_validation')].pivot(index='channel',columns='model',values=metric).reindex(CHANNELS)
            t=t[[m for m in MODELS if m in t.columns]]
            t.to_csv(OUT/'tables'/f'{scope}_{metric}.csv')
            t.to_excel(writer,sheet_name=(scope+'_'+metric.replace('_histogram',''))[:31])
for f in ['patch_balance.csv','patch_balance_per_epoch.csv','patch_balance_definition.json']:
    shutil.copy2(RUNS['v11']/'results'/f,OUT/'tables'/f)
shutil.copy2(RUNS['v11']/'sampling_audit.csv',OUT/'tables/sampling_audit.csv')

fig,axes=plt.subplots(2,3,figsize=(15,8))
for ax,m in zip(axes.flat,['v6','v7','v9','v10','v11']):
    h=history[m];ax.plot(h.epoch,h.train_loss,label='train');ax.plot(h.epoch,h.val_loss,label='validation');ax.set_title(m+' loss');ax.set_xlabel('epoch');ax.legend();ax.grid(alpha=.2)
axes.flat[-1].axis('off');fig.tight_layout();fig.savefig(OUT/'figures/loss_curves.png',dpi=150);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for m in ['v6','v7','v9','v10','v11']:
    h=history[m]
    for ax,key in zip(axes,['val_macro_iou','val_macro_f1']):ax.plot(h.epoch,h[key],label=m);ax.set_title(key);ax.set_xlabel('epoch');ax.grid(alpha=.2);ax.legend()
fig.tight_layout();fig.savefig(OUT/'figures/validation_curves.png',dpi=150);plt.close(fig)

pdfmetrics.registerFont(TTFont('Chinese','/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'))
pdfmetrics.registerFont(TTFont('Latin','/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
styles=getSampleStyleSheet()
for style in styles.byName.values():style.fontName='Chinese';style.wordWrap='CJK'
styles['Normal'].fontSize=10;styles['Normal'].leading=16
styles['Heading1'].fontSize=19;styles['Heading1'].leading=25
styles['Heading2'].fontSize=13;styles['Heading2'].leading=20
styles.add(ParagraphStyle(name='SmallCN',fontName='Chinese',fontSize=7,leading=10,wordWrap='CJK'))
story=[]
def mixed(text):
    return re.sub(r'[^\u2e80-\u9fff\uff00-\uffef]+',lambda m:'<font name="Latin">'+escape(m.group())+'</font>',str(text))
def para(text,style='Normal'):story.append(Paragraph(mixed(text),styles[style]));story.append(Spacer(1,6))
def title(text):para(text,'Heading1')
def table(frame,digits=4):
    frame=frame.copy()
    for c in frame:
        if pd.api.types.is_float_dtype(frame[c]):frame[c]=frame[c].map(lambda x:f'{x:.{digits}f}' if pd.notna(x) else '—')
    data=[[Paragraph(mixed(x),styles['SmallCN']) for x in frame.columns]]
    data += [[Paragraph(mixed(x),styles['SmallCN']) for x in row] for row in frame.to_numpy()]
    t=Table(data,repeatRows=1,hAlign='LEFT',colWidths=[750/len(frame.columns)]*len(frame.columns))
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dce8f5')),('VALIGN',(0,0),(-1,-1),'TOP'),('GRID',(0,0),(-1,-1),.25,colors.HexColor('#ccd3db')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f6f8fb')]),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]));story.append(t);story.append(Spacer(1,10))
def pic(path,width=740,height=430):
    from PIL import Image as PILImage
    with PILImage.open(path) as im:w,h=im.size
    factor=min(width/w,height/h);story.append(Image(str(path),width=w*factor,height=h*factor))
def page():story.append(PageBreak())

title('H&E → 多标志物虚拟免疫荧光：实验与结果报告')
para('导师汇报版｜证据截止：2026 年 10 月 2 日｜版本范围：v1–v11及中断实验')
para('研究目标：从 H&E 图像预测 16 个标志物的逐像素表达区域。当前任务为独立多标签二分类，保留共表达；不是互斥 softmax 多分类。报告以实际代码、冻结配置、训练日志和测试结果为依据，不将讨论中的方案当作已完成实验。')
para('主要结论','Heading2')
para('完整 38,218-patch 测试集上，验证集阈值下 v11 macro F1=0.5532，高于 v7 的 0.5493 与 v10 的 0.5511，但差距较小。AP 以 v10 的 0.5521 略高于 v11 的 0.5516。固定 0.5 阈值时 v7 F1=0.5537，优于 v10/v11，说明加权、采样与阈值存在相互影响。')
para('没有一个版本在所有通道上占优。增加稀疏阳性权重和采样没有根本解决 FOXP3、CD31、CD163 等通道。所有主要结论均为单次实验结果，未进行多随机种子或患者级置信区间检验，不能声称统计显著。')
para('本报告新增：补齐 v7/v10/v11 在共同 CRC02 测试集 6,638 张 patch 上的全量推理，现可与 v1–v6/v9 在相同像素掩膜下对照。共同测试只有一位患者，不能替代跨患者评估。')
para('交付物：本 PDF、完整逐通道 Excel/CSV、训练与采样统计、24 张 matched patch 的 72 张 PNG、可独立审阅的代码快照。原始图像与模型权重不纳入 Git 代码提交。')
page();title('1．数据来源、标签与质控边界')
para('数据沿用 MIPHEI 发布的 ORION CRC patch 与既有质控结果，未再次执行全量 StarDist 清洗。历史提出的 DAPI Dice<0.2 全局剔除规则没有用于当前训练；DAPI 阴性但其他通道阳性的 patch 保留。已有核掩膜的原始生成记录未完整核实，不能据此声称新增核一致性质控已完成。')
para('当前源数据共 318,450 个 patch，40 个患者分组：train 28 位/223,400 张，val 6 位/56,832 张，test 6 位/38,218 张；患者隔离，CRC33 多切片合并为同一患者。所有统计权重只由训练集拟合。历史模型使用的划分不完全一致，因此另设共同测试患者 CRC02。')
para('mIF 为逐通道标签，不是单一总标签。读取时支持 16/17 通道 uint8；17 通道经既定映射选择 16 个标志物。原始标签含强弱信息。当前二值视图将非零前景记为 1，0 记为 0，有效性独立存储。模型不回归染色强度。原生训练 patch 333×333，模型标签与 CNN 网格 256×256；ViT 输入 224×224。')
para('v6 及以后有效像素 = H&E 组织掩膜 AND 已记录的通道可用性。组织内所有标志物均为 0 的像素也纳入负监督；组织外和缺失通道忽略。“可靠”来自已有可用性记录，并不代表完成了新的人工染色质量审核。组织掩膜由 H&E 得到，不使用 DAPI 核区域裁掉核外标志物。')
para('背景在归一化后也置零，损失和指标忽略背景。卷积前向仍会处理整张张量，不能描述为背景完全不经过网络。模型亦不能仅凭 H&E 判断某一 GT 阴性到底是真阴性还是染色失败，这需要原图复核或外部染色证据。')
para('数据增强（训练时在线；验证/测试不随机增强）','Heading2')
table(pd.DataFrame([['直角旋转','0/90/180/270°等概率','H&E、标签、掩膜同步'],['翻转','50%水平翻转，结合旋转覆盖D4','同步'],['小幅仿射','35%；角度±10°，缩放0.95–1.05','图像线性、标签最近邻'],['光度变化','50%；对比度0.95–1.05，亮度±5/255，RGB各0.97–1.03','仅H&E'],['边界处理','仿射空白忽略；背景置零','不生成伪阴性标签']],columns=['操作','范围','说明']))
page();title('2．模型架构与实际训练范围')
para('当前共同骨架：H&E → 冻结 Virchow2 编码器＋LoRA → 第 8/16/24/32 层特征；各层 1×1 投影到 256 通道后拼接融合为 512 通道。四层均为同一 patch-grid 分辨率，是语义层级融合，不是四个不同空间分辨率。')
para('并行 CNN DetailCapture 提取 stride 1/2/4/8/16 特征，通道为 32/64/128/128/64。浅层使用不同膨胀率卷积分支。ViT 融合特征与 CNN s16 在 decoder 入口拼接，随后逐级上采样、拼接 CNN 跳连和卷积，最终输出 16 通道 logits，sigmoid 得到独立概率。')
para('32 个 Transformer 注意力层均注入 Q/V LoRA，rank=32、alpha=16；K 和 MLP 未加 LoRA。不能将“每层有 LoRA”误解为该层所有子模块均微调。LoRA 可训练参数约 5.243M，CNN/neck/decoder 也参与训练。')
para('未实施：每个 Transformer block 均与 CNN 双向融合、基于 StarDist 的核/胞质/细胞外三域先验。相关文件为设计蓝图，不能归因于现有实验收益。仅核分割无法可靠定义完整细胞质边界。')
para('近期实验共同训练条件','Heading2')
table(pd.DataFrame([['并行/精度','4卡DDP，BF16，SyncBatchNorm'],['batch','每卡64，全局256；873批/轮'],['优化器','AdamW；LoRA lr=1e-4，decoder lr=3e-4，weight decay=0.01'],['调度','2轮warmup，余弦退火，最低学习率比例0.05'],['训练上限/早停','40轮；patience=8，min_delta=0.001，前10轮不累计坏轮'],['模型选择','固定0.5阈值的验证 macro IoU；不是按最低val loss选best'],['阈值','固定best后仅在val选逐通道阈值；test不拟合阈值'],['保存','best/last，逐轮loss、指标、耗时、patch快照；v11另记采样统计']],columns=['项目','配置']))
page();title('3．实验版本总表')
versions=[['v1','历史回归','512输入，末层ViT，旧单层融合','有checkpoint；旧划分'],['v2','历史回归','512输入/448 ViT；四层融合＋多尺度CNN','完成'],['v3','log强度回归','256/224；log归一化，前景MSE权重11','完成'],['v4','MSE分类基线','sigmoid概率＋MSE/重叠损失；旧有效掩膜与采样','完成；原test口径不同'],['阳性Dice试验','MSE分类','仅阳性图像/通道计算Dice','第4轮中断，仅探索参考'],['v5','加权BCE＋Dice','旧有效掩膜；阳性=N-/N+；外层历史通道权重','完成'],['v6','BCE＋原版Dice','全阴性组织纳入；正负均1，通道均1，打乱patch','28轮，best20'],['v7','BCE＋原版Dice','v5冻结阳性比值开平方；其余沿用v6','18轮，best5'],['v8','BCE＋原版Dice','无阳性加权；像素驱动patch采样','已准备，未单独训练'],['v9','BCE＋分段重叠损失','v6基础：全阴性图像/通道用mean(p)','18轮，best9'],['v10','BCE＋原版Dice','当前N-/N+；权重min(16,max(1,r)^0.6)','18轮，best5'],['v11','v10损失＋v8采样','权重不变；有放回像素驱动patch采样','18轮，best8'],['emptydice_v6_trial','阴性Dice尝试','后撤回，恢复原v6继续训练','第1轮停止，无可比较checkpoint']]
table(pd.DataFrame(versions,columns=['版本','任务/损失','主要变化','状态']))
para('v6→v7为阳性权重变化；v7→v10同时改变权重统计口径、指数和上下限，不能拆解归因；v10→v11仅改变采样，最适合分析采样影响。v5→v6同时改变有效掩膜、阳性权重和通道外层权重，不是单因素消融。各版本均为独立实验，不代表顺序继承上一版已训练checkpoint。')
page();title('4．损失函数、梯度与权重')
para('对每个通道 c，V_c 为有效像素，p=sigmoid(z)，y∈{0,1}。BCE_c = Σ[-w⁺_c y log(p) - w⁻_c(1-y)log(1-p)] / |V_c|。分母是未加权有效像素数，不是权重和；正负权重改变会改变损失尺度。v6以后通道外层权重统一为1，通道平均后与重叠项相加。')
para('原版 Dice 在每个图像×通道内按空间维度求和：1-(2Σpy+ε)/(Σp+Σy+ε)，ε=1e-6。先对有效图像/通道配对平均，再对通道平均；没有把不同标志物展平混算。纯阴性配对仍被纳入，但其对概率的导数 ε/(Σp+ε)² 通常很小。')
para('v9：有阳性时沿用Dice；纯阴性时改为 mean(p)=Σp/N。其逐像素概率梯度为1/N，对logit的梯度为p(1-p)/N，随后还需考虑图像、通道平均。Σp的逐像素概率梯度虽为1，但参数梯度包含链式求导，不等于预测阳性像素个数。')
para('v10/v11：r=N-/N+ 按当前训练集组织内且通道可用的像素统计；w⁺=min(16,max(1,r)^0.6)，w⁻=1。BCE系数和Dice系数均1；这不保证数值贡献或梯度贡献相等。v11采样后仍保留自然训练集权重，因此采样与阳性加权叠加。')
para('选择Dice而未训练Lovasz-Hinge：当前目标是独立多标签概率，Dice可直接与sigmoid概率重叠监督配合；Lovasz-Hinge属于候选，尚无受控对照证明优劣。不能宣称当前选择已优于该替代项。')
w5=read(P/'experiments/binary_bce/runs/prepared/data/bce_pixel_weights.json')
w7=read(RUNS['v7']/'data/bce_pixel_weights.json');w10=read(RUNS['v10']/'data/bce_pixel_weights.json')
wt=pd.DataFrame({'标志物':CHANNELS,'v5':w5['positive_weights'],'v6/v9':1.,'v7':w7['positive_weights'],'当前r':w10['ratios'],'v10/v11':w10['positive_weights']})
wt.to_csv(OUT/'tables/positive_weights.csv',index=False);table(wt,3)
page();title('5．像素比例与patch采样')
para('v11沿用v8的训练集概率拟合：以各通道阳性像素占比靠近0.5为优化目标，加入分布正则与均匀混合；单patch最大相对概率10，限制分布集中度。每轮有放回抽223,400次，首轮独立patch数125,161；验证/测试不重采样。多通道共表达限制了所有通道同时达到1:1。')
sampling=pd.read_csv(RUNS['v11']/'sampling_audit.csv')
table(sampling[['channel','natural_positive_fraction','expected_sampled_positive_fraction','epoch1_sampled_positive_fraction']].rename(columns={'channel':'标志物','natural_positive_fraction':'自然阳性像素占比','expected_sampled_positive_fraction':'采样期望占比','epoch1_sampled_positive_fraction':'首轮实际占比'}))
para('上述占比为0–1小数。例如FOXP3仅由约0.006升至0.011，不能描述为已经达到像素1:1平衡。')
page();title('5.1．逐通道阳性/阴性patch比例')
para('阳性patch定义：该通道有效组织内至少一个阳性像素。统计为256网格、随机增强前。18轮4,021,200次抽样由固定随机种子重建，重复patch按出现次数计入。各通道原始204张零有效像素patch不计入比例。极少数阳性像素即可让patch被记为阳性，因此patch比例与像素比例不同。')
balance=pd.read_csv(RUNS['v11']/'results/patch_balance.csv')
table(balance[['channel','natural_positive_patches','natural_negative_patches','natural_positive_percent','sampled_positive_percent','sampled_negative_percent']].rename(columns={'channel':'标志物','natural_positive_patches':'原始阳性数','natural_negative_patches':'原始阴性数','natural_positive_percent':'原始阳性%','sampled_positive_percent':'采样阳性%','sampled_negative_percent':'采样阴性%'}),2)
page();title('6．训练轨迹与模型选择')
table(training.rename(columns={'model':'版本','completed_epochs':'完成轮数','best_epoch':'最佳轮','best_val_iou':'最佳val IoU','best_val_f1':'对应val F1','last_train_loss':'末轮train loss','last_val_loss':'末轮val loss','recorded_epoch_hours':'累计轮耗时(h)'}),4)
para('训练耗时来自轮日志累计，不包含所有准备、校准、测试时间。v4/v5历史训练日志可能含中断或续跑，不能视为完整墙钟用时。新版本的best均按验证macro IoU核对，不能以F1峰值替代。')
pic(OUT/'figures/validation_curves.png',height=270)
page();title('6.1．Loss曲线与过拟合迹象')
pic(OUT/'figures/loss_curves.png',height=390)
para('v7/v10最佳轮均较早，后续train loss继续下降而val loss上升，符合泛化不足的迹象。不同版本loss权重、掩膜及采样不同，不能横向以绝对loss高低判优。v11的训练Dice下降还受训练分布变化影响，必须结合自然验证/测试集指标解释。')
page();title('7．评估口径与公平性')
para('口径A：共同CRC02患者6,638张；v1/v2/v3保留原生输入预处理和输出强度逆变换，随后对齐256网格；分类模型取sigmoid概率。统一组织/通道有效掩膜、二值GT，独立通道计算指标。新增v7/v10/v11全量评估已纳入。')
para('口径B：当前完整测试集38,218张，用于v6/v7/v9/v10/v11比较。旧模型可能在这些患者上训练过，不将其混入同表。v4/v5原生测试虽保留在源文件中，但有效掩膜/划分不完全相同，不拿原生数字直接排名。')
para('每个通道F1与二值Dice数值相同；macro是通道等权平均。AUROC/AP用256桶直方图近似，稀疏通道AP比AUROC更能体现精度—召回权衡。没有进行患者bootstrap、多seed或显著性检验。')
para('表中own_validation为各模型既有验证阈值；历史验证群体不完全一致，尚未统一重做阈值标定。固定0.5是共同数值阈值，但回归输出是强度不是概率，对v1–v3不代表公平工作点。阈值选择及GT非零二值化存在弱信号敏感性。')
for scope,label in [('CRC02_6638','7.1．共同CRC02：整体指标'),('full_38218','7.2．完整测试集：整体指标')]:
    page();title(label)
    for policy in ['own_validation','fixed_05']:
        para('各自验证阈值' if policy=='own_validation' else '固定0.5阈值（历史强度模型需谨慎解释）','Heading2')
        s=summary[(summary.test==scope)&(summary.policy==policy)][['model','precision','recall','f1','iou','auroc_histogram','average_precision_histogram']]
        table(s.rename(columns={'model':'版本','auroc_histogram':'AUROC','average_precision_histogram':'AP'}))
for scope,label in [('CRC02_6638','共同CRC02'),('full_38218','完整测试集')]:
    for metric,pretty in [('precision','Precision'),('recall','Recall'),('f1','F1 / Dice'),('iou','IoU'),('average_precision_histogram','AP'),('auroc_histogram','AUROC')]:
        page();title(f'8．{label}逐通道：{pretty}')
        para('各模型验证集阈值；AP/AUROC不依赖该二值阈值。详细混淆矩阵计数、固定0.5及FPR见配套Excel。')
        t=pd.read_csv(OUT/'tables'/f'{scope}_{metric}.csv');table(t)
page();title('9．解释、限制与下一步')
para('为什么v3回归有时更好：保留连续强弱信息；log归一化和前景MSE权重约11；二值化把弱非零信号与明确阳性等同，可能放大标签噪声。v3与分类模型的划分、目标、采样和阈值并非完全一致，不能仅凭结果证明回归优于分类。')
para('可视化偏差：当前GT在共表达像素中用原始强度选主通道；回归模型argmax选择强度，分类模型argmax选择概率，比较量并不相同。颜色不同不一定意味着逐通道分割错误，概率图与二值独立通道指标才是补充证据。')
para('v10提高稀疏阳性权重后，固定0.5下召回增加、精度下降；验证阈值校准后总体F1小幅提高。v11进一步改变训练曝光，F1略升但AP略低于v10，表明收益不是所有操作点都成立。ECadherin/Pan-CK相反、CD163/SMA过染及CD68漏染需逐通道核对，不能由argmax图直接断定互斥关系或GT漏染。')
para('工程工作：原生缓存减少反复TIFF解码；验证集非填充DDP采样防止重复计数；验证规约与超时处理避免不均匀验证批导致同步等待；动态显存探测、完整epoch断点和模型/代码签名保护续训。v10启动前曾遇临时GPU占用，重试后正常完成；不是loss实现导致训练崩溃。')
para('建议下一步（尚未执行）：在固定患者划分下做多seed复验；只在验证集研究阈值稳定性并报告患者级分布；针对易误染/漏染通道人工复核原始mIF与配准；比较严格相同掩膜下连续目标、二值目标及弱信号阈值；保留自然采样对照，避免阳性采样与高权重过度叠加。')
para('StarDist核一致性、细胞结构先验、逐层ViT/CNN融合均需独立消融及标注质量确认；当前未证明其收益。测试集已被多轮查看用于研究反馈，最终泛化结论需要新的锁定患者测试集。')
page();title('9.1．中断实验与未执行方案')
para('阳性Dice实验在第4轮中断，不是收敛模型；共同CRC02固定0.5阈值 macro F1=0.4055，IoU=0.2987，AP=0.3328。下列逐通道结果仅供追溯，不能作为成熟方案与完整训练版本公平排名。')
table(partial_table[['channel','precision','recall','f1','iou','average_precision_histogram']])
para('早期 pixel_balance 方案仅准备数据/采样，未形成正式已训练模型；v8亦未单独训练。v6阴性Dice试跑在首轮中断且无可比较checkpoint。ddp_baseline_before_cache / interrupted 为同一基线的备份或中断阶段，不另算独立实验。')
page();title('10．可视化阅读说明与24张同patch对照')
para('样本组成：12张按通道/信号特征选取的代表patch，加12张固定随机样本；全部来自共同CRC02。不是跨6位测试患者的代表性抽样，不能用这些图片估计整体性能。每图使用冻结验证阈值、同通道固定颜色，背景及无通道通过阈值的位置为黑色。')
para('随后每页展示一张重点对照：H&E、GT、v3、v6、v7、v9、v10、v11。完整附件还包含v1/v2/v4/v5及中断实验的全部模型图，以及GT/v7/v10/v11逐通道概率图（固定0–1色标，忽略像素留白）。')
manifest=pd.read_csv(VIS/'preview_manifest.csv');shutil.copy2(VIS/'preview_manifest.csv',OUT/'tables/visualization_manifest.csv')
for row in manifest.itertuples():
    page();title(f'Patch {int(row.patch_id)} · {row.selection}')
    pic(VIS/'figures'/f'patch_{int(row.patch_id)}_focus.png',height=455)
page();title('11．证据索引与复现说明')
para('训练入口：vit_training_project/experiments/{v10,v11}/scripts/run.sh；对应configs与runs内submitted/resolved config记录实际参数。v6/v7/v8位于experiments/v6_v7_v8；v9位于v6_v7_v8_emptydice。不可直接运行旧的全套queue，否则可能触发未要求的实验。')
para('当前工程含本机绝对路径及本地数据依赖。代码发布不等于下载后即可无数据运行；需提供合法取得的ORION CRC数据、Virchow2权重、重建数据清单/缓存并修改路径。训练权重、原始图像和巨大缓存均不进入Git。')
para('v10/v11准备脚本依赖已有准备产物，不是独立原始数据导入器。CPU测试只能证明逻辑/通信流程；实际运行日志和完整测试产物用于证明本机四卡实验已完成。原历史README含阶段性状态，应以本报告及冻结运行记录为准。')
for s in sources:para(str(s.relative_to(ROOT)),'SmallCN')
para('主要实现：vit_seg/data.py、binary_targets.py、bce_balance.py、distributed.py、pixel_sampling.py、train_ddp.py、metrics.py；架构：vit_matte/vitmatte_unet.py、encoder.py、lora.py。统计与PDF生成代码：vit_training_project/review/build_mentor_report.py。','SmallCN')
def footer(canvas,doc):
    canvas.setFont('Latin',8);canvas.drawString(40,20,'H&E to mIF | Experiment Report | 2026-10-02');canvas.drawRightString(800,20,str(doc.page))
pdf=OUT/'导师报告_全部实验与结果_20261002.pdf'
SimpleDocTemplate(str(pdf),pagesize=landscape(A4),rightMargin=40,leftMargin=40,topMargin=32,bottomMargin=36,title='H&E到mIF全部实验与结果',author='项目实验记录').build(story,onFirstPage=footer,onLaterPages=footer)
(OUT/'source_manifest.json').write_text(json.dumps([{'path':str(s.relative_to(ROOT)),'sha256':hashlib.sha256(s.read_bytes()).hexdigest()} for s in sources],indent=2))
print('REPORT COMPLETE',pdf,flush=True)
