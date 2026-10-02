# ViT逐像素多标签分类

当前直接复用MIPHEI已质控的318450张patch，**不再清洗、不再运行StarDist、不按DAPI分数删除数据**。旧核质控预览与来源调查保留为历史证据，不是当前处理入口。

## 当前流程

1. 全部patch按ORION病例重新分为train/val/test，种子42；29/6/6例，226465/56832/35153张，患者不跨集合。
2. 在原生333×333网格统计每通道阳性像素和组织内覆盖率。true表示组织内至少一个处理后强度>0像素；这是可复现的信号阈值，不等同于经病理验证的生物学阳性。
3. 只用train true集估计阳性像素mean/std以及覆盖率均值；覆盖率高于均值为true1，小于等于为true2。val/test复用train阈值。
4. 各通道、各集合分别生成false:true1:true2=2:1:1不放回采样清单。没有足够负例或任一正例子组时标记不可行，不制造样本。训练混合可行通道的清单；自然分布val/test用于主评价，不能声称合并后的所有通道边际仍严格平衡。
5. 训练时在线温和增强，不复制大量增强文件。几何变换同步作用于HE、标签和组织mask，标签用最近邻。包括90°旋转/翻转，35%概率±10°旋转和0.95–1.05缩放，50%概率小幅亮度/对比度/颜色变化。
6. 16通道独立sigmoid输出，保留共表达；argmax仅用于展示。背景不计入loss和指标，新任务不计算PSNR/SSIM。

DAPI阴性但其他通道有信号的patch全部保留，DAPI不遮罩其他通道。切片中出现过该通道信号，才将其零信号patch列为可信阴性候选；整张切片通道全零暂记“可用性未知”、忽略该通道监督，不能据此断言通道缺失或实验失败。缺少原始实验可用性清单时，这是一项保守处理，可能减少真正阴性的样本。HE近乎空白的零信号patch同样不冒充可信阴性，但原patch仍保留。

HE玻片背景在归一化前后均设0。GT所有可用通道为0的像素在训练和评价时忽略；单个通道为0而其他通道阳性的像素是有效负例。推理只能使用HE组织mask，不借用GT；组织内全通道阴性区域无法在推理前被完美识别。常规稠密网络仍在背景位置执行计算，但背景不产生监督梯度。

## 运行

在项目根目录运行：

```bash
# 完整流程：统计、训练、独立test评价、最终图与概率输出
bash vit_versions/classification/run_pipeline.sh

# 只生成统计及采样清单；默认8个CPU分片，支持断点恢复
bash vit_versions/classification/prepare_all.sh
```

训练默认使用cuda:2，可通过DEVICE修改；不会启动新的融合或核先验。重复运行可从完整epoch的last.pt继续；配置或统计改变时拒绝混用checkpoint。没有完整统计标记不能启动正式训练。旧的scan、preview_qc入口只供历史诊断复现，主流程不会调用。

## 保留的输出

| 位置 | 内容 |
|---|---|
| data_prepared/patch_manifest.csv | 全部原始patch路径、病例、通道像素数/覆盖率/有效标记 |
| data_prepared/patient_split.csv | 患者划分及原始split |
| data_prepared/statistics.json | train-only mean/std、覆盖率阈值、逆标准差权重、全量计数 |
| data_prepared/slide_channel_health.csv | 切片通道信号观测及可用性依据 |
| data_prepared/channel_patch_counts.csv | 每通道每集合true/false/true1/true2和采样数量 |
| data_prepared/{train,val,test}.csv | 完整自然分布集合 |
| data_prepared/balanced/、train_balanced.csv | 通道独立采样清单及训练混合清单 |
| results/channel_patch_counts.png、augmentation_preview.png | 全量样本统计图和确定性增强示例 |
| results/model/ | best/last权重、训练曲线、验证指标、LoRA检查和进度 |
| results/test_metrics.json、test_class_metrics.csv/png | 独立test的分类指标和图 |
| results/predictions/ | 每个test病例前两张的16通道概率、多标签TIFF和展示图 |

生成中的SQLite和日志为断点恢复所需，完成并验证后才能清理；它们不是最终结果。数据统计完成不等于训练完成。旧回归权重和结果不会改名当作分类结果。

## Loss与评价

默认概率MSE＋逐图逐通道空间Dice；仅H/W聚合，不混合展平通道。可切换逐通道Lovasz-Hinge（输入原始logits）。train阳性像素强度1/σ经归一化、截断再归一化；无阳性统计的通道权重0，批内不可监督通道也不稀释其他通道loss。MSE:Dice=1:1为初始消融设置，未宣称最优。

评价包含各通道precision、recall、F1/Dice、IoU、specificity、阳性/阴性support、2×2混淆矩阵，以及macro指标、micro-F1、Hamming accuracy、完全可评价像素的exact-match、ECE、Brier、256-bin近似AUROC/AP。未定义项为null，argmax不参与主评价。

32层ViT的Q/V已全部配LoRA。decoder、多阶段融合架构、Dice/Hinge选择及核先验详见[方案分析](方案分析.md)；新增融合只保留[架构蓝图](fusion_blueprint.json)，未改主网络。MIPHEI来源与可借鉴做法见[MIPHEI可借鉴方案](MIPHEI可借鉴方案.md)，v1/v2/v3完整差异见[历史审查](../audit_20260925/代码检查与三版差异.md)。

verification.json记录实际测试状态；gpu_smoke.json仅证明真实patch前后向与梯度可运行，不是分类准确率。cleanup_manifest.json记录已清理内容。
