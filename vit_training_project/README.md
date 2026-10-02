# 四卡ViT多标签分类训练项目

**当前已按用户授权启动阳性Dice实验。** 项目包含独立源码、配置、运行脚本、验证和设计说明。运行入口默认只展示配置；显式提供`--approved`才开始处理或训练。最新方案和评估见本文件的2026-09-29更新及对应报告。

此前的单卡自动流程已停用。新实验使用独立运行目录，不接续旧分类训练权重；重新拟合修正患者划分后的训练统计，核验标签一致后复用原生数据缓存。原始MIPHEI发布数据和Virchow2预训练权重通过配置路径读取。

## 先读这三处

1. [完整训练方案](docs/训练方案.md)：数据、标签、采样、模型、loss、四卡策略和交付结果。
2. [实际配置](configs/train.json)：本次建议的训练参数。
3. [验证记录](review/verification.json)：已经做过的检查，以及审核后才执行的GPU验证。

关于0.5是否合适，见[阈值选择与本次复查](docs/阈值与一键运行复查.md)。现在训练阶段固定0.5选模型，固定best后在自然分布val上逐通道选择阈值，test同时报告0.5和选定阈值的成绩；推理自动加载同一份阈值文件。

```mermaid
flowchart LR
  A[审核通过] --> B[读取MIPHEI发布数据]
  B --> C[患者70/15/15划分]
  C --> D[train统计与逐通道平衡清单]
  D --> E[完整性和泄漏校验]
  E --> F[四卡显存实测与batch锁定]
  F --> G[在线增强与DDP训练]
  G --> H[val选best与早停]
  H --> T[val逐通道选阈值并锁定]
  T --> I[独立test双阈值评价]
  I --> J[概率、多标签图和报告]
```

## 默认训练参数

| 项目 | 配置 |
|---|---|
| GPU | 4卡DDP，一卡一个进程；BF16；SyncBatchNorm |
| 每卡minibatch | 审核后自动探测2～128间的最大可用偶数，四卡取共同可用值 |
| 显存目标 | 约90%，且保留至少2GiB；实际可达值受batch离散步长影响 |
| 梯度累积 | 自动使全局有效batch至少256；实际值为4×每卡batch×累积步数 |
| 优化器 | AdamW，β=(0.9,0.999)，ε=1e-8 |
| 初始学习率 | LoRA 1e-4；decoder 3e-4；不随探测batch自动放大学习率 |
| 权重衰减 | 0.01；bias及一维归一化参数不衰减 |
| 学习率变化 | 2轮线性warmup，随后余弦下降，最低为初始值的5% |
| 训练上限 | 40轮，每轮完整验证一次 |
| 早停 | val macro-IoU；patience=8；min_delta=0.001；前10轮不累计坏轮数 |
| 梯度裁剪 | 全局范数1.0 |
| checkpoint | 保存实际最高IoU的best，以及完整epoch的last |

“吃满显存”落实为实测最大稳定批量，不预先猜一个数字，也不把显存撑到100%导致通信或验证OOM。GPU探测将实际跑四卡模型前向、反向、AdamW更新，覆盖优化器状态和DDP缓冲区；**尚未执行探测，因此当前没有声称测得的batch数值**。

## 目录

```text
configs/             模型结构和训练配置
scripts/run.sh       审核/分阶段/完整流程入口
scripts/check_cpu.sh CPU合成验证入口，不用GPU
vit_seg/             数据、预处理、loss、指标、DDP、容量探测、推理与报告
vit_matte/           Virchow2、LoRA和现有CNN decoder
datacore/            通道定义和早停/曲线工具
docs/                完整方案、融合蓝图、MIPHEI来源分析
tests/               单元测试与四进程CPU通信/梯度验证
review/              本次审核记录
runs/                审核通过后才生成的运行数据、权重和结果
```

## 环境

本机使用`/home/weiyh/.conda/envs/virchow_env/bin/python`。默认数据和预训练权重路径写在配置里，迁移机器时修改这两项，并设置`PYTHON`。需要Python 3.10与支持BF16的四张CUDA GPU。依赖版本见`requirements.txt`，不需要TensorFlow或StarDist。

新环境示例（审核后自行安装）：

```bash
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## 审核与执行

```bash
# 只展示配置；不会统计数据、占用GPU或训练。
bash scripts/run.sh

# 仅CPU合成验证，可独立检查代码。
bash scripts/check_cpu.sh

# 以下命令仅在审核通过后执行：从预处理开始完整重跑。
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run.sh --approved --run-dir runs/ddp_baseline
```

需要按阶段核对时，可在审核后依次运行：

```bash
bash scripts/run.sh --approved --stage prepare --run-dir runs/ddp_baseline
bash scripts/run.sh --approved --resume --stage capacity --run-dir runs/ddp_baseline
bash scripts/run.sh --approved --resume --stage train --run-dir runs/ddp_baseline
bash scripts/run.sh --approved --resume --stage calibrate --run-dir runs/ddp_baseline
bash scripts/run.sh --approved --resume --stage test --run-dir runs/ddp_baseline
bash scripts/run.sh --approved --resume --stage report --run-dir runs/ddp_baseline
```

`--resume`保留已验证的数据、容量配置和完整epoch checkpoint。已早停或达到epoch上限的训练不会再多跑一轮。中途停止一轮时，该轮需要重做；不承诺GPU逐位一致复现。配置、代码、模型结构、依赖版本或预训练权重变化时拒绝续跑，应换新运行目录。GPU被其他进程占用时停止并报错，不结束其他任务。四卡GPU实测及完整流程尚待审核后执行，CPU测试不能代替GPU测试。

训练状态在`runs/<名称>/status.json`，各阶段日志和失败原因保存在同一目录。最终结果包括数据统计、增强预览、训练曲线、best/last、逐类分类指标、独立test结果以及示例概率与多标签图。历史回归PSNR/SSIM不进入新项目。逐轮记录见`results/model/train_log.csv`（train/val loss、macro指标、耗时），每个epoch的固定patch虚拟mIF快照见`results/model/snapshots/`。
# 2026-09-29 更新

当前默认配置已改为**仅阳性图像/通道计算Dice，全部有效正负像素计算MSE**，并修复CRC33两切片跨训练/测试集合的问题（真实患者28/6/6）。新运行：`runs/ddp_positive_dice`；旧基线：`runs/ddp_baseline`，其修改前代码保存在`source_at_completion/`。

[完整评估与训练说明](docs/整体评估与阳性Dice训练_20260929.md) · [四版本指标和24张同patch图](review/unified_eval_20260929/index.html)

新运行的启动/中断恢复入口：`bash scripts/run.sh --approved --resume --config configs/train_positive_dice.json --run-dir runs/ddp_positive_dice`。正在运行时不要重复启动，运行锁会拒绝第二个训练进程。
