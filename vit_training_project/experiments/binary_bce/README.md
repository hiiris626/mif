# 二值数据集＋BCE＋正负patch Dice

2026-09-30更新：已完成的v5检查点保持原训练定义，原源码及配置归档在`runs/prepared/source_at_completion`。当前工作代码的默认argmax已增加逐通道阈值，快照只显示二值GT和概率；新提交配置的外层通道权重设为1，不再使用强度σ。旧run不应以修改后配置恢复，新训练需新run。此次没有重训，也没有自动把不可靠的全零GT改为可信阴性。

此前版本按2026-09-29要求准备：有效阳性标签全部为1，有效阴性为0；原像素MSE替换为BCEWithLogitsLoss；Dice恢复计算所有有效图像/通道对，包括阴性patch/通道。当前运行中的旧实验不修改、不停止，本版本尚未启动GPU训练。

## 数据集

数据目录：`/home/weiyh/h2e_mif_benchmark/vit_training_project/datasets/binary_expression_v1`。

原训练在加载时已经将强度>0转为二值表达标签。这次把相同监督定义固化成独立数据集，而不是仅修改显示颜色或继续临时读取强度TIFF作为目标。

- 全部318450张patch保留，不重新StarDist筛选，不按DAPI删除其他通道阳性patch。
- 保持train/val/test患者28/6/6，patch223400/56832/38218。
- 每张目标为`uint8 [16,333,333]`，值只有0和1；有效监督掩膜另存，不能将掩膜外的0当负例。
- 采用逐patch无损压缩、分片打包存储，保留原生分辨率。训练再同步增强并缩放至256。
- `BinaryTargetStore.read(patch_id)`返回`target, valid, tissue`；target中不含255。为了兼容现有增强与loss，训练加载器在内存中将`valid=False`转换为ignore=255。
- 原H&E、原强度文件及正在运行的实验不覆盖。新训练读取新二值目标，H&E复用原缓存。
- CSV中的原`target_path`只保留源数据追溯意义；`binary_targets.json`指定实际新标签存储，加载器优先使用它。无缓存模式也不会再读原强度TIFF生成目标。
- 原`statistics.json`的阳性强度均值/σ仍是原强度统计，用于追溯并保持本次通道权重不变。没有拿二值阳性的零标准差重新计算1/σ。

数据可用性以`BINARY_DATASET_COMPLETE.json`为准，进度见`progress.json`。原文件逐记录检查ID及CRC，新数据逐记录检查压缩往返；另抽512条通过公开reader逐像素核对。训练数据/患者隔离检查记录在`VALIDATED.json`。

单patch可导出为常规TIFF：

```bash
/home/weiyh/.conda/envs/virchow_env/bin/python scripts/export_binary_patch.py \
  --dataset /home/weiyh/h2e_mif_benchmark/vit_training_project/datasets/binary_expression_v1 \
  --patch-id 314024 --out review/binary_examples
```

得到16通道0/1目标TIFF、0/1有效监督mask TIFF、组织mask和通道说明。存储的二值1不是255；如需肉眼展示可以只在显示时乘255。

## Loss

`p = sigmoid(logits)`。

`w_positive,c = N_negative,c / N_positive,c`，`w_negative,c = 1`。

`BCE_c = Σ有效像素 [y*w_positive,c + (1-y)] * BCEWithLogits(logits, y) / 有效像素数`

使用数值稳定的logits版本，等价于各通道指定上述 `pos_weight`。权重由全部223400张唯一训练patch在256网格、随机增强前的有效像素统计得到；不使用val/test拟合，不按batch重算，不截断。缺少任一类别时报错。阳性总权重与阴性总权重均为N_negative；它们的实际loss、梯度或每个batch不必相等。分母是未加权有效像素数，不再做均值为1的权重归一化。

完整权重见 [各通道像素计数与BCE权重](review/bce_pixel_weights.csv)。复用已核验来源的全训练集像素计数，并对新二值数据抽256条交叉核对；权重不是由256条样本估计。稀疏通道权重较大，后续需要通过验证集评估假阳性与阈值稳定性。

`DiceLoss_bc = 1 - (2Σ有效像素(p*y)+1e-6)/(Σ有效像素(p+y)+1e-6)`

按每张图、每个通道计算空间Dice，再对所有有效图像取平均。`dice_scope=all_valid`，阴性图像/通道也参与；未知通道和背景仍忽略。若整个patch在原监督规则下没有任何有效像素，它仍不参与loss，不能把缺失标签或全通道无信号的忽略区域重新当可信负例。

`总loss = Σ_c γ_c (1.0*BCE_c + 1.0*DiceLoss_c) / Σ_c γ_c`

历史v5的γ来自原训练集强度统计，实际约0.86～1.10；当前新配置`channel_weighting=uniform`令γ=1。旧检查点无此字段时仍按历史规则评价，保证复现。BCE内部采用上述正负像素权重，Dice不使用这些像素权重。四卡按全局分子分母归一化，未知/背景梯度为0。

全阴性图像的标准Dice约为`1-ε/(Σp+ε)`，通常接近1且梯度很弱；不能把新增这部分loss数值理解为有效假阳性惩罚同比增强，主要阴性约束来自BCE。这一版本按要求恢复原公式，未另行发明阴性Dice替代项。

## 配置、日志与启动边界

配置：`configs/train_binary_bce.json`；`configs/train.json`内容相同。四卡DDP、batch64/卡、AdamW、2轮warmup＋余弦退火、40轮上限、patience8等沿用现设置。训练改为全部223400张唯一patch每轮随机打乱、不放回采样，患者划分不变。四卡每卡55850张，不丢末尾batch，每轮873步，末尾每卡42张。旧分层采样表只作来源记录。相比原来每轮2824步，现在每轮更新更少，warmup和余弦调度按新步数计算。

日志字段改为`train_bce_loss`、`val_bce_loss`，同时保留总trainloss/valloss、Dice分量、分类指标、耗时和每轮快照。Brier指标可以继续保留用于概率误差评价，但它不再充当训练MSE loss。BCE总loss与旧MSE总loss不能直接比较大小。

准备入口：

```bash
/home/weiyh/.conda/envs/virchow_env/bin/python scripts/prepare_binary_run.py \
  --dataset /home/weiyh/h2e_mif_benchmark/vit_training_project/datasets/binary_expression_v1
/home/weiyh/.conda/envs/virchow_env/bin/python scripts/configure_random_bce.py
bash scripts/run.sh --config configs/train_binary_bce.json --run-dir runs/prepared
```

最后一条不带`--approved`，只检查配置，不启动训练。prepared中复用的H&E缓存视为只读。

本次只完成数据和代码准备及CPU验证。未来明确切换训练后，应从新的运行配置开始，先重新做真实模型GPU容量验证，不把旧MSE checkpoint作为本版本的无缝恢复点。已有prepared目录未来执行流程需`--resume`以复用准备阶段；这不意味着加载旧实验模型。

## 检查与实现位置

- `scripts/export_binary_dataset.py`：全量目标导出，支持分片恢复。
- `vit_seg/binary_targets.py`：0/1目标与单独mask读取，DataLoader进程各自管理文件句柄。
- `vit_seg/data.py`：优先新二值目标，保留增强、背景与通道可用性。
- `scripts/configure_random_bce.py`：设置随机采样，核验全训练集像素计数并生成固定BCE权重。
- `vit_seg/bce_balance.py`：拟合/加载阴性÷阳性权重，检查数据来源。
- `tests/test_random_balanced_bce.py`：权重、手算loss和梯度、四卡完整覆盖与尾batch。
- `vit_seg/distributed.py`：BCE及含阴性Dice、训练与评价一致的全局归约。
- `vit_seg/train_ddp.py`：BCE配置、日志、数据格式检查。
- `tests/test_binary_bce.py`：手算BCE值/梯度、极端logits、阴性Dice计数与梯度、ignore零梯度。
- `tests/ddp_cpu_check.py`：四进程与单进程全局目标及梯度一致。
- `tests/production_cpu_check.py`：真实二值导出/读取＋微型模型四进程训练、早停、恢复、校准、测试和预测输出。

CPU检查不等于真实模型GPU容量或精度验证；GPU训练尚未执行。
