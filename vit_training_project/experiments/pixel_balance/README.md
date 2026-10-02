# 像素平衡候选实验：仅准备，未启动

本目录隔离于正在运行的 `vit_training_project/runs/ddp_positive_dice`。保持现有患者集合不变，每轮重新组织训练抽样。当前实验继续，不在这里执行新训练。

先阅读 [方案、全通道统计与阈值检查](review/方案与检查结果.md)。关键数据在 `review/pixel_weight_table.csv`、`review/channel_cohort_counts.csv` 和 `review/threshold_stability/`。

## 逻辑顺序

1. `scripts/count_training_pixels.py`：从现有缓存统计训练标签网格，背景与未知通道不计入。
2. `scripts/prepare_proposal.py`：固定划分；生成false/true1/true2队列、三轮采样检查、冻结候选像素权重和配置。没有重新清洗数据。
3. `vit_seg/pixel_sampling.py`：自然数据覆盖、每通道额外4:1:1抽样、重复上限、每轮患者及patch顺序打乱、DDP分片。
4. `vit_seg/distributed.py`：按通道计算加权像素MSE与阳性图像Dice，正确归一化并忽略背景。
5. `scripts/patient_threshold_audit.py`：只用验证患者做留一患者阈值诊断；当前结果使用旧baseline模型。
6. `scripts/report_proposal.py`：生成报告与PNG/PDF图。

## 三份候选配置

- `configs/train_pixel_balance_no_positive_boost.json`：正负像素同权，MSE:Dice=1:0.5；优先对照，观察是否需要额外阳性权重。
- `configs/train_pixel_balance.json`：按训练像素比例拟合温和权重，上限2，MSE:Dice=1:0.5。
- `configs/train_pixel_balance_dice1_control.json`：相同温和权重，MSE:Dice=1:1。

三份均为待验证方案，不能预先保证精度或假阳性改善。只保证每通道**额外抽样**false:true1:true2=4:1:1，最终全部通道的混合监督分布以实测数据为准。

当前检查入口（仅检查配置，不启动任何训练）：

```bash
bash scripts/run.sh --config configs/train_pixel_balance_no_positive_boost.json --run-dir runs/prepared
```

## 已验证的边界

24项单元测试、四进程CPU DDP梯度测试、四进程微型模型完整生产流程，以及真实缓存批次CPU检查通过。CPU测试没有验证真实模型GPU显存；未来获准切换时仍须容量测试，不能直接复用旧容量结论。

`runs/prepared`保存已准备数据及状态，不含训练checkpoint。缓存和原数据清单通过硬链接复用，视为只读，不手工覆盖这些文件。新统计与验证回执单独写入。

当前任务明确要求继续原实验，**本次没有执行带`--approved`的启动指令**。未来不同候选应使用独立运行目录和对应配置，不应混用checkpoint。新配置改变目标函数，横向比较请看固定患者和固定阈值的分类指标，不直接比较不同定义的loss数值。
