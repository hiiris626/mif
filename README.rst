H&E → mIF：ViT / LoRA 多标志物实验
================================

这是截至 2026-10-02 的研究代码与冻结实验记录快照。任务是 16 通道逐像素
多标签预测；各通道独立 sigmoid，argmax 仅用于展示，不用于训练监督。

首先阅读
--------

* ``reports/experiment_report_20261002/全部实验与结果_20261002.pdf``：
  53 页中文实验报告，含版本设计、数据、损失、训练曲线、全部指标和 24 张可视化。
* ``reports/experiment_report_20261002/tables/全部实验_完整指标.xlsx``：
  共同 CRC02 测试集及完整 38,218-patch 测试集的指标，严格分表。
* ``experiment_records/``：冻结配置、权重统计、训练轨迹与测试结果。
* ``release_manifest.json``：发布文件的大小与 SHA256。

版本状态
--------

v1–v3 为历史强度回归；v4 为 MSE 分类基线；v5 为原比值加权 BCE；
v6 增加全阴性组织监督并使用单位权重；v7 使用 v5 权重平方根；
v8 是像素驱动 patch 采样方案，未单独训练；v9 修改纯阴性 Dice；
v10 使用当前训练集比值的截断幂权重；v11 在 v10 基础上加入 v8 采样。
v10、v11 均已完成。阳性 Dice 中断实验仅供参考，不算正式完成实验。

v11 完整测试 macro F1=0.5532（验证阈值），v10 为 0.5511，v7 为 0.5493。
这些差距没有多 seed / 患者 bootstrap 显著性验证，不应宣称确定性提升。
共同 CRC02 6,638-patch 测试已补齐 v7/v10/v11；不要混用两种测试范围。

代码位置
--------

* ``vit_versions/``、``vit_matte/``：历史版本归档与共用模型。
* ``vit_training_project/vit_seg/``：历史分类主工程。
* ``vit_training_project/experiments/v6_v7_v8/``：v6/v7/v8。
* ``vit_training_project/experiments/v6_v7_v8_emptydice/``：v9；其中旧 v7/v8 配置不是已训练版本。
* ``vit_training_project/experiments/v10/``、``v11/``：独立训练源码。
* ``vit_training_project/review/``：评估、可视化、统计与报告生成脚本。

复现前提
--------

仓库不包含 ORION CRC 原始数据、Virchow2 权重、训练 checkpoint、二值分片
和大型缓存。需自行获得数据与权重，并按其授权条款使用。没有在仓库中重新授权第三方资产。
代码保留原实验路径以便审计；迁移机器须修改配置和部分历史分析脚本中的本地路径。
已有 ``prepare_v10.py`` / ``prepare_v11.py`` 依赖前序准备结果，不是原始数据导入器。
``experiment_records`` 是只读证据，不是可直接训练的数据清单目录。

环境：Python 3.10，PyTorch 2.5.1、torchvision 0.20.1、CUDA/BF16 四卡；
完整训练依赖见 ``vit_training_project/requirements.txt``。
PDF 生成另需 reportlab、pandas、openpyxl、matplotlib、Pillow 和 CJK/Latin 字体。

在数据准备与路径配置完成后，以 v11 为例：

::

    cd vit_training_project/experiments/v11
    PYTHON=/path/to/python bash scripts/run.sh --config configs/v11.json
    # 上一行仅审阅配置。明确要执行时：
    PYTHON=/path/to/python bash scripts/run.sh --approved --resume \
      --config configs/v11.json --run-dir runs/v11

请勿直接运行历史队列脚本，否则会按历史顺序触发其他实验。
旧 README 中的“待启动”等段落是历史快照；当前状态以本页、实验报告和冻结记录为准。

验证
----

已重新运行 v6/v7/v8 源码目录的 35 项 CPU 单元测试，通过；
v11 新增的 4 项权重边界、权重篡改拒绝和 DDP 采样重建测试也通过。
发布时所有 Python 文件通过语法检查，未发现明显私钥或 GitHub token。
这些检查不是所有历史脚本的全环境集成验证；真实四卡运行证据见各实验记录。
``reports/experiment_report_20261002/code_checks.log`` 保存本次测试输出。

原始工作区未被移动或删除；本仓库是独立整理的发布快照。
