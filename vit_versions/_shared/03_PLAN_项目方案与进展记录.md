# H&E → mIF 虚拟染色：CNN / ViT / 扩散模型 三种架构范式对比方案

> 目标：在 **OrionCRC** 数据集上，对 **pix2pixHD（CNN）**、**Virchow2 + ViTMatte 风格 U-Net + LoRA（ViT）**、**I2SB 薛定谔桥（扩散）** 三种范式做统一训练与评估。
> 指标：图像质量（PSNR / SSIM / FID）+ 细胞分类 F1。
> 原则：**权重/模型尽量从发表论文的官方仓库原样复制**，仅在必要处做适配。

---

## 0. 已确认决策（2026-08-08）

1. **数据源**：MIPHEI 处理版（Zenodo 10.5281/zenodo.15340874）——已配准/去AF/归一化/带细胞伪标签的 256×256 tile，直接训练。
2. **mIF 通道**：全部 16 通道（15 marker + Hoechst），与 MIPHEI 基准完全可比。
3. **评估范围**：仅 OrionCRC 域内（train/val/test = 37/2/2 切片），暂不做域外验证。
4. **CNN 基线双轨**：① **pix2pixHD**（全局生成器+多尺度判别器）：严格 GAN 版 + 去判别器 generator-only 对照；② **原版 pix2pix**（UNet256+70×70 PatchGAN，复用本地已有 `train_pix2pix.py` / `HCCExplorer/C3UT` 代码）——与 MIPHEI 发表的 Pix2Pix 基准同构，数字可直接对齐。共三组 CNN 模型。
5. **FID 口径**：RGB 复合图主口径（16 通道标准配色合成），暂不做逐 marker 灰度 FID。
6. **旧肺癌 OMAP-7 逐 marker checkpoint**：不迁移（域不同，且为单 marker 模型），仅复用其代码结构与训练经验。
7. **存储策略**：所有大数据（数据集/权重/checkpoint/大体积输出）统一放根目录数据盘 `/data/weiyh/`（家目录已用 95%、仅剩 83G）；代码与脚本留在 `/home/weiyh/h2e_mif_benchmark/`。
8. **数据文件（已核实）**：MIPHEI 处理版 Zenodo 15340874 的主文件是 `ORIONCRC_dataset_tile_20x.zip`，**127 GB**，md5 `fdc3188206ac68576b4195cd039d9061`，下载链接 `https://zenodo.org/api/records/15340874/files/ORIONCRC_dataset_tile_20x.zip/content`。本地 `/data/weiyh/ORIONCRC_dataset_tile_20x.zip`（157 GB）是损坏的原始版，不可用，需重新下载。

---

## 1. 项目概述

- **任务**：同一组织切片（同一切面）的 H&E → 多色免疫荧光（mIF）逐像素预测，mIF 为 16 通道（15 个蛋白标记 + Hoechst 核染）。
- **三种范式**：

| 范式 | 模型 | 角色 | 权重来源 |
|---|---|---|---|
| CNN | pix2pixHD（全局生成器 + 多尺度 PatchGAN） | 经典 CNN 基线 | 官方 NVIDIA 仓库；任务权重需在 OrionCRC 上从头训 |
| ViT | Virchow2（ViT-H/14，632M）+ ViTMatte 风格 U-Net + LoRA | 自监督病理基础模型微调 | Virchow2 权重已在本机（CC-BY-NC-ND-4.0）；架构参照 MIPHEI-ViT |
| 扩散 | I2SB（Image-to-Image Schrödinger Bridge） | 扩散/桥模型 | 官方 NVlabs 仓库已在本地；网络用官方自动下载的 ADM 预训练 U-Net 初始化 |

- **核心参考论文**：MIPHEI-ViT（arXiv 2505.10294，Computers in Biology and Medicine 2026）。它正是用 OrionCRC 做 H&E→mIF 的公开基准，给出了 Pix2Pix / HEMIT / ROSIE / DiffusionFT / MIPHEI 的逐模型 PSNR / SSIM / Pearson / AUPRC / F1 参考数值，可直接作为我们结果的 sanity check。

---

## 2. 数据集：OrionCRC

### 2.1 数据概况（来源：MIPHEI-ViT 论文 §3.1, Table 1）

- **扫描方式**：ORION 扫描仪，同一组织切片**重染后**同时采集 H&E 与 18 通道 IF（同一切面 → 天然配准）。
- **规模**：41 张 WSI；按切片划分 **train 37 / val 2 / test 2**。
- **tile**：256×256 px @ 0.5 mpp（Aperio GT450 扫描仪）。
  - Train：295,096 tile；Val：12,402；Test：10,952。
- **预测通道（16）**：Hoechst, CD31, CD45, CD68, CD4, FOXP3, CD8a, CD45RO, CD20, PD-L1, CD3e, CD163, E-cadherin, Pan-CK, α-SMA, Ki67（PD-1 因信号质量差弃用）。

### 2.2 数据获取（已确认：MIPHEI 处理版）

**下载源**：Zenodo 10.5281/zenodo.15340874（MIPHEI-ViT Dataset，MIT 许可）。该 record 共 3 个文件，本项目只需主数据：

| 文件 | 大小 | 内容 | 需要 |
|---|---|---|---|
| `ORIONCRC_dataset_tile_20x.zip` | **127 GB**（md5 `fdc3188206ac68576b4195cd039d9061`） | 处理好的 tile 主数据：`he/`(H&E JPEG)、`if/`(8-bit TIFF，已去AF+归一化 mIF)、`nuclei/`(核掩码)、`csv_nuclei_pos/`(单细胞表型)、`slide_dataframe.csv` + `train/val/test_dataframe.csv`（**官方 tile 级划分直接可用**） | ✅ 必下 |
| `ORIONCRC_dataset_20x_he_norm.zip` | 18.5 GB | CycleGAN 归一化 H&E（论文用作增强，针对域外鲁棒性） | 暂不需要 |
| `HEMIT_nuclei_analysis.zip` | 0.58 GB | HEMIT 数据（本项目只做 OrionCRC 域内） | 不需要 |

**下载链接**：`https://zenodo.org/api/records/15340874/files/ORIONCRC_dataset_tile_20x.zip/content`

> ⚠️ 本地 `/data/weiyh/ORIONCRC_dataset_tile_20x.zip`（157 GB）是**损坏的**：它是原始 OrionCRC（Zenodo 7637988）的同名文件，下载不完整，zip 中央目录与实际数据偏移错位，任何成员都无法解压。**不可用**，需重新下载上表 127 GB 的正确文件（两者同名但版本/大小不同）。
> 许可：MIPHEI 处理版数据为 **MIT**（源 ORION-CRC 亦为 MIT）；论文与代码仓库另有条款，遵守学术非商业用途即可。
> 说明：论文用 256×256 tile 训练；处理版 zip 内 tile 尺寸以解压后为准（可能为 512×512），必要时统一重切/降采样到 256（详见 §3）。

### 2.3 数据预处理要点（若选 B，按 MIPHEI §3.6 复刻）

1. **配准**：同切面数据本身已对齐，无需配准。
2. **Tile 选择**：Otsu 组织区域检测 → 256×256 切块；用空 AF 通道阈值 + H-optimus-0 embedding 聚类过滤伪影 tile（约 10% 被剔除）；CNN 检测错位 tile。
3. **去自荧光（AF）**：$I_c^{corr} = \max(0, I_c^{IF} - \lambda_c \cdot I_{AF} + b_c)$，参数用 napari 工具人工调。
4. **通道归一化**：每个 marker 按训练集前景像素 99.9 分位 $q_c$，再做 log 变换 $I_{norm}^c = 255 \cdot \log(\min(I_c^{corr}, q_c)/q_c + 1)$。
5. **细胞伪标签**：DAPI 用 fine-tune 过的 Cellpose 分割核 → 核区膨胀 2µm 作细胞区域 → 逐细胞求各通道均值 → 分层 GMM 门控判阳性。

---

## 3. 三种模型详细方案

统一约定：H&E 输入归一化采用各编码器官方 mean/std；mIF 目标缩放到 $[-0.9, 0.9]$（配合 Tanh，MIPHEI 设定）；所有模型输出 16 通道 mIF。

### 3.1 CNN 基线（双轨）：pix2pixHD + 原版 pix2pix

#### 轨道 A：pix2pixHD（主基线，需 clone NVIDIA 仓库）

- **代码**：`git clone https://github.com/NVIDIA/pix2pixHD`（BSD 许可）。注意本机目前**没有** pix2pixHD 仓库本体，需要新下载。
- **适配改动**（在官方代码上加最小改动）：
  - `--label_nc 0`（输入直接用 RGB 图像，不用 label map）、`--no_instance`；
  - 数据集用 `train_A / train_B` 配对目录；
  - 输出改 16 通道：最后一层卷积 out=16 + Tanh；
  - 分辨率 256×256：用**全局生成器**（`--netG global --ngf 64 --ndf 64`），256 分辨率用不到 local 增强器；`--resize_or_crop none`（256 已经可被 32 整除）。
- **损失（GAN 版）**：GAN（多尺度 PatchGAN 判别器）+ VGG perceptual + feature matching（原论文设定）。
- **对照（generator-only 版）**：去掉判别器，仅用 L1/加权 MSE + VGG perceptual，用于解释判别器对像素指标与细胞 F1 的影响。
- 两个版本共用同一生成器架构与超参，只开关判别器与 GAN 损失。

#### 轨道 B：原版 pix2pix（与 MIPHEI 发表基准对齐）

- **代码**：复用本机已有代码——`hmu_output/HCCExplorer_reproduction/train_pix2pix.py`（UNet256 + 70×70 PatchGAN，LSGAN + L1×100）或 `HCCExplorer/C3UT/models/networks.py` 的 `unet_256` + PatchGAN 实现。
- **适配改动**：
  - 输出由 3 通道（肺癌时逐 marker 用 R 通道）改为 **16 通道直接输出**（OrionCRC 是多 marker 面板，一次输出全部通道，避免逐 marker 重复训）；
  - 数据源改为 OrionCRC 统一 dataset loader。
- **超参**：沿用其已验证配置（Adam lr=2e-4 β1=0.5，LSGAN+L1×100，200 epoch，batch 4，256×256），必要时按 16 通道调 batch。
- **作用**：该模型与 MIPHEI 发表的 "Pix2Pix" 基准（54.4M/6.49 GFLOPs）同构，OrionCRC 上的结果可直接与其 Table 3/4 对齐，作为双轨 CNN 的可比锚点。
- **旧 checkpoint 处理**：肺癌 OMAP-7 的逐 marker 权重（`03_c3ut/checkpoints/`）**不迁移**，仅参考代码结构与训练经验。
- **权重来源说明**：官方预训练权重是 Cityscapes 的（label→街景），与本任务无关，**需在 OrionCRC 上从头训练**。已发表的同任务参考数字是 MIPHEI 基准里的 "Pix2Pix"（Table 3/4，OrionCRC test 上细胞 AUPRC 0.287、F1 0.097 左右，供比对）。
- **预计训练**：256×256 很轻量，单卡/多卡 1 天内收敛（~200 epoch）。

### 3.2 ViT：Virchow2 编码器 + ViTMatte 风格 U-Net + LoRA

架构完全参照 **MIPHEI-ViT（§4.1）**，只把编码器从 H-optimus-0 换成 Virchow2：

- **编码器**：Virchow2（ViT-H/14，632M，SwiGLU，4 register tokens，LayerScale）。
  - 权重：**本机已有** `/home/weiyh/Virchow2/model.safetensors`（也可从 HF `paige-ai/Virchow2` 下载）。
  - 输入 224×224（与预训练一致；OrionCRC tile 256×256 先 resize 到 224）。
  - 冻结骨干 + **LoRA**（rank=8, α=1，加在注意力 Q/V 投影上，MIPHEI §4.3 设定），可训练参数 ≈ 6–7M。
  - 复用现有 `virchow2_pannuke_decoder/models/load_pipeline.py::load_virchow2` 的加载逻辑（本地 safetensors + timm 显式构建）。
- **ViTMatte 风格解码器**（MIPHEI 变体）：
  - 轻量卷积 **Detail Capture Module**：从输入 H&E 提取多尺度金字塔特征；
  - ViT patch token（1/16 分辨率）作为 bottleneck，bicubic 插值对齐到解码器需要的 1/16 分辨率；
  - 解码器：双线性上采样 + 跳连 + 3×3 卷积 + BN + ReLU；每个 marker 一个独立输出头 + Tanh；
  - 不做 trimap、不做卷积 neck、不做 window attention（MIPHEI 对原版 ViTMatte 的删减）。
- **损失**：逐 marker **加权 MSE**，权重为各通道标准差的倒数（MIPHEI §4.2）：$\mathcal{L} = \frac{\lambda}{M}\sum_j \frac{1}{\sigma_j}\mathrm{MSE}_j$。
- **训练**：Adam，lr 2e-4，weight decay 1e-5，grad clip max-norm 1，dropout 0.1，前 400 iter 线性 warmup 后 cosine 衰减；batch 16；与 MIPHEI 一致在单 A100 上跑，我们的 4×4090 上 batch 16/卡即可。
- **参考数字**：MIPHEI-ViT 在 OrionCRC test 上像素 PSNR ~31.9、SSIM ~0.95、细胞 F1（marker 级 0.05–0.93 区间）。我们的 Virchow2 版本应与之同量级。

### 3.3 扩散：I2SB（薛定谔桥）

- **代码**：官方 NVlabs 仓库已在本地 `/home/weiyh/I2SB`（i2sb conda env 已建好）。
- **网络**：ADM U-Net（`Image256Net`，256×256，res_blocks=2, head_channels=64，~114M 参数），**用官方 train.py 自动下载的 ADM（guided-diffusion）预训练 checkpoint 初始化** —— 这是官方公开可拷贝的预训练权重。
- **任务定义（核心适配）**：新增 `corruption/virtual_stain.py`，把"H&E→mIF"建模为桥：给定 mIF 目标 $y$，源 $x = H\&E$（确定性映射）。开启 `--cond-x1`（I2SB 官方对 pix2pix 类翻译任务推荐，弥补大信息损失）。
- **数据**：配对 H&E/mIF tile，组织成官方要求的 LMDB（或改 dataset 直接读配对目录）；图像归一化到 $[-1,1]$。
- **输出通道**：把 ADM U-Net 最后一层输出通道改为 16（多色 mIF）。若显存/收敛有问题，可先退化为 RGB 复合（Hoechst=蓝、Pan-CK=红等）再逐步扩展。
- **训练**：`--beta-max` 噪声调度、fp16、`--microbatch` 适配 4090 显存（论文默认 microbatch 2 / 32GB V100，4090 需按显存下调或用梯度累积）；`--ot-ode` 可选。
- **采样**：NFE 步数少（I2SB 2–8 步即出图），`--clip-denoise` 可选；推理逐 marker 一次即可（16 通道一次出，比逐 marker 扩散便宜）。
- **计算量**：这是三家中最重的（295k tile × 256×256，从头训桥模型）。预计 4×4090 fp16 下约 5–15 天；先训到中间 checkpoint 出图验证再决定跑满。

---

## 4. 统一评估协议（对齐 MIPHEI，保证与已发表数字可比）

### 4.1 图像质量（像素级）
- **逐通道**计算 PSNR、SSIM，然后**跨通道取平均**（mIF 稀疏，逐通道才反映各 marker）。
- **FID（已确认口径）**：16 通道按标准配色合成为 **RGB 复合图**（Hoechst=蓝、Pan-CK=红、α-SMA=绿等）后计算 FID，作为唯一主口径。FID 特征统计在 OrionCRC **train 集的 mIF 目标**上计算（同 I2SB/clean-fid 惯例）。
- **建议附加 Pearson 相关系数**（MIPHEI 明确指出：稀疏 mIF 下 PSNR/SSIM 对背景比例敏感，Pearson 更稳健；作为论文级评估建议纳入）。

### 4.2 细胞分类 F1（细胞级，任务核心指标）
按 MIPHEI §5.2.2 协议：
1. 核分割（优先用 MIPHEI 数据自带的 DAPI/Cellpose 掩码；外部再补 HoverFast/HoVer-Net）。
2. 每个细胞对**预测的 mIF** 逐通道求均值 → 得到单细胞表达向量。
3. 用该向量在 **val 集细胞**上训练一个 **logistic regression** 分类器（匹配细胞伪标签）。
4. 在 **test 集细胞**上评估：**逐 marker 的 F1 与 AUPRC**（mIF 细胞表型高度不平衡，F1/AUPRC 比准确率合适）。
5. 可选：1,000 次 bootstrap 给 95% CI。

### 4.3 统一设置
- 三模型同一 train/val/test 划分（37/2/2，按切片）、同一 tile、同一归一化、同一评估脚本，保证公平。
- 每模型固定 seed，记录 checkpoint + 全量超参（`options.pkl` 风格），便于复现。

---

## 5. 目录结构（建议）

> **大数据统一放 `/data/weiyh/`**（数据集/权重/checkpoint/大体积输出），项目内只放代码、脚本与小文件：

```
/data/weiyh/
├── orioncrc_miphei/            # 解压后的 MIPHEI 处理版 tile + 官方划分 dataframes
├── weights/                    # virchow2.safetensors、ADM init、各模型 checkpoint
└── results/                    # 各模型预测输出（大体积）

/home/weiyh/h2e_mif_benchmark/
├── PLAN.md                     # 本方案
├── README.md
├── envs/                       # 三个 conda env 的 yaml（pix2pixhd / virchow / i2sb）
├── datasets/                   # 统一 dataset loader（读 /data/weiyh/orioncrc_miphei）
├── pix2pixhd/                  # 官方仓库 + 16 通道适配
├── vit_matte/                  # Virchow2 + ViTMatte U-Net + LoRA
│   ├── encoder.py              #   （复用 load_pipeline.load_virchow2）
│   ├── detail_capture.py
│   ├── vitmatte_unet.py
│   ├── lora.py
│   ├── train.py / sample.py
│   └── configs/
├── i2sb/                       # 官方仓库（本地已有）+ corruption/virtual_stain.py
├── eval/
│   ├── metrics.py              # PSNR / SSIM / FID / Pearson
│   ├── cell_classify.py        # 细胞级 F1 / AUPRC
│   └── run_eval.sh
└── scripts/                    # 数据准备 / 训练 / 采样 / 评估 launch 脚本
```

---

## 6. 实施阶段与时间表（4× RTX 4090，单机）

| 阶段 | 内容 | 预估耗时 |
|---|---|---|
| P0 | 数据获取（建议 Zenodo 处理版）+ 统一 dataset + 划分 + 基线指标脚本 | ~1 周 |
| P1 | pix2pixHD（GAN + generator-only）+ 原版 pix2pix 适配与训练 + 出图 | ~1.5 周 |
| P2 | Virchow2 + ViTMatte + LoRA 训练 + 出图 | ~1–2 周 |
| P3 | I2SB 桥训练（最重） | ~2–4 周 |
| P4 | 统一评估（PSNR/SSIM/FID + 细胞 F1）+ 对比表 + 图 | ~1 周 |
| P5 | 论文图表 / 报告整理 | ~1 周 |
| **合计** | | **约 2–3 个月** |

> P1–P3 可并行推进（三种模型彼此独立）；I2SB 训练期间同时写 P4 评估代码。

---

## 7. 风险与注意事项

1. **许可**：Virchow2 = CC-BY-NC-ND-4.0；MIPHEI 处理版数据/代码 = 非商业；pix2pixHD = BSD；I2SB = MIT。全部限非商业学术研究，**不能用于任何商业/临床诊断目的**（Virchow2 条款尤其严格）。
2. **计算**：I2SB 是最大瓶颈；建议先用 1–2 个训练步数的中间 checkpoint 出图验证方向，再决定完整训练时长。
3. **稀疏目标**：mIF 目标大部分背景为 0，PSNR/SSIM 容易被"预测全零"骗高 —— 务必以**细胞级 F1/AUPRC + Pearson** 为主指标（MIPHEI 的结论）。
4. **Virchow2 分辨率**：tile 256 vs 模型 224，需 resize；ViT patch=14 对 224 才能整除（256/14 非整数）。
5. **I2SB 通道数**：ADM U-Net 改 16 输出通道即可；若收敛差，先试 RGB 复合再扩展。
6. **公平性**：三模型用同一数据管线/划分/评估脚本；pix2pixHD/原版 pix2pix 是 GAN（有对抗训练），MIPHEI 发现判别器会略微降低像素指标但对真实感有帮助——在结果表中如实报告，不调优掩盖。
7. **数据申请**：若选 B（原始 OrionCRC WSI），需向 LabSysPharm 申请 + 自行预处理（+1 个月），不推荐。

---

## 8. 决策记录

上述 5 项决策已于 2026-08-08 确认，见第 0 节。后续如需调整（例如追加域外数据集、逐 marker FID），只需改第 0 节并同步受影响阶段。

---

## 9. P0/P1 进展记录（2026-08-08 → 08-27 更新）

### ✅ 已完成

**数据准备**
- ✅ 数据下载 + md5 校验通过（127GB，与官方一致），解压到 `/data/weiyh/orioncrc_miphei/`。
- ✅ `verify_data.py` 核对通过；16 通道顺序确认（剔除 PD-1，同 MIPHEI 口径）。
- ✅ memmap 缓存构建：`/data/weiyh/orioncrc_cache`（256 分辨率，train/val/test，加速训练）。
- ✅ 数据集概况：**41 张 CRC WSI**（train 37 / val 2 / test 2），train 295,096 tiles、val 12,402、test 10,952（512×512 @20x）。每张 WSI 含配对 H&E + 17 通道 mIF（16 通道用）+ 细胞表型 CSV（`csv_nuclei_pos/`）+ 核分割 mask（`nuclei/`）。

**模型实现与训练**
- ✅ **ViT 路** `vit_matte/`：Virchow2 编码器 + ViTMatte 风格解码器 + LoRA(r=32,α=16)。512 输入训练 15 epoch 完成，可训练 13.42M。
- ✅ **CNN 路** `pix2pixhd16/`：pix2pixHD（GlobalGenerator 45.7M + MultiscaleD 8.3M，InstanceNorm2d），20 epoch 训练完成。
- ✅ **CNN 路** `pix2pix/`：原版 pix2pix（UNet256 54.4M + PatchGAN70，LSGAN+L1×100），200 epoch；**加权 L1 重训版** `pix2pix_16ch_w`（逐 marker 1/std 加权，30 epoch）完成。
- ✅ **扩散路** `I2SB/`：ADM U-Net 256（552M）+ 薛定谔桥，`--paired --cond-x1 --out-ch 16 --in-ch 19`，4 卡 DDP 训练 **20000/20000 iters 完成**（8/26，loss ~0.02）。
- ✅ 评估脚本：`eval/metrics.py`（逐通道 PSNR/SSIM/Pearson + RGB 复合 FID）、`eval/cell_classify.py`（逐 marker 二元分类 AUC/F1，val 训练 LR、test 评估）。
- ✅ 可视化：`scripts/visualize_virtual_stain.py`（patch + WSI 级多色荧光对比）、`scripts/wsi_panorama.py`（整张 WSI 坐标拼接全景，冒烟测试通过）。

### 📊 评估结果（test 集，10952 tiles，2026-08-27 全部完成）

| 模型 | 范式 | PSNR↑ | SSIM↑ | Pearson↑ | 细胞 AUC↑ | 细胞 F1↑ |
|---|---|---|---|---|---|---|
| **vit_matte_512** | ViT | **39.25** | 0.814 | **0.217** | **0.831** | **0.383** |
| pix2pixHD | CNN-GAN | 35.14 | **0.873** | 0.186 | 0.796 | 0.346 |
| I2SB | 扩散 | 36.76 | 0.763 | 0.173 | 0.741 | 0.295 |
| pix2pix_w | CNN-L1加权 | 27.26 | 0.824 | 0.038 | 0.515 | 0.070 |

**关键结论**
1. **ViT（vit_matte）图像质量与细胞级指标双第一**（PSNR 39.25 / AUC 0.831 / F1 0.383）。
2. **GAN 对抗损失对细胞级生物学信号至关重要**：pix2pixHD（GAN）AUC 0.796 vs pix2pix_w（纯 L1 加权）AUC 0.515 ≈ 随机。加权 L1 只提升像素质量（PSNR 26.5→27.26），学不到细胞特异性信号。
3. **I2SB 扩散排第三**：PSNR 36.76 良好，但 SSIM 最低（0.763）、细胞 AUC 0.741。诊断（见 §10）：免疫 marker 被系统性低估 2-7 倍、信号弥散（非单纯弱信号），根因在训练阶段而非采样；最终 ckpt 未保存（`latest.pt` 停在 15001/20000 iters）也影响其潜力。
4. 四范式排名：**ViT > CNN-GAN > 扩散 > CNN-L1**。
5. pix2pix 训练日志中 val PSNR 36.75 为**虚高**（train.py evaluate 用 `*0.5+0.5` 错误还原 [-0.9,0.9] 目标），`eval/metrics.py` 用 `(x+0.9)/1.8` 正确还原，真实 test PSNR 27.26。

### ✅ 可视化完成
- 整张 WSI 全景（`18459_LSP10353`，9324 tiles，152×108 网格，9728×6912 px）：
  - `wsi_panorama/wsi_18459_LSP10353_HE.png`（H&E）
  - `wsi_panorama/wsi_18459_LSP10353_GT.png`（GT 多色荧光）
  - `wsi_panorama/wsi_18459_LSP10353_{vit512,pix2pixhd,pix2pix}_pred.png`（三模型预测）
- patch 级对比图（8 张，覆盖 CD4/CD8a/FOXP3/CD20/CD68/CD163 不同免疫微环境）：`visualization/patch_comparisons/`，每张 [H&E | GT | vit512 | pix2pixHD | pix2pix | I2SB]，统一配色/增益（控制变量）。
- I2SB 改进验证对比图（4 张）：`visualization/i2sb_step1/`，每张 [GT | 原始 I2SB | ensemble×8 | ensemble+校准]。

### ⏳ 待办
- ✅ WSI 预测全景（3 模型并行）已完成。
- ✅ I2SB 第 1 步零成本验证（nfe 提升 / ensemble / per-marker 校准）已完成，结论见 §10。
- ⏳ I2SB 重训（第 2 步根治，见 §10.4 改进方案）。
- ✅ 科研总结报告撰写完成（2026-08-28，`h2e_mif_benchmark/科研总结报告.md`，含全部参数、指标计算口径、逐通道/逐 marker 结果明细）。

---

## 10. I2SB 问题诊断与改进方案（2026-08-28）

> 背景：I2SB 在 test 集上 PSNR 36.76 / SSIM 0.763 / 细胞 AUC 0.741，图像质量尚可但细胞级信号与可视化效果明显弱于 vit_matte 与 pix2pixHD。本节记录对 I2SB 输出的系统性诊断、验证实验与后续改进方案。

### 10.1 六大直观问题（4 张样例图全部复现）

| # | 问题 | 表现 |
|---|---|---|
| 1 | **全局褪色、荧光亮度偏移**（最突出） | 真实图有丰富多色信号（粉/绿/橙/红），I2SB 整图偏暗，大量蛋白信号丢失，仅蓝色细胞核明显，其他蛋白通道被压低 |
| 2 | **信号弥散、模糊，纹理丢失** | 蛋白信号"糊成一团"，细胞间精细纹理、斑点状荧光模式被抹平（扩散模型通病）；核边界尚可，蛋白亚细胞定位细节丢失 |
| 3 | **伪影：成片泛白/过曝** | 部分样本中间出现大片异常亮白区域，非生物学真实信号（采样步数不足 / EMA 不匹配易出现） |
| 4 | **跨通道串色错误** | 生成不存在的杂色（如不应有绿色的间质区出现大片绿色背景），真实 mIF 中不同 marker 有空间约束，I2SB 违反此约束 |
| 5 | **弱表达蛋白直接归零** | 低丰度蛋白（真实图微弱荧光斑点）在 I2SB 输出中直接消失，对生物分析不友好 |
| 6 | **局部小结构失真** | 微小腺腔、细小间质条索形态畸变，轮廓大体对但精细结构有偏差 |

### 10.2 根因归因（问题 → 机制）

| # | 现象 | 根本机制 |
|---|---|---|
| 1 | 全局褪色、蛋白丢失 | 扩散模型输出分布整体下移 + 稀疏信号被"概率平均"压掉 |
| 2 | 信号弥散、纹理丢失 | 扩散模型匹配**像素概率分布**，天然倾向平滑（"average"而非"mode"） |
| 3 | 泛白/过曝 | 采样时 EMA 不匹配 + 极端值未 clip + 步数不足 |
| 4 | 跨通道串色 | 网络对各通道**空间解耦不足**，损失对所有通道等权，未学到空间约束 |
| 5 | 弱蛋白生成不出 | 与 #1 同源：稀疏信号在 MSE 里贡献近零，梯度被主导通道淹没 |
| 6 | 微小结构失真 | 256 分辨率 + 高频细节重建不足 |

**核心矛盾**：MSE 去噪目标 = 让所有通道平均重建，导致"高频稀疏信号被牺牲以保主结构"。

### 10.3 第 1 步零成本验证实验与结论

**实验 1：采样步数 nfe 提升（100 → 999）**
- 结果：nfe=100 与 nfe=999 预测几乎相同（CD4 均值 5.1116 vs 5.1117，差异 <0.0001%）。
- 结论：**I2SB 采样在 nfe≥100 已收敛，采样步数不是瓶颈**（I2SB 少步高质量的设计优势）。增加 nfe 无用。

**实验 2：ensemble 采样（N=8 平均）**
- 结果：随机伪影略减，但免疫信号强度不变。
- 结论：ensemble 只平均掉随机噪声（`diffusion.py` `p_posterior` 中的 `randn`），**不解决系统性信号低估**。

**实验 3：per-marker 强度校准（p99 对齐）**

校准系数 = GT_p99 / 预测_p99（越大 = 预测越弱）：

| marker | 校准系数 | 含义 |
|---|---|---|
| CD163 | **6.71×** | M2 巨噬信号被低估 6.7 倍 |
| CD8a | **5.77×** | CD8+ T 杀伤被低估 5.8 倍 |
| CD68 | 3.36× | 巨噬被低估 3.4 倍 |
| Pan-CK | 2.94× | 上皮也偏弱 |
| CD4 | 2.42× | CD4+ T 被低估 |
| Hoechst | 1.08× | 细胞核（正常，几乎不偏） |

- 定量证实：**免疫 marker 被系统性低估 2-7 倍，而细胞核正常**（对应 #1 #5）。
- 但校准放大后**曝光过度、看不清基本结构**（效果最差）——因为 I2SB 信号是**弥散的低强度噪声**（#2），简单线性放大把弥散噪声一起放大铺满全图。
- **关键诊断**：I2SB 信号不是"弱"，而是"弥散"。GT 的免疫信号是稀疏强斑点（少数像素亮、大量 0），I2SB 是大面积低强度弥散。后处理（校准/增益）无法还原稀疏结构，**治标不治本**。

### 10.4 完整改进方案（分层）

#### 🔴 A. 损失层面（根治弥散 + 弱信号，最核心）

| 方案 | 解决 | 改动位置 | 说明 |
|---|---|---|---|
| A1 通道加权去噪 MSE | #1 #5 | `i2sb/runner.py` `loss=F.mse_loss(...)` | 按 x0 通道 std 反比加权，免疫 marker 获 5-7 倍梯度权重 |
| A2 对抗损失（GAN 判别器） | #2 #4 #5 | 新增判别器 + 对抗 loss | **最对症**：pix2pixHD 成功关键，强迫输出稀疏锐利的真实荧光而非弥散平均 |
| A3 感知损失（VGG） | #2 #6 | 加 VGG 特征 loss | 特征级损失保留纹理/结构 |

#### 🟠 B. 数据层面

| 方案 | 解决 | 改动位置 | 说明 |
|---|---|---|---|
| B1 mIF 非线性归一化（gamma/log） | #1 #5 | `dataset/orioncrc_paired.py` `x0 = mif/127.5-1` | 提升弱信号相对强度（三选一：gamma / log1p / per-marker 归一化） |
| B2 免疫富集 tile 过采样 | #5 | dataset 采样器 | 让模型多见免疫信号（当前随机采样，免疫 tile 占比低） |
| B3 数据增强 | 泛化 | dataset | 翻转/颜色抖动 |

#### 🟡 C. 训练策略

| 方案 | 解决 | 改动位置 | 说明 |
|---|---|---|---|
| C1 训练量 50000+ iters | #1 #2 #5 #6 | `--num-itr` | 当前 20000 ≈ 8.7 epochs，严重不足 |
| C2 ema 0.999 | #3 | `--ema` | 更平滑 |
| C3 fp16 + 大 batch | 加速 | `train.py` `opt.use_fp16=False` | 让大训练量可行 |

#### 🟡 D. 采样层面（零训练成本）

| 方案 | 解决 | 说明 |
|---|---|---|
| D1 DDIM 确定性采样 | #3 | 去随机噪声，减少泛白伪影 |
| D2 guided sampling（CFG） | #1 #5 | 条件引导增强信号强度 |

#### 🟢 E. 架构层面（成本高，进阶）

| 方案 | 解决 | 说明 |
|---|---|---|
| E1 512 分辨率 | #2 #6 | 需 512 cache + 显存×4 |
| E2 条件注入改进（多尺度/CrossAttn） | #4 | 通道空间解耦 |

### 10.5 推荐实施顺序与关键结论

```
第 1 步（已完成）：nfe/ensemble/校准 → 确认采样非瓶颈、根因在训练
第 2 步（待实施，根治）：A1 通道加权 + A2 对抗损失 + B2 免疫过采样 + B1 归一化 + C1 训练量
第 3 步（可选进阶）：E1 512 分辨率
```

**关键结论**：
1. I2SB 的核心问题是**信号弥散**（#2）而非单纯信号弱（#1 #5 是其表象）。
2. 简单后处理（校准/增益）会放大弥散噪声导致曝光过度，不可行。
3. **必须靠 GAN 对抗损失（A2）+ 通道加权（A1）**让模型学会输出稀疏锐利信号——这正是 pix2pixHD（AUC 0.796）碾压纯 L1 pix2pix（AUC 0.515）的同一原理。
4. 待改动的 bug 已修复：`runner.py` checkpoint 保存（`it % 5000 == 0 or it == num_itr-1`）。

---

## 12. vit_matte V3：正样本加权 + per-marker log 归一化 + 四卡 DDP（2026-09-06 → 09-09）

> 背景：V2 可视化偏暗根因——GT 阳性像素 p99≈106-111，V2 预测除 Hoechst(0.83×) 外仅 GT 的 2-16%。
> 机制：① per-marker log 归一化目标域（默认开）；② 正样本加权（fg_weight=10）；③ 验证指标 log 域口径。

### 12.1 机制
- **log 归一化**（`datacore/orioncrc_dataset.py::normalize_mif_log`）：`x0 = 2·log2(min(y,q_c)/q_c+1) − 1 ∈ [-1,1]`，q 来自 `marker_q.json`（同 I2SB v2 口径）；反变换 `denormalize_mif_np` 互逆
- **正样本加权**（`vit_matte/train.py`）：`loss = Σ_c mean_HW(se·(1+fg_weight·fg))·marker_w_c`，fg 为阳性像素 mask；log 域 marker_w=1（动态范围已平衡）
- **验证 log 口径**：val PSNR/SSIM 按论文公式 `I_log = 255·ln(min(I,q)/q+1)`（= LN2_255·(x0+1)/2）计算，与 `eval/metrics_log.py` 数值等价（实测差异 <1e-5）
- **采样自动识别**：`vit_matte/sample.py` 从 ckpt config 读 tile_size/vit_size/vit_layers/lora/log_norm

### 12.2 四卡 DDP 训练设施（2026-09-09 上线）
- `--ddp`（torchrun）+ DistributedSampler + 梯度累积 no_sync
- 每 1000 步：`{name}_step{it}.pt` 快照 + `viz_{name}/viz_{name}_step{it}.png` [H&E|GT|预测]；`--ckpt_keep 5` 自动清理
- `{name}_best.pt`：val PSNR（log 域）最优的 model+optimizer+config
- **正式训练已启动（2026-09-09 12:39）**：vit256（tile 256 / vit_size 224），batch 32/卡（全局 128），15 epoch ≈ 34.6k iters；4 卡 100% 利用率、约 27GB/卡
  - 日志 `/data/weiyh/logs/vitmatte_v3_256.log`；启动脚本 `scripts/launch_vit_v3_256_ddp.sh`
- 冒烟已验证：DDP 循环、best.pt、step 快照、viz 图全通；find_unused_parameters=True 实际无 unused（告警提示可改 False）
