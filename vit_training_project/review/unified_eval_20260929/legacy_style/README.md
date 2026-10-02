# 旧版多色荧光patch可视化

已按v1/v2/v3原始multicolor_composite函数重新生成：固定10通道颜色、固定GAIN、逐通道相加并裁剪RGB。没有argmax、概率阈值或逐图自动增强；共表达会混色。

沿用之前固定的6张测试切片、每张4个patch，共24张。主图为H&E | GT真实强度 | v1 | v2 | v3 | 分类基线。前三版为还原后的强度，分类基线为概率，后者的亮度不能解释为mIF强度。为便于核对，另附GT二值标签与分类概率的无增益配色图。所有图使用相同的H&E玻片背景掩膜，不用DAPI裁掉核外信号。

旧版函数仅展示10个marker，其余CD31、CD45、CD45RO、PDL1、ECadherin、Ki67没有加入合成图。这是忠实复用历史显示规则，16通道预测数据仍保留在原评估目录。

CRC02是四版本共同未训练患者；其他切片仅作定性查看，旧模型见过这些患者，CRC33还涉及旧基线另一切片的训练暴露。这里的分类基线是已完成的ddp_baseline，未替换成正在训练的阳性Dice模型。

文件：index.html为独立可浏览图册；patches/为24张六栏对照PNG；rgb/为各面板256像素PNG；probability_pairs/为24张标签/概率对照；patch_comparisons.pdf为24页图册。patch_manifest.csv记录全部patch身份，protocol.json记录配色、参数与预测输入哈希。

只重新渲染已有预测，没有重新推理，没有改变指标或正在进行的训练。
