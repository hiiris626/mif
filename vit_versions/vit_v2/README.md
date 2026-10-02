# ViT v2 历史最终结果

本目录保留v2选用权重、最终预测、指标和可视化。原重复源码已替换为指向项目共享实现的软链；历史配置见 `config.json`。新的训练任务统一使用 [classification](../classification/README.md)，不再用历史启动脚本训练。

- 选用权重：`virchow2_vitmatte_v2_512_epoch15.pt`。
- 历史预测/指标：`results/`；历史回归指标仅作归档，不与新分类指标直接比较。
- 复现历史推理：`bash vit_versions/vit_v2/sample.sh --out_dir <新输出目录>`。
- v1的legacy结构已修复；v2/v3继续按checkpoint配置加载。
- 原未选用快照软链、过程预览与旧启动脚本已清理。外部真实权重和原始数据没有删除。

完整历史差异见 [审查报告](../audit_20260925/代码检查与三版差异.md)。
