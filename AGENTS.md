@C:\Users\Edifier\.codex\RTK.md

# 项目研究上下文

用户希望后续创新改进持续参考已核实的机制与实验方案。涉及 WBT 混合训练、Stage 2 泛化、策略冻结、双机协作、奖励/观测/恢复训练的改动前，请读取：

- [训练机制 FAQ 与第一阶段代码阅读路线](docs/hybrid-training-faq-zh.md)
- [策略偏移与双 G1 协作搬运研究方案](docs/dual-humanoid-research-plan-zh.md)

以当前源码和最终配置为准；文档中的候选方法、能力假设与实验计划不表示已实现或已验证。用户已确认两台 Unitree G1，计划面向 RSS 2027 第一轮投稿；具体时间与资源条件在后续工作时重新核实。

更改相关机制时同步维护文档，尤其核对观测/normalizer/checkpoint 兼容性、参考片段 reset 与 episode reset、动态目标与一次性奖励、严格冻结与 FiLM 适配的区别。不根据研究计划自动启动训练或硬件运行；按用户当次任务范围执行。
