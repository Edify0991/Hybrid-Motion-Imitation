# 主张与证据

| ID | 主张 | 来源 | 强度 | 风险与状态 |
|---|---|---|---|---|
| E1 | std不能单独测行为保留 | PPOActor代码，std不依赖state；均值推理 | evidence-backed | 需看effective std而非仅raw参数 |
| E2 | 当前非仅第0帧随机化 | MotionCommand.reset与command配置 | evidence-backed | 不同噪声作用位置不要混淆 |
| E3 | 随机邻域与技能拼接已有研究 | SkillMimic-V2 2505.02094 Sec4.3/4.4 | evidence-backed | 不能把简单加噪作为贡献 |
| E4 | 单机到多人形搬运已有近邻 | decMHT 2609.17824 SecIV/V | evidence-backed | 与冻结全身交互策略不同 |
| E5 | 异构分层搬运已有近邻 | HA-DiPO 2610.33584 | evidence-backed | 不把不同硬件组合当独立创新 |
| E6 | 少量演示可能缺少侧向和转弯技能 | 当前两npz；需计算和执行能力测量 | hypothesis | 演示覆盖不等于可执行能力 |
| E7 | 均值偏移、访问分布和接触一致性比std更有诊断力 | 统计定义+待做对照 | hypothesis | 需留出集和干预证明 |
| E8 | 能力约束的接触协调能减少双机失效 | 本方案待检验 | hypothesis | 能力模型可能不能覆盖伙伴扰动 |
| E9 | 无关节的共享高层可跨形态迁移 | 本方案待检验 | hypothesis | 必须有能力上下文与独立低层 |
| E10 | RSS首轮允许初步实验但须完整方法 | 官方2027 CFP | evidence-backed | 不是录用保证；投稿前复核 |

来源链接与章节将在主方案参考文献表中维护。文献是定向检索而非穷尽综述；最新近邻的优先级高于泛泛列举分层RL。
