# 术语表

本手册用到的术语、它们在这里的含义，以及中文版的写法。标为“保留”的术语在中文正文里保持英文；
页面可以在第一次使用时在括号里加中文释义。

## 保留英文的术语

| 术语 | 在本手册中的含义 | 中文版写法 |
|---|---|---|
| batch-invariant | 一行的结果不取决于它所在的批次（L2 级） | 保留；释义 批不变 |
| L1–L5 | 五个不变性级别：run-to-run、批次、模式、跨引擎、并行（见[基础](0-foundations/README.md)） | 保留 |
| rollout | 强化学习的推理一侧：引擎生成 token 及其 logprob | 保留；释义 采样生成 |
| trainer | 强化学习的训练一侧：重新计算 logprob 并更新权重 | 保留；释义 训练端 |
| logprob | token 在策略下的对数概率 | 保留；释义 对数概率 |
| staleness | 生成 rollout 的权重与 trainer 权重之间的策略版本差 | 保留；释义 策略滞后 |
| kernel | 由算子启动的一个 GPU 函数 | 保留 |
| GEMM | 通用矩阵乘法 | 保留 |
| split-K | 把归约维度拆给多个 program，再把部分和相加 | 保留 |
| KV cache | 之前 token 的 key 和 value 的存储 | 保留 |
| prefill | 一次前向处理全部提示 token | 保留；释义 预填充 |
| decode | 每步依据 KV cache 生成一个 token | 保留；释义 解码 |
| TP/DP/EP/FSDP | 张量并行、数据并行、专家并行和全分片数据并行 | 保留 |
| SDC | 静默数据损坏：硬件返回错误的值却不报错 | 保留；释义 静默数据损坏 |
| fingerprint | 在设备上计算的张量比特小摘要（XOR） | 保留；释义 指纹 |
| OpGuard | 本手册使用的捕获与定位工具（[仓库](https://github.com/ziming-zh/OpGuard)） | 保留 |
| harness | OpGuard 面向 agent 的工具层；也指包在示例外面的测试驱动 | 保留 |

## 中文术语选择

| 英文 | 中文 | 说明 |
|---|---|---|
| alignment, aligned | 对齐 | bitwise alignment：逐位对齐 |
| bit for bit, bitwise | 逐位 | bit-equal：逐位相等 |
| invariance, invariant | 不变性, 不变 | |
| nondeterminism | 非确定性 | controlled / residual：受控 / 残余 |
| determinism, deterministic | 确定性 | |
| reduction, reduction order | 归约, 归约顺序 | reduction tree：归约树 |
| rounding, rounding point | 舍入, 舍入点 | |
| accumulate, accumulator | 累加, 累加器 | |
| operator | 算子 | operator boundary：算子边界 |
| first divergence | 首个分歧点 | diverge：出现分歧 |
| localize | 定位 | |
| preflight | 预检 | |
| pre-training | 预训练 | |
| RL, train–inference | 强化学习, 训推 | |
| inference engine, engine | 推理引擎, 引擎 | |
| collective | 集合通信 | |
| attention, linear attention | 注意力, 线性注意力 | |
| sampling, sampler | 采样, 采样器 | |
| adapter | 适配器 | LoRA 保留 |
| quantized | 量化 | |
| checkpoint, resume | 检查点, 恢复训练 | |
| optimizer, gradient, weights | 优化器, 梯度, 权重 | |
| regression, baseline | 回归, 基线 | |
| verdict | 判定 | |
| atomics | 原子操作 | |
| tile | tile | 保留 |
| seed | 种子 | |
| skill (agent skill) | skill（agent skill） | 保留 |
| open items | 待办事项 | |
| workflow, step | 工作流程, 步骤 | |
| report (an example's `REPORT.md`) | 报告, 实验报告 | 页面背后的证据 |
| known-answer demonstration | 已知答案的演示 | 原因是事先植入的 |
| emulated | 模拟 | |
| not yet shown | 尚未证明 | |

## 代价等级

页面按等级描述一个选项的代价；页面背后的报告给出测得的数字。

| 等级 | 含义 | 中文版写法 |
|---|---|---|
| free | 没有可测出的代价 | 免费 |
| small | 百分之几 | 小（百分之几） |
| moderate | 不到 2 倍 | 中等（不到 2 倍） |
| large | 2 倍或更多 | 大（2 倍或更多） |
