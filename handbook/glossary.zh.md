# 术语表

这一页列出本手册用到的术语，说明它们在这里指什么，以及中文版怎么写。中文 ML 系统圈子里很多技术词
习惯直接说英文，所以中文版保留这些英文词，不去生造译名；页面可以在第一次用到时用中文解释一下。

## 保留英文的术语

| 术语 | 在本手册中的含义 | 中文版写法 |
|---|---|---|
| bit, bitwise | 数值存储中的一个二进制位。两个结果的每个 bit 都相同，就叫逐 bit 一致 | 保留，统一写"逐 bit 一致"；不写"位"或"比特" |
| batch, batch size, token | 一次调用里一起计算的行或序列、它们的个数，以及序列里的一个位置 | 保留；不写"批次""令牌" |
| batch-invariant | 一行的结果不取决于它所在的 batch（L2 级） | 保留；解释为"结果不随 batch 变化" |
| L1–L5 | 五个不变性级别：run-to-run、batch、模式、跨引擎、并行（见[基础](0-foundations/README.md)） | 保留 |
| rollout | 强化学习里的推理部分：引擎生成 token 和它们的 logprob | 保留；写作 rollout 引擎、rollout 端 |
| trainer | 强化学习里的训练部分：重新计算 logprob 并更新权重 | 保留；写作 trainer 端 |
| logprob | token 在当前策略下的对数概率 | 保留；不写"对数概率" |
| staleness | 生成 rollout 所用的权重与 trainer 当前权重之间相差的策略版本数 | 保留；不写"陈旧度" |
| kernel | 算子启动的一个 GPU 函数 | 保留；不写"内核" |
| GEMM, tile, split-K | 通用矩阵乘法；一个 program 负责计算的那块输出；把归约维度拆给多个 program，再把部分和加起来 | 保留 |
| KV cache | 之前各个 token 的 key 和 value 的缓存 | 保留；不写"键值缓存" |
| prefill | 用一次前向处理全部提示 token | 保留；不写"预填充" |
| decode | 每一步根据 KV cache 生成一个 token | 保留；不写"解码" |
| TP, DP, EP, PP, SP, FSDP | 张量并行、数据并行、expert 并行、流水线并行、序列并行和全分片数据并行 | 保留；TP 的大小写作"TP 规模" |
| rank, shard | 并行组里的一个进程；一个 rank 持有的那部分张量 | 保留；不写"秩" |
| all-reduce, reduce-scatter, all-to-all | 跨 rank 求和、求和后再切分、所有 rank 之间互相交换数据的集合通信 | 保留 |
| torch.compile, CUDA graph, eager | PyTorch 的编译器；录制下来、可以重放的一串 kernel launch；不编译、逐个算子执行的方式 | 保留；不写"编译代码""CUDA 图""急切模式" |
| checkpoint | 一次运行保存下来的权重和优化器状态 | 保留；不写"检查点" |
| dtype, bf16, fp32 | 张量的数值格式，以及两种常见格式 | 保留；不写"数据类型" |
| LoRA, adapter | 低秩微调；加到模型上的一组低秩权重 | 保留；不写"适配器" |
| MoE, expert, top-k | mixture-of-experts 层；层里的一个 expert 网络；最大的 k 个分数 | 保留；不写"专家混合" |
| SDC | 静默数据损坏：硬件算出了错误的值，却不报任何错 | 保留；可解释为"静默数据损坏" |
| fingerprint | 在设备上对张量的 bit 算出的小摘要（XOR） | 保留；第一次出现时可括注"指纹" |
| OpGuard | 本手册用来采集张量并定位分歧的工具（[仓库](https://github.com/ziming-zh/OpGuard)）。capture 一律写"采集"，CUDA graph 的 capture 写"录制" | 保留 |
| harness | OpGuard 面向 agent 的工具层；也指包在示例外面的测试驱动 | 保留 |
| skill, agent | agent 使用的 skill，以及按 skill 工作的 AI agent | 保留 |

## 中文术语选择

下面这些词本来就是通用的中文技术词汇，所以中文版用中文。

| 英文 | 中文 | 说明 |
|---|---|---|
| alignment, aligned | 对齐 | bitwise alignment：逐 bit 对齐 |
| bit for bit, bitwise | 逐 bit | bit-equal 也写"逐 bit 一致"；不写"逐位""位都相同" |
| invariance, invariant | 不变性, 不变 | |
| nondeterminism | 非确定性 | controlled / residual：受控 / 残余 |
| determinism, deterministic | 确定性 | |
| reduction, reduction order | 归约, 归约顺序 | reduction tree：归约树 |
| rounding, rounding point | 舍入, 舍入点 | |
| accumulate, accumulator | 累加, 累加器 | |
| operator | 算子 | operator boundary：算子边界 |
| forward, backward | 前向, 反向 | |
| precision | 精度 | |
| parallel degree | 并行度 | |
| training, inference | 训练, 推理 | |
| side（rollout side、trainer side、both sides） | 端 | rollout 端、trainer 端、两端；不写"侧" |
| job | 任务, 训练任务 | 不写"作业" |
| first divergence | 首个分歧点 | diverge：出现分歧 |
| localize | 定位 | |
| preflight | 预检 | |
| pre-training | 预训练 | |
| RL, train–inference | 强化学习, 训推 | |
| inference engine, engine | 推理引擎, 引擎 | |
| collective | 集合通信 | 具体的操作保留英文名 |
| attention, linear attention | 注意力, 线性注意力 | |
| sampling, sampler | 采样, 采样器 | |
| quantized | 量化 | |
| resume | 恢复训练 | checkpoint 保留英文 |
| optimizer, gradient, weights | 优化器, 梯度, 权重 | |
| regression, baseline | 回归, 基线 | |
| verdict | 判定 | |
| atomics | 原子操作 | |
| seed | 种子 | |
| open items | 待办事项 | |
| workflow, step | 工作流程, 步骤 | |
| report（示例的 `REPORT.md`） | 报告, 实验报告 | 页面背后的证据 |
| known-answer demonstration | 已知答案的演示 | 原因是事先植入的 |
| emulated | 模拟 | |
| not yet shown | 尚未证明 | |

## 代价等级

页面按等级描述一个选项的代价，具体测得的数字在页面背后的报告里。

| 等级 | 含义 | 中文版写法 |
|---|---|---|
| free | 没有可测出的代价 | 免费 |
| small | 百分之几 | 小（百分之几） |
| moderate | 不到 2 倍 | 中等（不到 2 倍） |
| large | 2 倍或更多 | 大（2 倍或更多） |
