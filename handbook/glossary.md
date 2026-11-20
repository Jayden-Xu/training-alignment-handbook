# Glossary

This page lists the terms the handbook uses, says what each one means here,
and shows how the Chinese edition writes it. Chinese ML systems engineers say
many technical terms in English, so the Chinese edition keeps those terms in
English instead of coining a translation. A Chinese page may explain such a
term in Chinese the first time it uses it.

## Terms kept in English

| Term | Meaning here | Chinese edition |
|---|---|---|
| bit, bitwise | One binary digit of a stored value. Two results are bitwise equal when every bit matches | kept: 逐 bit 一致, 逐 bit 相等; never 位 or 比特 |
| batch, batch size, token | The rows or sequences one call computes together, their number, and one position in a sequence | kept; never 批次 or 令牌 |
| batch-invariant | A row's result does not depend on the batch it is computed in (level L2) | kept; explained as 结果不随 batch 变化 |
| L1–L5 | The five invariance levels: run-to-run, batch, mode, cross-engine and parallelism ([foundations](0-foundations/README.md)) | kept |
| rollout | The inference part of RL: the engine generates tokens and their logprobs | kept; rollout 引擎, rollout 端 |
| trainer | The training part of RL: it recomputes the logprobs and updates the weights | kept; trainer 端 |
| logprob | The log-probability of a token under the policy | kept; never 对数概率 |
| staleness | The gap in policy versions between the weights that generated a rollout and the trainer's weights | kept; never 陈旧度 |
| kernel | One GPU function that an operator launches | kept; never 内核 |
| GEMM, tile, split-K | A general matrix multiply; the block of the output that one program computes; splitting a reduction dimension across programs and then adding the partial sums | kept |
| KV cache | The stored keys and values of earlier tokens | kept; never 键值缓存 |
| prefill | Processing all prompt tokens in one forward pass | kept; never 预填充 |
| decode | Generating one token per step from the KV cache | kept; never 解码 |
| TP, DP, EP, PP, SP, FSDP | Tensor, data, expert, pipeline, sequence and fully sharded data parallelism | kept; the TP size is TP 规模 |
| rank, shard | One process in a parallel group; the part of a tensor that one rank holds | kept; never 秩 |
| all-reduce, reduce-scatter, all-to-all | The collectives that sum across ranks, sum and split, and exchange data between all ranks | kept |
| torch.compile, CUDA graph, eager | The PyTorch compiler; a recorded sequence of kernel launches that is replayed; running op by op without compiling | kept; never 编译代码, CUDA 图 or 急切模式 |
| checkpoint | The saved weights and optimizer state of a run | kept; never 检查点 |
| dtype, bf16, fp32 | A tensor's number format, and two common formats | kept; never 数据类型 |
| LoRA, adapter | Low-rank fine-tuning; one set of low-rank weights added to the model | kept; never 适配器 |
| MoE, expert, top-k | A mixture-of-experts layer; one expert network in it; the k largest scores | kept; never 专家混合 |
| SDC | Silent data corruption: the hardware returns wrong values and reports no error | kept; may be explained as 静默数据损坏 |
| fingerprint | A small summary (XOR) of a tensor's bits, computed on the device | kept; may be explained as 指纹 at first use |
| OpGuard | The capture and localization tool the handbook uses ([repository](https://github.com/ziming-zh/OpGuard)) | kept |
| harness | OpGuard's tool layer for agents; also a test driver around an example | kept |
| skill, agent | An agent skill, and the AI agent that follows it | kept |

## Chinese term choices

These words are ordinary Chinese technical vocabulary, so the Chinese edition
translates them.

| English | Chinese | Note |
|---|---|---|
| alignment, aligned | 对齐 | bitwise alignment: 逐 bit 对齐 |
| bit for bit, bitwise | 逐 bit | bit-equal: 逐 bit 相等; never 逐位 or 位都相同 |
| invariance, invariant | 不变性, 不变 | |
| nondeterminism | 非确定性 | controlled / residual: 受控 / 残余 |
| determinism, deterministic | 确定性 | |
| reduction, reduction order | 归约, 归约顺序 | reduction tree: 归约树 |
| rounding, rounding point | 舍入, 舍入点 | |
| accumulate, accumulator | 累加, 累加器 | |
| operator | 算子 | operator boundary: 算子边界 |
| forward, backward | 前向, 反向 | |
| precision | 精度 | |
| parallel degree | 并行度 | |
| training, inference | 训练, 推理 | |
| side (rollout side, trainer side, both sides) | 端 | rollout 端, trainer 端, 两端; never 侧 |
| job | 任务, 训练任务 | never 作业 |
| first divergence | 首个分歧点 | diverge: 出现分歧 |
| localize | 定位 | |
| preflight | 预检 | |
| pre-training | 预训练 | |
| RL, train–inference | 强化学习, 训推 | |
| inference engine, engine | 推理引擎, 引擎 | |
| collective | 集合通信 | the operations themselves keep their English names |
| attention, linear attention | 注意力, 线性注意力 | |
| sampling, sampler | 采样, 采样器 | |
| quantized | 量化 | |
| resume | 恢复训练 | checkpoint stays in English |
| optimizer, gradient, weights | 优化器, 梯度, 权重 | |
| regression, baseline | 回归, 基线 | |
| verdict | 判定 | |
| atomics | 原子操作 | |
| seed | 种子 | |
| open items | 待办事项 | |
| workflow, step | 工作流程, 步骤 | |
| report (an example's `REPORT.md`) | 报告, 实验报告 | the evidence behind a page |
| known-answer demonstration | 已知答案的演示 | the cause was planted in advance |
| emulated | 模拟 | |
| not yet shown | 尚未证明 | |

## Cost classes

Pages describe the cost of an option by its class. The report behind each page
has the measured numbers.

| Class | Meaning | Chinese edition |
|---|---|---|
| free | no measurable cost | 免费 |
| small | a few percent | 小（百分之几） |
| moderate | under 2x | 中等（不到 2 倍） |
| large | 2x or more | 大（2 倍或更多） |
