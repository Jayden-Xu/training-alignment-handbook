# Glossary

The terms the handbook uses, what they mean here, and how the Chinese edition
writes them. Terms marked "kept" stay in English in the Chinese text; a page may
add the Chinese gloss in parentheses the first time it uses one.

## Terms kept in English

| Term | Meaning here | Chinese edition |
|---|---|---|
| batch-invariant | A row's result does not depend on the batch it is computed in (level L2) | kept; gloss 批不变 |
| L1–L5 | The five invariance levels: run-to-run, batch, mode, cross-engine, parallelism ([foundations](0-foundations/README.md)) | kept |
| rollout | The inference side of RL: the engine generates tokens and their logprobs | kept; gloss 采样生成 |
| trainer | The training side of RL: recomputes logprobs and updates the weights | kept; gloss 训练端 |
| logprob | Log-probability of a token under the policy | kept; gloss 对数概率 |
| staleness | The policy version gap between the weights that generated a rollout and the trainer's weights | kept; gloss 策略滞后 |
| kernel | One GPU function launched by an operator | kept |
| GEMM | General matrix multiply | kept |
| split-K | Splitting a reduction dimension across programs, then adding the partial sums | kept |
| KV cache | Stored keys and values of earlier tokens | kept |
| prefill | Processing the prompt tokens in one forward | kept; gloss 预填充 |
| decode | Generating one token per step from the KV cache | kept; gloss 解码 |
| TP/DP/EP/FSDP | Tensor, data, expert and fully sharded data parallelism | kept |
| SDC | Silent data corruption: hardware returns wrong values with no error | kept; gloss 静默数据损坏 |
| fingerprint | A small device-side summary (XOR) of a tensor's bits | kept; gloss 指纹 |
| OpGuard | The capture and localization tool the handbook uses ([repository](https://github.com/ziming-zh/OpGuard)) | kept |
| harness | OpGuard's agent-facing tool layer; also a test driver around an example | kept |

## Chinese term choices

| English | Chinese | Note |
|---|---|---|
| alignment, aligned | 对齐 | bitwise alignment: 逐位对齐 |
| bit for bit, bitwise | 逐位 | bit-equal: 逐位相等 |
| invariance, invariant | 不变性, 不变 | |
| nondeterminism | 非确定性 | controlled / residual: 受控 / 残余 |
| determinism, deterministic | 确定性 | |
| reduction, reduction order | 归约, 归约顺序 | reduction tree: 归约树 |
| rounding, rounding point | 舍入, 舍入点 | |
| accumulate, accumulator | 累加, 累加器 | |
| operator | 算子 | operator boundary: 算子边界 |
| first divergence | 首个分歧点 | diverge: 出现分歧 |
| localize | 定位 | |
| preflight | 预检 | |
| pre-training | 预训练 | |
| RL, train–inference | 强化学习, 训推 | |
| inference engine, engine | 推理引擎, 引擎 | |
| collective | 集合通信 | |
| attention, linear attention | 注意力, 线性注意力 | |
| sampling, sampler | 采样, 采样器 | |
| adapter | 适配器 | LoRA kept |
| quantized | 量化 | |
| checkpoint, resume | 检查点, 恢复训练 | |
| optimizer, gradient, weights | 优化器, 梯度, 权重 | |
| regression, baseline | 回归, 基线 | |
| verdict | 判定 | |
| atomics | 原子操作 | |
| tile | tile | kept |
| seed | 种子 | |
| skill (agent skill) | skill（agent skill） | kept |
| open items | 待办事项 | |
| workflow, step | 工作流程, 步骤 | |
| report (an example's `REPORT.md`) | 报告, 实验报告 | the evidence behind a page |
| known-answer demonstration | 已知答案的演示 | the cause was planted in advance |
| emulated | 模拟 | |
| not yet shown | 尚未证明 | |

## Cost classes

Pages describe the cost of an option by class; the report behind the page has
the measured numbers.

| Class | Meaning | Chinese edition |
|---|---|---|
| free | no measurable cost | 免费 |
| small | a few percent | 小（百分之几） |
| moderate | under 2x | 中等（不到 2 倍） |
| large | 2x or more | 大（2 倍或更多） |
