---
name: engine-parity
description: 在真实模型上，让 trainer 前向和推理引擎（vLLM、SGLang、自定义服务）达到跨引擎逐位相等（L4）。盘点两侧的前向，写出两侧的舍入序列，先检查引擎自身的 L1/L2/L3，一次把 trainer 的一个舍入点改成引擎的做法并在每次之后测量不同 token 的数量，逐算子、逐层定位剩余差异，并对每个差异决定是改 trainer、改引擎，还是接受并记录。当强化学习 rollout 的 logprob 与 trainer 的 logprob 不一致时，或在声称 trainer 和引擎计算的是同一个策略之前使用。
---

# 引擎对齐

一次只负责一对：一个 trainer（训练端）前向和一种引擎配置，在一个模型、GPU 和软件栈上。
把这一对从“未知”带到“在声明的提示集和批大小范围上，实际 token 的 logprob（对数概率）
逐位相等”（L4），或者带到一个指名的剩余算子，附上证据以及每种解决方式的代价。
交付两侧的舍入序列、步骤表、每个差异的决定，以及最终引擎配置的吞吐代价。

背景和级别：[skills/batch-invariant-kernel/SKILL.md](../batch-invariant-kernel/SKILL.md)。
目前找到的每个舍入点，以及每一侧的做法：
[references/rounding-points.md](references/rounding-points.md)。
一个完整案例（Qwen2.5-0.5B、vLLM 0.19.1、一个 GPU）：
[handbook/2-rl/engine-parity-vllm.md](../../handbook/2-rl/engine-parity-vllm.md) 和
[examples/cross_engine/](../../examples/cross_engine/)，其中的 `trainer.py`
和 `cross_engine.py` 是第 4 步和第 5 步的模板。

## 测量强化学习实际使用的量

比较每个实际 token 的对数概率：提示 token（prompt logprobs）和采样得到的 token。
把引擎自己的 token 以 teacher forcing 方式送入 trainer，这样不同的 token 就不会掩盖数值差异。
比较 fp32 的比特，绝不使用容差。报告不同 token 的比例、最大的 |Δlogp|，
以及在能拿到两侧完整分布时的平均 KL（hook 引擎的 logits processor）。
logprob 相等是要声明的结论；logits 相等是更强的事实，用来解释它。

在做其他事之前先检查一件事：引擎报告的 logprob 等于你捕获的 logits 的
`log_softmax(fp32(engine logits))`，并且计算方式与引擎相同（同一设备：在这里检查的值上，
CPU 和 GPU 的 `log_softmax` 相差约 2.5e-5）。如果不相等，说明引擎对 logits
做了后处理（temperature、惩罚项、“processed” logprobs 模式），你比较的对象就错了。

## 1. 盘点两侧的前向

对每一侧，依据源码和实际运行的内容：
- 模型代码：有哪些模块，哪些被打包（QKV、gate/up），哪些被融合
  （残差相加 + norm），bias 在哪里加入；
- 分派：自定义 CUDA 算子、编译器（`torch.compile`/Inductor 会改写原生算子；
  vLLM 在编译时关闭它的自定义算子）、CUDA graph、实际选中的注意力后端；
- 每个会改变 kernel 的开关：batch-invariant 模式、eager 模式、注意力后端、split-KV、
  cascade attention、前缀缓存、分块 prefill；
- 这个开关在你的 GPU 上实际覆盖了什么。阅读对应你的计算能力的代码路径。
  测得：vLLM 0.19.1 的 batch-invariant 模式只在列出的一组计算能力上覆盖
  `aten::mm/addmm/linear`（vLLM `batch_invariant.py` 中的 `enable_batch_invariant_mode`）；
  在所测的 GPU 上（不在该组内），lm_head 仍走 cuBLAS，而 decoder 的 linear 用 vLLM 的 Triton GEMM。

打印已加载的引擎模型的模块树和后端类名；不要相信你传入的配置。

## 2. 写出两侧的舍入序列

对每一侧，按顺序每个舍入点写一行：算子、每个输入和累加器的 dtype、每次类型转换、
什么被融合进了什么、已知的归约顺序。把
[references/rounding-points.md](references/rounding-points.md) 中的表作为检查清单。
每一条标明“读自源码（文件、版本）”或“测得”。
源码不可得的地方（编译好的 CUDA kernel），要说明，并用算子重放来确定（第 5 步）。

## 3. 先检查引擎自身

在与 trainer 做任何比较之前，先在提示集上单独测量引擎：
- L1：同一批运行两次；
- L2：每个提示单独运行，与批大小 4、16、64 比较；
- L3：decode 与同样 token 的 prefill 比较（第 5 步中对注意力的算子重放会逐层做这件事）。

如果引擎不是 L2，那么无论 trainer 怎么做，批量 rollout 的 trainer–引擎 L4 都不可能成立。
把它报告为引擎的变化，然后要么把引擎切换到 L2 的模式，要么把结论限制在批大小 1。
测得（Qwen2.5-0.5B，[报告](../../examples/cross_engine/REPORT.md)）：默认的 vLLM 在批大小 1 和批大小 64 之间改变了 96% 的
实际 token logprob；batch-invariant eager 模式改变了 7%（全部来自该模式在所测 GPU 上
未覆盖的 cuBLAS lm_head），把 lm_head 移到 vLLM 自己的 batch-invariant GEMM 后则为零。

## 4. 一次对齐一个舍入点

从 trainer 的现状开始。把一个舍入点改成引擎的做法，重新运行 teacher forcing 比较，
记下不同 token 的数量，然后改下一个。优先在 trainer 中使用引擎自己的 kernel（导入它），
而不是重新实现：除非 tile 形状、归约顺序和编译器都一致，否则相同的源码并不等于相同的比特。
然后从最终状态做 leave-one-out：单独撤回每个点，再计数。在当前提示集上撤回后不改变任何结果的点，
并没有被证明无关；要说明这一点。

计数可能在几步内保持不变，而在最后一步下降。logits 上游的每个舍入点都必须一致，
才会有任何 token 相等，所以显示进展的是下面的逐层定位，而不是 token 计数。

## 5. 定位剩余差异

两个工具，按这个顺序：
- **算子重放。** hook 引擎的每个模块（前向的 pre-hook 和 post-hook，在原地 kernel 运行前
  克隆输入），记录一次 prefill 和几步 decode，然后把每个捕获的输入送入每个 trainer 候选实现，
  统计逐位相等的输出。这能独立于其他舍入点确定每一个舍入点，并显示 L3 效应
  （把 decode 步与 trainer 对整个序列的 prefill 比较）。
- **逐层流。** 在捕获的提示上运行 trainer，找到残差流或 MLP 输出与引擎不同的第一层。

hook 只在 eager 模式下触发。对于编译或 graph 捕获的引擎，改为比较边界张量的
逐向量 fingerprint（指纹）：OpGuard 的 `lastdim-u32le-v1` 向量，在纯 torch 中可通过
`tools/opguard_bridge.py xor_vectors` 获得；以及 OpGuard 的 `rl-alignment` skill
（OpGuard 仓库，`harness/skills/rl-alignment/SKILL.md`），它把 logprob 差距拆分为
staleness（策略滞后）和数值误差，并沿首个分歧点追到一个算子和 token。

## 6. 对每个差异做决定

| 选择 | 何时 | 需要说明的代价 |
|---|---|---|
| 改 trainer | 引擎的算子是 batch-invariant 的，且 trainer 可以调用（Triton kernel、自定义算子） | trainer 吞吐；反向：引擎的 kernel 可能没有反向 |
| 改引擎 | 引擎的算子不是 batch-invariant 的，或 trainer 无法调用它（闭源 kernel、编译器输出） | 引擎吞吐，在 decode 批大小上测量 |
| 接受并记录 | 两种算法上不同的形式（编译器融合、不同的注意力算法），且两侧都无法改动 | 测得的差距：token 比例、最大 |Δlogp|、KL |

把决定连同其测得的效果写在每个舍入点旁边。

## 7. 报告

- 模型、GPU、两侧的版本、引擎参数、提示集和批大小；
- 每种配置下引擎的 L1/L2；
- 两侧的舍入序列；
- 步骤表和 leave-one-out 表；
- 算子重放表；
- 最终的 L4 结果（比较的 token 数、不同的 token 数、最大 |Δlogp|、KL）；
- 最终引擎配置相对默认配置的吞吐；
- 已知限制：未测试的提示和长度、未覆盖的模式（前缀缓存、投机解码、TP），
  未覆盖反向。

## 防护规则

- 没有在相同提示和批上得到引擎 L2，绝不声明 L4。
- 绝不比较不同 token 的 logprob：用 teacher forcing 送入引擎的 token。
- 引擎的开关不等于它的覆盖范围。阅读对应你的 GPU 计算能力的代码路径；测量它改变了什么。
- 编译后的引擎是另一个引擎。Inductor 会融合原生算子并去掉中间的舍入；
  eager 的结果不能沿用到编译模式。
- 相同的源码不等于相同的比特：在另一个框架或另一种 tile 形状中重新实现 kernel
  会改变归约顺序。导入 kernel。
- 在一个模型上一致的舍入点在另一个模型上可能不一致：bias、共享的 embedding、
  恰好为 1 的 norm 权重或 2 的幂的缩放会掩盖差异。使用真实的检查点。
- batch-invariant 引擎模式有吞吐代价。在 decode 批大小上测量，并在对齐结论旁边报告。
- 不改动生产默认值。对齐模式和补丁需要显式开启。
