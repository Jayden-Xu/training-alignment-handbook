---
name: engine-parity
description: 在真实模型上，让 trainer 的前向和推理引擎（vLLM、SGLang、自研的推理服务）算出逐 bit 一致的结果（跨引擎一致，L4）。做法是：先弄清两端的前向实际跑了什么，写下两端各自在哪里舍入，再确认引擎自己和自己一致（L1/L2/L3）；然后每次把 trainer 的一个舍入点改成引擎的做法，每改一次就数一遍有多少 token 不同；接着逐算子、逐层定位剩下的差异；再对每个差异决定是改 trainer、改引擎，还是接受并记录下来；最后给对齐后的前向配一个正确的反向，并用 fp32 参考检查梯度。当强化学习的 rollout logprob 和 trainer 的 logprob 对不上，或者在声称 trainer 和引擎算的是同一个策略之前，使用这个 skill。
---

# 引擎对齐

这个 skill 一次只处理一对组合：一个 trainer 前向和一种引擎配置，限定在一个模型、
一种 GPU 和一套软件栈上。开始时你并不知道两者是否一致。目标是 L4：在你声明的
prompt 集和 batch size 范围内，每个实际出现的 token，其 logprob 在两端逐 bit 一致。
如果做不到，就要指出仍然不一致的那个算子，给出证据，并说明消除这个差异的每种办法
各要付出什么代价。最后交付：

- 两端各自的舍入序列；
- 步骤表；
- 对每个差异做出的决定；
- 最终采用的引擎配置的吞吐代价；
- 反向的梯度检查结果。

背景知识和各级不变性的定义见
[skills/batch-invariant-kernel/SKILL.md](../batch-invariant-kernel/SKILL.md)。
[references/rounding-points.md](references/rounding-points.md) 列出了目前找到的
所有舍入点，以及两端在每个点上各自怎么做。
一个完整的案例（Qwen2.5-0.5B、vLLM 0.19.1、一块 GPU）写在
[handbook/2-rl/engine-parity-vllm.md](../../handbook/2-rl/engine-parity-vllm.md)，
代码在 [examples/cross_engine/](../../examples/cross_engine/)。其中的
`trainer.py` 和 `cross_engine.py` 是第 4 步和第 5 步的模板。

## 测量强化学习真正用到的量

强化学习用的是每个实际出现的 token 的 logprob，所以要比较的就是它，既包括 prompt
token（prompt logprobs），也包括采样出来的 token。用 teacher forcing 把引擎自己生成的
token 送进 trainer，这样就不会因为 token 不同而掩盖数值上的差异。逐 bit 比较 fp32 值，
绝不设容差。报告不同 token 所占的比例、最大的 |Δlogp|；能拿到两端完整分布时，再报告
平均 KL（引擎端的分布可以通过 hook 它的 logits processor 拿到）。对外声明的结论是
logprob 相等；logits 相等是更强的事实，它解释了 logprob 为什么相等。

在做其他任何事之前，先检查一件事。引擎报告的 logprob，必须等于你在采集到的 logits 上
计算 `log_softmax(fp32(engine logits))` 的结果，而且要按引擎的方式、在同一种设备上算
（在这里检查过的值上，CPU 和 GPU 的 `log_softmax` 相差约 2.5e-5）。如果两者不相等，
说明引擎对 logits 做了后处理（temperature、惩罚项、“processed” logprobs 模式），
你比较的就不是同一个量。

## 1. 弄清两端的前向实际跑了什么

对每一端，既看源码，也看实际运行的情况，整理出下面这些内容：

- 模型代码里有哪些模块，哪些投影被打包成一个 GEMM（QKV、gate/up），哪些算子被融合
  在一起（残差相加 + norm），bias 在哪里加上；
- 算子怎样分派，包括自定义 CUDA 算子、编译器（`torch.compile`/Inductor 会改写原生
  算子，vLLM 在编译时会关掉自己的自定义算子）、CUDA graph，以及实际选中的注意力后端；
- 每个会改变所用 kernel 的开关，包括 batch-invariant 模式、eager 模式、注意力后端、
  split-KV、cascade attention、prefix caching 和 chunked prefill；
- 这些开关在你的 GPU 上实际覆盖了什么。要读与你的 GPU compute capability 对应的
  代码路径。实测中，vLLM 0.19.1 的 batch-invariant 模式只在列出的几种 compute
  capability 上覆盖 `aten::mm/addmm/linear`（见 vLLM `batch_invariant.py` 中的
  `enable_batch_invariant_mode`）。所测的 GPU 不在其中，所以它的 lm_head 仍然走
  cuBLAS，而 decoder 里的 linear 层用的是 vLLM 的 Triton GEMM。

打印引擎实际加载的模型的模块树，以及它选中的后端的类名。不要相信你传进去的配置。

## 2. 写下两端各自的舍入序列

对每一端，按执行顺序，每个舍入点写一行。每一行写明是哪个算子、每个输入和累加器的
dtype、每一次类型转换、哪些运算被融合在一起，以及已知的归约顺序。把
[references/rounding-points.md](references/rounding-points.md) 里的表当作检查清单。
每一条都要标明是“读自源码（文件、版本）”还是“实测”。拿不到源码的地方
（已编译的 CUDA kernel）要写明，并在第 5 步用算子重放来确定。

## 3. 先确认引擎自己和自己一致

在和 trainer 做任何比较之前，先在 prompt 集上单独测引擎：

- L1：同一个 batch 跑两次；
- L2：每个 prompt 单独跑一次，再和 batch size 为 4、16、64 时的结果比较；
- L3：把 decode 的结果和对同样的 token 做 prefill 的结果比较（第 5 步对注意力做
  算子重放时，会逐层完成这项比较）。

如果引擎达不到 L2，那么无论 trainer 怎么改，trainer 和引擎在成 batch 的 rollout 上
都不可能达到 L4。把这一点作为引擎内部的不一致报告出来，然后要么把引擎切到满足 L2
的模式，要么把结论限定在 batch size 为 1 的情况。实测（Qwen2.5-0.5B，
[报告](../../examples/cross_engine/REPORT.md)）：从 batch size 1 换到 64，
默认配置的 vLLM 改变了 96% 的实际 token logprob。batch-invariant eager 模式下改变了
7%，全部来自 lm_head，因为在所测的 GPU 上这个模式没有覆盖 lm_head，它仍然走 cuBLAS。
把 lm_head 换成 vLLM 自己的 batch-invariant GEMM 之后，就没有 logprob 再变了。

## 4. 一次只对齐一个舍入点

从 trainer 的现状出发。把一个舍入点改成引擎的做法，重新跑一遍 teacher forcing
的比较，记下有多少 token 不同，然后再改下一个。在 trainer 里，优先直接导入引擎
自己的 kernel，而不是重写一遍：源码相同并不保证结果逐 bit 一致，除非 tile 形状、
归约顺序和编译器也都一样。到达最终状态后，再做一轮 leave-one-out：每次只把一个点
改回去，重新计数。如果在这个 prompt 集上改回某个点之后什么都没变，也不能证明这个点
无关紧要；要在报告里写明这一点。

不同 token 的个数可能连续几次改动都纹丝不动，到最后一次才一下子降下来，因为只有
logits 上游的所有舍入点都一致之后，token 才可能相等。所以要看进展，应该看下面的
逐层定位，而不是 token 计数。

## 5. 定位剩下的差异

按顺序用两个工具：

- **算子重放。** 给引擎的每个模块挂上前向的 pre-hook 和 post-hook，并在原地计算的
  kernel 覆盖输入之前先把输入克隆下来。记录一次 prefill 和几步 decode。然后把采集到的
  每个输入分别送进 trainer 的每个候选实现，统计有多少输出逐 bit 一致。这样每个舍入点
  都能单独确定，不受其他舍入点影响。把 decode 步和 trainer 对整个序列做的 prefill
  比较，还能看出 L3 的影响。
- **逐层比较残差流。** 在采集到的 prompt 上运行 trainer，找出残差流或 MLP 输出第一次
  和引擎不同的那一层。

hook 只在 eager 模式下才会触发。如果引擎经过编译，或者跑的是录制好的 CUDA graph，
就改为比较算子边界上张量的逐向量 fingerprint（指纹）。OpGuard 把它们算成
`lastdim-u32le-v1` 向量；在纯 torch 里，同样的向量可以用
`tools/opguard_bridge.py xor_vectors` 得到。OpGuard 的 `rl-alignment` skill
（[OpGuard 仓库](https://github.com/ziming-zh/OpGuard/tree/main/harness/skills/rl-alignment)，`harness/skills/rl-alignment/SKILL.md`）
就是基于这些向量工作的：它把 logprob 的差距拆成 staleness 和数值误差两部分，
并把首个分歧点定位到具体的算子和 token。

## 6. 逐个决定怎样处理差异

| 选择 | 什么时候选 | 需要说明的代价 |
|---|---|---|
| 改 trainer | 引擎的算子是 batch-invariant 的，而且 trainer 能调用它（Triton kernel、自定义算子） | trainer 的吞吐。如果用于训练，要注意引擎的 kernel 可能没有反向实现，这时按第 7 节给它配一个反向 |
| 改引擎 | 引擎的算子不是 batch-invariant 的，或者 trainer 调用不了它（闭源 kernel、编译器生成的代码） | 引擎的吞吐，在 decode 常用的 batch size 下测量 |
| 接受并记录 | 两端用的是算法上不同的形式（编译器融合、不同的注意力算法），而且哪一端都改不了 | 测得的差距：不同 token 的比例、最大 \|Δlogp\| 和 KL |

把每个决定和测得的效果一起写在对应的舍入点旁边。

## 7. 给对齐后的前向配一个正确的反向

logprob 必须逐 bit 一致，梯度只需要正确，所以反向不必复现前向 kernel 的 bit。
小算子手写 fp32 导数。注意力和 MoE 在前向里保留引擎 kernel 算出的值，梯度则用保存下来的
输入在 PyTorch 里重算得到。

把每个参数的梯度和一个独立的 fp32 参考比较，每个张量的相对 L2 误差都不超过 1e-3，
检查才算通过。先在缩短版的模型上跑这项检查：保留几层，覆盖每种层类型。完整的例子见
[examples/verl_fsdp2_vllm/REPORT.md](../../examples/verl_fsdp2_vllm/REPORT.md)。

## 8. 报告

报告包含：

- 模型、GPU、两端的软件版本、引擎参数、prompt 集和 batch size；
- 每种配置下引擎的 L1/L2 结果；
- 两端的舍入序列；
- 步骤表和 leave-one-out 表；
- 算子重放表；
- 最终的 L4 结果，即比较了多少 token、有多少不同、最大 |Δlogp| 和 KL；
- 最终引擎配置和默认配置的吞吐对比；
- 梯度检查，包括检查了哪些张量、最大的相对 L2，以及在哪个模型上跑的；
- 已知限制，包括没有测过的 prompt 和长度，以及没有覆盖的模式（prefix caching、
  speculative decoding、TP）。

## 约束

- 引擎在相同的 prompt 和 batch 上达不到 L2 时，绝不声明 L4。
- 绝不比较不同 token 的 logprob。要用 teacher forcing 把引擎的 token 送进 trainer。
- 开关的名字说明不了它覆盖了什么。要读与你的 GPU compute capability 对应的代码路径，
  并测量这个开关到底改变了什么。
- 编译后的引擎就是另一个引擎。Inductor 会融合原生算子、去掉中间的舍入，
  所以在 eager 模式下测得的结果不能直接套用到编译模式。
- 源码相同不等于结果逐 bit 一致。用另一个框架或另一种 tile 形状重写 kernel，
  会改变归约顺序，所以要直接导入原来的 kernel。
- 一个舍入点在某个模型上一致，换一个模型未必一致。bias、tied embedding、
  恰好为 1 的 norm 权重或 2 的幂的缩放，都可能掩盖差异，所以要用真实的 checkpoint 来测。
- 引擎的 batch-invariant 模式会降低吞吐。要在 decode 常用的 batch size 下测出这项代价，
  并写在一致性结论旁边。
- 不改动生产环境的默认值。对齐模式和补丁都必须显式开启。
