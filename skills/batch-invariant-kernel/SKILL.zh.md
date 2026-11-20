---
name: batch-invariant-kernel
description: 在没有现成实现时，编写、改造或验证一个 batch-invariant（批不变）的 GPU kernel。固定归约树；去掉 split-K、split-KV、原子操作和依赖形状的分派；让 MoE combine 和量化 GEMM 的顺序稳定；然后证明它在批大小、行位置、同批内容、prefill/decode 和分块上逐位不变。当模型带来一个没有 batch-invariant 实现的算子时使用：新的注意力变体（MLA、稀疏/DSA、线性/GDN）、MoE dispatch 和 combine、FP8/FP4 GEMM，或闭源 kernel。
---

# Batch-invariant kernel

一次只负责一个算子。把它从“未知或有变化”带到“在声明的级别上已验证不变”，
或者带到一个有文档的回退方案。交付改动（配置、替换或 kernel 修改）、验证报告、
吞吐代价，以及一条手册条目。选用现成的 batch-invariant 算子是下面的第 2 步。
本 skill 用于这还不够的情况。

各类算子的做法：[references/operator-recipes.md](references/operator-recipes.md)。
验证流程以及如何解读失败：
[references/verification.md](references/verification.md)。
检查工具：[tools/bi_check.py](../../tools/bi_check.py)。
可以作为起点的参考 kernel：[tools/bi_kernels.py](../../tools/bi_kernels.py)，
在一种 GPU 型号上验证过：
- 单一配置的 GEMM；
- 固定分块的因果注意力，带不用原子操作的反向。

更多已验证的 kernel 随案例研究一起提供，在 `examples/` 下：
- 分页和变长注意力；
- 循环形式和分块形式的 GDN；
- 吸收形式的 MLA 核心；
- 稳定的 top-k 稀疏流水线；
- 固定槽位的专家并行 combine；
- 针对逻辑 TP 大小的 fp32 输出 GEMM，以及固定顺序的词表并行
  log_softmax；
- 按行选择适配器的 LoRA linear；
- 以计数器为键的采样器（[tools/bi_sampling.py](../../tools/bi_sampling.py)）；
- 用于反向 scatter 的排序、升序分段求和；
- 逻辑分片的数据并行梯度。

[tools/bi_audit.py](../../tools/bi_audit.py) 列出一个模块运行的每个算子，
并给出每个模块的 L2 判定：用它找出哪些算子需要本 skill。
[证据表](references/operator-recipes.md#evidence-behind-each-recipe)
说明哪些做法已验证、哪些未测试。把未测试的做法当作假设，在依赖之前先验证。

## 级别：说明你交付的是哪一级

| 级别 | 成立条件 | 常见破坏因素 |
|---|---|---|
| L1 run-to-run | 相同输入，每次运行得到相同的比特 | 原子操作、竞争、读取未初始化内存 |
| L2 batch-invariant | 一行的结果不取决于批大小、它的位置或其他行的内容 | 依赖形状的 tile 或分派、split-K、整批动态缩放 |
| L3 mode-invariant | prefill、decode、分块 prefill 和分页布局在每个位置给出相同的比特 | split-KV、不同的块迭代、每种模式用不同的 kernel、代数上相等但形式不同 |
| L4 cross-engine | trainer 和推理引擎给出相同的比特 | 不同的舍入点、融合、kernel 或公式，即使每一侧都是 L2/L3 |
| L5 parallel-invariant | 不同的 TP/EP/SP 度给出相同的比特 | 分片的归约、集合通信树、专家 combine 顺序 |

L1 不意味着 L2。在一种 GPU 型号上用 torch 2.5.1 测得：bf16 的 `torch.mm`
每次运行逐位相同，但当批达到 31 行时，某一行的结果变了（4096 个元素中有 4 个不同）。
L2 和 L3 不意味着 L4。两个引擎还必须在相同的位置舍入（见手册的强化学习部分）。

## 规则

每个输出元素的浮点归约必须是该元素自身归约长度的固定函数。这包括运算顺序、
分组方式、硬件指令以及每一步的精度。它不能取决于行数或 token 数、该行的位置、
其他行的值、grid 大小、SM 数量或占用率、由调度器决定的页或块边界，或时序。

## 1. 找到变化来源

阅读 kernel 的启动和分派代码，然后运行 `bi_check.check_batch`；对类似注意力的算子，
再运行 `check_modes`。查找：

- 根据 M 选择的 tile、warp 或 stage 配置，或以形状为键的自动调优；
- split-K、stream-K 或持久化调度，其 K 拆分取决于 tile 数量；
- split-KV（flash-decoding），其拆分数由批大小或 SM 数量推出；
- 按形状分派的快速路径：M=1 时的 GEMV、小 M 的 kernel、库的启发式规则；
- 按参数分派的后端，例如带显式 mask 的注意力调用走的后端与
  `is_causal=True` 不同；
- 浮点原子操作：`atomicAdd`，以及向共享输出做 `index_add_`/`scatter_add_`；
- 行很少时，一行被拆到多个 block 上；
- 整批的统计量：FP8 的逐张量动态 amax、NVFP4 的动态全局缩放、MoE 容量上限、
  整批上的 top-k；
- 算法或协议取决于消息大小的集合通信；
- 依赖对齐的向量宽度，当一行的起始对齐取决于它在批中的位置时；
- 按形状特化的编译器或图层：`torch.compile` 分桶，
  每个填充后的批大小一个 CUDA graph。

失败的模式指向原因。[verification.md](references/verification.md#reading-failures)
中的表把模式对应到可能的机制。

## 2. 选择能成立的最便宜策略

1. **配置。** 使用库暴露的开关：单一拆分（`num_splits=1`）、
   关闭 split-K、固定算法、确定性反向、固定的集合通信算法和协议。然后重新验证。
2. **选用**现有的 batch-invariant 实现，例如 Thinking Machines 的
   `batch_invariant_ops` 或 vLLM 的 `VLLM_BATCH_INVARIANT=1` 集合。在你自己的形状、
   dtype 和 GPU 上重新验证：不同算子的覆盖情况不同。
3. **为每个 M 固定一种配置。** 从自动调优中去掉形状键，接受小 M 时较低的利用率。
4. **修改 kernel**，按其类别的做法进行。
5. **当存在两种代数上相等的形式时，为每种模式选择同一种算法**：
   分块与循环形式的线性注意力，吸收与显式形式的 MLA。说明代价：它可能很大
   （据报告，只用循环形式的 GDN 前向使 trainer 吞吐差 2–3x）。
6. **回退**，当 kernel 是闭源且没有开关时。替换它，或者保留一个逐行运行的慢速包装，
   只作为调试用的正确性参照，并把该算子排除在对齐声明之外。

## 3. 保持与要匹配的一方完全相同的运算

批不变性不能移动舍入点。修改之前，写下你要匹配的引擎的参考序列：
- 类型转换发生在哪里；
- 哪些被融合，哪些是单独的算子；
- 累加器的 dtype；
- epilogue 的顺序：bias、激活、残差、norm 权重。

除非两侧一起改，否则保持不变。一个舍入方式与另一个引擎不同的 batch-invariant kernel
能通过 L2，但仍然在 L4 上失败。

## 4. 验证

运行 [verification.md](references/verification.md) 中的完整矩阵：
- 批大小，包括阈值附近的值；
- 行位置；
- 填充内容；
- 模式；
- 重复运行；
- 反向，当训练用到它时；
- 部署时用的包装（`torch.compile`、CUDA graph 捕获大小）。

比较通过整数视图逐位进行。零不匹配为通过；其他任何结果都是失败，没有容差。

## 5. 集成

- 注册它，让编译器无法再次融合它：用 `torch.library` 覆盖或不透明的自定义算子。
  测得：直接调用时，Triton kernel 让 Dynamo 对每个批大小重新编译。交给 inductor 的
  norm 和激活保持了 L2，但与 eager 相比改变了 65% 的元素。要与 eager 一致，
  就把每个会舍入的算子都设为不透明。
- 放在一个开关后面，默认关闭，除非用户另有决定。
- 对于 L4，让两个引擎调用同一个实现，或调用已证明共享归约树的 kernel。
  与真实引擎对齐见 [engine-parity skill](../engine-parity/SKILL.md)。
- 在有代表性的 prefill M 和 decode M（1–64）上测量吞吐，
  并把代价记录在基线旁边。

## 6. 报告

交付一条手册条目，包括：
- 算子和形状；
- 交付的级别；
- 变化来源；
- 策略和改动；
- 验证矩阵和结果，以及 GPU 和版本；
- 吞吐代价；
- 已知限制，例如未测试的大小或未覆盖的模式。

如果 kernel 是由 agent 编写或修改的，记录迭代次数和需要的人工修改。
公开报告很少说明这一点。

## 防护规则

- 永远不要从少数几个大小得出不变的结论。列出测试了什么，并包括你在代码中
  找到的每个阈值附近的大小。
- 永远不要对输出做舍入或后处理来强行相等，也永远不要使用容差。
- 确定性模式（`torch.use_deterministic_algorithms`）给出 L1，而不是 L2，
  并且只在 `warn_only=False` 时如此。在 `warn_only=True` 下，SDPA 的反向只发出警告，
  仍然是非确定性的（在一种 GPU 型号上测得）。确定性也不等于写明的顺序：严格模式下的 scatter
  和 embedding 反向会随行宽或索引数量改变顺序。要跨框架一致，就自己定义顺序。
- 用至少三项的和来测试：MoE top-k ≥ 3，三个或更多 rank。两项的和满足交换律，
  所以会掩盖顺序错误。测得：原子操作的 combine 和逐 rank 预归约在 top-2 时都通过，
  在 top-3 时都失败；交给 NCCL 的 DP 梯度从 3 个 rank 起失败，即使在 Tree 下也是如此。
- 用不是 2 的幂的缩放和系数来测试：2 的幂的缩放在任何位置应用都是精确的，
  会掩盖舍入点的差异。
- 改变被测的布局：页大小、打包偏移、消息大小。块跟随页或打包的 kernel
  在任一设置下都与自身一致，但在不同设置之间会变化。
- 固定集合通信的算法不等于不变性。在许多大小的消息中检查一个固定的切片。
  测得：在每种协议下固定 Ring 仍然失败。
- 确定性不等于并行不变。当要比较的运行之间并行度可能不同时，单独检查 L5。
  DP 不变性需要固定的 micro-batch 数量和包含跨 rank 求和的固定树。
  前向中的 L5 修复不能修复反向：每个列并行层的输入梯度都是跨 rank 的和。
- 检查第一次迭代和一次恢复训练，而不只是稳定状态：torch DDP 在第一次迭代
  （以及每次重启）时用单个 bucket 归约，在 Ring 下这会改变梯度的比特。
- 固定优化器实现（fused、foreach 或 single-tensor）和损失归约方式：
  每一种都是不同的运算，而且 mean 不是布局不变的。
- 随机流是算子的一部分。检查重算、回滚、其他请求的 n、投机分块大小和新进程之后的重放，
  而不只是批大小。`torch.cumsum` 既不是 batch-invariant 的，对长行来说也不是确定性的：
  用整数求和。
- 在每种服务模式下检查整个模型，并把它的舍入序列与 trainer 的比较。
  只有一个舍入点不同的 L2/L3 kernel 改变了 86–100% 的 logits。
- 引擎的 batch-invariant 开关并不覆盖每个 GPU 上的每个算子。
  测得：在某些 GPU 架构上，vLLM 0.19.1 的这个模式把 lm_head 留在 cuBLAS 上，
  这在 7% 的 logprob 上破坏了它的 L2。检查它在你的 GPU 上覆盖哪些算子。
- 在 `torch.compile` 内部插探针可能改变你要探测的东西。给探针一个精确的逐元素副本
  （`tools/graph_probe.py`，`isolate=True`），并通过带探针和不带探针分别编译来确认。
  在编译前用 eager 构建惰性缓存（RoPE 表）：如果在图内构建，它们来自 Inductor 的计算，
  并会保留给之后的 eager 运行使用。
- 算法上的差异（同一数学的两种形式）是一种选择，而不是 bug。把它作为选择连同代价一起报告。
- 不改动生产默认值。对齐模式需要显式开启。
