---
name: batch-invariant-kernel
description: 没有现成实现时，编写、改造或验证一个 batch-invariant 的 GPU kernel。要做的是：固定归约树；去掉 split-K、split-KV、原子操作和随形状变化的分派；让 MoE combine 和量化 GEMM 的求和顺序固定下来；然后证明无论 batch size、行的位置、同一 batch 里其他行的内容、prefill 还是 decode、怎样分 chunk，结果都逐 bit 不变。当模型带来一个还没有 batch-invariant 实现的算子时使用，比如新的注意力变体（MLA、稀疏/DSA、线性/GDN）、MoE 的 dispatch 和 combine、FP8/FP4 GEMM，或者闭源 kernel。
---

# Batch-invariant kernel

一次只处理一个算子。起点是一个不知道是否 batch-invariant、或者已知会随 batch 变化的算子；
终点是它在声明的级别上验证过不变，或者有一个记录在案的回退方案。最后交付四样东西：
改动（配置、替换，或者对 kernel 的修改）、验证报告、吞吐代价，以及一条 handbook 条目。
直接选用现成的 batch-invariant 算子是下面的第 2 步；这个 skill 处理的是现成实现不够用的情况。

[references/operator-recipes.md](references/operator-recipes.md) 按算子类别给出了做法。
[references/verification.md](references/verification.md) 讲验证流程，以及怎样解读失败。
检查工具是 [tools/bi_check.py](../../tools/bi_check.py)。
可以拿来做起点的参考 kernel 在 [tools/bi_kernels.py](../../tools/bi_kernels.py) 里，
都在一种 GPU 上验证过：
- 只用一种配置的 GEMM；
- 固定分块的 causal attention，反向不用原子操作。

更多验证过的 kernel 随案例研究一起放在 `examples/` 下：
- paged attention 和 varlen attention；
- 递归形式和分块形式的 GDN；
- 吸收式的 MLA 核心计算；
- 用稳定 top-k 的稀疏注意力流水线；
- 把返回的行放进固定槽位再求和的 expert 并行 combine；
- 针对逻辑 TP 规模、输出 fp32 的 GEMM，以及按固定顺序求和的词表并行
  log_softmax；
- 每一行各自选择 adapter 的 LoRA linear；
- 以计数器为键的采样器（[tools/bi_sampling.py](../../tools/bi_sampling.py)）；
- 用于反向 scatter 的分段求和：先按索引排序，再按升序相加；
- 按逻辑 shard 计算的数据并行梯度。

[tools/bi_audit.py](../../tools/bi_audit.py) 会列出一个模块运行的所有算子，
并给出每个模块的 L2 判定。用它找出哪些算子需要这个 skill。
[证据表](references/operator-recipes.md#evidence-behind-each-recipe)
写明了哪些做法已经验证、哪些还没测过。没测过的做法只能当作假设，依赖它之前先验证。

## 先说清你交付的是哪一级不变性

| 级别 | 什么情况下成立 | 通常是什么破坏了它 |
|---|---|---|
| L1 run-to-run | 同样的输入，每次运行的结果都逐 bit 一致 | 原子操作、竞争、读到未初始化的内存 |
| L2 batch-invariant | 一行的结果不受 batch size、这一行的位置和其他行内容的影响 | 根据形状选 tile 或分派、split-K、在整个 batch 上算出的动态缩放 |
| L3 mode-invariant | prefill、decode、chunked prefill 和 paged 布局在每个位置上的结果都逐 bit 一致 | split-KV、各模式遍历 block 的方式不同或者用了不同的 kernel、代数上相等但算法不同的写法 |
| L4 cross-engine | trainer 和推理引擎的结果逐 bit 一致 | 舍入点、融合方式、kernel 或公式不同，即使两端各自都满足 L2/L3 |
| L5 parallel-invariant | 换 TP/EP/SP 规模后结果仍逐 bit 一致 | 拆到多个 shard 上的归约、集合通信的归约树、expert 输出的合并顺序 |

L1 不意味着 L2。在一种 GPU 上用 torch 2.5.1 测过：bf16 的 `torch.mm` 每次运行的结果
逐 bit 一致，但 batch 一到 31 行，某一行的结果就变了（4096 个元素里有 4 个不同）。
L2 和 L3 也不意味着 L4：两个引擎还必须在同样的地方做舍入（见 handbook 的 RL 部分）。

## 规则

产生每个输出元素的浮点归约，只能由这个元素自己的归约长度决定。这里的“决定”包括
运算的顺序、怎样分组、用哪条硬件指令，以及每一步的精度。这些都不能随下面任何一项变化：
行数或 token 数、这一行的位置、其他行的值、grid 大小、SM 数量或 occupancy、
调度器决定的 page 或 chunk 边界，以及时序。

## 1. 找出结果从哪里开始随 batch 变化

先读 kernel 的启动和分派代码，再运行 `bi_check.check_batch`；对类似注意力的算子，
还要运行 `check_modes`。重点找这些地方：

- 根据 M 选择 tile、warp 或 stage 配置，或者自动调优以形状为键；
- split-K、stream-K 或 persistent 调度，K 怎么拆取决于 tile 的数量；
- split-KV（flash-decoding），拆成几份由 batch size 或 SM 数量推出来；
- 按形状分派的快速路径：M=1 时走 GEMV、小 M 专用的 kernel、库的启发式规则；
- 按参数分派的后端：比如带显式 mask 的注意力调用，走的后端可能和
  `is_causal=True` 时不同；
- 浮点原子操作：`atomicAdd`，以及往共享的输出里做 `index_add_`/`scatter_add_`；
- 行数很少时，把一行拆到多个 block 上算；
- 整个 batch 上的统计量：FP8 的逐张量动态 amax、NVFP4 的动态全局缩放、MoE 的 capacity 上限、
  在整个 batch 上取 top-k；
- 集合通信的算法或协议随消息大小变化；
- 向量宽度取决于对齐，而一行起始地址的对齐又取决于它在 batch 中的位置；
- 按形状特化的编译器或 graph 层：`torch.compile` 的分桶，
  每个 padding 后的 batch size 各用一个 CUDA graph。

失败的规律会指向原因。[verification.md](references/verification.md#reading-failures)
里的表列出了每种规律最可能对应的机制。

## 2. 选最便宜、又能成立的做法

1. **调配置。** 用库提供的开关：只拆一份（`num_splits=1`）、关掉 split-K、固定算法、
   用确定性的反向、固定集合通信的算法和协议。改完重新验证。
2. **选用**现成的 batch-invariant 实现，比如 Thinking Machines 的
   `batch_invariant_ops`，或者 vLLM 用 `VLLM_BATCH_INVARIANT=1` 打开的那一组算子。
   在你自己的形状、dtype 和 GPU 上重新验证，因为各个算子的覆盖情况不一样。
3. **所有 M 都用同一种配置。** 把形状从自动调优的键里去掉，接受小 M 时利用率低一些。
4. **修改 kernel**，按它所属类别的做法来改。
5. **如果有两种代数上相等的写法，所有模式都用同一种算法**，比如线性注意力的
   分块形式和递归形式，MLA 的吸收式和显式。要写明代价，
   它可能很大：有报告说，GDN 前向只用递归形式时，trainer 吞吐低了 2–3 倍。
6. **kernel 闭源又没有开关时，只能退而求其次。** 替换它；或者保留一个逐行运行的慢速包装，
   只在调试时当作正确性的参照，并且不把这个算子算进任何对齐结论。

## 3. 运算方式要和你要对齐的那一方完全一样

让 kernel 变得 batch-invariant，不能顺带挪动舍入点。动手改之前，先把参考顺序写下来，
也就是你要对齐的那个引擎是怎么算的：
- 在哪里做类型转换；
- 哪些运算融合在一起，哪些单独算；
- 累加器用什么 dtype；
- epilogue 的顺序：bias、激活、残差、norm 权重。

除非两端一起改，否则保持这个顺序不变。一个 batch-invariant kernel 如果舍入方式和
另一个引擎不同，能通过 L2，照样过不了 L4。

## 4. 验证

跑完 [verification.md](references/verification.md) 里的整个矩阵，它包括：
- 各种 batch size，包括每个阈值两边的值；
- 行的位置；
- 同一 batch 里其他行的内容；
- 各种模式；
- 重复运行；
- 反向（训练用到时）；
- 部署时套在外面的那一层（`torch.compile`、CUDA graph 录制时用的尺寸）。

比较时把张量当作整数来看，逐 bit 比较。一个不匹配都没有才算通过，
其他结果都算失败，不设容差。

## 5. 投入使用

- 注册这个 kernel，让编译器没法再把它融合进去：用 `torch.library` 覆盖，
  或者做成不透明的自定义算子。实测：直接调用 Triton kernel 时，Dynamo 会对每个
  batch size 重新编译一次。交给 Inductor 处理的 norm 和激活仍然满足 L2，但和 eager
  相比有 65% 的元素变了。要和 eager 一致，就把每个会舍入的算子都做成不透明的。
- 用一个开关控制它，默认关闭，除非用户另有决定。
- 要做到 L4，就让两个引擎调用同一个实现，或者调用已经证明归约树相同的 kernel。
  怎样和真实的引擎对齐，见 [engine-parity skill](../engine-parity/SKILL.md)。
- 在有代表性的 prefill M 和 decode M（1–64）上测吞吐，
  把代价记在基线旁边。

## 6. 报告

交付一条 handbook 条目，内容包括：
- 算子和形状；
- 交付的级别；
- 结果随 batch 变化的来源；
- 采用的做法和改动；
- 验证矩阵和结果，以及 GPU 和版本；
- 吞吐代价；
- 已知的限制，比如没测过的大小、没覆盖的模式。

如果 kernel 是 agent 写的或改的，记下迭代了几轮、需要人工改哪些地方。
公开的报告很少写这些。

## 防护规则

- 不要只凭几个大小就说它不变。列出测了哪些大小，并包括你在代码里找到的
  每个阈值附近的大小。
- 不要对输出做舍入或后处理来凑成相等，也不要用容差。
- 确定性模式（`torch.use_deterministic_algorithms`）只能给出 L1，给不了 L2，
  而且只有 `warn_only=False` 时才成立。`warn_only=True` 时，SDPA 的反向只发警告，
  结果仍然不确定（在一种 GPU 上测得）。确定性也不代表顺序是明确规定好的：严格模式下的
  scatter 和 embedding 反向，顺序会随行宽或索引数量变化。要在不同框架之间一致，
  就自己规定顺序。
- 测试时求和至少要有三项：MoE 的 top-k ≥ 3，rank 数在三个以上。两项相加满足交换律，
  会把顺序上的 bug 藏起来。实测：原子操作的 combine 和每个 rank 先做局部归约的做法，
  在 top-2 时都通过，到 top-3 都失败；交给 NCCL 做的 DP 梯度归约，从 3 个 rank 起失败，
  即使用 Tree 也一样。
- 用不是 2 的幂的缩放和系数来测。2 的幂的缩放在哪里乘都是精确的，
  会掩盖舍入点的差异。
- 改变被测的布局：page 大小、pack 偏移、消息大小。如果 kernel 的 block 跟着 page
  或 pack 走，那么在任何一种设置下它都和自己一致，换一种设置结果就变了。
- 固定集合通信的算法，并不等于不变。要在许多不同大小的消息里检查同一个固定的切片。
  实测：在每种协议下固定 Ring，仍然失败。
- 确定性不等于并行不变。如果要比较的两次运行可能用不同的并行规模，就单独检查 L5。
  DP 要做到不变，需要固定 micro-batch 的数量，并用一棵固定的归约树，跨 rank 的求和
  也要包含在这棵树里。前向里修好了 L5，反向不会跟着好，因为每个列并行层的输入梯度
  都是跨 rank 的和。
- 除了稳定运行时的状态，还要检查第一次迭代和恢复训练之后。torch DDP 在第一次迭代
  （以及每次重启后）用单个桶做归约，在 Ring 下这会让梯度变掉。
- 固定优化器的实现（fused、foreach 或 single-tensor）和 loss 的归约方式：
  每一种都是不同的运算，而且 mean 不是布局不变的。
- 随机数流也是算子的一部分。除了 batch size，还要检查重算、回滚、其他请求的 n、
  speculative decoding 的不同分块大小和换新进程之后能不能重放出来。`torch.cumsum` 不是
  batch-invariant 的，对长行来说甚至不是确定性的：改用整数求和。
- 在每种推理模式下检查整个模型，并把它的舍入顺序和 trainer 的对照。
  只有一个舍入点不同的 L2/L3 kernel，就让 86–100% 的 logits 变了。
- 引擎的 batch-invariant 开关并不在每种 GPU 上覆盖每个算子。
  实测：在某些 GPU 架构上，vLLM 0.19.1 的这个模式把 lm_head 留在 cuBLAS 上，
  导致 7% 的 logprob 不满足 L2。要在你的 GPU 上查清它覆盖了哪些算子。
- 在 `torch.compile` 里插 probe，可能会改变被探测的东西。给 probe 一份逐元素的精确副本
  （`tools/graph_probe.py`，`isolate=True`），并分别带 probe 和不带 probe 编译来确认。
  惰性缓存（RoPE 表）要在编译前用 eager 建好：如果在 graph 里建，它们会按 Inductor 的
  算法算出来，之后的 eager 运行也会沿用。
- 算法上的差异（同一个数学的两种写法）是一种选择，不是 bug。把它当作选择来报告，
  并写明代价。
- 不改生产环境的默认值。对齐模式需要显式开启。
