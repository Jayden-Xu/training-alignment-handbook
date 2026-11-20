# 各类算子的做法

每条做法都讲清两件事：是什么让一个算子的结果随 batch 变化（batch-variant），以及怎么修。
一条做法能达到哪一级，用的是 [SKILL.md](../SKILL.md#levels-say-which-one-you-deliver)
里定义的级别。每条做法最后都要走同一套验证（[verification.md](verification.md)），
因为光读代码永远证明不了修复真的有效。

## 每条做法背后的证据

单 GPU 的案例都在同一种 GPU 型号上测量。多 GPU 的案例（集合通信、EP、DP 梯度、
TP 下的整个模型）换了另一种 GPU 型号，在一到两个节点上运行，每个节点 8 块 GPU。
节点内的 GPU 通过 PCIe 相连，节点之间走 25 GbE 的 socket，没有 RDMA。除非某一行
另有说明，所有测量都用 torch 2.5.1 和 Triton 3.1（跨引擎那一行用的是 vLLM 自带的
软件栈：torch 2.10.0、Triton 3.6.0）。每个案例的硬件和原始结果都在它自己的
`examples/<case>/REPORT.md` 里。案例研究包括：
- [MoE](../../../examples/moe_combine/REPORT.md#one-gpu)；
- [EP（expert 并行）下的 MoE](../../../examples/moe_ep/REPORT.md#under-expert-parallelism)；
- [注意力、FP8 和线性注意力](../../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)；
- [注意力的反向](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv)；
- [paged KV、varlen 打包、torch.compile 和 CUDA graph](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)；
- [MLA 和稀疏注意力](../../../handbook/3-kernels/mla-and-sparse.md)；
- [线性注意力（GDN）](../../../handbook/3-kernels/linear-attention.md)；
- [集合通信和 TP](../../../handbook/3-kernels/collectives.md)；
- [整个模型](../../../handbook/2-rl/whole-model.md)和 [TP 下的整个模型](../../../handbook/2-rl/parallelism.md)；
- [DP 和 FSDP 式分片下的梯度（模拟实现：扁平梯度、reduce-scatter、分片的 AdamW）](../../../handbook/1-pretraining/data-parallel.md)；
- [采样](../../../handbook/2-rl/sampling.md)；
- [LoRA 和混有多个 adapter 的 batch](../../../handbook/2-rl/lora.md)；
- [训练 step](../../../handbook/1-pretraining/training-step.md)；
- [trainer 与 vLLM 的对比](../../../handbook/2-rl/engine-parity-vllm.md)。

| # | 类别 | 状态 | 证据 | 还没覆盖的 |
|---|---|---|---|---|
| 1 | 按行归约 | 在原生 torch 中观察到 | 用 torch 算子拼出的 RMSNorm 和 fp32 的 log_softmax，在 hidden 4096、vocab 32000、batch 1–2048 下都满足 L2。torch 的 mean 和 sum 在 fp32 下会随行数换一棵归约树（hidden 1024 时从 8 行起，4096 时从 2 行起），但 bf16 RMSNorm 的输出并没有变。固定归约树的 Triton 行求和在 1–2048 行、宽度 8–4096 下满足 L2（整模型案例）。 | 行数少时把一行拆给多个 block 的情况还没有复现出来。LayerNorm 和 cross-entropy 也还没有覆盖。 |
| 2 | 稠密 GEMM | **已实现并验证** | 单一配置的 Triton GEMM 在 3 种形状上、M=1–2048 时都满足 L2。cuBLAS 在这 3 种形状上都会随 batch 变化。 | 调优、大形状、融合的 epilogue、固定 cuBLASLt 的算法，以及 Hopper。 |
| 3 | grouped GEMM，`bmm` 与 `mm` | **已实现（用循环）并验证** | 每个 expert 用一个单一配置的 GEMM，在两种 GPU 型号上都满足 L2（[单 GPU](../../../examples/moe_combine/REPORT.md)、[8 GPU](../../../examples/moe_ep/REPORT.md)）。cuBLAS 的 expert GEMM 在两种型号上都会随 batch 变化，在固定大小的行块上调用 `bmm` 也一样。 | 一次 launch 就完成的 grouped kernel。 |
| 4 | softmax 注意力 | **已实现并验证**（前向、反向、paged、varlen） | 固定 block 的 Triton kernel 在 prefill、decode 和各个 chunk 之间满足 L2 和 L3，而 SDPA 既不满足 L2（batch ≥ 3），也不满足 L3。不用原子操作的反向满足 L1 和 L2，右 padding 的情况也包括在内，耗时是 SDPA flash 的 1.38x。SDPA flash 不满足 L1（问题出在 dQ），在严格确定性模式下不满足 L2。paged KV（页大小 16–128）和 varlen 打包（前向和反向）在 block 按每个序列的绝对位置划分时，结果逐 bit 一致；如果 block 跟着页或打包偏移走，就做不到。 | 固定切分大小的 split-KV、滑动窗口，以及 paged KV 的反向。 |
| 5 | MLA 的吸收式与显式两种形式 | **已实现并验证** | 两种形式精度相当，却有 65% 的元素不同。只用一种形式、并且每个部件都不变时，满足 L2 和 L3。显式 prefill 搭配吸收式 decode，从第 2 层起就和 trainer 对不上了。光有不变的注意力核心还不够，解压 GEMM 和输出的 `bmm` 也必须不变。 | FlashMLA、吸收式的 split-KV、反向，以及对引擎的 L4。 |
| 6 | 稀疏注意力（top-k） | **已实现并验证** | 每一次选择翻转（约四分之一的位置）都来自 cuBLAS 的投影 GEMM。投影换成单一配置、再用稳定的 top-k 之后，选中的集合和输出都满足 L2 和 L3。`torch.topk` 遇到并列时在每次测试里都取了最小的索引，但文档并没有写明这个顺序。 | 完整的 DSA 规模（top-2048）、FP8 indexer，以及反向。 |
| 7 | 线性注意力 / SSM | **已实现并验证**（GDN 前向） | 递归 kernel 满足 L1–L3。分块 kernel 满足 L1/L2，只有 chunk 边界落在 C 的绝对倍数上时才满足 L3。用 bf16 做矩阵乘时，分块形式和递归形式有 55.5% 的输出不同；用 fp32 时只有 0.03%。递归形式的耗时在 T ≤ 4096 时是分块形式的 0.9–1.5x，在 8192 时是 2.5–3.6x。 | 反向、生产环境用的分块 kernel（fla），以及 varlen。 |
| 8 | MoE 的路由和 combine | **已实现并验证**（单 GPU 上，以及两个 8 GPU 节点上的 EP 1–16；[报告](../../../examples/moe_ep/REPORT.md)） | 固定顺序的 fp32 combine 加上单一配置的 GEMM，满足 L1 和 L2。在 EP 下，固定槽位的 combine 在 EP 1、2、4、8 和 16 时结果逐 bit 一致，不管放在一个节点上还是拆到两个节点上（L5）。先在每个 rank 上做归约满足 L1/L2，但从 top-3 起不满足 L5。用原子操作做 combine 会破坏 L1（EP 案例中从 top-3 起）。按 capacity 上限丢 token 不满足 L2 和 L5。 | EP 下的反向、shared expert、DeepEP 风格的 kernel、NVLink、RDMA、两个以上的节点，以及对 trainer 的 L4。 |
| 9 | 量化 GEMM | scale 已验证（模拟） | 动态的 per-tensor scale 不满足 L2；per-token scale 和 1x128 的 block scale 能通过。 | 真实 FP8 GEMM 的 promotion interval（需要有 FP8 tensor core 的 GPU），以及 NVFP4（需要有 FP4 tensor core 的 GPU）。 |
| 10 | 集合通信 | **已测量**（一个和两个 8 GPU 节点；[报告](../../../examples/collectives_tp/REPORT.md)） | all-reduce 用默认设置不满足 L2，固定 Ring 并试遍所有协议也不满足。`NCCL_ALGO=Tree` 在一个节点上能通过（代价 1.0–1.3x，1 MB 时 5.3x），在两个节点上也能通过（16 个 rank：每个节点先沿自己的链求和，两个节点的和再相加一次）。reduce-scatter 的顺序取决于哪个 rank 拥有这一行，所以开了序列并行的 TP 不满足。all-to-all 之后在 fp32 下按 rank 顺序求和，在一个节点上能通过，代价 1.04–1.20x。跨节点时，两级的固定顺序（1.06–1.8x）可以替代扁平的回退方案（2.1–8.8x）。逻辑 TP 规模能给出 L5：取 8 时在 TP 1–8 之间一致，取 16 时在两个节点上的 TP 1–16 之间一致。在整个模型里，逻辑 TP（C = 16）加上固定顺序的词表 log_softmax，在 TP 1–16 下都给出同一个结果，prefill 和 decode 用不同的 TP 时也一样。DP 梯度方面，在 Ring 下桶的布局会让梯度变掉（换成 Tree，或者用一个有序的 all-reduce，代价 1.05–1.20x，就能修好）。逻辑 shard（S=16，固定的树）在 DP 1–16 下给出完全相同的梯度、范数和 10 步之后的 AdamW 权重，一个节点和两个节点、复制和分片、弹性恢复训练前后都一样。 | 三个或更多节点、NVLink/NVLS、RDMA、torch.distributed.fsdp 本身，以及激活梯度之外的 TP 反向。 |
| 11 | 闭源 kernel | 已测量 | 已经摸清 cuBLAS 和 SDPA 在哪些情况下会变。在固定大小的块上循环调用 cuBLAS 可以当作参照（oracle），慢约 33x。严格确定性模式下，SDPA 的 memory-efficient 反向在测过的形状上通过了 L1 和 L2，耗时是 flash 的 4.4x。 | cuDNN 和 FlashAttention 的二进制包。 |
| 12 | 部署时的包装 | **已测量** | 编译后的 block 仍然满足 L2，但和 eager 相比有 65% 的元素不同，直到把每个会舍入的算子都做成不透明的自定义算子为止。直接调用 Triton kernel 时，每换一个 batch size 就重新编译一次（18 个 graph；注册成自定义算子后只有 2 个）。CUDA graph 的分桶不会破坏本来就不变的 kernel，而 cuBLAS 在 7 个桶上给出了 6 种不同的结果。在编译后的代码和 graph 里插的 probe（`tools/graph_probe.py`）只要隔离好，就不会改变编译后的结果。 | 编译 paged、varlen 和反向路径，以及 `max-autotune`。 |
| 13 | 采样 | **已实现并验证** | 每个请求各用一个生成器的做法满足 L2，但遇到重算、回滚、n>1 和 speculative decoding 的分块都会失败。cumsum 不满足 L2（V=151,936 时连 L1 也不满足）。用计数器作 key、权重用 int64 的采样器通过了 L1–L3 和卡方检验，耗时是原生采样器的 1.3x（B=256）。 | 融合的噪声 kernel、min-p、torch.compile 和 CUDA graph，以及真实的引擎。 |
| 14 | 混用多个 adapter 的 LoRA | **已实现并验证**（前向、反向） | grouped cuBLAS 在 723 种同 batch 组合里有 486–506 种不满足 L2，即使 base GEMM 不变，也还有 22–100/723 种不满足。先 gather 再做 bmm 同样不满足。在 bf16 下 merge 会改变约 50% 的 bit，而且当增量只有输出的 1e-2–1e-3 时，会丢掉 18–78% 的 ΔW。改变 scale 乘上去的位置，会让 5–6% 的输出变化。我们的 kernel 通过了 L1、L2、prefill 对 decode、trainer 对引擎，以及带 padding 的反向。 | vLLM 本身、打包的切片、TP、torch.compile，以及 QLoRA。 |
| 15 | 训练 step（反向算子、优化器、loss、恢复训练） | **已实现并验证**（单 GPU） | index_add_、scatter_add_、index_select 的反向、带权的 bincount 和 1-D cumsum 都不满足 L1。先按索引排序、再在 fp32 下按升序做分段求和，满足 L1，并且和 NumPy 循环逐 bit 一致。严格模式遇到 cumsum、bincount(weights) 和 histc 会直接报错。AdamW 的 fused、foreach 和 single-tensor 三条路径结果互不相同（foreach 和 single-tensor 只差在偏差修正的那一次除法上）。保存并恢复 7 项状态，恢复训练就能逐 bit 一致。 | 让 fused 优化器的顺序完全对上（目前 0.15% 的 exp_avg 差 1 ulp）、多 GPU（见第 10 行），以及 torch.compile。 |
| 16 | 跨引擎舍入（L4） | **已测量**（vLLM 0.19.1，[报告](../../../examples/cross_engine/REPORT.md)） | 投影的打包方式、bias 的位置、norm 的写法、RoPE 表、注意力 kernel、lm_head kernel 和 log_softmax kernel，任何一项不同都会破坏相等。RoPE 公式和 SiLU 本来就一致。trainer 调用 vLLM 的 kernel 之后，不同的 token 数是 0/76,155。 | 编译后的 vLLM、反向、TP，以及其他模型。 |

## 1. 按行归约（RMSNorm、LayerNorm、softmax、log_softmax、行求和、cross-entropy）

按行归约的 kernel 在下面几种情况下会随 batch 变化：
- 行数少时，为了把 GPU 填满，把一行拆给多个 block 去算；
- 根据行数或指针对齐来选 block 大小或向量宽度；
- 对小 batch 走一条单独的快速路径。

修复方法：
- 不管 batch 多大，每一行都由它自己的 program（一个 block 或一个 warp）处理。
- 固定沿行方向的 block 大小和每个线程的累加方式，然后在 block 内用一棵固定的树把各线程的
  结果合起来。
- 如果一行太长、一个 block 放不下，就把它切成若干 chunk，chunk 的个数只取决于行长；
  第二遍再按固定顺序把这些 chunk 合起来。任何时候都不要用原子操作。
- 固定向量宽度。如果一行起始地址的对齐会随它在 batch 中的位置变化（hidden size 不是
  向量宽度的倍数时就会这样），就全部走标量路径，或者把行的 stride padding 到对齐。
- 保留参考实现的舍入点，比如归一化后的值是在乘权重之前还是之后转成 bf16。

这条做法能达到 L2。在一种 GPU 型号、torch 2.5.1、hidden 4096、vocab 32000 的条件下，
用 torch 算子拼出的 RMSNorm 和 fp32 的 `log_softmax` 在 batch size 1–2048 之间本来就
满足 L2。但换了形状、版本或 GPU 就没有这个保证，所以要在你自己的环境上验证。

## 2. 稠密 GEMM（bf16/fp16）

稠密 GEMM 在下面几种情况下会随 batch 变化：
- 按 M 来选 tile、指令或 kernel；
- 用了 split-K 或 stream-K；
- 由库的启发式规则挑选 kernel。

在一种 GPU 型号、torch 2.5.1 下，`torch.mm`（cuBLAS）在 M=16 及以下满足 L2，
从 M=31 起就开始随 batch 变化。

修复方法：
- 对所有 M 都用同一套 tile 配置（block M/N/K、warp 数、stage 数、MMA 指令）。
- 不用 split-K，也不用 stream-K。K 循环按升序累加到 fp32 累加器里，所有 M 都用
  同一个 epilogue。
- 可以另配一个专门处理小 M 的 kernel，但前提是它对每个元素的 K 切分方式和所用指令都和
  主 kernel 一样，并且要逐 bit 证明它和主 kernel 的结果一致。
- cuBLASLt 可以固定算法和它的 split-K 设置。即便如此也要在不同的 M 上验证，因为固定了
  算法并不等于有保证。
- 关掉 TF32，除非两端用 TF32 的方式完全一样。

这条做法能达到 L2。要达到 L4，两个引擎还必须跑同一个 kernel。代价是小 M 时（也就是
decode 时）GPU 利用率较低。

## 3. batched GEMM 和 grouped GEMM（MoE expert，`bmm` 与 `mm`）

- 一个 expert 的 GEMM 看到的 M 是 M_e，也就是路由到这个 expert 的 token 数，而 M_e
  会随 batch 里其他内容变化。所以每个 expert GEMM 都必须对 M_e 不变（见第 2 节）。
  grouped kernel 常常按最大的 M 或总的 M 来定 tile 大小，要把 tile 大小固定下来。
- 如果一端对 3-D 张量调用 `bmm`，另一端对 2-D 张量调用 `mm`，两端跑的就是不同的
  kernel。有一个公开报告的例子出在 MoE router 上：gate 分数稍有不同，同一个 token 就
  可能被路由到不同的 expert，这是离散的分歧。要让两端都走同一个不变的 GEMM。

## 4. softmax 注意力（prefill、decode、chunked prefill、paged KV）

在一种 GPU 型号、torch 2.5.1 下，SDPA 有两处结果会变：
- prefill 和逐 token 的 decode 相比，约 10% 的输出元素不同；
- 带显式 mask 的 chunked prefill 有 1 个元素不同。带 mask 的调用很可能走了和
  `is_causal=True` 不同的后端。

修复的目标是让位置 t 上的 query 在每种模式下都按同样的方式计算：
- **所有模式按同一种方式遍历 key。** 以固定大小 B 的 block 访问 KV，从位置 0 开始按
  绝对位置的顺序，逐个 block 做 online softmax 更新。prefill、chunked prefill 和
  decode 用同一个 B、同一个顺序。
- **不用 split-KV**（`num_splits=1`）。如果一定要切，就按固定的 token 数来切，并对齐到
  绝对位置，绝不按切分个数来切；而且 prefill 和 decode 要用同样的切分与合并方式。
- **paged KV：** block 边界跟着绝对位置走（B 的倍数），与页大小无关，也与 prefill 的
  chunk 从哪里开始无关。在 kernel 运行之前，先把当前 chunk 的 K/V 写进 cache，这样
  kernel 只从一个统一的来源读数据。
- **mask：** 被 mask 掉的分数，贡献必须恰好为零。部分被 mask 的 block，在每种模式下
  都用同样的方式计算 running max 和归一化因子。
- 每种模式都用同一条指数计算路径（例如先乘 log2(e) 再用 `exp2`）、同一个 scale 和
  同一种 GQA head 映射。
- 用 varlen 打包（`cu_seqlens`）时，一个序列的遍历方式不能依赖于和它打包在一起的
  其他序列。
- **反向：** 常见的 kernel 用原子操作把各个 key block 的 program 算出的 dQ 贡献加起来，
  这会破坏 L1。改成两遍计算，都不用原子操作：
  - dK/dV：每个 key block 一个 program，先遍历它所在 GQA 组的各个 query head，再遍历
    各个 query block，顺序固定；
  - dQ：每个 query block 一个 program，从位置 0 开始遍历各个 key block。

  两遍都在 fp32 寄存器里累加，只写一次。GQA 组内的求和在 program 内部用 fp32 完成。
  前向的 log-sum-exp 由前向的一个变体顺带存下来，并逐 bit 检查这个变体的输出和普通
  前向一致。在[注意力反向案例](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv)
  中测得：
  - L1 和 L2 都成立，用带 mask 的 loss 做右填充时也成立；
  - 前向加反向的耗时是 SDPA flash 的 1.38x。

  厂商提供的确定性反向需要 `warn_only=False`，而且它的不变性不会比对应的前向更好：
  flash 在这种模式下仍然不满足 L2。

这条做法能达到 L2 和 L3。代价是没有了 split-KV，序列很少时的长上下文 decode 会变慢。

在一种 GPU 型号上测得（[paged、varlen 和 compile 案例](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)）：
- **paged KV。** 页大小取 16–128、页随机摆放时，每种模式都逐 bit 复现了连续存储时的
  结果。如果改成按页大小来划分 block，每种页大小都会得到一个不同的结果。这些结果各自在
  不同模式之间又是一致的，所以只测一种页大小发现不了问题，要多测几种。页比 block
  小时需要一次 gather，在 `num_stages=1` 下多花 +4%。
- **varlen 打包。** 当 block 从每个序列自己的起点开始划分时，单独运行和打包运行的
  前向、反向结果完全相同。如果网格按打包偏移对齐，只要偏移不是 block 大小的倍数就会
  失败。

## 5. MLA（latent attention）

很多实现在 decode 时用吸收式（把权重折进 query 和 output 投影），在 prefill
时用显式，也就是把 K/V 展开。两种形式在精确算术下相等，但舍入不同。这是算法上的
差异，所以要选定一种形式，在所有模式下都用它；要达到 L4，trainer 也得用同一种。然后对
注意力核心套用第 4 节的做法，并说明所选形式的吞吐代价。

在一种 GPU 型号上测得（[MLA 和稀疏注意力案例](../../../handbook/3-kernels/mla-and-sparse.md)）：
- **两种形式的结果不同。** 它们有 65% 的输出元素不一致，但相对 fp64 的精度一样。
- **单用一种形式可以做到不变。** 当每个 GEMM、norm 和注意力核心都不变时，任一形式
  单独使用都满足 L2 和 L3。
- **混用两种形式会和 trainer 对不上。** 显式 prefill 搭配吸收式 decode，从第二层
  起就和 trainer 不一致了，不管 trainer 用哪种形式。只有一层的模型可能碰巧一致，因为
  latent cache 与形式无关。
- **只让注意力核心不变还不够。** 不变的注意力核心配上 cuBLAS 的投影，仍然会失败。
  显式里的解压 GEMM（它的 M 是 cache 中的 token 数）和每个 head 输出的 `bmm`
  也必须不变。
- **代价。**
  - 用吸收式做 prefill，耗时是显式 prefill 的 1.77x（B=4，T=1024）。
  - 用显式做 decode、每一步都重新解压 K/V 时，耗时是吸收式 decode 的
    6.9–27x（B=16–64）。
  - 用显式做 decode、每个 head 单独缓存 K/V 时，耗时是 1.17–3.0x，但每个
    token 的 cache 要占 7.2x 的显存。

## 6. 先选 key 再计算的稀疏注意力（indexer 加 top-k）

在这类算子里，分数一旦随 batch 变化，就会变成离散的翻转：kernel 选中的 key 集合变了，
输出也就差得很远。

- 用第 1–2 节的做法让 indexer 的分数不变。
- top-k 遇到并列时要有确定的规则：先比分数，再比索引。
- 按排好序的索引顺序在选中的 key 上累加，具体按第 4 节来做。
- 先确认不同 batch 下选中的集合完全一样，再去比较数值。

在一种 GPU 型号上测得（[MLA 和稀疏注意力案例](../../../handbook/3-kernels/mla-and-sparse.md)）：
- **翻转都来自投影 GEMM。** 在 chunked prefill 和 batched decode 下，约四分之一的
  位置发生了选择翻转，而每一次翻转都来自 cuBLAS 的投影 GEMM。只换成稳定的 top-k，
  或者只让 indexer 不变，一个翻转都消除不了。投影换成单一配置之后，全部通过。
- **稳定排序还是要保留。** `torch.topk` 遇到并列时在每次测试中都选了最小的索引，但文档
  并没有保证这个顺序。稳定排序在 T=2048 时多花 0.87 ms。
- **先比集合，再比数值。** 基线里大部分数值不一致，都出现在选中集合并没有变化的位置上。

## 7. 线性注意力和 SSM（GDN、Mamba 类）

分块形式（用于训练和 prefill）和递归形式（用于 decode）是两种不同的算法，
归约顺序也不同。一次只处理一个序列的递归 kernel 天然就是 batch-invariant 的，
因为它从不跨序列做归约。

有两种选择：
- 所有地方都用递归形式做前向，分块形式只用于反向。有报告说这样做与 vLLM
  的 logprob 差异为 0，代价是 trainer 吞吐差 2–3x；
- 或者接受这个差异，并把它作为一种选择来报告，而不是当成故障。

只要 chunk 大小和 chunk 内部的顺序是固定的，分块 kernel 就满足 L2。

在一种 GPU 型号上用 gated-delta-rule 层测得（[线性注意力案例](../../../handbook/3-kernels/linear-attention.md)）：
- **递归形式。** 递归形式的 Triton kernel（每个序列、head 和 value tile 各一个
  program）不需要额外处理就满足 L1、L2 和 L3。
- **分块形式。** 分块 kernel 满足 L1 和 L2。只有每次调用都把 chunk 放在 C 的绝对
  倍数上，才满足 L3。decode 可以从上一个 chunk 边界的状态出发，重算还没填满的那个
  chunk，这样能与完整的 prefill 逐 bit 一致，耗时是递归 decode 一步的 3.4–3.7x。
- **精度比算法影响更大。** 矩阵乘的操作数用 fp32 时，分块和递归在 bf16 输出
  上只有 0.03% 不同；操作数用 bf16 时（常见的分块 kernel 就是这样），有 55.5% 不同。
- **递归形式的代价随 T 增长。** 和一个没有调优的分块 kernel 相比，递归前向在
  T ≤ 4096 时耗时 0.9–1.5x，在 T = 8192 时是 2.5–3.6x。
- **配置不同，结果也不同。** 两种递归 tile 配置的结果彼此不同，所以 trainer 和
  引擎必须用同一种。

## 8. MoE 的路由、dispatch 和 combine

- **router：** 用不变的 GEMM（第 2–3 节）、按行的 softmax 或 sigmoid（第 1 节），以及
  稳定的 top-k（先按分数，再按 expert id）。routing 权重要保持参考实现的 dtype：有一个
  公开报告的不一致，原因仅仅是把这些权重转成了 bf16。
- **不要按 batch 丢 token。** capacity 上限会让一个 token 是否被丢弃取决于它的邻居。
  要用不丢 token 的路由（dropless），或者用不依赖其他 token 的 capacity 规则。
- **dispatch** 只搬数据，所以是精确的。expert 分段里的 padding 行不能改变真实的行；
  只要 expert GEMM 不变，它们就不会改变。
- **combine：** 对每个 token，用一个 program 把它的 top-k 个 expert 输出按固定顺序
  （按 top-k 名次或按 expert id）、以固定精度加起来。绝不要用 scatter-add 或原子的
  index-add 往输出里累加。在 EP 下，先把各 rank 返回的行放进固定的槽位再求和，这样
  它们到达的先后就无关紧要。shared expert 也要加在这个求和里的固定位置上。
- **测量结果**（[MoE 案例研究](../../../examples/moe_combine/REPORT.md#one-gpu)，一种 GPU 型号）：
  - cuBLAS 的 expert GEMM 从 2 个 token 起就破坏了 L2；
  - 确定性模式只修好了 L1；
  - 即使 GEMM 都不变，用原子操作做 combine 也会破坏 L1；
  - 单一配置的 Triton GEMM 加上每个 token 固定顺序的 fp32 combine，通过了 L1 和 L2，
    耗时比同样的 GEMM 配原子 combine 多约 +11%；
  - 在固定大小的块上做 batched matmul，结果并不是不变的。
- **EP 下的测量结果**（[EP 案例](../../../examples/moe_ep/REPORT.md#under-expert-parallelism)，
  一个和两个 8 GPU 节点）：
  - 固定槽位的 combine 在 EP 1、2、4、8 和 16 下结果逐 bit 一致，一个节点和跨两个节点
    都是如此，与 GPU 怎么摆放无关（L5）。不需要任何 NCCL 设置，因为 all-to-all 里
    没有归约；
  - 在每个 rank 上先求和再返回，满足 L1 和 L2，但从 top-3 起不满足 L5，因为分组方式
    取决于哪些 expert 落在同一个 rank 上。它的不一致取决于 EP 规模，而不取决于 rank
    摆在哪里；
  - top-2 时任何顺序的结果都逐 bit 一致，所以要用 top-k ≥ 3 来测；
  - 有 capacity 丢弃时，哪些 token 能留下来会随邻居、EP 规模或本地顺序而变。

## 9. 量化 GEMM（FP8 block scaling、MXFP8、NVFP4）

- **激活的 scale 不能把各行耦合在一起。** 用 per-token、per-token-block（例如 1x128）
  或静态 scale，绝不要用在整个 batch 上算出的 per-tensor amax。NVFP4 的第二级 scale 也
  一样，要么 per-token，要么静态。
- 量化 kernel 本身就是一个按行归约（求 amax），所以按第 1 节处理。
- GEMM 按第 2 节处理，另外还要固定 promotion interval，也就是在 tensor core 精度下
  累加多少个 MMA 步之后，把部分结果加进 fp32 累加器。scale 要在固定的位置、按固定的
  顺序乘上去，并且不用 split-K。
- 在检查 GEMM 输出之前，先验证量化后的值本身（数据和 scale）在不同 batch 中逐行完全
  相同。

## 10. 集合通信（TP / EP / SP）

在 8 块 GPU 上测得（一个 PCIe 节点，NCCL 2.21.5，[集合通信案例](../../../handbook/3-kernels/collectives.md)）：

- **只固定算法和协议还不够。** 固定 Ring、试遍所有协议之后，同一段固定的行在不同
  大小的消息里仍然按不同的顺序归约。每个元素从哪个 rank 开始累加取决于 chunk 边界，
  而 chunk 边界会随消息大小和偏移移动。行并行的 TP 4 和 8 从 M = 2 起就不满足 L2。
- **在一个节点上，只有一种顺序的算法是可行的。** 在这个节点上，`NCCL_ALGO=Tree` 对
  任意大小下的每个元素都用同一个顺序归约，通过了 L2，代价是 all-reduce 的 1.0–1.3x，
  1 MB 时是 5.3x。
- **跨两个节点也成立。** 在 25 GbE 上用 16 个 rank，固定 Tree 仍然通过。每个元素先在
  各节点内沿链求和，再把两个节点的和相加一次，而两个操作数的加法满足交换律。三个或
  更多节点很可能会破坏这一点（尚未测试），所以要在你自己的拓扑上验证。
- **reduce-scatter 对每个输出 chunk 只有一种顺序，但这还不够。** 这个顺序取决于哪个
  rank 拥有这一行。开了序列并行之后，归属又取决于位置和 batch，所以 TP 加 SP 在任何
  设置下都失败了。
- **回退方案。** 交换部分和（用 all-to-all 或 all-gather），然后在本地按 rank 顺序用
  fp32 相加。这种做法从构造上就是不变的。all-to-all 版本从 1 MB 起的代价是 all-reduce
  的 1.04–1.20x。它改变了运算本身，所以要达到 L4，两个引擎都得用它。
- **跨节点时用两级。** 跨节点时，扁平的回退方案代价是 all-reduce 的 2.1–8.8x。两级的
  固定顺序同样不变，代价只有 1.06–1.8x：先在每个节点内按 rank 顺序求和，再在节点之间
  做一次 fp32 交换。
- **只有两个 rank 时顺序无所谓**，因为两个操作数的和满足交换律。要用 3 个或更多 rank
  来测。
- **并行度不同，归约树就不同。** 在 TP 2–8 下，行并行和不切分相比有 37.5–59% 的元素
  不同。用一个逻辑 TP 规模可以达到 L5：
  - 把 K 切成固定数量的 chunk（这里是 8 个）；
  - 用同一个 kernel 计算每个 chunk 的部分和；
  - 收集所有部分和，按 chunk 顺序相加。

  这样 TP 1、2、4 和 8 就逐 bit 一致了；用树形式实现时，TP 8 下的耗时是行并行的
  1.6–1.9x。用 16 个 chunk 时，TP 1、2、4、8 和 16 在两个节点上也一致。
- **所有地方都用同一个 chunk 数。** 逻辑 TP 8 和逻辑 TP 16 的结果不同。逻辑 TP 8 要在
  TP 16 上运行，只能把每个 chunk 的输出列再切开；如果改切 K，结果就变了。
- **整个模型**（whole-model-tp 案例）：每个行并行层都用逻辑 TP（C = 16），再加上按固定
  chunk 求和的词表并行 log_softmax，在 TP 1、2、4、8 和 16 下都给出同一个结果，prefill
  用 TP a、decode 用 TP b 时也一样。反向里列并行层的输入梯度也是跨 rank 的求和，所以
  也要用同样的方式归约（NCCL all-reduce 在检查过的全部 9 个点上都破坏了 L5）。代价是：
  在通过 PCIe 互联的 8 块 GPU 上，TP 2–8 时 prefill 为 1.4–1.6x，decode 为 1.6–1.9x。
  逻辑 TP 和单 GPU 的运算不同（TP 1 时有 84.7% 的 logits 不同），所以 trainer 和引擎
  都必须用它。
- **数据并行的梯度**（training-dp-fsdp 案例，两个 8 GPU 节点）：在 Ring 下，桶的
  布局会让梯度变掉，而 torch DDP 的第一次迭代（以及每次重启）会把所有梯度放进一个
  桶来归约。改用 Tree 或有序的 all-reduce 就能修好这个打包问题（1.05–1.20x）。
  reduce-scatter 的顺序跟着拥有该行的 rank 走。要在不同的 DP 规模之间达到 L5，就把 batch
  切成固定数量的逻辑 shard（这里是 16 个），每个 shard 单独算梯度，再用一棵固定的树把
  它们加起来，而且这棵树要包含跨 rank 的那一步求和。把这一步交给 NCCL 去做，从 3 个
  rank 起就失败了，即使用 Tree 也一样。

## 11. 闭源或预编译的 kernel

- 用 `check_batch` 扫很多种大小，摸清 kernel 在哪个范围内不变。padding 到固定的 M 只有在
  每次调用真的都以这个 M 运行时才有用。CUDA graph 的分桶会打破这一点，因为每个
  桶都按自己的 M 运行。
- 如果没有哪个开关管用，不变的范围也覆盖不了你要用的大小，就替换这个 kernel。如果替换
  不了，就保留一个逐行运行的包装作为调试用的参照（oracle），并把这个算子排除在一致性
  结论之外。

## 12. `torch.compile` 和 CUDA graph 这类部署时的包装

在一种 GPU 型号上测得（[paged、varlen 和 compile 案例](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)）：
- **把不变的 kernel 注册成不透明的自定义算子。** 直接调用这些 Triton kernel 时，
  Dynamo 每换一个 batch size 就重新编译一次，一共 18 个 graph，即使加了 `dynamic=True`
  也一样；注册成自定义算子之后只需要 2 个。
- **编译可以保住 L2，结果却仍然会变。** Inductor 把 RMSNorm、残差加法和 SiLU·up 融合在
  一起之后，编译后的 block 和 eager 相比有 65% 的元素不同。把每个会舍入的算子都做成
  不透明的之后，compiled = eager。一个必须和 eager trainer 一致的引擎就需要这样做。
- **只有 kernel 本身不变，CUDA graph 才能保持不变性。** padding 到桶的尺寸之后，cuBLAS
  对同一个序列在 7 个桶上给出了 6 种不同的结果。graph replay 本身在同一个形状下
  和 eager 逐 bit 一致。
- **自定义算子有 host 端开销：** 这里在小 batch 下每次调用约 0.65 ms，除非把它放进
  graph 里运行。
- **在编译后的代码里采集 fingerprint**（[compile-graph-capture 案例](../../../handbook/1-pretraining/capture-compiled-code.md)）：
  `tools/graph_probe.py` 可以从编译后的代码和 graph replay 中记录 OpGuard 格式的
  fingerprint。要给 probe 一份精确的逐元素拷贝（`isolate=True`），因为如果读取这个张量的
  是一个非逐元素的算子，Inductor 可能改用 `addmm`，结果也就变了。会惰性构建的 cache
  （比如 RoPE 表）要在编译之前先用 eager 建好。

## 13. 采样

噪声要用基于计数器的生成器来产生，以 (seed, absolute position, purpose, sample index)
作为 key，绝不依赖生成器的运行状态。分布要做成精确的：用整数权重
floor(exp(z−max)·2^S)，这样求和与 cumsum 就和顺序无关。遇到并列时用稳定排序按
(value desc, index asc) 排列，这样 greedy decoding 会选到最小的索引。如果要求 token 与
不用草稿模型时 decode 的结果一致，验证草稿 token 时就要检查它是否等于目标
模型自己采样出的 token。以上在一种 GPU 型号上测得（[采样案例](../../../handbook/2-rl/sampling.md)；
参考采样器是 [tools/bi_sampling.py](../../../tools/bi_sampling.py)）。

## 14. LoRA 与混有多个 adapter 的 batch

在 kernel 内部为每一行选出它自己的 adapter；绝不为了做归约而按 adapter 重新给行分组。
base GEMM 用单一配置运行。shrink 和 expand 两步都按固定的 rank 顺序和 K 顺序进行，LoRA
的增量保持 fp32，再用一次 fp32 FMA 加到 base 的结果上，最后只做一次类型转换。

trainer（只有一个 adapter）和引擎（一个 batch 里混着多个 adapter）必须用同一个 kernel。
scale 在哪一步乘上去要固定下来。rank 较小的 adapter 在 fp32 累加里补零。trainer 计算
logprob 时要关掉 dropout。如果引擎必须和一个不把 adapter merge 进权重的 trainer 一致，
就不要在 bf16 下做 merge。

反向时，dX 用同一个 kernel 计算；dA/dB 按 adapter 分别计算，只遍历梯度非零的行，按升序
进行，不用原子操作。以上在一种 GPU 型号上测得（[LoRA 案例](../../../handbook/2-rl/lora.md)）。

## 15. 训练 step（反向里的 scatter、优化器、loss、恢复训练）

- 把 `index_add_`、`scatter_add_`、embedding 的反向和带权的 `bincount` 换成你自己写定的
  顺序：先按索引排序，再在 fp32 下按升序做分段求和。只开严格模式并不能给出写定的顺序，
  因为它的 scatter 和 embedding 反向会随行宽或索引个数改变顺序。
- 固定优化器的实现路径（fused、foreach 或 single-tensor）和 loss 的归约方式。mean 不是
  布局不变的。
- 保存并恢复一个训练 step 会读到的每一项状态；这个案例列出了 7 项。
- 在一种 GPU 型号上测得（[训练 step 案例](../../../handbook/1-pretraining/training-step.md)）；
  多 GPU 下的梯度见第 10 节。

## 16. 跨引擎的舍入差异（L4）

即使两端各自都满足 L2 和 L3，它们也不一定彼此一致，所以 L4 要单独去做。把两端的舍入
序列都写下来（投影的打包方式、bias 的位置、norm 的写法、RoPE 表，以及注意力、lm_head
和 log_softmax 的 kernel）。先检查引擎自己的 L1 和 L2，然后一次对齐一个舍入点。具体
流程见 [engine-parity skill](../../engine-parity/SKILL.md)。在
[trainer 与 vLLM 的案例](../../../handbook/2-rl/engine-parity-vllm.md)中（vLLM 0.19.1，
一种 GPU 型号），只有让 trainer 调用 vLLM 自己的 kernel，才做到了相等。

## 文中引用数字的出处

- Thinking Machines，"Defeating Nondeterminism in LLM Inference"（2025）。文章指出，
  推理中的非确定性根源在于缺少 batch invariance，并给出了不变的 RMSNorm、矩阵乘和注意力。
- Yichuan Wang，"Closing the Training–Inference Mismatch for Linear Attention in
  RL"。文中前向用递归形式的 GDN，MoE router 用 batch-invariant 的 `bmm`，
  trainer 的代价是 2–3x。
- verl-omni issue #637。它让 vLLM 逐个算子对齐 HF 的数值（norm 的类型转换顺序、残差
  加法的精度、RoPE 公式、router 权重的类型转换）；报告的 rollout 吞吐约为 37%。
- IsoExec（SkyRL，vLLM 博客）。它提出了一份跨框架共享的执行约定，以及并行不变的
  kernel。

这些链接都在 handbook 的[参考文献](../../../handbook/references.md)页面里。本页的观察
结果来自 `tools/bi_check.py --demo`（结果存在 `tools/` 下的一个 demo JSON 里），以及
各个案例研究；每个案例的脚本和结果 JSON 都放在 `examples/` 下。
