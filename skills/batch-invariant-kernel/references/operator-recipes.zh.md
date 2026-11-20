# 算子配方

每个配方说明是什么让该算子变成批相关（batch-variant），以及如何修复。
它达到的级别依据 [SKILL.md](../SKILL.md#levels-say-which-one-you-deliver)。
每个配方最后都走同一套验证（[verification.md](verification.md)）；
没有一个能只靠读代码来证明。

## 每个配方背后的证据

测量：单 GPU 案例用一种 GPU 型号。多 GPU 案例（集合通信、专家并行、DP 梯度、
TP 下的整个模型）在一到两个各有 8 个 GPU（另一种型号）的节点上运行：节点内 PCIe，
节点之间 25 GbE 套接字，无 RDMA。全部使用 torch 2.5.1 和 Triton
3.1。例外在对应行中注明（跨引擎一行使用 vLLM 自己的
技术栈：torch 2.10.0、Triton 3.6.0）。每个案例的硬件和原始结果在它的
`examples/<case>/REPORT.md` 中。案例研究：
- [MoE](../../../examples/moe_combine/REPORT.md#one-gpu)；
- [专家并行下的 MoE](../../../examples/moe_ep/REPORT.md#under-expert-parallelism)；
- [注意力、FP8、线性注意力](../../../examples/kernel_cases/REPORT.md#forward-prefill-decode-and-chunked-prefill-l2-l3)；
- [注意力反向](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv)；
- [分页 KV、varlen、compile 和 CUDA graph](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)；
- [MLA 和稀疏注意力](../../../handbook/3-kernels/mla-and-sparse.md)；
- [线性注意力（GDN）](../../../handbook/3-kernels/linear-attention.md)；
- [集合通信和 TP](../../../handbook/3-kernels/collectives.md)；
- [整个模型](../../../handbook/2-rl/whole-model.md)和 [TP 下的整个模型](../../../handbook/2-rl/parallelism.md)；
- [DP 和 FSDP 风格（模拟：扁平梯度、reduce-scatter、分片 AdamW）的梯度](../../../handbook/1-pretraining/data-parallel.md)；
- [采样](../../../handbook/2-rl/sampling.md)；
- [LoRA 和多适配器批次](../../../handbook/2-rl/lora.md)；
- [训练步](../../../handbook/1-pretraining/training-step.md)；
- [trainer 对比 vLLM](../../../handbook/2-rl/engine-parity-vllm.md)。

| # | 类别 | 状态 | 证据 | 仍未解决 |
|---|---|---|---|---|
| 1 | 行内归约 | 原生实现即观察到 | torch 算子 RMSNorm 和 fp32 log_softmax 在 hidden 4096、vocab 32000、batch 1–2048 下为 L2；fp32 的 torch mean/sum 会随行数改变其归约树（hidden 1024 从 8 行起，4096 从 2 行起），而 bf16 RMSNorm 的输出没有变化；固定树的 Triton 行求和在行数 1–2048、宽度 8–4096 下为 L2（整模型案例） | 少行拆分的情况未复现；LayerNorm、cross-entropy |
| 2 | 稠密 GEMM | **已构建并验证** | 单配置 Triton GEMM 在 3 种形状上于 M=1–2048 为 L2；cuBLAS 在全部 3 种上都随批变化 | 调优、大形状、融合 epilogue、cuBLASLt 固定、Hopper |
| 3 | 分组 GEMM，`bmm` 对比 `mm` | **已构建（循环）并验证** | 逐专家单配置 GEMM 在两种 GPU 型号上都为 L2（[单 GPU](../../../examples/moe_combine/REPORT.md)、[8 GPU](../../../examples/moe_ep/REPORT.md)）；cuBLAS 专家 GEMM 在两者上都随批变化；在固定行块上的 `bmm` 随批变化 | 单次启动的分组 kernel |
| 4 | Softmax 注意力 | **已构建并验证**（前向、反向、分页、varlen） | 固定块 Triton 在 prefill、decode 和分块之间为 L2 和 L3；SDPA 不满足 L2（batch ≥ 3）和 L3。无原子操作的反向为 L1 和 L2，包括右填充，耗时为 SDPA flash 的 1.38x；SDPA flash 不满足 L1（dQ），在严格确定性模式下不满足 L2。当块按每个序列的绝对位置划分时，分页 KV（页大小 16–128）和 varlen 打包（前向和反向）保持比特不变；按页或按打包偏移划分的块则不行 | 固定大小的 split-KV、滑动窗口、分页反向 |
| 5 | MLA absorbed 对比 explicit | **已构建并验证** | 两种形式在 65% 的元素上不同，精度相同；每个部分都不变的单一形式为 L2 和 L3；explicit prefill 配 absorbed decode 从第 2 层起不再与 trainer 一致；只有不变的核心还不够（解压 GEMM、输出 `bmm`） | FlashMLA、absorbed 形式的 split-KV、反向、对引擎的 L4 |
| 6 | 稀疏注意力（top-k） | **已构建并验证** | 每次选择翻转都来自 cuBLAS 投影 GEMM（约四分之一的位置）；用单配置投影和稳定的 top-k，选中集合和输出为 L2 和 L3；`torch.topk` 在每次测试中都在并列时取最小索引，这是未写入文档的顺序 | DSA 规模（top-2048）、FP8 indexer、反向 |
| 7 | 线性注意力 / SSM | **已构建并验证**（GDN 前向） | 循环 kernel 为 L1–L3；分块为 L1/L2，只有分块边界位于 C 的绝对倍数时才是 L3；bf16 分块矩阵乘与循环形式在 55.5% 的输出上不同，fp32 时为 0.03%；在 T ≤ 4096 时循环耗时为分块的 0.9–1.5x，在 8192 时为 2.5–3.6x | 反向、生产级分块 kernel（fla）、varlen |
| 8 | MoE 路由与合并 | **已构建并验证**（1 GPU；在两个 8 GPU 节点上 EP 1–16，[报告](../../../examples/moe_ep/REPORT.md)） | 固定顺序的 fp32 合并加单配置 GEMM 为 L1 和 L2；在 EP 下，固定槽位合并在 EP 1、2、4、8 和 16 下给出相同比特，无论在一个节点上还是分在两个节点上（L5）；逐 rank 预归约为 L1/L2，从 top-3 起不满足 L5；原子合并破坏 L1（EP 案例中从 top-3 起）；容量丢弃不满足 L2 和 L5 | EP 下的反向、共享专家、DeepEP 风格 kernel、NVLink、RDMA、两个以上节点、对 trainer 的 L4 |
| 9 | 量化 GEMM | 缩放已验证（模拟） | 逐张量动态缩放不满足 L2；逐 token 和 1x128 通过 | 真实 FP8 GEMM 的提升间隔（在有 FP8 张量核心的 GPU 上）、NVFP4（在有 FP4 张量核心的 GPU 上） |
| 10 | 集合通信 | **已测量**（8 GPU 的节点，一个和两个节点；[报告](../../../examples/collectives_tp/REPORT.md)） | all-reduce 在默认设置下、以及固定 Ring 并配合每种协议时都不满足 L2；`NCCL_ALGO=Tree` 在一个节点上通过（1.0–1.3x，1 MB 时 5.3x），在两个节点上也通过（16 个 rank：先是每个节点的链，再是一次两操作数相加）；reduce-scatter 的顺序跟随所属 rank，因此带序列并行的 TP 不满足；all-to-all 加 fp32 按 rank 顺序求和在一个节点上以 1.04–1.20x 通过；跨节点时，两级固定顺序（1.06–1.8x）替代扁平的后备方案（2.1–8.8x）；逻辑 TP 大小给出 L5：8 在 TP 1–8 之间，16 在两个节点上的 TP 1–16 之间；在整个模型中，逻辑 TP（C = 16）加固定顺序的词表 log_softmax 在 TP 1–16 下、以及不同 TP 的 prefill/decode 之间给出同一结果。DP 梯度：在 Ring 下桶布局会改变比特（Tree 和有序 all-reduce 以 1.05–1.20x 修复）；逻辑分片（S=16，固定树）在一个或两个节点上、复制或分片、以及弹性恢复训练之间，于 DP 1–16 给出相同的梯度、范数和 10 步 AdamW 权重 | 三个或更多节点、NVLink/NVLS、RDMA、torch.distributed.fsdp 本身、激活梯度之外的 TP 反向 |
| 11 | 闭源 kernel | 已测量 | cuBLAS 和 SDPA 的变化已摸清；固定分块的 cuBLAS 循环可作为参照（约慢 33x）；严格确定性模式下的 SDPA memory-efficient 反向在所测形状上通过 L1 和 L2（flash 的 4.4x） | cuDNN、FlashAttention 二进制 |
| 12 | 部署包装层 | **已测量** | 编译后的块仍为 L2，但在 65% 的元素上与 eager 不同，直到每个舍入算子都成为不透明的自定义算子；直接调用的 Triton kernel 会按每个 batch 大小重新编译（18 个图；作为自定义算子为 2 个）；CUDA graph 分桶让不变的 kernel 保持不变，而 cuBLAS 在 7 个桶上给出 6 个结果；编译代码和图内部的探针（`tools/graph_probe.py`）在隔离时不改变编译后的比特 | 分页、varlen 和反向路径的 compile；`max-autotune` |
| 13 | 采样 | **已构建并验证** | 逐请求生成器为 L2，但在重算、回滚、n>1 和推测分块时不满足；cumsum 不满足 L2（在 V=151,936 时也不满足 L1）；用 int64 权重、以计数器为键的采样器通过 L1–L3 和卡方检验，耗时为原生的 1.3x（B=256） | 融合噪声 kernel、min-p、compile/graph、真实引擎 |
| 14 | LoRA / 多适配器 | **已构建并验证**（前向、反向） | 分组 cuBLAS 不满足 L2（486–506/723），基座不变时仍有 22–100/723；gather 后的 bmm 不满足；bf16 合并改变 ~50% 的比特，并在 update/output 1e-2–1e-3 时丢失 18–78% 的 ΔW；缩放位置改变 5–6%；我们的实现通过 L1、L2、prefill/decode、trainer 对比引擎、反向填充 | vLLM 本身、打包切片、TP、compile、QLoRA |
| 15 | 训练步（反向算子、优化器、loss、恢复训练） | **已构建并验证**（1 GPU） | index_add_/scatter_add_/index_select 反向/带权 bincount/1-D cumsum 不满足 L1；先排序再升序 fp32 分段求和为 L1，并与 NumPy 循环逐位一致；严格模式在 cumsum、bincount(weights)、histc 上报错；AdamW fused/foreach/single 互不相同（foreach 与 single 只在偏差修正的除法上不同）；7 项状态给出逐位一致的恢复训练 | 融合优化器的精确顺序（0.15% 的 exp_avg 差 1 ulp）、多 GPU（见第 10 行）、compile |
| 16 | 跨引擎舍入（L4） | **已测量**（vLLM 0.19.1，[报告](../../../examples/cross_engine/REPORT.md)） | 打包、bias 位置、norm 形式、RoPE 表、注意力 kernel、lm_head kernel 和 log_softmax kernel 各自都会破坏相等；RoPE 公式和 SiLU 本来就相等；trainer 调用 vLLM 的 kernel 得到 0/76,155 | 编译的 vLLM、反向、TP、其他模型 |

## 1. 行内归约：RMSNorm、LayerNorm、softmax、log_softmax、行求和、cross-entropy

变化来自：
- 行数少时，为了填满 GPU，把一行拆给多个块；
- 根据行数或指针对齐来选块大小或向量宽度；
- 小 batch 的快速路径。

修复：
- 每个 program（块或 warp）处理一行，无论 batch 多大。
- 沿行方向固定块大小；每个线程的累加方式固定，然后是固定的
  块内归约树。
- 如果一行对一个块来说太长，就把它切成若干段，段数只取决于
  行长，并在第二遍中按固定顺序合并。
  永远不要用原子操作。
- 固定向量宽度。如果行起始地址的对齐随它在 batch 中的位置而变
  （hidden 大小不是向量宽度的倍数），就在所有地方都走标量路径，
  或者对行跨度做填充。
- 保持参考的舍入点：例如归一化后的值是在乘以权重之前还是之后
  转换为 bf16。

达到 L2。在一种 GPU 型号、torch 2.5.1、hidden 4096、vocab 32000 下观察到：
torch 算子 RMSNorm 和 fp32 `log_softmax` 在 batch 大小
1–2048 之间本来就是 L2。对其他形状、版本或 GPU 并无保证，所以要验证。

## 2. 稠密 GEMM（bf16/fp16）

变化来自：
- 按 M 选择 tile、指令或 kernel；
- split-K 或 stream-K；
- 库的启发式规则。

在一种 GPU 型号、torch 2.5.1 下观察到：`torch.mm`（cuBLAS）在 M=16 及以下为 L2，
从 M=31 起随批变化。

修复：
- 对每个 M 都用同一个 tile 配置（block M/N/K、warp 数、stage 数、MMA 指令）。
- 不用 split-K，不用 stream-K。K 循环按升序累加到 fp32
  累加器，epilogue 相同。
- 只有当第二个小 M kernel 复现相同的逐元素 K 分段和指令时，才允许使用它。
  用逐位比较证明它与主 kernel 一致。
- cuBLASLt 可以固定一个算法及其 split-K 设置。仍要在不同 M 上验证：
  固定的算法并不是保证。
- 关闭 TF32，除非两侧以完全相同的方式使用它。

达到 L2。L4 需要两个引擎使用同一个 kernel。代价：小 M（decode）时
利用率较低。

## 3. 批量与分组 GEMM：MoE 专家，`bmm` 对比 `mm`

- 专家的 GEMM 看到的是 M_e，即路由到它的 token 数，而它会随
  batch 里的其他内容变化。每个专家 GEMM 都必须对 M_e 不变
  （配方 2）。分组 kernel 常按最大或总 M 选 tile 大小：把它们
  固定下来。
- 一侧对 3-D 张量调用 `bmm`，另一侧对 2-D 张量调用 `mm`，运行的是
  不同的 kernel。一个已报告的案例是 MoE router：不同的门控分数
  会把 token 路由到不同的专家，这是离散的分歧。让两侧都走
  同一个不变的 GEMM。

## 4. Softmax 注意力：prefill、decode、分块 prefill、分页 KV

在一种 GPU 型号、torch 2.5.1 SDPA 下观察到：
- prefill 与逐 token decode 在约 10% 的输出元素上不同；
- 带显式 mask 的分块 prefill 在 1 个元素上不同。带 mask 的调用
  很可能走了与 `is_causal=True` 不同的后端。

修复，对位于位置 t 的查询：
- **处处使用相同的 key 迭代。** KV 按固定大小 B 的块访问，
  从 0 开始按绝对位置顺序，逐块应用 online-softmax 更新。
  prefill、分块 prefill 和 decode 使用相同的 B 和相同的
  顺序。
- **不用 split-KV**（`num_splits=1`）。或者按固定的 token 数切分，并对齐
  到绝对位置，永远不要按切分个数来切。然后在 prefill 中使用与 decode
  相同的切分与合并。
- **分页 KV：** 块边界跟随绝对位置（B 的倍数），
  与页大小以及 prefill 分块从哪里开始无关。在 kernel 运行之前，把
  当前分块的 K/V 写入 cache，使 kernel 读取
  统一的单一来源。
- **掩码：** 被掩掉的分数贡献恰好为零。部分被掩的块
  在每种模式下都以相同方式计算滑动最大值和归一化因子。
- 在每种模式下使用相同的指数路径（例如带 log2(e) 预缩放的 `exp2`）、相同的缩放
  和相同的 GQA head 映射。
- 变长打包（`cu_seqlens`）：一个序列的迭代不依赖于
  打包中的相邻序列。
- **反向：** 常见 kernel 用原子操作把来自 key 块 program 的 dQ 贡献
  加起来，这会破坏 L1。改用两遍无原子操作的计算：
  - dK/dV：每个 key 块一个 program，按固定顺序先遍历 GQA 组的查询 head，
    再遍历查询块；
  - dQ：每个查询块一个 program，从位置 0 开始遍历 key 块。

  两者都在 fp32 寄存器中累加并只写一次。在 program 内以 fp32 对 GQA 组
  求和。把前向的 log-sum-exp 存为一个变体，其输出要与普通前向
  逐位比对。在
  [注意力反向案例](../../../examples/attention_backward/REPORT.md#backward-dq-dk-dv)中测得：
  - L1 和 L2，包括带掩码 loss 的右填充；
  - 前向+反向为 SDPA flash 的 1.38x。

  厂商的确定性反向需要 `warn_only=False`，并且不会比其前向
  提供更多不变性：flash 这样运行时不满足 L2。

达到 L2 和 L3。代价：去掉 split-KV 会拖慢序列很少时的长上下文
decode。

在一种 GPU 型号上测得（[分页、varlen 和 compile 案例](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)）：
- **分页 KV。** 在页大小 16–128、页随机放置时，连续存储的
  结果在每种模式下都逐位复现。改为按页大小分块计算，
  则每种页大小都给出不同的结果。每个结果在各模式之间与
  自身一致，所以要测试多个页大小。比块还小的页需要 gather；
  在 `num_stages=1` 时它多花 +4%。
- **变长打包。** 当块从每个序列的起点开始时，前向和反向在单独运行和打包运行时
  完全相同。与打包偏移对齐的网格在偏移不是块大小的倍数时
  都会失败。

## 5. MLA（潜在注意力）

实现常对 decode 使用 absorbed 形式（权重折叠进
query/output 投影），对 prefill 使用显式 K/V 形式。两者在精确算术下
相等，但舍入不同。这是算法上的差异：
在每种模式下都使用同一种形式，并在 trainer 中也这样做以达到 L4。然后对注意力核心应用
配方 4。说明所选形式的吞吐代价。

在一种 GPU 型号上测得（[MLA 和稀疏案例](../../../handbook/3-kernels/mla-and-sparse.md)）：
- **两种形式不同。** 它们在 65% 的输出元素上不一致，相对 fp64 的
  精度相同。
- **单一形式是不变的。** 当每个 GEMM、norm 和注意力核心都
  不变时，每种形式单独使用都是 L2 和 L3。
- **混用形式会破坏与 trainer 的一致。** explicit prefill 配 absorbed decode
  从第二层起就不再与 trainer 一致，无论 trainer 用哪种形式。
  只有一层时可能碰巧一致，因为潜在
  cache 不取决于形式。
- **只有核心还不够。** 不变的注意力核心配 cuBLAS
  投影仍然失败。explicit 形式的解压（其 M 是
  缓存的 token 数）和逐 head 的输出 `bmm` 也必须不变。
- **代价。**
  - absorbed prefill：explicit 的 1.77x（B=4，T=1024）。
  - 每步重新解压 K/V 的 explicit decode：在 B=16–64 时为 absorbed 的
    6.9–27x。
  - 带逐 head K/V cache 的 explicit decode：1.17–3.0x，但每个 token 的 cache
    占用 7.2x 的内存。

## 6. 带选择的稀疏注意力（indexer 加 top-k）

随批变化的分数会变成离散翻转：选中的 key 集合不同，
这是很大的分歧。

- 用配方 1–2 让 indexer 的分数不变。
- 采用确定性并列处理的 top-k：先比分数，再比索引。
- 用配方 4 按排序后的索引顺序在选中的 key 上累加。
- 先检查不同 batch 下选中的集合完全相同，再检查
  数值。

在一种 GPU 型号上测得（[MLA 和稀疏案例](../../../handbook/3-kernels/mla-and-sparse.md)）：
- **翻转来自投影 GEMM。** 在分块 prefill 和批量 decode 下，约
  四分之一位置上的每一次选择翻转都来自
  cuBLAS 投影 GEMM。单独用稳定的 top-k 或不变的 indexer 一个也
  消除不了。用单配置投影后全部通过。
- **仍然保留稳定排序。** `torch.topk` 在每次测试中都在并列时选了最小索引，
  但这个顺序没有写入文档。稳定排序在 T=2048 时额外花费
  0.87 ms。
- **先比较集合，再比较数值。** 大多数基线数值不一致出现在
  集合没有变化的位置上。

## 7. 线性注意力与 SSM（GDN、Mamba 风格）

分块形式（训练、prefill）和循环形式（decode）是
归约顺序不同的不同算法。逐序列的循环
kernel 天然是 batch-invariant（批不变）的：它没有跨序列的归约。

选项：
- 处处使用循环前向，只在反向中使用分块。据
  报告这样相对 vLLM 达到 logprob 差为 0，trainer
  吞吐代价为 2–3x；
- 或者接受这个差异，并把它报告为一种选择而不是故障。

当分块大小和块内顺序固定时，分块 kernel 为 L2。

在一种 GPU 型号上用 gated-delta-rule 层测得（[线性注意力案例](../../../handbook/3-kernels/linear-attention.md)）：
- **循环。** 循环 Triton kernel（每个序列、head 和
  value tile 一个 program）无需额外工作就是 L1、L2 和 L3。
- **分块。** 分块 kernel 为 L1 和 L2。只有当每次调用的
  分块都位于 C 的绝对倍数上时才是 L3。decode 可以从最后一个边界状态重算
  未闭合的分块，并与完整 prefill 逐位一致，代价为
  循环 decode 一步的 3.4–3.7x。
- **精度比算法更重要。** 用 fp32 矩阵乘操作数时，
  分块与循环在 0.03% 的 bf16 输出上不同。用 bf16 操作数时（
  常见分块 kernel 就这样用），在 55.5% 上不同。
- **循环的代价随 T 增长。** 相对一个未调优的分块 kernel，
  循环前向在 T ≤ 4096 时耗时 0.9–1.5x，在 T = 8192 时为 2.5–3.6x。
- **配置也会不同。** 两种循环 tile 配置彼此
  不同，所以 trainer 和引擎必须共用一种。

## 8. MoE 路由、分发与合并

- **Router：** 不变的 GEMM（配方 2–3）、行内 softmax 或 sigmoid
  （配方 1）、稳定的 top-k（先分数，再专家 id）。路由权重保持参考的
  dtype：一个已报告的不一致仅仅来自把它们转换为
  bf16。
- **不按 batch 丢弃。** 容量上限让一个 token 的去留取决于
  它的相邻 token。使用无丢弃路由，或者不依赖其他 token 的
  容量规则。
- **分发**只搬运数据，所以是精确的。专家分段中的填充行
  不能改变真实行；当专家 GEMM 不变时它们不会改变。
- **合并：** 对每个 token，在每个 token 一个 program 内，按固定顺序（按
  top-k 名次或专家 id）、以固定精度对它的 top-k 专家输出求和。
  永远不要 scatter-add 或原子 index-add 到输出中。在专家
  并行下，求和之前先把返回的行放进固定槽位，使到达
  顺序无关紧要。把共享专家加在该求和中的固定位置。
- **测得**（[MoE 案例研究](../../../examples/moe_combine/REPORT.md#one-gpu)，一种 GPU 型号）：
  - cuBLAS 专家 GEMM 从 2 个 token 起破坏 L2；
  - 确定性模式只修复了 L1；
  - 即使 GEMM 不变，原子合并也破坏 L1；
  - 单配置 Triton GEMM 配逐 token 固定顺序的 fp32 合并通过 L1
    和 L2，比同样的 GEMM 配原子合并多约 +11%；
  - 在固定大小分块上的批量矩阵乘不是不变的。
- **专家并行下测得**（[EP 案例](../../../examples/moe_ep/REPORT.md#under-expert-parallelism)，
  一个和两个各有 8 个 GPU 的节点）：
  - 固定槽位合并在 EP 1、2、4、8 和 16 下给出相同比特，在一个
    节点上和跨两个节点都是如此，与 GPU 放置无关（L5）。不需要任何 NCCL 设置，
    因为 all-to-all 不包含归约；
  - 返回之前按 rank 先求和为 L1 和 L2，但从 top-3 起不满足 L5：
    分组跟随哪些专家共享同一个 rank。其不一致取决于
    EP 度数，而不取决于 rank 位于何处；
  - 在 top-2 时任何顺序都给出相同比特，所以要用 top-k ≥ 3 测试；
  - 容量丢弃使哪些 token 留下随相邻 token、EP
    度数或本地顺序而变。

## 9. 量化 GEMM（FP8 分块缩放、MXFP8、NVFP4）

- **激活缩放不能把行耦合起来。** 使用逐 token、逐 token 块
  （例如 1x128）或静态缩放，永远不要用在整个 batch 上计算的逐张量 amax。
  NVFP4 的第二级缩放同样如此：逐 token 或静态。
- 量化 kernel 本身就是一个行内归约（amax）：配方 1。
- GEMM 遵循配方 2，外加固定的提升间隔：在加入 fp32 累加器之前，
  有多少个 MMA 步骤以 tensor core 精度累加。
  在固定的位置、按固定顺序应用缩放。不用 split-K。
- 在检查 GEMM 输出之前，先验证量化值本身（数据和缩放）在不同 batch 之间
  逐行完全相同。

## 10. 集合通信（TP / EP / SP）

在 8 个 GPU 上测得（一个 PCIe 节点，NCCL 2.21.5，[集合通信案例](../../../handbook/3-kernels/collectives.md)）：

- **固定算法和协议还不够。** 在每种协议下固定 Ring 时，
  一段固定的行在不同大小的消息中仍然得到不同的归约顺序。
  每个元素的起始 rank 跟随分块边界，而这些边界随消息大小和偏移移动。
  行并行 TP 4 和 8 从 M = 2 起不满足 L2。
- **单一顺序的算法在一个节点上有效。** 在该节点上，
  `NCCL_ALGO=Tree` 对每种大小下的每个元素都给出同一个顺序，并通过
  L2。代价是 all-reduce 的 1.0–1.3x，1 MB 时为 5.3x。
- **跨两个节点也成立。** 在 25 GbE 上用 16 个 rank，固定 Tree
  仍然通过。每个元素先是每个节点内的链，再把两个
  节点的和相加一次，而两操作数相加满足交换律。三个或更多节点很可能破坏
  这一点（未测试）：要在你的拓扑上验证。
- **reduce-scatter 每个输出分块只有一个顺序，这还不够。** 这个
  顺序取决于哪个 rank 拥有该行。在序列并行下，这又
  取决于位置和 batch，带 SP 的 TP 在每种设置下都失败。
- **后备方案。** 交换部分和（all-to-all 或 all-gather），并在本地
  按 rank 顺序以 fp32 求和。这在构造上就是不变的。
  all-to-all 形式从 1 MB 起代价为 all-reduce 的 1.04–1.20x。它改变了
  算术，所以要达到 L4，两个引擎都必须使用它。
- **跨节点时，用两级。** 扁平的后备方案在那里代价为 all-reduce 的 2.1–8.8x。
  两级固定顺序同样不变，代价为 1.06–1.8x：先在
  每个节点内按 rank 顺序求和，再在节点之间做一次 fp32 交换。
- **两个 rank 与顺序无关**，因为两操作数的和满足交换律。用 3
  个或更多 rank 测试。
- **不同的并行度就是不同的归约树。** 在 TP 2–8 时，行并行与
  不分片在 37.5–59% 的元素上不同。逻辑 TP 大小可达到
  L5：
  - 把 K 切成固定数量的分块（这里是 8）；
  - 用同一个 kernel 计算每个分块的部分和；
  - 收集部分和，并按分块顺序求和。

  这样 TP 1、2、4 和 8 逐位一致，在 TP 8 时以树形式耗时为行并行的
  1.6–1.9x。用 16 个分块时，TP 1、2、4、8 和 16 在两个节点之间一致。
- **处处固定分块数。** 逻辑 TP 8 和 16 彼此
  不同。逻辑 TP 8 只有通过切分每个分块的输出
  列才能与 TP 16 一致。改为切分 K 则会改变结果。
- **整个模型**（案例 whole-model-tp）：每个行并行层都用逻辑 TP
  （C = 16），再加上分块和固定的词表并行 log_softmax，在
  TP 1、2、4、8 和 16 下，以及以 TP a 做 prefill、以 TP b 做 decode 时都给出同一个
  结果。反向中列并行的输入梯度也是跨 rank 求和：
  用同样的方式归约它们（NCCL all-reduce 在检查的全部 9 个点上都破坏了 L5）。
  代价：在通过 PCIe 连接的 8 个 GPU 上 TP 2–8 时，prefill 为 1.4–1.6x，decode 为 1.6–1.9x。逻辑 TP
  与单 GPU 是不同的算术（TP 1 时 84.7% 的 logits 不同），所以
  trainer 和引擎都必须使用它。
- **数据并行梯度**（案例 training-dp-fsdp，两个 8 GPU 节点）：在 Ring 下，
  桶布局会改变比特，而 torch DDP 在第一次迭代
  （以及每次重启）时用单个桶归约。Tree 或有序 all-reduce 可修复
  打包问题（1.05–1.20x）。reduce-scatter 的顺序跟随所属 rank。要在不同
  DP 度数之间达到 L5，把 batch 切成固定数量的逻辑分片（这里是 16），
  单独计算每个分片的梯度，并用一棵固定的树求和，
  这棵树要包含跨 rank 求和；把这个求和交给 NCCL 从 3 个 rank 起就失败，
  即使在 Tree 下也是如此。

## 11. 闭源或预编译 kernel

- 用 `check_batch` 在多种大小上摸清不变区域。填充到
  固定的 M 只有在每次调用真的都以该 M 运行时才有帮助：CUDA graph 分桶
  会在桶之间破坏这一点。
- 如果没有任何开关或区域成立，就替换该 kernel。否则保留一个逐行的
  包装作为调试参照，并把该算子排除在一致性声明之外。

## 12. 部署包装层：`torch.compile` 和 CUDA graph

在一种 GPU 型号上测得（[分页、varlen 和 compile 案例](../../../examples/attention_serving/REPORT.md#serving-layers-paged-kv-varlen-packing-compile-and-cuda-graphs)）：
- **把不变的 kernel 注册为不透明的自定义算子。** 直接调用时，
  Triton kernel 会让 Dynamo 在每个 batch 大小上重新编译：18 个图，即使用了
  `dynamic=True`。作为自定义算子：2 个。
- **编译可以保持 L2，但仍然改变比特。** Inductor 融合了 RMSNorm、
  残差相加和 SiLU·up，编译后的块在 65% 的
  元素上与 eager 不同。让每个会舍入的算子都不透明后，得到 compiled = eager。
  必须与 eager trainer 一致的引擎需要这样做。
- **CUDA graph 只有配合不变的 kernel 才保持不变性。** 填充到
  桶时，cuBLAS 对同一个序列在 7 个桶上给出 6 个不同的结果。
  在相同形状下，graph 重放本身给出 eager 的比特。
- **自定义算子有主机端开销：** 这里在小 batch 时每次调用约 0.65 ms，
  除非放进 graph。
- **编译代码内部的 fingerprint（指纹）**（[compile-graph-capture 案例](../../../handbook/1-pretraining/capture-compiled-code.md)）：
  `tools/graph_probe.py` 从编译代码和 graph 重放中记录 OpGuard 格式的
  fingerprint。给探针一个精确的逐点复制（`isolate=True`）：
  非逐点的使用者可能让 Inductor 切换到 `addmm` 并改变比特。在编译之前
  于 eager 中构建惰性 cache（RoPE 表）。

## 13. 采样

用基于计数器的生成器，以 (seed, absolute position, purpose, sample index) 为键生成噪声，
永远不要依赖生成器的运行状态。让
分布精确：整数权重 floor(exp(z−max)·2^S)，使求和与 cumsum
与顺序无关。并列按 (value desc, index asc) 用稳定排序排序；greedy
选最小索引。如果 token 必须与无草稿解码一致，就通过与
目标模型自身样本相等来验证推测解码。在一种 GPU 型号上测得
（[采样案例](../../../handbook/2-rl/sampling.md)；参考采样器
[tools/bi_sampling.py](../../../tools/bi_sampling.py)）。

## 14. LoRA 和多适配器批次

在 kernel 内为每一行选择它的适配器；永远不要为了
归约而重新分组行。基座 GEMM 用一个配置运行，shrink 和 expand 按固定的
rank 和 K 顺序进行，update 保持 fp32，并在一次 fp32 FMA 中与基座结果相加，
之后只做一次类型转换。trainer（一个适配器）和
引擎（混合适配器）使用同一个 kernel；固定缩放进入的位置；在 fp32
累加内部对 rank 做零填充；trainer 计算 log-probs 时关闭 dropout；当
引擎必须与未合并的 trainer 一致时，不要在 bf16 中合并。反向：dX 用同一个 kernel，
dA/dB 按每个适配器在其非零梯度行上按升序计算，不用
原子操作。在一种 GPU 型号上测得（[LoRA 案例](../../../handbook/2-rl/lora.md)）。

## 15. 训练步：反向 scatter、优化器、loss、恢复训练

- 把 `index_add_`、`scatter_add_`、embedding 反向和带权
  `bincount` 替换为写明的顺序：按索引排序，然后做升序 fp32
  分段求和。只开严格模式不算写明的顺序：它的 scatter 和
  embedding 反向会随行宽或索引数改变顺序。
- 固定优化器路径（fused、foreach 或 single-tensor）和 loss
  归约方式。mean 不是布局不变的。
- 保存并恢复一步会读取的每一项状态；该案例列出了 7 项。
- 在一种 GPU 型号上测得（[训练步案例](../../../handbook/1-pretraining/training-step.md)）；
  多 GPU 梯度见配方 10。

## 16. 跨引擎舍入（L4）

每侧各自的 L2 和 L3 不能带来 L4。写下两侧的舍入序列
（打包、bias 位置、norm 形式、RoPE 表、注意力、lm_head 和
log_softmax kernel），先检查引擎自身的 L1 和 L2，然后一次对齐一个
舍入点。工作流程见
[engine-parity skill](../../engine-parity/SKILL.md)。在一种 GPU 型号上用 vLLM 0.19.1
测得（[trainer 对比 vLLM 案例](../../../handbook/2-rl/engine-parity-vllm.md)）：
要达到相等，trainer 需要调用 vLLM 自己的 kernel。

## 已报告数值的来源

- Thinking Machines，"Defeating Nondeterminism in LLM Inference"（2025）：
  把批不变性作为推理非确定性的根因；不变的 RMSNorm、
  矩阵乘和注意力。
- Yichuan Wang，"Closing the Training–Inference Mismatch for Linear Attention in
  RL"：GDN 循环前向、用于 MoE router 的 batch-invariant `bmm`、2–3x
  trainer 代价。
- verl-omni issue #637：vLLM 逐算子与 HF 数值对齐（norm 转换顺序、
  残差相加精度、RoPE 公式、router 权重转换）；rollout
  吞吐约为 37%。
- IsoExec（SkyRL，vLLM 博客）：跨框架执行契约和
  并行不变的 kernel。

链接在手册的[参考文献](../../../handbook/references.md)中。上面的
观察来自 `tools/bi_check.py --demo`（结果在
`tools/` 下的 demo JSON 中）以及各案例研究，每个案例的脚本和
结果 JSON 都在 `examples/` 下。
