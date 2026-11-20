# 验证

一个 kernel 只有通过下面这个矩阵，才能算在某个级别上是 batch-invariant 的，
而且要在你实际发布时用的 GPU、dtype 和版本上测。通过的标准是一个逐 bit 的不匹配都没有。
检查工具是 [tools/bi_check.py](../../../tools/bi_check.py)。

## 比较 bit，而不是比较数值

`torch.equal` 和 `==` 认为 `-0.0` 和 `+0.0` 相等，又认为 NaN 不等于它自己。
所以要改成比较同样宽度的整数视图，用 `bi_check.bit_equal(a, b)`。零的符号不同是真实的差异：
如果一段被 mask 掉的尾部在一条路径上加的是 `+0.0`，在另一条路径上加的是 `-0.0`，
这个 kernel 的结果就会变，而且这个差异可能影响下游（`1/x`、`copysign`）。

## 测试矩阵

| 维度 | 取值 | 能发现什么问题 |
|---|---|---|
| batch size M | 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1024, 2048；另外加上 SM 数量及其倍数，各自 ±1；代码里看到的每个阈值；CUDA graph 录制时用的每一种尺寸 | 分派阈值，它们通常落在 2 的幂、tile 大小的倍数和 SM 数量的倍数上 |
| 行的位置 | 第一行、中间、最后一行 | 取决于对齐或向量宽度的代码路径，以及尾部处理 |
| 其他行的内容 | 随机值、全零、很大的值（×1e3） | 取决于内容的统计量：动态缩放、capacity、top-k |
| 模式（类似注意力的算子） | prefill；逐 token 的 decode；用几种 chunk 大小做 chunked prefill，包括不能整除 block 大小的；几种 KV page 大小 | split-KV 和 block 的遍历方式，以及按调用参数选择的后端 |
| 重复 | 同一个 batch 至少跑 5 次 | 原子操作和竞争（L1） |
| 反向 | 训练用到这个算子时，在梯度上跑同一个矩阵；另外把一个序列单独运行的结果，和它在右边补 padding 后放进 batch 运行的结果做比较，padding 位置的输出梯度设为零；`warn_only=False` 的确定性模式 | 确定性模式下行为不同的反向 kernel、dQ 里的原子操作、padding 的处理 |
| 布局 | page 大小取 16、32、64、128，随机放置；不是 block 整数倍的 pack 偏移和长度 | block 跟着 page 或 pack 走，而不是跟着绝对位置走 |
| 并行度（L5） | TP 和 EP 取 1、2、4、8；几种不同的 token 切分方式；在 1 KB–256 MB 的消息上做集合通信；top-k ≥ 3，rank 数在 3 个以上 | 按 rank 或 shard 分组的归约，以及跟着 chunk 边界走的 ring 顺序 |
| 部署时的包装 | eager、用部署设置的 `torch.compile`、部署大小下的 CUDA graph；把编译后的输出和 eager 逐 bit 比较 | 按形状特化，以及编译器把 norm 和激活重新融合 |
| 跨引擎（L4） | 同样的行分别经过 trainer 的算子和引擎的算子 | 舍入点，以及两端跑的是不是同一个 kernel |

## 使用检查工具

```python
from bi_check import check_batch, check_modes, check_repeat

rep = check_batch(lambda x: my_gemm(x, W), rows)    # rows: [R, K] on the GPU
print(rep.summary())                                # PASS, or failing sizes

rep = check_modes({"prefill": run_prefill, "decode": run_decode,
                   "chunk-16": run_chunked(16), "chunk-7": run_chunked(7)})
```

`python tools/bi_check.py --demo` 会检查几个原生 torch 算子，结果写在
`tools/` 下的一个 demo JSON 里。下表的实测结果，除非某一行另有说明，都是在一种 GPU 上
用 torch 2.5.1 测的（多 GPU 的那几行用的是另一种 GPU，每个节点 8 块）。细节（包括硬件）
见案例研究和每个案例的 `examples/<case>/REPORT.md`。要在你自己的软件栈上重新检查每一行，
运行[回归套件](../../../handbook/4-operations/regression.md)。

| 算子 | 结果 |
|---|---|
| `torch.mm` bf16（cuBLAS） | L1 成立。从 M=31 起 L2 不成立（一行里 4/4096 个元素不同） |
| 一次只算一行的 matmul | L2 成立（这是对照组，按构造就不变） |
| 用 torch 算子拼出来的 RMSNorm，以及 fp32 的 `log_softmax` | 在测过的形状上 L2 成立 |
| SDPA causal | L3 不成立：decode 和 prefill 相比有 ~10% 的元素不同；带显式 mask 的 chunked prefill 和 prefill 相比有 1 个元素不同 |
| SDPA causal，一个序列和其他序列放在同一个 batch 里（GQA 8/2，T=256） | L2 不成立：同一个序列在 batch 为 3、8 和 16 时结果变了（17.6% 的元素），batch 为 2 时没变 |
| 固定分块的 Triton 注意力（`tools/bi_kernels.py`） | L2 和 L3 成立：prefill、decode 以及 chunk 大小为 7/16/64/100 时，结果逐 bit 一致 |
| 它的反向（不用原子操作） | dQ/dK/dV 上 L1 和 L2 成立，覆盖 batch 2–16，以及右边补 padding 到 256/320 |
| SDPA 反向 | flash 的 L1 不成立（dQ）。严格确定性模式下的 flash，L1 成立、L2 不成立，因为它的前向从 batch 3 起就变了。严格确定性模式下的 memory-efficient，在测过的形状上 L1 和 L2 都成立 |
| paged KV（page 大小 16–128）和 varlen packing，配合固定分块的注意力 | L2 和 L3 成立。block 跟着 page 或 pack 偏移走的实现不成立 |
| 只用一种形式、每一部分都不变的 MLA | L2 和 L3 成立。显式 prefill 配吸收式 decode，从第 2 层起就和 trainer 对不上 |
| 稀疏注意力，投影只用一种配置，并用稳定的 top-k | L2 和 L3 成立。用 cuBLAS 做投影，会让选中的集合发生翻转 |
| GDN 的递归和分块两种形式 | 递归形式 L1–L3 都成立。分块形式只有在 chunk 边界落在绝对位置上时，L3 才成立 |
| MoE 固定槽位的 combine，EP 1–16（两个 8 GPU 节点；[报告](../../../examples/moe_ep/REPORT.md)） | 单节点和跨两个节点时 L5 都成立。每个 rank 先局部归约的做法，从 top-3 起 L5 不成立 |
| NCCL all-reduce（8 和 16 个 rank，一个和两个节点） | 用默认设置或固定 Ring 时 L2 不成立。在一个或两个节点上用 Tree 时成立；用按 rank 顺序求和的回退方案也成立 |
| 编译后的 block（`torch.compile`） | L2 成立。只有每个会舍入的算子都不透明时，结果才和 eager 相同 |
| torch.compile 和 CUDA graph 里的 fingerprint probe | probe 给出的 fingerprint 和 opguard_bridge 的相同；原生 graph 和 eager 相同。原生的分桶 8/16 和分桶 1 在 o_proj 处不同，而 bi 分桶 1/8/16 都相同。用 bi_opaque_all 时，编译结果和 eager 相同（3 种模式）。隔离后的 probe 不改变编译后的 logits |
| 采样：逐请求的 generator、torch.cumsum 和 bi_sampling | 逐请求的 generator 满足 L2，但在抢占、回滚和 speculative decoding 分块之后无法重放。cumsum 不满足 L2，在 V=151,936 时连 L1 也不满足。bi_sampling 满足 L1–L3，卡方检验也通过 |
| LoRA linear，base 只用一种配置，每行选自己的 adapter，只做一次 cast | L1、L2、prefill/decode、trainer 对引擎都成立。分组的 cuBLAS 和 gather 之后的 bmm 不满足 L2。vLLM 式的 split-K 原子操作在 fp32 缓冲区里不满足 L1 |
| 训练 step，4 层的 LM，跑 20 步 | 默认设置下，第一个差异出现在第 0 步（SDPA dQ）。打开严格模式并换上修好的 kernel 后，不同进程之间结果相等 |
| Embedding 反向 | L1 成立。L2 不成立：在 3072 个索引处，kernel 会在两种顺序之间切换 |
| DP 梯度 all-reduce 和 FSDP 风格的 reduce-scatter（模拟实现：扁平梯度、reduce-scatter、按 shard 存放的 AdamW），DP 1–16（两个 8 GPU 节点；[报告](../../../examples/training_dp/REPORT.md)） | 按桶打包时，Ring 下 L2 不成立，Tree 或有序 all-reduce 下成立。按行的归属看，NCCL reduce-scatter 的 L2 不成立，有序版本成立。L5 方面，任何按 rank 分批算反向的做法都不成立；用逻辑 shard 时成立，而且端到端覆盖了 AdamW 和弹性恢复训练 |
| 整个模型（dense、MoE、wide、Qwen2 形状），用 bi kernel | 在所有引擎模式下 L1–L3 都成立，激活梯度满足 L2。用原生 kernel 时每个 logprob 都不同，KL 在 1e-3 到 1.4e-2 之间 |
| 整个模型，TP 1–16（每个节点 8 块 GPU，16 时用两个节点；[报告](../../../examples/whole_model_tp/REPORT.md)） | 用逻辑 TP C=16 加上固定顺序的词表 log_softmax，L5 在前向、各种引擎模式、不同 TP 下的 prefill 和 decode，以及激活梯度（TP 2、4）上都成立。用原生 NCCL TP 时，每种并行规模下每个实际 token 的 logprob 都不同 |
| vLLM 0.19.1 对 model.py，Qwen2.5-0.5B（[报告](../../../examples/cross_engine/REPORT.md)） | 默认、eager、Triton、FlashInfer 和 Flex 都满足 L1、不满足 L2（单独跑和 batch 64 相比，96% 的 logprob 不同）。设置 VLLM_BATCH_INVARIANT=1 后，在测过的 GPU 上 L2 仍不成立，因为这个模式在那里把 lm_head 留在 cuBLAS 上（7%）；把 lm_head 换到 vLLM 的 Triton GEMM 上就成立了。对一个调用 vLLM kernel 的 trainer，L4 成立（0/76,155），但只在 eager 模式下 |

## 解读失败

| 规律 | 可能的机制 |
|---|---|
| 某个 M 以下通过，到这个 M 及以上就失败，而且不管在哪个位置、其他行填什么内容都一样 | 按形状分派的 kernel 或 tile 配置（启发式阈值） |
| 在零散的几个大小上失败 | 由 tile 数量驱动的 split-K 或 stream-K，或者随 grid 大小变化的拆分 |
| 只在某些位置失败 | 取决于对齐的向量宽度，或者尾部处理 |
| 只在其他行填某些内容时失败 | 取决于内容的统计量：整个 batch 上的 amax、按 capacity 丢 token、在整个 batch 上取 top-k |
| 同一个 batch 重复运行就失败 | 原子操作或竞争。L1 已经坏了，先修它 |
| 不同模式之间有很多元素不同 | block 的遍历方式不同，或者 decode 用了 split-KV |
| 不同模式之间只有少数元素不同 | 调用参数让它选了另一个后端，或者对部分被 mask 的 block 处理方式不同 |
| eager 下通过，编译或用 graph 后失败 | 按形状特化，或者编译器做了融合；把这个算子注册成不透明的 |
| 编译后满足 L2，但和 eager 不同 | 融合挪动了舍入点（norm、激活）；把每个会舍入的算子都做成不透明的 |
| 不同 page 大小或 pack 偏移之间结果不同，但每种设置各自是一致的 | 计算用的 block 跟着布局走，而不是跟着绝对位置走 |
| top-2 或 2 个 rank 时通过，从 3 起失败 | 求和依赖顺序，只是两项相加满足交换律，把问题藏住了 |
| L1 和 L2 都通过，但结果随并行规模变化 | 按 rank 或 shard 分组的归约（L5） |
| 固定了算法的集合通信，在更大的消息里失败 | chunk 边界随消息大小移动，改变了每个元素在 ring 上的求和顺序 |
| 离散的输出发生翻转（选中的 key、路由到的 expert、被丢掉的 token） | 上游某个分数随 batch 变化，或者有 capacity 规则；先比较集合，再比较数值 |
| 稳定运行时通过，第一次迭代或恢复训练后失败 | torch DDP 在第一次迭代（以及每次重启后）用单个桶做归约，在 Ring 下这就是另一种顺序 |
| 采样出的 token 在不同 batch size 下一致，但在抢占、回滚、其他请求的 n 变化或 speculative decoding 分块之后变了 | 噪声来自 generator 的运行状态，而不是由（seed、位置、用途、样本序号）决定 |
| 两端各自都满足 L2/L3，但 trainer 和引擎的结果不同（L4） | 某个舍入点不同（packing、bias、norm 形式、RoPE 表），或者 kernel 不同；对照两端的舍入顺序 |

## 算子通过之后

1. 做模型级别的检查：同一个序列单独运行，和放进几种大小的 batch 里运行，
   每个 token 的 logprob 都必须相同。
2. 对于 L4，trainer 和引擎在权重相同、staleness 为零时，
   给出的 logprob 必须相同。
3. 如果不同，就沿着数据流用逐向量的 fingerprint 定位差异（OpGuard，
   `OPGUARD_XOR_ROWS=1`），做法和 RL 部分一样。
4. 在有代表性的 prefill M 和 decode M（1–64）上测吞吐，
   和结果一起记下来。
