# 验证

一个 kernel 只有通过这个矩阵，才在某个级别上是 batch-invariant（批不变）的；
测量要在你实际发布的 GPU、dtype 和版本上进行。通过的标准是零个逐位不匹配。
检查工具是 [tools/bi_check.py](../../../tools/bi_check.py)。

## 比较比特，而不是数值

`torch.equal` 和 `==` 把 `-0.0` 和 `+0.0` 视为相等，并把 NaN 视为不等于自身。
改为比较相同宽度的整数视图：`bi_check.bit_equal(a, b)`。零的符号差异是真实的：
一个被 mask 的尾部在一条路径上加 `+0.0`、在另一条路径上加 `-0.0`，这就是一种变化，
它可能影响下游（`1/x`、`copysign`）。

## 矩阵

| 维度 | 取值 | 原因 |
|---|---|---|
| 批大小 M | 1, 2, 3, 4, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 1024, 2048；加上 SM 数量及其倍数 ±1；代码中出现的每个阈值；每个 CUDA graph 捕获大小 | 分派阈值位于 2 的幂、tile 的倍数和 SM 数量的倍数处 |
| 行的位置 | 第一行、中间、最后一行 | 对齐和向量宽度路径、尾部处理 |
| 同批内容 | 随机、全零、大值（×1e3） | 依赖内容的统计量：动态缩放、容量、top-k |
| 模式（类似注意力的算子） | prefill；逐 token decode；多种块大小的分块 prefill（包括不能整除 block 大小的）；多种 KV 页大小 | split-KV 和块迭代、按参数选择后端 |
| 重复 | 同一批至少运行 5 次 | 原子操作和竞争（L1） |
| 反向 | 当训练用到该算子时，在梯度上运行同一矩阵；另外比较一个序列单独运行与在批中右填充、填充位置输出梯度为零的情况；`warn_only=False` 的确定性模式 | 确定性反向的变体；dQ 中的原子操作；填充处理 |
| 布局 | 页大小 16、32、64、128，随机放置；不是 block 倍数的打包偏移和长度 | 块跟随页或打包，而不是绝对位置 |
| 并行度（L5） | TP 和 EP 1、2、4、8；多种 token 分片方式；1 KB–256 MB 消息上的集合通信；top-k ≥ 3 且 3 个或更多 rank | 按 rank 或分片分组的归约；跟随 chunk 边界的 ring 顺序 |
| 部署包装 | eager、使用部署设置的 `torch.compile`、部署大小下的 CUDA graph；编译结果与 eager 逐位比较 | 按形状特化、norm 和激活的再融合 |
| 跨引擎（L4） | 同样的行分别经过 trainer 的算子和引擎的算子 | 舍入点、kernel 是否相同 |

## 使用检查工具

```python
from bi_check import check_batch, check_modes, check_repeat

rep = check_batch(lambda x: my_gemm(x, W), rows)    # rows: [R, K] on the GPU
print(rep.summary())                                # PASS, or failing sizes

rep = check_modes({"prefill": run_prefill, "decode": run_decode,
                   "chunk-16": run_chunked(16), "chunk-7": run_chunked(7)})
```

`python tools/bi_check.py --demo` 检查几个原生 torch 算子，结果在
`tools/` 下的 demo JSON 中。以下为测得的结果，除非另有说明，均在一种 GPU 型号上用 torch 2.5.1
测得（多 GPU 的行：由另一种型号的 8 GPU 组成的节点）；细节（包括硬件）见案例研究和每个案例的
`examples/<case>/REPORT.md`。要在你自己的软件栈上重新检查每一行，运行
[回归测试套件](../../../handbook/4-operations/regression.md)：

| 算子 | 结果 |
|---|---|
| `torch.mm` bf16 (cuBLAS) | L1 是；L2 否：从 M=31 起失败（一行中 4/4096 个元素不同） |
| 逐行 matmul | L2（对照：按构造不变） |
| 用 torch 算子写的 RMSNorm、fp32 `log_softmax` | 在测试的形状上 L2 |
| SDPA causal | L3 否：decode 与 prefill ~10% 的元素不同；分块（显式 mask）与 prefill 相差 1 个元素 |
| SDPA causal，一个序列与其他序列同批（GQA 8/2，T=256） | L2 否：同一序列在批大小 3、8 和 16 时改变（17.6% 的元素），但在 2 时不变 |
| 固定分块的 Triton 注意力（`tools/bi_kernels.py`） | L2 和 L3 是：prefill、decode 和分块 7/16/64/100 逐位相等 |
| 它的无原子操作反向 | dQ/dK/dV 上 L1 和 L2 是：批大小 2–16，以及右填充到 256/320 |
| SDPA 反向 | flash：L1 否（dQ）。严格确定性的 flash：L1 是，L2 否，因为它的前向从批大小 3 起改变。严格确定性的 memory-efficient：在测试的形状上 L1 和 L2 是 |
| 分页 KV（页大小 16–128）和变长打包，固定分块注意力 | L2 和 L3 是；块跟随页或打包偏移的实现失败 |
| MLA，单一形式且每个部分都不变 | L2 和 L3 是；显式 prefill 配吸收形式 decode，与 trainer 相比从第 2 层起失败 |
| 稀疏注意力，单一配置的投影加稳定的 top-k | L2 和 L3 是；cuBLAS 投影会翻转选中的集合 |
| GDN 循环 / 分块 | 循环形式 L1–L3；分块形式只有在块边界位于绝对位置时才 L3 |
| MoE 固定槽位 combine，EP 1–16（两个 8 GPU 节点；[报告](../../../examples/moe_ep/REPORT.md)） | 单节点和跨两个节点 L5 是；逐 rank 预归约从 top-3 起 L5 失败 |
| NCCL all-reduce（8 和 16 个 rank，一个和两个节点） | 默认设置或固定 Ring 时 L2 否；一个或两个节点上用 Tree 时是；用 rank 顺序回退方案时是 |
| 编译后的 block（`torch.compile`） | L2 是；只有在每个会舍入的算子都不透明时才等于 eager |
| 编译代码和 CUDA graph 中的 fingerprint（指纹）探针 | 探针 = opguard_bridge fingerprint；原生 graph = eager；原生分桶 8/16 与 1 在 o_proj 处不同；bi 分桶 1/8/16 相等；bi_opaque_all 编译 = eager（3 种模式）；探针不改变编译后的 logits（隔离方式） |
| 采样：逐请求 generator / torch.cumsum / bi_sampling | generator：L2 是，抢占后重放、回滚、投机分块：否；cumsum：L2 否，在 V=151,936 时 L1 否；bi_sampling：L1–L3 是，卡方检验通过 |
| LoRA linear，单一配置的基础 GEMM + 按行选择的适配器，一次转换 | L1、L2、prefill/decode、trainer 与引擎：是；分组 cuBLAS 和 gather 后的 bmm 在 L2 失败；vLLM 风格的 split-K 原子操作在 fp32 缓冲区中 L1 失败 |
| 训练步，4 层 LM，20 步 | 默认：第一个差异在第 0 步（SDPA dQ）；严格和固定：跨进程相等 |
| Embedding 反向 | L1 是；L2 否：两种顺序在 3072 个索引处分开 |
| DP 梯度 all-reduce / FSDP 风格 reduce-scatter（模拟；扁平梯度、reduce-scatter、分片 AdamW），DP 1–16（两个 8 GPU 节点；[报告](../../../examples/training_dp/REPORT.md)） | 打包 L2：Ring 下否，Tree 或有序 all-reduce 下是；owner L2：NCCL reduce-scatter 否，有序时是；L5：任何逐 rank 批量反向都否；用逻辑分片时是，端到端包含 AdamW 和弹性恢复训练 |
| 整个模型（dense、MoE、wide、Qwen2 形状），bi kernel | 在所有引擎模式下 L1–L3；激活梯度 L2；原生：每个 log p 都不同，KL 1e-3 到 1.4e-2 |
| 整个模型，TP 1–16（8 GPU 的节点，16 时用两个节点；[报告](../../../examples/whole_model_tp/REPORT.md)） | 逻辑 TP C=16 + 固定顺序的词表 log_softmax：前向、引擎模式、不同 TP 下的 prefill/decode 以及激活梯度（TP 2、4）上 L5 是；原生 NCCL TP：在每个并行度上每个实际 log-prob 都不同 |
| vLLM 0.19.1 与 model.py，Qwen2.5-0.5B（[报告](../../../examples/cross_engine/REPORT.md)） | 默认、eager、Triton、FlashInfer、Flex：L1 是，L2 否（单独与批大小 64 相比，96% 的 logprob 不同）；VLLM_BATCH_INVARIANT=1：在所测 GPU 上 L2 否，该模式在那里把 lm_head 留在 cuBLAS 上（7%），把 lm_head 放到 vLLM 的 Triton GEMM 上时是；对一个调用 vLLM kernel 的 trainer，L4 是（0/76,155），仅 eager |

## 解读失败

| 模式 | 可能的机制 |
|---|---|
| 某个 M 以下通过，在它及以上失败，在每个位置和填充下都一样 | 按形状分派的 kernel 或 tile 配置（启发式阈值） |
| 在零散的大小上失败 | 由 tile 数量驱动的 split-K 或 stream-K；依赖 grid 大小的拆分 |
| 只在某些位置失败 | 依赖对齐的向量宽度；尾部处理 |
| 只在某些填充下失败 | 依赖内容的统计量：整批 amax、容量丢弃、整批上的 top-k |
| 同一批重复运行时失败 | 原子操作或竞争：L1 已破坏，先修复它 |
| 模式之间许多元素不同 | 不同的块迭代，或 decode 中的 split-KV |
| 模式之间少数元素不同 | 按调用参数选择了不同的后端，或部分 mask 块的处理不同 |
| eager 下通过，编译或 graph 下失败 | 按形状特化或再融合；把该算子注册为不透明 |
| 编译后是 L2，但与 eager 不同 | 再融合移动了舍入点（norm、激活）；把每个会舍入的算子都设为不透明 |
| 不同页大小或打包偏移之间不同，但各自自洽 | 计算块跟随布局，而不是绝对位置 |
| top-2 或 2 个 rank 时通过，从 3 起失败 | 一个依赖顺序的求和，因为两项的和满足交换律而被掩盖 |
| L1 和 L2 通过，但结果随并行度改变 | 按 rank 或分片分组的归约（L5） |
| 固定了算法的集合通信在更大的消息内部失败 | chunk 边界随消息大小移动，改变了每个元素的 ring 顺序 |
| 离散输出翻转（选中的 key、路由的专家、丢弃的 token） | 上游一个随批变化的分数或容量规则；先比较集合，再比较数值 |
| 稳定状态下通过，第一次迭代或恢复训练后失败 | torch DDP 在第一次迭代（以及每次重启）时用单个 bucket 归约；在 Ring 下那是另一种顺序 |
| 采样的 token 在不同批大小下通过，但在抢占、回滚、其他请求的 n 或投机分块之后改变 | 噪声来自 generator 的运行状态，而不是来自（seed、位置、用途、样本） |
| 每一侧都是 L2/L3，但 trainer 和引擎不同（L4） | 不同的舍入点：打包、bias、norm 形式、RoPE 表，或不同的 kernel；比较两侧的舍入序列 |

## 算子通过之后

1. 运行模型级检查：同一个序列单独运行，以及放在几种大小的批中运行，
   必须给出相同的逐 token 对数概率。
2. 对于 L4，trainer 和引擎在相同权重、staleness（策略滞后）为零时，
   必须给出相同的对数概率。
3. 如果不同，沿数据流用逐向量 fingerprint 定位（OpGuard，
   `OPGUARD_XOR_ROWS=1`），做法同强化学习部分。
4. 在有代表性的 prefill M 和 decode M（1–64）上测量吞吐，
   并与结果一起记录。
