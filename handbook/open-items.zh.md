# 待办事项

三条流程的每一页上，凡是会改变读者做法的缺口，都用一行"尚未证明"点出来。完整的列表放在本页，
包括本手册还没跑、跑不了或还没设计的所有事项。这些事项来自第 1–3 部分各主题页的"局限"小节、
[回归](4-operations/regression.md#limits)、[硬件覆盖](4-operations/hardware.md#untested)和
[审计](../AUDIT.md)。细节留在各主题页上，本页只保留列表。某一项完成后，从这里删掉，并更新它的主题页。

## 需要硬件，或需要这里装不上的软件

据 OpGuard 的作者报告
（[OpGuard 论文](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)），
OpGuard 本身（采集、fingerprint、找首个分歧点的比较）已经在前四项列出的平台上用于生产环境的
LLM 训练。还没做的，是本手册自己在这些平台上的测量，包括 batch-invariant kernel 和各种修复、
固定的集合通信树，以及引擎一致性的那套搭建。

- 其他 GPU 架构和其他加速器：这里一台都没有。测过哪些软件栈，以及 OpGuard 自己的结果和本手册的
  结果怎么区分，见[硬件覆盖](4-operations/hardware.md#untested)。
- FP8 tensor-core GEMM 及其精度提升间隔、NVFP4；其他 GPU 架构上的 FlashAttention 3、FlashMLA 和
  FlashInfer。目前的 FP8 scale 都是模拟的。
- NVLink、NVSwitch 和 NVLS；RDMA、InfiniBand 和 RoCE；两个以上的节点。本手册所有多 GPU 的结果，
  节点内走的都是 PCIe，两个节点之间走的是普通以太网 socket。
- 手册里的 kernel 在其他版本的 torch、Triton、NCCL 和 CUDA 上的表现。目前只测了一套软件栈，
  跨引擎案例另外用了 vLLM 自带的软件栈，verl × FSDP2 × vLLM 案例也有自己的一套软件栈
  （[报告](../examples/verl_fsdp2_vllm/REPORT.md)）。
- L4 下的 SGLang：这里的引擎环境缺少 SGLang 需要的 CUDA 库。
- 在引擎里使用真实的预训练 checkpoint 或 MoE checkpoint：一个真实的 MoE checkpoint 已经在完整的
  RL 循环里、TP 1 下达到 L4（[报告](../examples/verl_fsdp2_vllm/REPORT.md)）。带 TP 或 EP 的 L4、
  prefix caching 和 speculative decoding 还没有验证：这需要下载权重。
  在[整模型案例](../examples/whole_model/REPORT.md)里，只用随机权重检查过一个真实架构，而且只检查了其中几层。
- 把 `fla`（flash-linear-attention）作为 GDN kernel 的生产基线：还没有安装。

## 需要在现有机器上花 GPU 时间

- 真正的 `torch.distributed.fsdp`：probe 已经写好，但从来没有启动过。
  [DP 案例](1-pretraining/data-parallel.md)是模拟出来的 FSDP 风格（扁平梯度、reduce-scatter、
  分片的 AdamW）。那里也还没测 bf16 梯度、混合精度和 loss scaling。
- 套件的 full 模式：从来没有通过 `tools/bi_suite.py` 运行过。最近加入的几个条目的 full 命令没有
  验证过，目前只有 quick 模式的基线。
- CI 工作流（`.github/workflows/bi-suite.yml`）从来没有运行过，需要在自托管的 GPU runner 上跑一次。
- `perfetto/token_match.py`（OpGuard）还从来没有在 [rl-loop](2-rl/rl-loop.md) 导出的结果上运行过。
- 强化学习重放：batch-invariant 的重放还没有和按原始 batch 组成的重放比较过；混合了多个策略版本
  的 batch 也还没有重新采集。
- OpGuard 的 `tests/test_xor_vectors.py`：只在 GPU 上跑的那些路径（原生 kernel 对比回退实现、
  每种 dtype 的 v1 采集、经过 tracer 的工作负载），在最近一次修改 XOR 向量之后没有重新运行。
- 审计里的 GPU 抽查重跑：对照 `~/bi-check-suite` 的校验和，以及在第二种 GPU 上运行
  `lora_bi.py --quick` 和 `sampling_bi.py --quick` 并与它的基线比较
  （[AUDIT.md 第 5 节](../AUDIT.md#5-gpu-re-run)）。
- 有几个原生 kernel 的失败（kernel 案例、LoRA、MLA、MoE）在第二种 GPU 上用 quick 尺寸跑时反而通过了，
  需要在那里用 full 尺寸再跑一遍。在那之前，这些失败只算在第一种 GPU 上成立
  （[测得的覆盖范围](../suite/COVERAGE.md)）。
- 用 OpGuard 做预训练对齐时还没测的部分：跨多个 rank 的采集、CUPTI 和 CUDA-API 两种调用点后端，
  以及 fused AdamW 与 foreach 版本的对比。
- 只在 eager 下测过的 kernel，用 `torch.compile` 和 CUDA graph 时的结果：注意力反向、paged kernel
  和 varlen kernel、线性注意力、LoRA、MLA 和稀疏注意力、采样、整个模型，以及 TP。
- 还没跑过的反向路径：EP 下的 MoE、TP 的参数梯度，以及 KV cache 路径。

## 需要和引擎或框架集成

- 在 vLLM、SGLang、verl 和 OpenRLHF 里加上写记录的 hook（`RecordWriter.forward`、`rollout_batch`、
  `rows`）。目前只有 `examples/rl_loop/rl_core.py` 里的玩具引擎写出过记录。
- 与 vLLM 匹配的 trainer 反向：vLLM 的 Triton GEMM、RMSNorm、`log_softmax` 和它自己构建的
  FlashAttention 在这里都没有反向，所以匹配 vLLM 的 trainer 只有前向。
- 与编译后的 vLLM 对齐：用了 torch.compile 和 CUDA graph 的 vLLM（这是它的默认设置）还没有对齐。
  要在那里定位差异，需要在编译后的图里用 OpGuard 的逐向量 fingerprint，这还没有在任何引擎上
  演示过。
- 超出已测案例的 vLLM L4：prefix caching、speculative decoding、带温度的采样、batch-invariant 模式
  下的 CUDA graph、TP、MoE 和 LoRA。
- kernel 案例与 trainer 之间的 L4：MoE（combine 和 EP）、MLA 和稀疏注意力、线性注意力，以及
  注意力 kernel。
- DeepSpeed ZeRO-1/2/3、Megatron 的 kernel 和它自定义的 all-reduce。

## 需要先做设计

- 盲测：让一个不知道事先植入了什么原因的 agent 按 `alignment-pipeline` 走一遍（预训练和强化学习
  两条路径都要走），同时记下它启动了什么、调用了哪些 harness。到目前为止，所有端到端案例都是
  已知答案的演示。
- fused AdamW 和 fused SGD：它们确切的求和顺序还没有弄清楚。我们对它们顺序的建模很接近，但并不
  精确（[训练 step](1-pretraining/training-step.md)）。
- OpGuard 的覆盖范围：`torch.nn.Linear` 内部的 GEMM 不会被采集，因为它们是从 torch 自己的文件里
  启动的；模型代码里的单字母变量名也会妨碍包装
  （[预训练案例](1-pretraining/localize-with-opguard.md)）。
- 还没写的 kernel：递归 GDN 前向对应的反向、paged 注意力的反向，以及用于长 decode 的固定大小 split-KV。
- MoE 路由规则：在 expert 所在的 rank 上按 capacity 丢弃 token，以及分数完全相同时按
  (score, expert id) 决定先后。
