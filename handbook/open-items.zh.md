# 待办事项

三条流程的每一页，都用一行简短的"尚未证明"写出会改变读者做法的事项。
本页保留完整列表：本手册还没运行、无法运行或尚未设计的所有事项。来源：第 1–3
部分各专题页面的“局限”小节、[回归](4-operations/regression.md#limits)、[硬件覆盖](4-operations/hardware.md#untested)以及
[审计](../AUDIT.md)。细节留在各专题页面；本页只保留列表。某一项完成后，从这里删除，并更新它的专题页面。

## 需要硬件（或这里无法安装的软件）

OpGuard 本身（捕获、fingerprint、首个分歧点比较）已经在前四项所列的平台上用于生产环境的 LLM 训练，
这是它的作者报告的（[OpGuard 论文](https://www.usenix.org/conference/osdi26/presentation/zhou-ziming)）。待办的是本手册自己在那些平台上的测量：
它的 batch-invariant kernel 和修复、固定的集合通信树，以及引擎一致性的搭建。

- 其他 GPU 架构和其他加速器：这里没有。测过的软件栈和上面这种区分见
  [硬件覆盖](4-operations/hardware.md#untested)。
- FP8 tensor-core GEMM 及其提升间隔、NVFP4；其他 GPU 架构上的 FlashAttention 3、
  FlashMLA 和 FlashInfer（FP8 缩放因子是模拟的）。
- NVLink、NVSwitch、NVLS；RDMA、InfiniBand、RoCE；两个以上节点。本手册所有
  多 GPU 结果在节点内走 PCIe，在两个节点之间走普通以太网 socket。
- 手册中的 kernel 在其他 torch、Triton、NCCL 和 CUDA 版本上的情况（只测了
  一个软件栈，跨引擎案例另加 vLLM 自带的软件栈）。
- L4 下的 SGLang：这里的引擎环境缺少 SGLang 需要的 CUDA 库。
- 在引擎中使用真实的预训练或 MoE 检查点（带 TP/EP 的 L4、前缀缓存、
  投机解码）：需要下载权重。一个真实架构只用随机权重检查过其中几层
  （[整模型报告](../examples/whole_model/REPORT.md)）。
- 把 `fla`（flash-linear-attention）作为 GDN kernel 的生产基线：
  未安装。

## 需要现有机器上的 GPU 时间

- 真正的 `torch.distributed.fsdp`：探针已写好，从未启动。
  [DP 案例](1-pretraining/data-parallel.md)是 FSDP 风格的（模拟：扁平梯度、
  reduce-scatter、分片 AdamW）。那里也未测试：bf16 梯度、混合精度、loss scaling。
- 套件完整模式：从未通过 `tools/bi_suite.py` 运行；最后加入的那些条目的完整命令
  未经验证，只有 quick 基线。
- CI 工作流（`.github/workflows/bi-suite.yml`）：从未运行；需要在自托管
  GPU runner 上跑一次。
- `perfetto/token_match.py`（OpGuard）在 [rl-loop](2-rl/rl-loop.md)
  导出结果上：从未运行。
- 强化学习重放：batch-invariant 与按原始组成重放的结果未作比较；
  混合版本批次未重新捕获。
- OpGuard `tests/test_xor_vectors.py`：仅 GPU 的路径（原生 kernel 对比
  回退实现、每种 dtype 的 v1 捕获、经过 tracer 的工作负载）在最近一次
  XOR 向量改动后没有重新运行。
- 审计中的 GPU 抽查重跑：对照 `~/bi-check-suite` 的校验和，以及在第二种 GPU 上
  运行 `lora_bi.py --quick` 和 `sampling_bi.py --quick` 并对照其基线
  （[AUDIT.md 第 5 节](../AUDIT.md#5-gpu-re-run)）。
- 原版失败项在第二种 GPU 上的完整尺寸（它们在 quick 尺寸下在那里通过：
  kernel 案例、LoRA、MLA、MoE）：这些失败只针对第一种 GPU 声明
  （[测得的覆盖范围](../suite/COVERAGE.md)）。
- 通过 OpGuard 做预训练：多 rank 捕获、CUPTI 和 CUDA-API
  调用点后端、fused AdamW 对比 foreach。
- 只在 eager 下测过的 kernel 在 `torch.compile` 和 CUDA graph 下的比特：
  注意力反向、paged 和 varlen kernel、线性注意力、LoRA、MLA 和
  稀疏注意力、采样、整个模型、TP。
- 未运行的反向路径：EP 下的 MoE、TP 参数梯度、KV cache 路径。

## 需要引擎或框架集成

- 为 vLLM、SGLang、verl 和 OpenRLHF 提供记录钩子（`RecordWriter.forward`、`rollout_batch`、`rows`）。
  记录只从 `examples/rl_loop/rl_core.py` 中的玩具引擎写出过。
- 与 vLLM 匹配的 trainer 反向：vLLM 的 Triton GEMM、RMSNorm、`log_softmax`
  及其 FlashAttention 构建在这里都没有反向，因此匹配的 trainer
  只有前向。
- 编译模式下的 vLLM 一致性：编译和 CUDA graph 模式的 vLLM（其默认模式）没有
  匹配；定位它需要在编译图内部使用 OpGuard 逐向量 fingerprint（指纹），
  这还没有在任何引擎上展示过。
- 超出已测案例的 L4 vLLM：前缀缓存、投机解码、温度采样、batch-invariant
  模式下的 CUDA graph、TP、MoE、LoRA。
- kernel 案例对照 trainer 的 L4：MoE（combine 和 EP）、MLA 和
  稀疏注意力、线性注意力、注意力 kernel。
- DeepSpeed ZeRO-1/2/3、Megatron 的 kernel 和自定义 all-reduce。

## 需要设计工作

- 盲测：一个不被告知预设原因的 agent 按照 `alignment-pipeline`
  （预训练和强化学习两条路径）执行，记录启动和 harness 调用。
  到目前为止所有端到端案例都是已知答案的。
- Fused AdamW 和 fused SGD：它们确切的求和顺序尚未确定；
  顺序模型接近但不精确（[训练步](1-pretraining/training-step.md)）。
- OpGuard 覆盖：`torch.nn.Linear` 内部的 GEMM 不会被捕获（从
  torch 自己的文件启动），模型代码中的单字母变量名会阻碍包装
  （[预训练案例](1-pretraining/localize-with-opguard.md)）。
- 尚未构建的 kernel：循环 GDN 前向的反向、paged
  注意力反向、用于长 decode 的固定大小 split-KV。
- MoE 路由规则：在专家 rank 上执行基于容量的丢弃，以及针对完全并列的
  (score, expert id) 决胜规则。
