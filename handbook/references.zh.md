# 参考文献

本手册使用以下来源。标有 * 的链接来自我们的工作笔记，
发布前应重新核对。完整的分类列表正在兄弟仓库 `awesome-bitwise-alignment` 中整理。

## 工具与方法

- OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM
  Training, OSDI '26. https://www.usenix.org/conference/osdi26/presentation/zhou-ziming
- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025)：把
  批不变性视为推理非确定性的原因；不变的 RMSNorm、矩阵乘法
  和注意力。https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ *
- `batch_invariant_ops`（Thinking Machines）。https://github.com/thinking-machines-lab/batch_invariant_ops *
- PyTorch 可复现性说明。https://pytorch.org/docs/stable/notes/randomness.html
- cuBLAS 结果可复现性。https://docs.nvidia.com/cuda/cublas/#results-reproducibility *

## 强化学习训推对齐

- IsoExec: Unified Execution to Eliminate Trainer-Inference Mismatch in SkyRL
  （vLLM blog，2026-08-21）。https://vllm.ai/blog/2026-08-21-isoexec
- Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven
  （vLLM blog，2026-10-14）。https://vllm.ai/blog/2026-10-14-train-inference-consistency
- Yichuan Wang et al., Closing the Training–Inference Mismatch for Linear
  Attention in RL（异步强化学习，GDN）。https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/
- verl-omni issue #637，用于逐位 rollout/actor 一致性的 HF 数值模式。
  https://github.com/verl-project/verl-omni/issues/637
- Yingru Li et al., When Speed Kills Stability: Demystifying RL Collapse from the
  Training–Inference Mismatch. https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

## 本仓库中的测量

每个结果文件都放在它的代码旁边，它的报告写明了测量所用的软件栈。
结果文件及其硬件和软件的列表见[测得的覆盖范围](../suite/COVERAGE.md#result-files)；
哪些因素会让比特在软件栈之间发生变化，见[硬件覆盖](4-operations/hardware.md)。
