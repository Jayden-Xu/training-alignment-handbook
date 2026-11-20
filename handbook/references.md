# References

The handbook uses these sources. Links marked * come from our working notes and
should be re-checked before publication. The comprehensive, categorized list is
being built in the sibling repository `awesome-bitwise-alignment`.

## Tools and methods

- OpGuard: Bitwise Alignment for Precise and General Debugging of Production LLM
  Training, OSDI '26. https://www.usenix.org/conference/osdi26/presentation/zhou-ziming
- Thinking Machines, "Defeating Nondeterminism in LLM Inference" (2025): batch
  invariance as the cause of inference nondeterminism; invariant RMSNorm, matmul
  and attention. https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ *
- `batch_invariant_ops` (Thinking Machines). https://github.com/thinking-machines-lab/batch_invariant_ops *
- PyTorch reproducibility notes. https://pytorch.org/docs/stable/notes/randomness.html
- cuBLAS results reproducibility. https://docs.nvidia.com/cuda/cublas/#results-reproducibility *

## RL train–inference alignment

- IsoExec: Unified Execution to Eliminate Trainer-Inference Mismatch in SkyRL
  (vLLM blog, 2026-08-21). https://vllm.ai/blog/2026-08-21-isoexec
- Finish RL Training Faster: Bitwise-Consistent, High-Performance, Agent-Driven
  (vLLM blog, 2026-10-14). https://vllm.ai/blog/2026-10-14-train-inference-consistency
- Yichuan Wang et al., Closing the Training–Inference Mismatch for Linear
  Attention in RL (async RL, GDN). https://yichuan-w.github.io/blog/GDN-train-inference-mismatch-asyncRL/
- verl-omni issue #637, HF-numerics mode for bitwise rollout/actor parity.
  https://github.com/verl-project/verl-omni/issues/637
- Yingru Li et al., When Speed Kills Stability: Demystifying RL Collapse from the
  Training–Inference Mismatch. https://yingru.notion.site/When-Speed-Kills-Stability-271211a558b7808d8b12d403fd15edda

## Measurements in this repository

Every result file sits next to its code, and its report names the stack it was
measured on. The list of result files with their hardware and software is in
[measured coverage](../suite/COVERAGE.md#result-files); what changes bits from
one stack to another is in [hardware coverage](4-operations/hardware.md).
