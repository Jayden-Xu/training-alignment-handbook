---
hide:
  - toc
---

# 训练对齐手册

如何让两次训练或推理运行逐位一致，并找到它们第一次不一致的地方。

## 三条流程

本手册教三条工作流程。其中两条从一个症状出发：两次本应一致的预训练运行不一致（第 1 部分）；
或者 rollout 引擎和 trainer 对同样的 token 给出不同的对数概率（第 2 部分）。
如果原因是某个算子的结果取决于它周围的 batch，两条流程的结局相同：
把这个算子交给第三条流程，让它满足不变性（第 3 部分）。
[第 0 部分](0-foundations/README.md)讲三条流程共同依赖的概念。

<figure class="flows" aria-label="三条流程" markdown>
<p class="flows-base" markdown>建立在[第 0 部分：基础](0-foundations/README.md)之上：L1–L5 各个级别，以及在哪里比较、怎样比较。</p>
<div class="flow-lane" data-lane="pre" data-side="above" markdown>
<p class="flow-head" markdown><span class="flow-part">第 1 部分</span> [预训练](1-pretraining/README.md) <span class="flow-goal">找到两次运行分开的地方</span></p>
<ol class="flow-steps" style="--steps: 7" markdown>
<li markdown="span">[确定性与预检](1-pretraining/determinism-and-preflight.md)<span class="flow-desc">打开让运行可重复的设置，并检查两次运行确实都用上了。</span></li>
<li markdown="span">[数据对齐](1-pretraining/data-alignment.md)<span class="flow-desc">检查两次运行的每个 rank 在每一步拿到的 batch 都相同。</span></li>
<li class="flow-handoff" markdown="span">[用 OpGuard 定位](1-pretraining/localize-with-opguard.md)<span class="flow-desc">采集两次运行，找到第一个出现差别的算子，从它走到原因，修复并确认。</span></li>
<li markdown="span">[训练步](1-pretraining/training-step.md)<span class="flow-desc">原因在反向、优化器、损失或恢复训练时：让它们逐位重复。</span></li>
<li markdown="span">[数据并行](1-pretraining/data-parallel.md)<span class="flow-desc">原因是跨 rank 求和或分片的梯度时：让它保持相等。</span></li>
<li markdown="span">[编译代码](1-pretraining/capture-compiled-code.md)<span class="flow-desc">作业运行 `torch.compile` 或 CUDA graph 时：在 eager 采集看不到的内部取 fingerprint。</span></li>
<li markdown="span">[SDC](1-pretraining/sdc.md)<span class="flow-desc">软件已被排除时：找出静默返回错误值的硬件。</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="kernel" data-delay="1" data-target markdown>
<p class="flow-head" markdown><span class="flow-part">第 3 部分</span> [batch-invariant kernel](3-kernels/README.md) <span class="flow-goal">让一个算子满足不变性</span></p>
<ol class="flow-steps" markdown>
<li markdown="span">[GEMM 与量化 GEMM](3-kernels/gemm-and-quantized.md)<span class="flow-desc">对每个 batch size 固定同一种 tile 和归约顺序。</span></li>
<li markdown="span">[注意力](3-kernels/attention.md)<span class="flow-desc">prefill、decode 和反向对 key 用同一种切分。</span></li>
<li markdown="span">[MLA 与稀疏注意力](3-kernels/mla-and-sparse.md)<span class="flow-desc">对压缩后的 key 和 top-k 选择做同样的事。</span></li>
<li markdown="span">[线性注意力](3-kernels/linear-attention.md)<span class="flow-desc">循环状态用同一种分块顺序。</span></li>
<li markdown="span">[MoE](3-kernels/moe.md)<span class="flow-desc">路由、专家 GEMM 和 combine，与哪些 token 同在一个 batch 无关。</span></li>
<li markdown="span">[集合通信](3-kernels/collectives.md)<span class="flow-desc">跨 GPU 固定归约顺序。</span></li>
</ol>
</div>
<div class="flow-lane" data-lane="rl" data-side="below" markdown>
<p class="flow-head" markdown><span class="flow-part">第 2 部分</span> [强化学习训推](2-rl/README.md) <span class="flow-goal">让 rollout 与 trainer 一致</span></p>
<ol class="flow-steps" markdown>
<li class="flow-handoff" markdown="span">[整个模型](2-rl/whole-model.md)<span class="flow-desc">让一侧对一条序列给出相同的 logits：单独计算、放在 batch 里、任何服务模式下都一样。</span></li>
<li markdown="span">[并行](2-rl/parallelism.md)<span class="flow-desc">在张量并行和 prefill/decode 分离部署下保持这一点。</span></li>
<li markdown="span">[采样](2-rl/sampling.md)<span class="flow-desc">让选出的 token 能从 logits 和种子重放出来。</span></li>
<li markdown="span">[LoRA](2-rl/lora.md)<span class="flow-desc">一个 batch 混用多个适配器时，保持不变性。</span></li>
<li class="flow-handoff" markdown="span">[引擎一致性](2-rl/engine-parity-vllm.md)<span class="flow-desc">让真实引擎按 trainer 的舍入步骤计算，两侧才能一致。</span></li>
<li markdown="span">[RL 循环](2-rl/rl-loop.md)<span class="flow-desc">记录每次 rollout，把 staleness 和不一致分开，离线定位差距。</span></li>
</ol>
</div>
<p class="flows-next" markdown>之后，[第 4 部分：运维](4-operations/README.md)在软件栈或硬件变化时重新检查结果。</p>
<figcaption><span class="flow-key"></span>虚线框的步骤把它找到的算子交给第 3 部分。把鼠标移到某一步上或用键盘聚焦它，可以看到这一步做什么。</figcaption>
</figure>

不确定从哪里开始？[alignment-pipeline skill](../skills/alignment-pipeline/SKILL.md)
会提出问题，替你分流。全书用到的词见[术语表](glossary.md)。

## 第 1 部分：两次预训练运行分开了

[概览](1-pretraining/README.md) · agent skill：
[pretraining-alignment](../skills/pretraining-alignment/SKILL.md)

1. [确定性与预检](1-pretraining/determinism-and-preflight.md)：
   打开让运行可重复的设置，并检查两次运行确实都用上了。
2. [数据对齐](1-pretraining/data-alignment.md)：检查两次运行的每个 rank 在每一步拿到的 batch 都相同。
3. [用 OpGuard 定位](1-pretraining/localize-with-opguard.md)：采集两次运行，找到第一个出现差别的算子，从它走到原因，修复并确认。

然后去首个分歧点指向的那一页：

- [训练步](1-pretraining/training-step.md)：让反向、优化器、损失以及从检查点恢复训练逐位重复。
- [数据并行](1-pretraining/data-parallel.md)：梯度跨 rank 求和或分片时，保持相等。
- [作业运行编译代码时](1-pretraining/capture-compiled-code.md)：在 eager 采集看不到的
  `torch.compile` 和 CUDA graph 内部取 fingerprint（指纹）。
- [SDC](1-pretraining/sdc.md)：静默返回错误值的硬件。

## 第 2 部分：rollout 与 trainer 不一致

[概览](2-rl/README.md) · agent skill：
[engine-parity](../skills/engine-parity/SKILL.md)；OpGuard 的 `rl-alignment`

1. [整个模型](2-rl/whole-model.md)：让一侧对一条序列给出相同的 logits，
   无论单独计算、放在 batch 里，还是在任何一种服务模式下。
2. [并行](2-rl/parallelism.md)：在张量并行和 prefill/decode 分离部署下保持这一点。
3. [采样](2-rl/sampling.md)：让选出的 token 能从 logits 和种子重放出来。
4. [LoRA](2-rl/lora.md)：一个 batch 混用多个适配器时，保持不变性。
5. [引擎一致性](2-rl/engine-parity-vllm.md)：让真实引擎按 trainer 的舍入步骤计算，两侧才能一致。
6. [RL 循环](2-rl/rl-loop.md)：记录每次 rollout，把 staleness（策略滞后）和不一致分开，离线定位差距。

## 第 3 部分：让一个算子满足不变性

[标准](3-kernels/README.md) · agent skill：
[batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md)

1. [GEMM 与量化 GEMM](3-kernels/gemm-and-quantized.md)：对每个 batch size 固定同一种 tile 和归约顺序。
2. [注意力](3-kernels/attention.md)：prefill、decode 和反向对 key 用同一种切分。
3. [MLA 与稀疏注意力](3-kernels/mla-and-sparse.md)：对压缩后的 key 和 top-k 选择做同样的事。
4. [线性注意力](3-kernels/linear-attention.md)：循环状态用同一种分块顺序。
5. [MoE](3-kernels/moe.md)：路由、专家 GEMM 和 combine，与哪些 token 同在一个 batch 无关。
6. [集合通信](3-kernels/collectives.md)：跨 GPU 固定归约顺序。

## 第 5 部分：其他对齐场景

[概览](5-scenarios/README.md) · [框架迁移](5-scenarios/framework-migration.md) · [检查点转换](5-scenarios/checkpoint-conversion.md) · [离线后训练](5-scenarios/offline-post-training.md)：同一套方法用于其他配对。

## 第 4 部分与参考

- [第 4 部分：运维](4-operations/README.md)：在[软件栈变化](4-operations/regression.md)之后
  和[换新硬件](4-operations/hardware.md)时重新检查结果。
- Skills：[alignment-pipeline](../skills/alignment-pipeline/SKILL.md)、
  [pretraining-alignment](../skills/pretraining-alignment/SKILL.md)、
  [engine-parity](../skills/engine-parity/SKILL.md)、
  [batch-invariant-kernel](../skills/batch-invariant-kernel/SKILL.md)、
  [handbook-writing](../skills/handbook-writing/SKILL.md)。
- [待办事项](open-items.md) · [参考文献](references.md) ·
  [术语表](glossary.md) · [审计](../AUDIT.md) · [工具](../tools/)

每一页上的结论都链接到对应的报告，报告里有测量结果和测量所用的软件栈。
