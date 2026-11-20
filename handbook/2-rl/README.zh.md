# 2. 强化学习：训推对齐

## 我们在哪里

在强化学习中，rollout（采样生成）引擎（vLLM、SGLang）采样 token 并报告它们的
logprob（对数概率），trainer（训练端）（FSDP、Megatron、TorchTitan）为策略梯度比率
重新计算同样的 logprob。两边运行同一个策略，但代码、kernel、批处理、decode（解码）
模式和并行方式都不同。所以即使权重相同，logprob 也不同，训练在没人选择的情况下变成了
off-policy。这一部分讲两件事：怎样让两边逐位一致；不一致时怎样找到出错的地方。
它以[第零部分](../0-foundations/README.md)的不变性级别 L1–L5 为前提。

首先把一致性当作调试和审计工具：在一个短窗口里，基础设施给出的差距必须恰好为零。
是否在生产中一直开着，是一个代价问题。

## 工作流程

<figure class="fx" data-fig="rl-workflow" aria-label="强化学习训推对齐工作流" markdown>
<div class="fx-stage" markdown>
<div class="fx-phases" style="--cols: minmax(0, 1fr) minmax(0, 0.75fr) minmax(0, 1.25fr)" markdown>
<div class="fx-phase" markdown="span"><span class="fx-label">每一侧各自不变</span><span class="fx-seq is-col"><span class="fx-step" data-hot="0">[1. 审查算子](whole-model.md)</span><span class="fx-step" data-hot="1">[2. 整个模型](whole-model.md)<span class="fx-note">L1–L3</span></span><span class="fx-step" data-hot="2">[3. 并行度](parallelism.md)<span class="fx-note">L5</span></span><span class="fx-step" data-hot="3">[4. 采样](sampling.md)</span><span class="fx-step" data-hot="4">[5. LoRA](lora.md)</span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">两侧一致</span><span class="fx-seq is-col"><span class="fx-step" data-hot="5 9">[6. 引擎一致性](engine-parity-vllm.md)<span class="fx-note">L4</span></span></span></div>
<div class="fx-phase" markdown="span"><span class="fx-label">看守运行中的循环</span><span class="fx-seq is-col"><span class="fx-step" data-hot="6">[7. 在线记录](rl-loop.md)</span><span class="fx-step" data-hot="7">[8. 拆分差距](rl-loop.md)<span class="fx-note">staleness 和数值误差</span></span><span class="fx-fork is-pair"><span class="fx-leg" data-hot="8"><span class="fx-when">numeric ≠ 0</span><span class="fx-step to-bad" data-hot="8">[9. 离线定位](rl-loop.md)<span class="fx-note" data-hot="9">然后回到第 6 步</span></span></span><span class="fx-leg" data-hot="10"><span class="fx-when">numeric = 0</span><span class="fx-step to-ok" data-hot="10">[10. 回归](../4-operations/regression.md)</span></span></span></span></div>
</div>
</div>
<ol class="fx-say">
<li>1. 审查算子：得到输出依赖 batch 或模式的模块清单。</li>
<li>2. 整个模型：单侧在每种 batch 和服务模式下对一个序列给出相同的位（L1–L3）。</li>
<li>3. 并行度：每个 TP 度下位都相同，分离式 prefill 与 decode 之间也相同（L5）。</li>
<li>4. 采样：采出的 token 在批处理、抢占、n > 1 和投机解码之后仍可重放。</li>
<li>5. LoRA：adapter 在训练 batch 和混合服务 batch 里以相同方式舍入。</li>
<li>6. 引擎一致性：trainer 的 logprob 等于引擎的（L4）。</li>
<li>7. 在线记录正在运行的循环。</li>
<li>8. 把每个 token 的差距拆成 staleness 和数值误差。</li>
<li>numeric ≠ 0：9. 离线定位，一直定位到算子。</li>
<li>然后带着找到的原因回到第 6 步。</li>
<li data-hold="2">numeric = 0：10. 把结果作为基线保留，新版本必须复现它。</li>
</ol>
<figcaption>图：第 1–5 步让每一侧各自满足不变性，第 6 步让两侧一致，第 7–10 步看守正在运行的循环。每一步都链接到它的页面。</figcaption>
</figure>

[alignment-pipeline skill](../../skills/alignment-pipeline/SKILL.md) 会把一个问题分派到对应的步骤。

## 思路：先单侧，再两侧

两个数不同有两种原因：每一侧自身不稳定；或者两侧都稳定，但舍入方式不同。
按这个顺序修。如果引擎对某个 token 的 logprob 会因为另一个请求加入批次而改变，
那么任何 trainer 都对不上它。所以第 2–5 步先让每一侧各自不变，第 6 步才比较两侧。

有些差异不是 kernel 缺陷，而是同一数学的两种精确形式，舍入方式不同：
分块与循环形式的线性注意力、吸收式与显式的 MLA、split-KV decode 与单遍 prefill。
调 kernel 消除不了它们。为两侧选定同一种形式，并承担它的代价；
kernel 的编写见[第三部分](../3-kernels/README.md)。

## 思路：先拆分差距，再调试

在异步循环中，rollout 用的是策略版本 v，trainer 现在处于版本 c。
观察到的差距混合了两样东西，而且可以精确拆开：

```
gap       = logp_rollout(v) - logp_trainer(c)
numeric   = logp_rollout(v) - logp_trainer(v)   # same weights: an infrastructure fault, should be 0
staleness = logp_trainer(v) - logp_trainer(c)   # policy lag: corrected by IS / TIS / masking
```

一个小例子。某个 token 在版本 3 时采样，trainer 现在处于版本 5。
在 trainer 上用版本 3 重新计算这个 token 的 logprob 并比较：

| Token | logp_rollout(3) | logp_trainer(3) | logp_trainer(5) | gap | numeric | staleness |
|---|---|---|---|---|---|---|
| A | -1.20 | -1.20 | -1.35 | 0.15 | 0 | 0.15 |
| B | -1.20 | -1.21 | -1.35 | 0.15 | 0.01 | 0.14 |

两个 token 的差距相同。A 的差距全部是策略滞后，正是重要性采样要校正的部分；
基础设施没有问题。B 的差距中有一部分是基础设施故障，任何校正都不应吸收它，
这部分才值得定位。只有 `numeric` 是基础设施要修的，而且只有在 rollout 自己的版本上
重放才能测出它。

## 同步与异步运行

- **同步（staleness 为 0）。** `numeric` 就是全部差距。两侧都开启一致性时，
  目标是在一个短的 on-policy 窗口里每个 token 的差都恰好为 0。如果差距为 0
  而训练仍然有问题，问题就在数据或算法里。
- **异步（staleness > 0）。** 在线记录，拆分每个 token 的差距，只定位 `numeric`
  非零的 token。跨越一次权重更新生成的 token（KV cache 被保留）属于两个版本，
  无法拆分，先放到一边。

## 可能出错的地方

!!! warning "在每一侧稳定之前就比较"
    拿 trainer 去和一个不具备批不变性的引擎比较。这时每个 trainer 改动看起来都
    部分正确，因为引擎自身的波动和差距一样大。先检查引擎自身。

!!! warning "把数值误差当作 staleness 来校正"
    重要性采样校正会吸收任何差距，包括基础设施故障。这个故障随后会悄悄地训练策略。
    在应用校正之前，先在 rollout 的版本上拆分差距。

!!! warning "相信版本标签"
    标为某个版本的 rollout 批次，可能实际用的是另一个版本的权重。它的数值误差
    看起来是真的，但重放时不会复现。为每个批次实际使用的权重记录一个指纹。

## 小结

- 先让每一侧各自不变（L1–L3、L5、采样、LoRA），再让两侧一致（L4）。
- 有些不一致是同一数学的两种精确形式：为两侧选同一种。
- 把每个差距拆成 staleness 和数值误差；只有数值误差是基础设施故障。
- 非零的数值误差沿数据流离线定位，修好之后保存为回归基线。

## 深入阅读

- 各步骤页面：[整个模型](whole-model.md)、[并行](parallelism.md)、
  [采样](sampling.md)、[LoRA](lora.md)、
  [引擎一致性](engine-parity-vllm.md)、[RL 循环](rl-loop.md)。
- 其他团队报告的结果（IsoExec、verl-omni、TorchTitan RL）以及一个单元素定位的
  实例：[报告](../../examples/cross_engine/REPORT.md#context-moved-from-the-part-2-overview)；
  来源见[参考文献](../references.md)。
- Skill：[alignment-pipeline](../../skills/alignment-pipeline/SKILL.md)、
  [engine-parity](../../skills/engine-parity/SKILL.md)、
  [batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)；
  OpGuard 的 [`rl-alignment` skill](https://github.com/ziming-zh/OpGuard/tree/main/harness/skills/rl-alignment)（OpGuard 仓库中的 `harness/skills/rl-alignment`）
  及其协议 `docs/RLAlignment.md`。
