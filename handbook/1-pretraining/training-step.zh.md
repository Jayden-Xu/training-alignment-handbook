# 训练 step：backward、优化器、loss、恢复训练

## 我们在哪里

你已经能采集一对运行，并从首个分歧点走到原因（[用 OpGuard 定位](localize-with-opguard.md)）。当这条路走进训练 step 时，就用本页，对应[预训练工作流](README.md)的第 5 步（分类）：某个 backward 算子、loss 归约、优化器、AMP、梯度累加或检查点恢复。本页也列出训练 step 中要固定什么，才能让两个 trainer 一致，这属于第 2 步（控制）：在采集新的一对运行之前先读。[预检](determinism-and-preflight.md)检查的是设置；本页讲的是设置背后的算术。

## 思路：step 中每个浮点和都需要一个写明的顺序

训练 step 里的求和比前向多：

- 每个把梯度 scatter 回去的算子的 backward（embedding、gather、index_select、MoE 合并）；
- 对 token 和序列的 loss 归约；
- 优化器的更新，带着它自己的舍入点；
- 跨 micro-batch 的累加。

**小例子：原子操作。** 三个 token 共用一行 embedding。它们的梯度行 g1、g2、g3 被加进权重梯度。用原子操作时，顺序取决于谁先到：一次运行是 (g1 + g2) + g3，下一次是 (g3 + g1) + g2。第 0 部分的三个数已经说明这会改变比特。这破坏 **L1**（运行间）。

**小例子：确定但不通用。** 基于排序的 scatter 没有原子操作，所以能重复。但它的顺序是实现细节：一个 kernel 在索引少时每块行之后舍入，索引多时在 fp32 中求和、只舍入一次；另一个随行宽改变顺序。各自都能重复；两个框架只有共用这个 kernel 才会一致。这破坏 **L4**（跨实现），也可能让一个 token 的梯度取决于批次。

**规则。** 对 step 中的每个浮点和，顺序必须是定义的属性，而不是时序或 kernel 的属性：例如按源位置升序、在 fp32 中累加、只舍入一次。这就是把第 0 部分的原则用到 backward、loss 和优化器上。

step 的每个部分在哪里出问题，以及怎么修：

| 部分 | 何时破坏 L1（运行间） | 何时破坏 L4（两个 trainer） | 修复 |
|---|---|---|---|
| scatter 式 backward（embedding、`index_select`、重复取值的 `gather`、MoE 合并、带权 `bincount`） | 用原子操作累加各行 | 确定性 kernel 的顺序取决于索引个数或行宽 | 每个 scatter 式求和使用同一个写明的顺序 |
| 长的 1-D `cumsum` | scan 的顺序随时序变化 | scan 的分块布局不同 | 从左到右的 fp32 scan |
| 注意力 backward | dQ 用原子操作累加 | kernel 不同 | 无原子操作的 kernel，或严格模式 |
| loss 归约 | 很少 | `mean` 取决于布局（padding、打包、拆分），Python 数作除数与张量作除数的舍入不同 | 固定的树，计数放在张量里 |
| 优化器 | 很少 | fused、foreach 和 single-tensor 路径在不同位置舍入 | 固定一条路径和一个构建 |
| AMP | fp16 GradScaler：缩放改变哪些值保持为正规数 | 缩放日程不同 | bf16 不需要 scaler；fp16 下固定日程 |
| 梯度累加 | 很少 | micro-batch 的切分或顺序不同 | 固定切分和顺序 |
| 恢复训练 | 缺少任何一项状态 | 同左 | 恢复下面列出的每一项 |

基于排序的算子（embedding backward、cross-entropy、各种 norm、`linear`）在默认模式下能重复；基于原子操作的则不能（测量结果：见[报告](../../examples/training_step/REPORT.md#results)）。

## 步骤

### 固定训练 step

**目标：** 两个 trainer，或一个 trainer 跑两次，做同样的算术。

**做法：** 逐项过清单。固定顺序的 kernel 在 [ts_kernels.py](../../examples/training_step/ts_kernels.py) 里：`seg_sum`（每个 scatter 式求和：按目标稳定排序，然后按源位置升序在 fp32 中相加各行，舍入一次）、`seq_cumsum`（从左到右的 scan）和 `tree_mean_loss`（loss 归约）。

- [ ] **任何浮点和里都没有原子操作。** 使用 `warn_only=False` 的严格模式，或固定顺序的 kernel。严格模式遇到浮点 `cumsum`、带权 `bincount` 和 `histc` 会报错：先替换这三个。
- [ ] **要跨框架一致，每个 scatter 式求和使用同一个写明的顺序：** 源位置升序、fp32、一次舍入。
- [ ] **注意力 backward：** 无原子操作的 kernel，或严格模式下的 SDPA（[注意力](../3-kernels/attention.md)）。
- [ ] **优化器路径：** fused、foreach 或 single-tensor，并使用同一个 torch 构建。默认值在不同版本间不稳定。
- [ ] **loss 归约**用固定的树，包括计数怎样作除数。不要用 `mean`，它取决于布局。
- [ ] **micro-batch 的切分和顺序。**
- [ ] **fp16 GradScaler：** 缩放日程也要固定。bf16 不需要。
- [ ] **检查点的每一项状态：** 模型；含 step 计数器的优化器；scheduler；GradScaler；`torch.get_rng_state()`；`torch.cuda.get_rng_state()`；sampler 的 epoch、位置和排列。
- [ ] 两边都**关闭 TF32，并使用相同的 `CUBLAS_WORKSPACE_CONFIG`**。

| 选项 | 给出什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 原生 torch，默认模式 | 没有任何保证；基于原子操作的算子会变化 | 免费 | 对齐时永远不选 |
| 严格确定性模式 | torch 覆盖的每个算子满足 L1；其余报错 | 整个 step 中等（2 倍以内）；单个算子可能慢得多 | 单个框架，打开即可 |
| 有针对性的固定顺序 kernel（仅注意力） | 在模型实际变化的地方满足 L1 | 小（几个百分点） | 模型唯一的 L1 破坏者是注意力 |
| 注意力、embedding 和 loss 的固定顺序 kernel | L1，以及另一个框架能复现的顺序（L4） | 中等（2 倍以内） | 两个框架或两套代码必须一致 |

**成功：** 两次运行在不同进程中，每一步的 loss 比特、梯度 fingerprint 和权重都相等；从检查点恢复与不间断运行给出相同的各步。

这在一种 GPU、一个 torch 构建上得到证明：上面每种设置在测试的步数内都给出相等的运行，固定的 scatter 顺序与按同一定义写的 NumPy 循环逐位一致，覆盖了不同行宽、索引个数和数据类型（[报告](../../examples/training_step/REPORT.md#the-fix-one-written-order-for-every-scatter-style-sum)）。

### 给 step 内的首个分歧点分类

**目标：** 把 step 中的 pivot 对应到上表中的某个部分。

**做法：** 看 pivot 在哪里。

| pivot | 可能的部分 |
|---|---|
| backward 之后第一个梯度消费者，前向相等 | 上游有非确定的 backward（注意力、scatter） |
| 更新之后的权重或 Adam 矩，梯度相等 | 优化器路径或其状态 |
| loss，逐 token 的 loss 相等 | loss 归约 |
| 恢复训练后紧接着的 embedding 查表，token id 相等 | 恢复的权重或优化器状态 |
| 恢复训练后在 dropout 或重新洗牌处的 loss | 某个 RNG 或 sampler 状态 |

**成功：** 指出一个部分，以及一个只改变这个部分的检验。

## 可能出错的地方

!!! warning "用 Python 数作除数"
    `n` 是 Python int 时，`total / n` 与 `total / tensor(n)` 的舍入不同。两个看起来相同的归约给出不同的比特。在定义中写明除数的形式，并把计数放在张量里。

!!! warning "优化器交给默认值"
    torch 自己选择优化器路径，而 fused、foreach 和 single-tensor 路径在不同位置舍入。版本升级可能因此改变每个参数。固定路径和构建。

!!! warning "把累加当成无代价"
    一个含四条序列的批和四个各含一条序列的 micro-batch 是求和的不同分组，反转 micro-batch 的顺序也不同。梯度从第一步就不同。两边固定切分和顺序。

!!! warning "以为 fp16 缩放是精确的"
    在 fp16 中，缩放会保留那些不缩放时会成为次正规数或零的比特，所以缩放后的 backward 与不缩放的不同。在 bf16 中，2 的幂的缩放什么都不改变。fp16 下固定缩放日程。

!!! warning "RNG 状态恢复到了错误的设备"
    `torch.set_rng_state` 需要 CPU 上的字节张量；把检查点加载到 GPU 会导致恢复失败。运行在若干步之后的下一次重新洗牌处分歧。在 CPU 上加载 RNG 状态。

## 小结

- step 中每个浮点和都需要一个由定义固定的顺序：scatter、scan、注意力 backward、loss。
- 对两个 trainer 来说确定性不够：kernel 特有的顺序、优化器路径和 loss 除数的形式都会改变比特。
- 严格模式以中等代价给出 L1；固定顺序 kernel 给出 L1 和可复现的顺序。
- 固定优化器路径、loss 树、micro-batch 切分和每一项恢复状态。

## 深入阅读

- [报告：训练 step](../../examples/training_step/REPORT.md)：测量的每个算子、整步运行、优化器舍入点表、loss 布局、AMP、累加、裁剪、恢复训练和代价。
- 代码：[examples/training_step/](../../examples/training_step/)。
- [数据并行](data-parallel.md)：同样的问题，换成跨 rank 求和的梯度。
- 注意力 backward 见[注意力](../3-kernels/attention.md)。
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)。
