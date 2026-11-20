# 训练 step

## 我们在哪里

你已经能采集一对运行，并从首个分歧点追到原因（[用 OpGuard 定位](localize-with-opguard.md)）。如果在[预训练流程](README.md)的第 5 步（判断原因类别）追到了训练 step 里面，比如某个算子的反向、loss 的归约、优化器、AMP、梯度累加，或者从 checkpoint 恢复训练，就读这一页。这一页还列出了训练 step 里要固定哪些东西，两个 trainer 才能一致；这部分属于第 2 步（控制），所以在采集新的一对运行之前也应该先读。[预检](determinism-and-preflight.md)检查的是设置；这一页讲的是设置背后的算术。

## 思路：step 里的每一个浮点求和，都需要一个写明的顺序

一个训练 step 里的求和，比前向计算多得多：

- 凡是要把梯度 scatter 回输入的算子，它们的反向（embedding、gather、index_select、MoE 的 combine）；
- 对 token 和序列做的 loss 归约；
- 优化器的更新，它有自己的舍入点；
- 跨 micro-batch 的梯度累加。

**小例子：原子操作。** 三个 token 共用 embedding 的同一行。它们的梯度行 g1、g2、g3 要加到权重梯度上。用原子操作时，谁先到就先加谁：这次运行是 (g1 + g2) + g3，下次可能是 (g3 + g1) + g2。第 0 部分那三个数的例子已经说明，这样结果就可能不同。这破坏的是 **L1**（运行间一致）。

**小例子：确定，但各家实现不同。** 基于排序的 scatter 不用原子操作，所以每次运行都能重复自己。但它的求和顺序只是实现上的细节：有的 kernel 在索引少时每加完一组行就舍入一次，索引多时则在 fp32 里一次加完；另一个 kernel 的顺序会随行宽变化。它们各自都能重复，但两个框架只有用的是同一个 kernel 时才会一致。这破坏的是 **L4**（不同实现之间一致），还可能让一个 token 的梯度随 batch 而变。

**规则。** step 里的每一个浮点求和，顺序都必须由求和的定义决定，而不是由时序或 kernel 决定。例如：按源位置升序相加，在 fp32 里累加，最后只舍入一次。这就是把第 0 部分的原则用到反向计算、loss 和优化器上。

step 的每个部分在什么情况下会出问题，以及怎么修：

| 部分 | 什么时候破坏 L1（运行间） | 什么时候破坏 L4（两个 trainer 之间） | 修法 |
|---|---|---|---|
| scatter 类算子的反向（embedding、`index_select`、同一行被多次选中的 `gather`、MoE 的 combine、带权重的 `bincount`） | 用原子操作累加各行 | 确定性 kernel 的顺序取决于索引个数或行宽 | 所有 scatter 类求和都用同一个写明的顺序 |
| 很长的 1-D `cumsum` | scan 的顺序随时序变化 | scan 对输入分块的方式不同 | 在 fp32 里从左到右做 scan |
| 注意力的反向 | dQ 用原子操作累加 | 用的 kernel 不同 | 换成不用原子操作的 kernel，或者打开严格模式 |
| loss 归约 | 很少 | `mean` 的结果取决于数据布局（padding、packing 或拆分），而且除以 Python 数和除以张量的舍入不同 | 用固定的归约树，计数放在张量里 |
| 优化器 | 很少 | fused、foreach 和 single-tensor 三条代码路径在不同的地方舍入 | 固定一条代码路径和一个 torch 构建 |
| AMP | fp16 下 GradScaler 的缩放系数决定了哪些值还是正规数 | 缩放系数的变化规则不同 | bf16 不需要 scaler；fp16 下固定缩放规则 |
| 梯度累加 | 很少 | micro-batch 的切分方式或顺序不同 | 固定切分方式和顺序 |
| 恢复训练 | 漏掉了任何一项状态 | 同左 | 恢复下面清单里的每一项状态 |

基于排序的算子（embedding 的反向、cross-entropy、各种 norm、`linear`）在默认模式下就能逐 bit 重复；基于原子操作的则不能（实测见[报告](../../examples/training_step/REPORT.md#results)）。

## 步骤

### 把训练 step 的每个部分都固定下来

要让两个 trainer，或者同一个 trainer 跑两次，做的算术完全相同，就对照下面的清单逐项检查。固定顺序的 kernel 都在 [ts_kernels.py](../../examples/training_step/ts_kernels.py) 里：`seg_sum` 负责所有 scatter 类求和（先按目标位置做稳定排序，再按源位置升序在 fp32 里把各行加起来，只舍入一次），`seq_cumsum` 是从左到右的 scan，`tree_mean_loss` 负责 loss 归约。

- [ ] **所有浮点求和都不用原子操作。** 打开严格模式并设 `warn_only=False`，或者改用固定顺序的 kernel。严格模式遇到浮点 `cumsum`、带权重的 `bincount` 和 `histc` 会直接报错，所以先把这三个换掉。
- [ ] **要让不同框架一致，所有 scatter 类求和都用同一个写明的顺序：** 按源位置升序，在 fp32 里累加，只舍入一次。
- [ ] **注意力的反向：** 用不带原子操作的 kernel，或者在严格模式下用 SDPA（[注意力](../3-kernels/attention.md)）。
- [ ] **优化器的代码路径：** 在 fused、foreach 和 single-tensor 中选定一条，并用同一个 torch 构建。默认选哪条在不同版本之间并不稳定。
- [ ] **loss 归约**用固定的归约树，包括计数怎样参与除法。不要用 `mean`，它的结果取决于数据布局。
- [ ] **micro-batch 的切分方式和顺序。**
- [ ] **fp16 的 GradScaler：** 缩放系数的变化规则也要固定。bf16 不需要 scaler。
- [ ] **checkpoint 里的每一项状态：** 模型；优化器，包括它的 step 计数器；scheduler；GradScaler；`torch.get_rng_state()`；`torch.cuda.get_rng_state()`；sampler 的 epoch、位置和排列。
- [ ] 两端都**关闭 TF32，并使用相同的 `CUBLAS_WORKSPACE_CONFIG`**。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 原生 torch，默认模式 | 没有任何保证；基于原子操作的算子每次结果都可能不同 | 免费 | 需要对齐时，永远不要选 |
| 严格确定性模式 | torch 覆盖到的每个算子都满足 L1，其余算子直接报错 | 整个 step 中等（不到 2 倍）；个别算子可能慢得多 | 只有一个框架，想要一个打开就能用的办法 |
| 只给注意力换固定顺序的 kernel | 在模型实际出现变化的地方满足 L1 | 小（百分之几） | 模型里破坏 L1 的只有注意力 |
| 给注意力、embedding 和 loss 都换固定顺序的 kernel | 满足 L1，而且求和顺序能被另一个框架复现（L4） | 中等（不到 2 倍） | 两个框架或两套代码必须一致 |

做到了的标志是：在不同进程里跑的两次运行，每一步的 loss 逐 bit 一致，梯度的 fingerprint 相等，权重也相等；从 checkpoint 恢复的运行和一直没停的运行，每一步都相同。

这一点在一种 GPU、一个 torch 构建上得到了证明：在测试的步数内，上面每种设置跑出的两次运行都一致；固定的 scatter 顺序和按同一个定义写出的 NumPy 循环逐 bit 一致，覆盖了不同的行宽、索引个数和 dtype（[报告](../../examples/training_step/REPORT.md#the-fix-one-written-order-for-every-scatter-style-sum)）。

### 判断 step 内的首个分歧点指向哪个部分

看 pivot 落在哪里，把它对应到上表中的某个部分。指出了一个部分、并且有一个只改动这个部分的检验，这一步就完成了。

| pivot 落在哪里 | 最可能是哪个部分 |
|---|---|
| 反向计算之后第一个用到梯度的算子，而前向结果相等 | 上游有不确定的反向计算（注意力、scatter） |
| 更新之后的某个权重或 Adam 矩，而梯度相等 | 优化器的代码路径或它的状态 |
| loss，而每个 token 的 loss 都相等 | loss 归约 |
| 恢复训练后紧接着的 embedding 查表，而 token id 相等 | 恢复出来的权重或优化器状态 |
| 恢复训练后，在 dropout 或重新 shuffle 处的 loss | 某一项 RNG 或 sampler 状态 |

## 可能出错的地方

!!! warning "用 Python 数作除数"
    当 `n` 是 Python int 时，`total / n` 和 `total / tensor(n)` 的舍入方式不同。两个看起来一样的归约，结果却不是逐 bit 一致。在定义里写明除数的形式，并把计数放在张量里。

!!! warning "优化器用默认设置"
    torch 会自己选择优化器的代码路径，而 fused、foreach 和 single-tensor 三条路径在不同的地方舍入。于是一次版本升级就可能改变每一个参数。要固定代码路径和 torch 构建。

!!! warning "以为梯度累加不会改变结果"
    一个含四条序列的 batch，和四个各含一条序列的 micro-batch，对求和的分组方式不同；把 micro-batch 的顺序倒过来，结果也会变。于是梯度从第一步起就不同了。两端都要固定切分方式和顺序。

!!! warning "以为 fp16 的缩放是精确的"
    在 fp16 下，缩放能保住那些不缩放就会变成次正规数或零的 bit，所以缩放过的反向计算和没缩放的结果不同。在 bf16 下，按 2 的幂缩放不会改变任何东西。fp16 下要固定缩放规则。

!!! warning "RNG 状态恢复到了错误的设备上"
    `torch.set_rng_state` 需要一个 CPU 上的字节张量，所以把 checkpoint 加载到 GPU 上，RNG 状态就恢复不了。运行会在若干步之后、下一次重新 shuffle 时出现分歧。要在 CPU 上加载 RNG 状态。

## 小结

- step 里的每一个浮点求和，都需要一个由定义决定的顺序：scatter、scan、注意力的反向和 loss 都是如此。
- 要让两个 trainer 一致，光有确定性不够：kernel 各自的求和顺序、优化器的代码路径、loss 除数的形式，都会改变结果。
- 严格模式以中等代价换来 L1；固定顺序的 kernel 既满足 L1，顺序也能被别人复现。
- 固定优化器的代码路径、loss 的归约树、micro-batch 的切分，以及恢复训练所需的每一项状态。

## 深入阅读

- [报告：训练 step](../../examples/training_step/REPORT.md)：测量过的每个算子、整个 step 的运行、优化器舍入点的对照表、各种 loss 布局、AMP、梯度累加、梯度裁剪、恢复训练和代价。
- 代码在 [examples/training_step/](../../examples/training_step/)。
- [数据并行](data-parallel.md)讨论同样的问题，对象换成跨 rank 求和的梯度。
- 注意力的反向见[注意力](../3-kernels/attention.md)。
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)。
