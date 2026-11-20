# 让每个训练 step 都逐 bit 可重复

## 我们在哪里

你已经能用 OpGuard 采集一对运行，并从首个分歧点追到原因（[用 OpGuard 定位](localize-with-opguard.md)）。
很常见的情况是：两次运行的前向完全一致，第一处不同直到参数更新之后才出现。这时原因就在训练 step
内部：反向、loss、优化器、梯度累加、混合精度，或者从 checkpoint 恢复训练。这一页讲这几部分为什么会
改变结果，以及要固定哪些东西，才能让一个 step 完全可重复。采集新的一对运行之前也值得先读一遍：
两个 trainer 要对齐，靠的也是同一份清单。

## 问题

两次运行从同一个 checkpoint 出发，数据和 seed 都相同。跑完一个 step，两边的权重应该逐 bit 一致吗？

只有在 step 里的每一次求和，两边都按同样的顺序计算时，才会一致。[第 0 部分](../0-foundations/README.md)
已经说明，同一组数换个顺序相加，结果可能不同。而一个训练 step 里的求和比前向多得多，其中很多求和，
你不明确规定顺序，它就没有固定的顺序：

- 凡是要把梯度 scatter 回输入的算子，它的反向（embedding、`gather`、`index_select`、MoE 的 combine）；
- loss，要在 token 和序列上求平均；
- 优化器的更新，它有自己的舍入点；
- 多个 micro-batch 的梯度相加。

这些求和出问题的方式有两种。

**同一份代码，每次跑的结果都不一样（破坏 L1）。** 三个 token 用到 embedding 表的同一行，
它们的梯度行 g1、g2、g3 要加到同一个权重梯度行上。用原子加的 kernel，哪个线程先算完就先加哪个：
这次是 (g1 + g2) + g3，下次可能是 (g3 + g1) + g2，最后一个 bit 就可能不同。

**同一份代码每次都一样，但换一种实现就对不上（破坏 L4）。** 基于排序的 scatter 不用原子加，
所以每次都能重复自己。可它的求和顺序只是这个 kernel 的内部细节：有的 kernel 索引少时每加完一组行
就舍入一次，索引多时在 fp32 里一次加完；有的 kernel 的顺序会随行宽变化。它们各自都能重复，
但两个框架只有恰好用了同一个 kernel 才会一致。优化器和 loss 也是这样：fused、foreach、
single-tensor 三条优化器代码路径的舍入位置不同；同样的 token，padding 和 packing 两种排法下，
`mean` 算出的 bit 也不同。

**规则。** step 里的每一次浮点求和，顺序都要由它的定义决定，而不是由线程时序或 kernel 内部决定。
例如：按源位置从小到大把各行加起来，在 fp32 里累加，最后只舍入一次。这就是第 0 部分的原则，
用到反向、loss 和优化器上。

| step 的哪一部分 | 同一份代码，两次运行会不同吗 | 两种实现会不同吗 | 怎么固定 |
|---|---|---|---|
| scatter 类算子的反向（embedding、`index_select`、同一行被多次选中的 `gather`、MoE 的 combine、带权重的 `bincount`） | 会，如果用原子加 | 会，顺序取决于索引个数或行宽 | 所有 scatter 类求和都用同一个写明的顺序 |
| 很长的 1-D `cumsum` | 会，scan 的顺序跟着线程时序走 | 会，scan 对输入的分块方式不同 | 在 fp32 里从左到右做 scan |
| 注意力的反向 | 会，dQ 用原子加累积 | 会，kernel 不同 | 换成不用原子加的 kernel，或者打开严格模式 |
| loss | 很少 | 会：`mean` 的结果取决于 padding、packing 和切分方式，而且除以 Python 数和除以张量的舍入不同 | 用固定的归约树，计数放在张量里 |
| 优化器 | 很少 | 会：fused、foreach、single-tensor 三条路径的舍入位置不同 | 固定一条代码路径和一个 torch 构建 |
| 混合精度（fp16 的 GradScaler） | 缩放系数决定了哪些值还是正规数 | 缩放规则可能不同 | bf16 不需要 scaler；用 fp16 时固定缩放规则 |
| 梯度累加 | 很少 | 会，micro-batch 的切分方式或顺序不同 | 固定切分方式和顺序 |
| 从 checkpoint 恢复训练 | 会，只要漏存了任何一项状态 | 会，原因相同 | 恢复下面清单里的每一项状态 |

基于排序的算子（embedding 的反向、cross-entropy、各种 norm、`linear`）在默认模式下就能重复；
基于原子加的则不能（实测见[报告](../../examples/training_step/REPORT.md#results)）。

## 步骤

### 先让同一份代码能重复自己

从这里开始：一个 trainer 自己跑两次都对不上，就谈不上两个 trainer 对齐。办法是去掉所有用原子加的
浮点求和：要么打开 torch 的严格模式，要么只替换需要替换的那几个算子。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 原生 torch，默认模式 | 没有任何保证；用原子加的算子每次结果都可能不同 | 免费 | 需要对齐时，永远不要选 |
| 严格确定性模式 | torch 覆盖到的算子都满足 L1，其余算子直接报错 | 整个 step 中等（不到 2 倍）；个别算子可能慢得多 | 只有一个框架，想要一个开关就能解决 |
| 只给注意力换固定顺序的 kernel | 在模型实际出问题的地方满足 L1 | 小（百分之几） | 模型里破坏 L1 的只有注意力 |
| 给注意力、embedding 和 loss 都换固定顺序的 kernel | 满足 L1，而且求和顺序能被另一个框架复现（L4） | 中等（不到 2 倍） | 两个框架或两套代码必须一致 |

- 严格模式：打开时设 `warn_only=False`。它遇到浮点 `cumsum`、带权重的 `bincount` 和 `histc`
  会直接报错，所以先把这三个换掉。
- 固定顺序的 kernel 在 [ts_kernels.py](../../examples/training_step/ts_kernels.py) 里：
  `seg_sum` 负责所有 scatter 类求和（先按目标位置稳定排序，再按源位置从小到大在 fp32 里把各行加起来，
  只舍入一次），`seq_cumsum` 是从左到右的 scan，`tree_mean_loss` 负责 loss 的归约。
- 注意力的反向：用不带原子加的 kernel，或者在严格模式下用 SDPA（[注意力](../3-kernels/attention.md)）。
- 两次运行都关闭 TF32，并使用相同的 `CUBLAS_WORKSPACE_CONFIG`。

### 再让两个 trainer 做同样的算术

每个 trainer 都能重复自己之后，再把另一种实现可能做得不一样的地方固定下来：

- 所有 scatter 类求和用同一个写明的顺序：按源位置从小到大，在 fp32 里累加，只舍入一次；
- 固定一条优化器代码路径（fused、foreach 或 single-tensor）和一个 torch 构建，因为默认路径在
  不同版本之间可能会变；
- loss 用固定的归约树，把除法写进定义里；不要用 `mean`，它的结果取决于数据怎么排；
- micro-batch 的切分方式和顺序相同；
- 用 fp16 时，GradScaler 的缩放规则相同（bf16 不需要 scaler）。

做到的标志是：两次运行放在不同进程里，每一步的 loss 逐 bit 一致，梯度的 fingerprint 相等，
权重也相等。上面每一种设置都在一种 GPU、一个 torch 构建上验证过；固定的 scatter 顺序和按同一个定义
写出的 NumPy 循环，在测试过的每种行宽、索引个数和 dtype 下都逐 bit 一致
（[报告](../../examples/training_step/REPORT.md#the-fix-one-written-order-for-every-scatter-style-sum)）。

### 保存恢复训练需要的全部状态

恢复后的运行要和一直没停的运行一致，checkpoint 里必须存下后面各步会读到的全部状态：

- [ ] 模型；
- [ ] 优化器，包括它的 step 计数器；
- [ ] 学习率 scheduler；
- [ ] GradScaler（用 fp16 时）；
- [ ] CPU 的随机数状态，`torch.get_rng_state()`；
- [ ] GPU 的随机数状态，`torch.cuda.get_rng_state()`；
- [ ] sampler 的 epoch、位置和排列。

做到的标志是：从 checkpoint 恢复的运行和一直没停的运行，恢复之后的每一步都完全相同。

### 判断首个分歧点落在 step 的哪一部分

看首个分歧点落在哪里，把它对应到 step 的某一部分。能指出是哪一部分，并且有一个只改动这一部分的检验，
这一步就完成了。

| 首个分歧点落在哪里 | 最可能是哪一部分 |
|---|---|
| 反向之后第一个用到梯度的算子，而前向结果一致 | 上游有不确定的反向（注意力、scatter） |
| 更新之后的某个权重或 Adam 的动量，而梯度一致 | 优化器的代码路径或它的状态 |
| loss，而每个 token 的 loss 都一致 | loss 的归约 |
| 恢复训练后紧接着的 embedding 查表，而 token id 一致 | 恢复出来的权重或优化器状态 |
| 恢复训练后，dropout 处或下一次重新 shuffle 时的 loss | 漏存了某个随机数状态或 sampler 状态 |

## 可能出错的地方

!!! warning "除数是 Python 数"
    当 `n` 是 Python int 时，`total / n` 和 `total / tensor(n)` 的舍入方式不同。两个看起来一样的
    归约，最后一个 bit 却不同。把除法写进定义里，计数放在张量里。

!!! warning "优化器用默认路径"
    优化器走哪条代码路径由 torch 自己选，而 fused、foreach、single-tensor 三条路径的舍入位置不同。
    升级一次版本，就可能改变每一个参数。要固定代码路径和 torch 构建。

!!! warning "以为梯度累加不会改变结果"
    一个含四条序列的 batch，和四个各含一条序列的 micro-batch，求和的分组方式不同；把 micro-batch 的
    顺序倒过来，结果也会变。梯度从第一步起就不同了。两端都要固定切分方式和顺序。

!!! warning "以为 fp16 的缩放是精确的"
    在 fp16 下，缩放能保住那些不缩放就会变成次正规数或零的 bit，所以缩放过的反向和没缩放的结果不同。
    在 bf16 下，按 2 的幂缩放不会改变任何东西。用 fp16 时要固定缩放规则。

!!! warning "随机数状态恢复到了错误的设备上"
    `torch.set_rng_state` 要的是 CPU 上的字节张量，所以把 checkpoint 加载到 GPU 上，随机数状态就
    恢复不了。两次运行会在几步之后、下一次重新 shuffle 时分开。要在 CPU 上加载随机数状态。

## 小结

- 一个训练 step 里的求和比前向多得多，每一次求和都需要一个由定义决定的顺序。
- 两种问题看起来很像：原子加让同一份代码每次跑都不同（L1）；顺序固定但各不相同，让两种实现对不上（L4）。
- 先让一个 trainer 能重复自己（严格模式，或换掉几个固定顺序的 kernel），再固定优化器路径、loss 的归约
  和 micro-batch 的切分，让两个 trainer 一致。
- 恢复训练要一致，checkpoint 必须存下每一项状态，包括两个随机数状态和 sampler。

## 深入阅读

- [报告：训练 step](../../examples/training_step/REPORT.md)：测过的每个算子、整个 step 的运行、优化器的
  舍入点对照、loss 的各种排法、混合精度、梯度累加、梯度裁剪、恢复训练和代价。
- 代码在 [examples/training_step/](../../examples/training_step/)。
- [数据并行](data-parallel.md)：同样的问题，对象换成跨 rank 求和的梯度。
- 注意力的反向见[注意力](../3-kernels/attention.md)。
- [batch-invariant-kernel skill](../../skills/batch-invariant-kernel/SKILL.md)。
