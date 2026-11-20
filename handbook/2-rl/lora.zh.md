# LoRA：多 adapter 混合 batch

## 我们在哪里

经过[整个模型](whole-model.md)、[并行](parallelism.md)和[采样](sampling.md)这几步，
rollout 端和 trainer 端的基础模型与采样器都已经各自满足不变性。RL 微调常常不训练全部权重，
而是训练 LoRA adapter，每个 adapter 又会带来自己的 kernel。两端运行这些 kernel 的方式
差别很大：trainer 在一个 batch 里只用一个 adapter，每一行乘的都是同一组 A 和 B；推理引擎
却要在同一个 batch 里同时服务很多 adapter，要么用 grouped 或 gather 形式的 kernel，
让每个 token 取自己的 adapter，要么事先把 adapter 合并进基础权重。这一步要让 LoRA 层
在两端的舍入完全相同。

## 思路

LoRA 层计算 y = x·W + s·(x·A)·B。除了基础 GEMM x·W，它还包括一个低秩的小更新
(x·A)·B、一个缩放系数 s，以及把更新加回基础结果的那一步。在基础 GEMM 之外，
这三处都可能让两端的结果不再逐 bit 一致：

| 来源 | 小例子 | 破坏哪一级 |
|---|---|---|
| 有多少 token 共用这个 adapter | 引擎把使用同一个 adapter 的 token 归成一组，每组做一次 GEMM。一个只有 3 个 token 的请求单独运行时，组里就只有这 3 个 token；放进繁忙的 batch 里，同一个 adapter 的组可能有 300 个 token，GEMM 库可能因此换用另一个 kernel | L2 |
| 缩放系数 s 在哪一步乘进去 | s·(x·A)·B、(x·A)·(s·B) 和 (s·x·A)·B 在精确计算下相等；但如果每次乘完都舍入到 bf16，三种写法会得到三个不同的数，除非 s 恰好是 2 的幂 | L4 |
| 在哪里把更新加回基础结果 | 两个 bf16 值相加，要经历两次舍入；把 fp32 的更新直接加进基础 GEMM 的 fp32 累加器，最后只转换一次，就只舍入一次 | L4，也影响精度 |

此外，引擎里的 prefill 和 decode 常常使用不同的 LoRA kernel，这又会破坏 L3。

把 adapter 合并进基础权重是一种设计选择，本身不算错误，但要付出精度代价。W + s·B·A
以 bf16 存储时，凡是小于对应权重最后一个 bit 所代表数值一半的更新元素，都会在相加时被舍掉。
因此，部署合并权重的引擎不可能与 adapter 单独保存的 trainer 一致；更新越小，丢掉的部分越多。

所以规则是：两端运行同一个 LoRA kernel，并且这个 kernel 要做到三点。第一，每一行自己
查找自己的 adapter；第二，归约顺序与 batch 里还有哪些行无关；第三，更新通过一次 fp32
融合乘加（FMA）加进基础累加器，最后只转换成 bf16 一次。只转换一次同时也是最精确的
做法：它的精度等同于把精确结果一次性舍入到 bf16。

## 步骤

1. **关掉引擎无法复现的部分。** trainer 计算 logprob 时关闭 LoRA dropout，引擎部署时不合并
   adapter，这样 trainer 重新算的就是引擎生成时用的同一个函数。检查两端都不做 dropout，也都不合并 adapter。
2. **trainer 和引擎使用同一个 LoRA 线性层**，即 [lora_kernels.py](../../examples/lora/lora_kernels.py)
   中的 `lora_linear` 和 `lora_linear_train`。前向时每一行选取自己的 adapter，降维乘法 x·A（即 shrink）
   按固定的 fp32 顺序累加，再用一次 fp32 FMA 把更新加进基础累加器，只转换一次；反向按同样的顺序累加，
   并跳过梯度为零的行，这样 padding 就不会移动分块边界。这样 trainer 与引擎、prefill 与 decode
   的结果都逐 bit 一致：[lora_bi.py](../../examples/lora/lora_bi.py) 中的检查全部通过。
3. **测试那些会让 grouped kernel 出错的 batch 排布。** 拿单独运行的请求，与同一个请求在这几种情形下的结果比较：
   旁边有其他 adapter、旁边有不用 adapter 的行、处在 batch 中的不同位置、夹在 padding 行之间。
   组大小要覆盖 GEMM 库每个切换 kernel 的阈值，阈值上下都要测到；还要包括 rank 用零补齐过的 adapter。
   所有排布下的结果都要相同（L2）。
4. **检查 padding 会不会改变 trainer 的反向。** 把单独一条序列的梯度，与同一条序列在右边补上 padding 后
   （padding 部分的输出梯度为零）的梯度比较，再与它和其他序列放在同一个 batch 里时的梯度比较。
   dX、dA 和 dB 都不能变，这说明梯度是 batch-invariant 的（L2），而不只是每次运行都能复现（L1）。

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 把 adapter 合并进权重 | 只做一次 GEMM，最快；但小的更新在 bf16 中会丢失，无法与 adapter 单独保存的 trainer 一致 | 免费 | 只做推理部署、不需要与 trainer 一致时 |
| 使用库里的 grouped 或 gather GEMM | adapter 保持独立，但一个请求的结果取决于同一个 batch 里还有哪些 token | 免费到中等（host 端开销随 adapter 数量增长） | 两端不需要一致时 |
| 两端使用同一个不变的 LoRA kernel | 前向和反向都达到 L1 到 L4，精度达到 bf16 所能达到的最好水平 | 前向为中等（不到 2 倍）到大（2 倍或更多）；训练 step 为大；adapter 很多时反而比 grouped GEMM 快 | 用 LoRA 做 RL、trainer 和引擎必须一致时 |

已证明：使用同一个 LoRA kernel 时，trainer 和引擎在 prefill 和 decode 中的结果都逐 bit
一致，所有排布测试都通过，padding 也不改变反向结果。这是在一种 GPU 型号、方形的层上
得到的（[报告](../../examples/lora/REPORT.md#results)）。尚未证明：真实引擎的 LoRA
路径（引擎的几种 kernel 写法是照着源码建模的，并没有实际运行）、打包的 QKV 切片、
TP，以及量化的基础权重。

## 可能出错的地方

!!! warning "只让基础 GEMM 不变还不够"
    LoRA 的 GEMM 有自己的 M，也就是共用这个 adapter 的 token 数，这一点和 MoE expert
    的 GEMM 一样。只让基础 GEMM 不变，排布测试仍然会失败。更新路径也必须不变。

!!! warning "缩放系数是 2 的幂时，看不出它在哪一步乘入"
    乘以 2 无论在哪一步做都是精确的，所以每种写法都一致。用这样的缩放系数测试会通过，
    换成真实的 adapter 却会失败。测试时要用不是 2 的幂的缩放系数。

!!! warning "用原子操作做 split-K，连 L1 都达不到"
    如果引擎的 shrink 把 K 维切开，再用原子操作把部分和加进 fp32 缓冲区，这个缓冲区
    每次运行得到的值都不一样。转换到 bf16 能掩盖大部分差异，但掩盖不了全部。
    应改用顺序固定的归约。

!!! warning "确定性不等于不变性"
    经过库 GEMM 的 trainer 反向，可能每次运行都完全相同，但 batch 一加 padding，结果就变了。
    梯度要做 L2 检查，不能只做 L1 检查。

!!! warning "补齐 rank 会改变 K"
    用零把小 adapter 的 rank 补大，它的 GEMM 的 K 就变了，GEMM 库可能因此换用另一个
    kernel。顺序固定的 FMA 循环加上的只是精确的零，不受影响。

## 小结

- LoRA 新增了三处可能让舍入不同的地方：共用 adapter 的 token 数、缩放系数在哪一步乘入、
  更新在哪里加回基础结果。
- 在 bf16 中合并 adapter 会丢掉一部分更新，所以部署合并权重的引擎无法与 adapter
  单独保存的 trainer 一致。
- 两端运行同一个 kernel，用一次 fp32 FMA 加入更新、只转换一次。这样既满足不变性，
  又是最精确的写法。
- trainer 重新计算 logprob 时，要关闭 LoRA dropout。

## 深入阅读

- [实验报告](../../examples/lora/REPORT.md)：逐一比较的各种写法、合并会丢掉什么的
  对照表、排布扫描、舍入点、精度和代价，以及 vLLM 的 LoRA kernel 如何舍入（根据源码分析）。
- 代码：[lora_kernels.py](../../examples/lora/lora_kernels.py)、
  [lora_bi.py](../../examples/lora/lora_bi.py)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
- 回归基线：[回归](../4-operations/regression.md)。
- 下一步：[引擎一致性](engine-parity-vllm.md)。
