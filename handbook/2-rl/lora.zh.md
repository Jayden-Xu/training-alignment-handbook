# LoRA：训练批次里一个适配器，服务批次里多个

## 我们在哪里

完成[整个模型](whole-model.md)、[并行](parallelism.md)和[采样](sampling.md)几步后，
每一侧的基础模型和采样器都已不变。RL 微调常常训练 LoRA 适配器，而适配器带来自己的 kernel。
trainer 在整个批次上运行一个适配器。引擎在一个批次中服务多个适配器，用分组或 gather 的
kernel，或者直接服务合并后的权重。这一步让 LoRA 层在两侧以相同方式舍入。

## 思路

LoRA 层计算 y = x·W + s·(x·A)·B：一个基础 GEMM、一个小的更新、一个缩放 s，
以及两者相遇的位置。除了基础 GEMM，它还新增了三个位可能改变的地方：

| 来源 | 小例子 | 破坏什么 |
|---|---|---|
| 适配器分组的大小 | 引擎把共用一个适配器的 token 分到一组，每组跑一次 GEMM。一个 3 个 token 的请求，单独时组大小是 3，在繁忙的批次里是 300，GEMM 可能选另一个 kernel | L2 |
| s 在哪里乘入 | s·(x·A)·B、(x·A)·(s·B) 和 (s·x·A)·B 在精确数学中相等；每次乘积后都舍入到 bf16 时，它们是三个不同的数（除非 s 是 2 的幂） | L4 |
| 基础部分和更新在哪里相加 | 两个 bf16 值相加会舍入两次；把 fp32 更新加到 fp32 的基础累加器上再转换一次，只舍入一次 | L4，以及精度 |

引擎中 prefill 和 decode 往往用不同的 LoRA kernel，这也会破坏 L3。

合并是一种形式的选择，不是故障。以 bf16 存储的 W + s·B·A，会丢掉每个小于 W 最后一位
一半的更新元素。所以服务合并权重的引擎无法与不合并的 trainer 一致，而且更新越小，丢得越多。

规则：两侧使用同一个 LoRA kernel。在这个 kernel 中，每一行自己选择适配器，
归约顺序从不依赖还有哪些行，基础部分和更新在一次 fp32 融合乘加中相遇，之后只有一次转换。
只转换一次也是最精确的形式：它达到精确结果的 bf16 舍入。

## 步骤

1. **关掉引擎无法复现的东西。**
   - 目标：trainer 计算 logprob 时算的就是引擎算的。
   - 做法：trainer 的 logprob 计算关闭 LoRA dropout，服务不合并的适配器。
   - 成功标准：两侧都没有 dropout，也没有合并。
2. **trainer 和引擎使用同一个 LoRA linear。**
   - 目标：trainer 的位 = 引擎的位，prefill = decode。
   - 做法：[lora_kernels.py](../../examples/lora/lora_kernels.py) 中的 `lora_linear` 和
     `lora_linear_train`：按行选择适配器、fp32 顺序固定的 shrink，一次 fp32 FMA 加到
     基础累加器，一次转换；反向使用相同顺序，并跳过梯度为零的行，这样 padding 不会移动分块边界。
   - 成功标准：[lora_bi.py](../../examples/lora/lora_bi.py) 中的检查通过。
3. **检查会让分组 kernel 失效的位置。**
   - 目标：请求在其他适配器之间时达到 L2。
   - 做法：把单独的请求和同一个请求与其他适配器、无适配器的行、不同位置和填充一起比较，
     组大小覆盖 GEMM 库的每个 kernel 切换阈值；rank 用零填充。
   - 成功标准：每种位置都相等。
4. **检查 trainer 的反向对 padding 是否敏感。**
   - 目标：梯度达到 L2，而不只是 L1。
   - 做法：把单独的序列和右侧填充（输出梯度为零）的同一序列比较，再和批次中有其他序列时比较。
   - 成功标准：dX、dA 和 dB 都不变。

| 选项 | 得到什么 | 代价类别 | 何时选择 |
|---|---|---|---|
| 合并权重 | 一次 GEMM，最快；在 bf16 中丢掉小的更新，无法与不合并的 trainer 一致 | 免费 | 只做服务，不需要一致性 |
| 分组或 gather 的库 GEMM | 不合并，但请求的位取决于一起批处理的 token | 免费到中等（主机开销随适配器数量增长） | 不需要一致性 |
| 两侧使用同一个不变的 LoRA kernel | 前向和反向达到 L1–L4，精度达到 bf16 下限 | 前向中等（不到 2x）到大（2x 或更多）；训练步大；适配器很多时比分组 GEMM 快 | 使用 LoRA 的 RL，trainer 和引擎必须一致 |

已证明：使用同一个 LoRA kernel 时，trainer 和引擎在 prefill 和 decode 中给出相同的位，
每种位置都通过，反向对 padding 不变；在一种 GPU 型号上、方形层
（[报告](../../examples/lora/REPORT.md#results)）。尚未证明：真实引擎的 LoRA 路径
（引擎的形式是根据源码建模的）、打包的 QKV 切片、张量并行、量化的基础权重。

## 可能出错的地方

!!! warning "基础 GEMM 不变还不够"
    LoRA 的 GEMM 有自己的 M，即共用这个适配器的 token 数，和 MoE 的专家一样。
    只让基础部分不变时，仍有位置失败。更新路径也要不变。

!!! warning "2 的幂次缩放会掩盖缩放位置"
    缩放为 2 时，不论在哪里乘入都是精确的，所以每种位置都一致。
    用这样的缩放测试会通过，真实的适配器却会失败。用不是 2 的幂的缩放来测试。

!!! warning "带原子操作的 split-K 连 L1 都不满足"
    引擎中把 K 切开、用原子操作加到 fp32 缓冲区的 shrink，每次运行都会改变这个缓冲区。
    转换到 bf16 掩盖了大部分，但不是全部。使用顺序固定的归约。

!!! warning "确定性不等于不变性"
    经过库 GEMM 的 trainer 反向，可能多次运行完全相同，却在批次被填充时改变。
    对梯度做 L2 检查，而不只是 L1。

!!! warning "rank 填充会改变 K"
    用零把小适配器的 rank 填充后，GEMM 的 K 变了，GEMM 库可能为它选另一个 kernel。
    顺序固定的 FMA 循环加上的是精确的零，不受影响。

## 小结

- LoRA 新增三个舍入来源：分组大小、缩放在哪里乘入、基础部分和更新在哪里相加。
- 在 bf16 中合并是一种有损的形式选择；服务合并权重的引擎无法与不合并的 trainer 一致。
- 两侧同一个 kernel、一次 fp32 FMA、一次转换：既不变，又是最精确的形式。
- trainer 计算 logprob 时关闭 LoRA dropout。

## 深入阅读

- [实验报告](../../examples/lora/REPORT.md)：比较过的每种形式、合并损失表、位置扫描、
  舍入点、精度、代价，以及 vLLM 的 LoRA kernel 怎样舍入（根据源码阅读）。
- 代码：[lora_kernels.py](../../examples/lora/lora_kernels.py)、
  [lora_bi.py](../../examples/lora/lora_bi.py)。
- Skill：[batch-invariant-kernel](../../skills/batch-invariant-kernel/SKILL.md)。
- 回归基线：[回归](../4-operations/regression.md)。
- 下一步：[引擎一致性](engine-parity-vllm.md)。
