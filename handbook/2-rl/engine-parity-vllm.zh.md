# 让两侧一致：引擎一致性（L4）

## 我们在哪里

经过前面几步，每一侧各自都已不变：[整个模型](whole-model.md)在每种服务模式下、
每个[并行度](parallelism.md)下，加上可重放的[采样器](sampling.md)和 [LoRA](lora.md)。
在 RL 中，引擎通常不是你的代码：它是 vLLM 或 SGLang，有自己的模型代码、kernel 和编译器。
这一步让 trainer 的 logprob 与引擎的逐位相等（L4）。

## 思路

两个不变的侧仍然可能不同，因为它们在不同的位置舍入。**舍入点**是指把值转换到更窄类型的
任何位置，或者把几个运算合在一次舍入中的位置。一个来自 RMSNorm 的小例子，
x̂ 是 fp32 的归一化行，w 是 bf16 权重：

| 一侧 | 计算的是什么 | 舍入 |
|---|---|---|
| A | bf16(x̂) · w，再转 bf16 | 两次：x̂ 的转换，然后是乘积 |
| B | bf16(x̂ · w) | 一次：fp32 中相乘，然后转换 |

两者都正确，但很多元素的位不同。一个模型有几十个这样的点：投影怎样打包、偏置在哪里加、
norm、RoPE 表和公式、注意力 kernel、lm_head、log_softmax。logits 上游的每一个都对上之后，
才会有 logprob 相等。所以不同 token 的数量可能在几次修复中一直接近全部，
到最后一次才降到零。

方法由此而来：

1. **先检查引擎自身。** 如果其他请求加入批次时，引擎对某个 token 的 logprob 会改变，
   就没有 trainer 能对上它。
2. **写下两侧的舍入序列**，按顺序，每个舍入点一行。
3. **一次对齐一个舍入点**，每次之后都测量，并观察第一个不同的层向后移动，
   而不只看 token 数。
4. **在 trainer 中优先使用引擎自己的 kernel**，而不是重新实现。
   源码相同不等于位相同，除非 tile 形状、归约顺序和编译器也相同。

一般原则和 kernel 一样：结果相等需要算术相等，而不是公式相等。

## 步骤

[engine-parity skill](../../skills/engine-parity/SKILL.md) 把这些动作作为一个工作流程来执行。

1. **盘点两侧的前向。**
   - 目标：每一侧实际运行的是什么。
   - 做法：从源码和加载后的模型中找：打包和融合的模块、自定义算子、编译器和 CUDA graphs、
     实际选中的注意力后端、每个会改变 kernel 的开关（skill 第 1 节）。
   - 成功标准：模块树和后端是从运行中的引擎打印出来的，而不是取自你传入的配置。
2. **写下两侧的舍入序列。**
   - 目标：一份差异检查清单。
   - 做法：使用[舍入点](../../skills/engine-parity/references/rounding-points.md)中的表；
     每条标明“根据源码阅读”或“测量所得”。
   - 成功标准：两侧之间的每个差异都是有名字的一行。
3. **检查引擎自身。**
   - 目标：在与 trainer 比较之前，引擎先达到 L1 和 L2。
   - 做法：同一批次跑两次；每个 prompt 单独跑，再与几种大小的批次比较；
     如果引擎有批不变模式就切换过去，并阅读这个模式在你的 GPU 上覆盖了哪些算子。
   - 成功标准：每个 prompt 的 logprob 在单独和批处理时都相等。否则把它报告为引擎波动，
     在引擎中修。
4. **一次一个舍入点地对齐 trainer。**
   - 目标：实际 token 的 logprob 达到 L4。
   - 做法：把引擎自己的 token 以 teacher forcing 方式送入 trainer，比较 fp32 logprob 的位，
     不设容差；改一个点，再数一次；
     [trainer.py](../../examples/cross_engine/trainer.py) 和
     [cross_engine.py](../../examples/cross_engine/cross_engine.py) 是模板。
   - 成功标准：不同的 token 数为零；然后用逐一回退的方式看出哪些点有影响。
5. **定位剩下的差异。**
   - 目标：仍然不同的第一个算子。
   - 做法：算子重放（把捕获的每个引擎输入喂给每个 trainer 候选实现）和逐层比较残差流；
     对编译过的引擎，用边界指纹代替 hook（[RL 循环](rl-loop.md)）。
   - 成功标准：一个有名字的算子，并有来自两侧的证据。

| 途径 | 做什么 | 代价类别 | 何时选择 |
|---|---|---|---|
| 让 trainer 改用引擎的 kernel | trainer 调用引擎自己的 GEMM、norm、注意力和 log_softmax | 引擎侧因其不变模式而中等（不到 2x）到大；trainer 前向免费 | 引擎固定，trainer 归你管；trainer 还需要前向位相同的反向 kernel |
| 让引擎改用 trainer 的舍入 | 拆开并重排引擎的算子，让类型转换跟随 trainer；两侧都用不变算子 | rollout 吞吐量中等（不到 2x）到大（报告值） | trainer 固定，你可以给引擎打补丁 |
| 两侧共用一份模型定义 | 引擎只保留缓存管理；前向共享，只剩 kernel 需要对齐 | 某些架构上 trainer 吞吐量代价大（2x 或更多）（报告值） | 你能修改两个框架 |
| 一份执行契约 | 固定两侧每个与舍入有关的选择并用哈希检查；并行不变的 kernel | 中等（不到 2x）（报告值） | 多个框架需要长期保持对齐 |

已证明：调用引擎自己 kernel 的 trainer，在不同批大小下与一个真实引擎的实际 token logprob
逐位相等；一个模型、一种 GPU 型号、eager 模式、只有前向
（[报告](../../examples/cross_engine/REPORT.md#results)）。其他途径的报告结果收集在
[报告](../../examples/cross_engine/REPORT.md#three-ways-to-reach-parity)中。尚未证明：
编译过的引擎、L4 下的张量并行、带匹配反向 kernel 的训练步。

## 可能出错的地方

!!! warning "引擎的批不变模式不一定在每种 GPU 上覆盖每个算子"
    一个模式开关可能只把部分算子换成不变的 kernel，其余仍留在 GEMM 库上，这取决于
    GPU 架构。这时引擎仍随批次变化，trainer 在追噪声。阅读你的 GPU 对应的代码路径，
    并自己检查引擎的 L2。

!!! warning "比较了错误的 logprob"
    引擎报告的 logprob 可能经过了温度、惩罚项或其他处理，或者在和你不同的设备上计算
    log_softmax。比较结果就会显示一个不在模型里的差距。先确认报告的 logprob 等于
    按引擎的方式对捕获的 logits 计算的 log_softmax。

!!! warning "hook 在编译图里不会触发"
    编译器会把 norm、RoPE 和激活函数融合进相邻的算子，引擎在编译时还可能关掉自己的
    自定义算子。模块级捕获在那里什么也看不到。使用边界指纹，或者先对齐 eager 引擎。

!!! warning "token 数不变不等于这一步失败"
    logits 上游的每个舍入点都对上之后，才会有 token 相等。用 token 数评判每次修复，
    会放弃正确的修复。每次改动后跟踪第一个不同的层。

!!! warning "回退后没有变化，证明不了什么"
    在逐一回退中，回退后没有 token 改变的点，可能在其他 prompt 或更长的位置上有影响
    （RoPE 表可能只在某个位置之后才不同）。说“这里未显示有影响”，不要说“无关”。

## 小结

- 不变的两侧仍可能不同：它们必须在相同的位置舍入。
- 先检查引擎自身；不满足 L2 的引擎无法对齐。
- 写下两侧的舍入序列，一次对齐一个点，并跟踪第一个不同的层。
- 在 trainer 中调用引擎自己的 kernel，而不是重新实现。

## 深入阅读

- [实验报告](../../examples/cross_engine/REPORT.md)：一个真实引擎和模型、步骤表、
  逐一回退、引擎自身的波动、它的不变模式在哪种 GPU 上覆盖哪些算子、代价，
  以及其他团队报告的结果。
- Skill：[engine-parity](../../skills/engine-parity/SKILL.md) 和
  [舍入点](../../skills/engine-parity/references/rounding-points.md)。
- 代码：[examples/cross_engine/](../../examples/cross_engine/)。
- 下一步：[RL 循环](rl-loop.md)。
