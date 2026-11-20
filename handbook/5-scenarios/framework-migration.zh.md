# 框架迁移与库升级

## 我们在哪里

你有一个模型，原来用一套软件栈训练或推理，现在要搬到另一套软件栈上：从 Megatron 搬到
TorchTitan，从自研 trainer 换成某个库，或者还是同一个框架，只是升级了 torch、CUDA 或某个
融合 kernel 库。随之而来有两个问题：新的软件栈算的还是不是同一个东西？算得不一样的地方，
是 bug 还是有意的选择？本页回答这两个问题。做法是沿用[预训练工作流程](../1-pretraining/README.md)，
以旧软件栈为参考；再用[引擎一致性](../2-rl/engine-parity-vllm.md)里对齐舍入点的方法，
对每一处差异做出决定。

## 思路：同一个模型，写成了两个程序

实现同一个模型的两套软件栈，其实是两个不同的程序。它们划分模块边界的位置不同，融合的算子
不同，舍入的地方也不同。看一个小例子，一个注意力块：

| 步骤 | 软件栈 A | 软件栈 B |
|---|---|---|
| 注意力之前 | 先做 RMSNorm，再做 q、k、v 三个投影 | 一个融合 kernel，同时完成 norm 和打包的 QKV 投影 |
| 残差相加 | 用 bf16 做 | 用 fp32 做，之后统一转换一次 |
| loss | 在整个词表上做 log_softmax | 词表并行的交叉熵，跨 rank 求和 |

从这张表可以看出三件事。

1. **两边的 trace 对不齐。** 软件栈 A 有一个 norm 边界和三个投影边界，软件栈 B 只有一个边界。
   如果按位置给事件配对，就会凭空造出差异。应该按内容配对：只要一行在两边逐 bit 一致，
   它就是同一个值，不管出现在哪里。
2. **每一处差异要么是 bug，要么是选择。** 用错了 RoPE 变体是 bug；残差用 fp32 是选择。
   两者表现出来都是结果不一致，所以要对每一处差异单独做决定，并写下它的代价。
3. **升级是最简单的情况。** 同一个框架的两个版本，边界都相同，只有一处变化；首个分歧点
   通常就指向新版本换掉的那个 kernel。

所以规则是：**只在两套软件栈共有的边界上比较，按内容给行配对，再把剩下的每一处差异写成
一个有名字、有决定的舍入点。**这就是第 1 部分的对照实验，只是用在了两个实现之间；同时也是
引擎一致性的原则：要结果相同，需要的是运算过程相同，而不只是公式相同。

<figure class="fx" data-fig="migration" aria-label="框架迁移这一步，以一个注意力块为例">
<div class="fx-stage">
<div class="fx-steps"><span class="fx-step" data-hot="0">1. 比较预检报告</span><span class="fx-step" data-hot="1">2. 对齐数据和权重</span><span class="fx-step" data-hot="2">3. 采集两个软件栈</span><span class="fx-step is-key" data-hot="3">4. 找首个分歧点</span><span class="fx-step" data-hot="4">5. 列出舍入点</span><span class="fx-step" data-hot="5">6. 逐处做决定</span><span class="fx-step" data-hot="6">7. 留作回归检查</span></div>
<div class="fx-grid" style="--cols:4">
<span class="fx-label">软件栈 A</span><span class="fx-node" data-hot="2">RMSNorm</span><span class="fx-node" data-hot="2">q, k, v</span><span class="fx-node" data-hot="3">注意力</span><span class="fx-node to-bad" data-hot="3 4">残差相加，bf16</span>
<span></span><span class="fx-pair" data-at="2" style="grid-column: span 2">没有共同的边界，所以不比较</span><span class="fx-pair" data-at="3">共同的边界，按内容给行配对</span><span class="fx-swap fx-pair" data-at="3"><span data-to="4">共同的边界，按内容给行配对</span><span class="fx-pair is-bad" data-at="4">一个舍入点：一边 bf16，一边 fp32</span></span>
<span class="fx-label">软件栈 B</span><span class="fx-node is-ghost" data-hot="2" style="grid-column: span 2">融合的 norm + 打包的 QKV</span><span class="fx-node" data-hot="3">注意力</span><span class="fx-node to-bad" data-hot="3 4">残差相加，fp32</span>
</div>
</div>
<ol class="fx-say">
<li>1. 比较两份预检报告，确认两次运行只有软件栈不同。</li>
<li>2. 让两个软件栈看到同样的 token，并从同样的权重开始。</li>
<li data-hold="1.5">3. 在两个软件栈共有的边界上采集。软件栈 B 把 norm 和投影融合在了一起，所以这两处不参与比较。</li>
<li data-hold="1.5">4. 在共有的边界上按内容给行配对，找出第一个输入行相同、输出行却不同的边界。</li>
<li data-hold="1.5">5. 把两个前向之间的每一处差异都列成一个舍入点，比如残差相加一边用 bf16、一边用 fp32。</li>
<li>6. 对每一处差异做决定：它是要修的 bug，还是要保留的选择，并记下测得的影响。</li>
<li data-hold="2">7. 把这一对运行留作回归检查，下次升级时就能看出哪里变了。</li>
</ol>
<figcaption>图：只在两个软件栈共有的边界上比较，按内容给行配对，再把剩下的每一处差异写成一个有名字、有决定的舍入点。</figcaption>
</figure>

## 步骤

1. **比较预检报告**，确认两次运行只有软件栈不同。在每个训练任务里运行
   [determinism_preflight.py](../../tools/determinism_preflight.py) 的 `report`，再比较两份报告，
   并声明软件栈就是有意引入的那个差异（[确定性与预检](../1-pretraining/determinism-and-preflight.md)）。
   两个框架对同一个控制项常常叫法不同（确定性开关、集合通信算法、融合 kernel 的开关），所以要按
   含义来比较，并列出每一端各自设置了什么。diff 里应当只有软件栈本身和它自己的开关，没有别的。
2. **对齐数据和权重**，让两套软件栈按相同的顺序看到相同的 token，并从相同的权重开始。按
   [数据对齐](../1-pretraining/data-alignment.md)里的方法对齐 token 流。如果 checkpoint 格式不同，
   就只转换一次，并用 [ckpt_compare.py](../../tools/ckpt_compare.py) 逐 bit 检查这次转换
   （[checkpoint 转换](checkpoint-conversion.md)）。前几步的 token id 两边必须相等，每个权重的
   每一行都必须找到对应的行。
3. **采集两套软件栈**，在两边都有的每个边界上取得 fingerprint：embedding 的输出、每一层注意力和
   MLP 的输出、残差流、logits，训练时还有每个参数的梯度。使用
   [opguard_bridge.py](../../tools/opguard_bridge.py) 里的 `opguard_bridge.run_pair`，并打开
   OpGuard 的逐向量 fingerprint（`OPGUARD_XOR_ROWS=1`），这样才能按内容给行配对；如果代码用了
   torch.compile 或 CUDA graph，就用 [graph probe](../1-pretraining/capture-compiled-code.md)。
   `opguard_bridge.coverage(run)` 应当显示两边都采到了这些边界。
4. **找首个分歧点**，也就是第一个共有边界，在那里相同的输入行给出了不同的输出行。两边调用顺序
   一致时，OpGuard 会自动给事件配对（[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md)），
   但跨软件栈时，调用顺序往往对不上。这时就用 OpGuard 的 token 匹配，按 fingerprint 给行配对，
   从两边一致的锚点（比如 embedding 的输出）出发，沿着数据流往下走，就像它的 `rl-alignment`
   skill 那样。你依赖的每一处对应关系，都要连同证据记成 OpGuard alignment map 里的一条记录
   （见 OpGuard 的 `docs/AlignmentMaps.md`）。最后你会找到一个 pivot，它在两边的输入相等、输出
   不同，同时有一条 alignment map 记录说明这两个边界为什么是对应的。
5. **写出舍入点表**，把两个前向之间（训练时还有两个反向之间）的每一处差异都写成有名字的一行。
   按 [engine-parity skill](../../skills/engine-parity/SKILL.md) 的第 1、2 步来做：先盘点每一端
   实际运行的是什么，再以[舍入点](../../skills/engine-parity/references/rounding-points.md)表为
   检查清单，写出两边的舍入序列，每一行都标明是“根据源码阅读”还是“测量所得”。pivot 处的那个
   差异，必须是表里的某一行。
6. **逐处做决定**，并实测它的影响。像[引擎一致性](../2-rl/engine-parity-vllm.md)那样，每次只在
   一端改一个舍入点，然后重新采集；每改一次，第一个出现差异的边界都应该往模型后面移。不能留下
   任何一处没有决定的差异。
7. **留作回归检查**，下一次升级时就能看出哪里变了。对每个决定，挑出最小的、能体现它的一对运行
   （一层、一步），作为示例放进[回归](../4-operations/regression.md)套件，并把这些决定写在旁边。
   回归套件跑它时会报告一致，或者指出变化的那一行。

| 决定 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 把新软件栈改成和旧的一样舍入 | 与参考逐 bit 一致 | 小到中等，按舍入点计 | 旧软件栈是已经验证过的参考，或者这处差异本来就是 bug |
| 把旧软件栈改成和新的一样舍入 | 两边都认可的新参考 | 中等，因为旧结果要重新确立 | 新写法更好（舍入更少，或更快），而且你打算淘汰旧软件栈 |
| 接受差异并记录下来 | 一个已知、已测量的差异 | 免费 | 两种写法都是同一个数学计算的精确形式，而且两边都改不了（闭源 kernel、编译器做的融合） |

这些代价等级是根据每个决定额外增加的工作量做出的判断，并不是从真实迁移中测出来的。

如果只是**同一个框架内的升级**，通常第 1、3、4 步就够了。先跑一遍[回归](../4-operations/regression.md)
回归套件，因为它会告诉你哪些原生 kernel 变了；再采集新旧两个版本，看 pivot 在哪里。

## 可能出错的地方

!!! warning "跨软件栈按位置给事件配对"
    两套软件栈的边界不同，一端的第 n 个事件，并不是另一端的第 n 个事件。这样硬配出来的对，
    会报告并不存在的差异。要按内容给行配对，找不到对应项的事件就当作记录里的空缺。

!!! warning "权重或数据其实并不相同"
    转换时改动了几行，或者 shuffle 方式不同，都会让从第一个边界开始的每个边界都出现差异。
    这时 pivot 指向的是 embedding 或第一个投影，而不是真正的原因。采集之前，先检查转换结果
    和 token id。

!!! warning "把选择当成 bug"
    融合的 kernel 和同一个数学计算的非融合写法，本来就会舍入不同。硬要它们一致，付出的代价
    可能比这处差异本身的影响还大。把两种写法都写清楚，测出差距，再做决定。

!!! warning "只拿 loss 曲线当证据"
    两条 loss 曲线可能完全吻合，而某个边界其实是错的；也可能只因为一个无害的舍入点，在很多步
    之后慢慢分开。曲线不会告诉你差异在哪里。要在边界上比较。

## 小结

- 两套软件栈是两个程序，trace 对不齐，所以要按内容给行配对，并把每一处对应关系连同证据记下来。
- 先排除其他所有差异：比较预检报告，使用相同的数据，逐 bit 检查权重。
- 找到第一个“输入相同、输出不同”的边界，再把这处差异命名为一个舍入点。
- 对每一处差异做决定：要么改其中一端，要么接受它，并连同实测影响一起记录。最后把这一对运行
  留作回归检查。

## 深入阅读

- [预训练工作流程](../1-pretraining/README.md)和[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md)：
  讲采集、配对和首个分歧点。
- [引擎一致性](../2-rl/engine-parity-vllm.md)和 [engine-parity skill](../../skills/engine-parity/SKILL.md)：
  讲本页沿用的对齐舍入点的方法。
- 权重见[checkpoint 转换](checkpoint-conversion.md)，token 见[数据对齐](../1-pretraining/data-alignment.md)。
- 跨实现的配对，见 OpGuard 的 alignment map（`docs/AlignmentMaps.md`）和它的 `rl-alignment` skill
  （`harness/skills/rl-alignment`）。
- 尚未证明：两个训练框架之间的完整迁移。[跨引擎案例的报告](../../examples/cross_engine/REPORT.md)
  展示了在 trainer 和引擎之间对齐舍入点的方法。
