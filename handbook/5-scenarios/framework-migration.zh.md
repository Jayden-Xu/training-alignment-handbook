# 框架迁移与库升级

## 我们在哪里

你有一个由某个技术栈训练或服务的模型，现在要把它搬到另一个技术栈：从 Megatron 到 TorchTitan，
从自研 trainer 到某个库，或者同一个框架在升级 torch、CUDA、融合 kernel 库之后。随之而来两个问题：
新技术栈算的是不是同一个东西？算得不一样的地方，是 bug 还是选择？本页用
[预训练工作流程](../1-pretraining/README.md)回答这两个问题，以旧技术栈作参考；
再用[引擎一致性](../2-rl/engine-parity-vllm.md)的舍入点方法，对每一处差异做出判断。

## 思路：一个模型，两个程序

实现同一个模型的两个技术栈是两个程序。它们的模块边界不同，融合的算子不同，舍入的位置也不同。
一个小例子，一个注意力块：

| 步骤 | 技术栈 A | 技术栈 B |
|---|---|---|
| 注意力之前 | RMSNorm，然后 q、k、v 三个投影 | 一个融合 kernel：norm 加打包的 QKV 投影 |
| 残差相加 | bf16 | fp32，之后统一转换一次 |
| 损失 | 在整个词表上做 log_softmax | 词表并行的交叉熵，跨 rank 求和 |

这张表说明了三件事。

1. **trace 对不齐。**技术栈 A 有一个 norm 边界和三个投影边界，技术栈 B 只有一个。
   按位置配对事件会凭空制造差异。改为按内容配对：两边位相同的一行就是同一个值，不管它出现在哪里。
2. **差异要么是 bug，要么是选择。**用错了 RoPE 变体是 bug；fp32 残差是选择。两者都表现为位不同，
   所以要对每处差异单独判断，并写下它的代价。
3. **升级是简单的情况。**同一个框架的两个版本边界相同，只有一处变化；首个分歧点通常就指向新版本换掉的那个 kernel。

规则是：**在两个技术栈共有的边界上比较，按内容配对行，把剩下的每处差异写成一个有名字、有决定的舍入点。**
这就是第 1 部分的对照实验用在两个实现上，也是引擎一致性的原则：结果相等需要算术相等，而不是公式相等。

<figure class="fx" data-fig="migration" aria-label="框架迁移这一步，以一个注意力块为例">
<div class="fx-stage">
<div class="fx-steps"><span class="fx-step" data-hot="0">1. 预检 diff</span><span class="fx-step" data-hot="1">2. 对齐数据和权重</span><span class="fx-step" data-hot="2">3. 两边都采集</span><span class="fx-step is-key" data-hot="3">4. 首个分歧点</span><span class="fx-step" data-hot="4">5. 舍入点表</span><span class="fx-step" data-hot="5">6. 逐处决定</span><span class="fx-step" data-hot="6">7. 回归</span></div>
<div class="fx-grid" style="--cols:4">
<span class="fx-label">栈 A</span><span class="fx-node" data-hot="2">RMSNorm</span><span class="fx-node" data-hot="2">q, k, v</span><span class="fx-node" data-hot="3">注意力</span><span class="fx-node to-bad" data-hot="3 4">残差相加，bf16</span>
<span></span><span class="fx-pair" data-at="2" style="grid-column: span 2">没有共同边界：不比较</span><span class="fx-pair" data-at="3">共同边界：按内容配对</span><span class="fx-swap fx-pair" data-at="3"><span data-to="4">共同边界：按内容配对</span><span class="fx-pair is-bad" data-at="4">舍入点：bf16 对 fp32</span></span>
<span class="fx-label">栈 B</span><span class="fx-node is-ghost" data-hot="2" style="grid-column: span 2">融合的 norm + 打包的 QKV</span><span class="fx-node" data-hot="3">注意力</span><span class="fx-node to-bad" data-hot="3 4">残差相加，fp32</span>
</div>
</div>
<ol class="fx-say">
<li>1. 比较两份预检报告：两次运行只在软件栈上不同。</li>
<li>2. 两个栈看到同样的 token，从同样的权重开始。</li>
<li data-hold="1.5">3. 在两个栈共有的边界上采集。栈 B 融合了 norm 和投影，所以它们不参与比较。</li>
<li data-hold="1.5">4. 在共同边界上按内容给行配对；取第一个输入行相同而输出行不同的边界。</li>
<li data-hold="1.5">5. 舍入点表：两个前向之间的每个差异，比如残差相加用 bf16 还是 fp32。</li>
<li>6. 逐处决定：是要修的 bug，还是要保留的选择，并写下测得的影响。</li>
<li data-hold="2">7. 把这一对作为回归检查保留，下次升级时就能看出什么变了。</li>
</ol>
<figcaption>图：在两个栈共有的边界上比较，按内容给行配对，把剩下的每个差异变成一个有决定的、有名字的舍入点。</figcaption>
</figure>

## 步骤

1. **比较预检报告。**
   - 目标：两次运行只在技术栈上不同。
   - 做法：在每个作业里运行 [determinism_preflight.py](../../tools/determinism_preflight.py)
     的 `report`，比较两份报告，把技术栈声明为差异（[确定性与预检](../1-pretraining/determinism-and-preflight.md)）。
     两个框架对同一种控制的叫法不同（确定性开关、集合通信算法、融合 kernel 开关）：按含义比较，
     列出每一侧设置了什么。
   - 成功标准：diff 只列出技术栈本身和它自己的开关。
2. **对齐数据和权重。**
   - 目标：两个技术栈以相同顺序看到相同的 token，并从相同的权重开始。
   - 做法：按[数据对齐](../1-pretraining/data-alignment.md)对齐 token 流。检查点格式不同时，转换一次，
     并用 [ckpt_compare.py](../../tools/ckpt_compare.py) 逐位检查这次转换（[检查点转换](checkpoint-conversion.md)）。
   - 成功标准：前几步的 token id 两边相等，每个权重的每一行都匹配。
3. **两个技术栈都采集。**
   - 目标：在两边都有的边界上取得指纹：embedding 输出、每层注意力和 MLP 的输出、残差流、logits，
     训练时还有每个参数的梯度。
   - 做法：使用 [opguard_bridge.py](../../tools/opguard_bridge.py) 里的 `opguard_bridge.run_pair`，
     打开 OpGuard 的逐向量指纹（`OPGUARD_XOR_ROWS=1`），这样行可以按内容配对；
     编译后的代码用 [graph probe](../1-pretraining/capture-compiled-code.md)。
   - 成功标准：`opguard_bridge.coverage(run)` 显示两边都采到了这些边界。
4. **取首个分歧点。**
   - 目标：第一个共有边界，在那里相同的输入行给出不同的输出行。
   - 做法：调用顺序一致时，OpGuard 会自动配对事件（[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md)）。
     跨技术栈时往往不一致：用 OpGuard 的 token 匹配按指纹配对行，从两边一致的锚点（比如 embedding 输出）
     出发沿数据流走，就像它的 `rl-alignment` skill 那样。把你依赖的每一处对应关系连同证据，
     记成 OpGuard alignment map 的一条记录（OpGuard 的 `docs/AlignmentMaps.md`）。
   - 成功标准：一个两边输入相等、输出不同的 pivot，以及一条说明这两个边界为什么对应的 alignment map 记录。
5. **写出舍入点表。**
   - 目标：两个前向（训练时还有反向）之间的每一处差异都成为有名字的一行。
   - 做法：[engine-parity skill](../../skills/engine-parity/SKILL.md) 的第 1、2 步：
     清点每一侧实际运行的内容，再以[舍入点](../../skills/engine-parity/references/rounding-points.md)表为清单，
     写出两边的舍入序列；每行标明"读自源码"或"实测"。
   - 成功标准：pivot 处的差异是其中一行。
6. **逐处决定。**
   - 目标：每一行都有决定和实测的影响。
   - 做法：像[引擎一致性](../2-rl/engine-parity-vllm.md)那样，一次改一侧的一个舍入点，然后重新采集；
     每改一次，第一个不同的边界应该往后移。
   - 成功标准：没有一处差异没有决定。
7. **作为回归检查保留。**
   - 目标：下一次升级时能看出什么变了。
   - 做法：把能体现每个决定的最小配对（一层、一步）作为示例放进[回归](../4-operations/regression.md)套件，
     并在旁边写下这些决定。
   - 成功标准：套件运行它并报告相等，或者报告变化的那一行。

| 决定 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 把新技术栈改成旧技术栈的舍入方式 | 与参考逐位相等 | 小到中等，按舍入点计 | 旧技术栈是已验证的参考，或者这处差异是 bug |
| 把旧技术栈改成新技术栈的方式 | 两边都认可的新参考 | 中等：旧结果要重新确立 | 新形式更好（舍入更少、更快），而且你会淘汰旧技术栈 |
| 接受并记录 | 一个已知、已测量的差异 | 免费 | 同一数学的两种精确形式，两边都无法改动（闭源 kernel、编译器融合） |

这些代价等级是根据每个决定增加的工作量做出的判断，不是对某次迁移的测量。

对**同一框架内的升级**，通常第 1、3、4 步就够了：先跑[回归](../4-operations/regression.md)套件，
它会告诉你哪些标准 kernel 变了；再采集新旧两个版本，读 pivot。

## 可能出错的地方

!!! warning "跨技术栈按位置配对事件"
    两个技术栈的边界不同，一侧的第 n 个事件不是另一侧的第 n 个。强行配对会报告并不存在的差异。
    按内容配对行，没有伙伴的事件留作空缺。

!!! warning "权重或数据并不相同"
    一次改动了几行的转换，或者不同的 shuffle，会让从第一个边界开始的每个边界都不同。
    pivot 于是指向 embedding 或第一个投影，而不是原因。采集之前先检查转换和 token id。

!!! warning "把选择当成 bug"
    融合 kernel 和同一数学的非融合算子链，本来就舍入不同。强行让它们一致，代价可能超过这处差异本身的影响。
    写下两种形式，测量差距，然后做决定。

!!! warning "只看损失曲线"
    两条损失曲线可能一致，而某个边界是错的；也可能因为一个无害的舍入点，在很多步之后分开。
    曲线不会告诉你在哪里。要在边界上比较。

## 小结

- 两个技术栈是两个程序：trace 对不齐，所以按内容配对行，并把每处对应关系连同证据记下来。
- 先去掉其他所有差异：预检 diff、相同的数据、逐位检查过的权重。
- 取第一个相同输入给出不同输出的边界，再把这处差异命名为一个舍入点。
- 逐处决定：改一侧，或者接受并连同实测影响一起记录；把这对配置保留为回归检查。

## 深入阅读

- [预训练工作流程](../1-pretraining/README.md)和[用 OpGuard 定位](../1-pretraining/localize-with-opguard.md)：
  采集、配对和首个分歧点。
- [引擎一致性](../2-rl/engine-parity-vllm.md)和 [engine-parity skill](../../skills/engine-parity/SKILL.md)：
  本页复用的舍入点方法。
- 权重见[检查点转换](checkpoint-conversion.md)，token 见[数据对齐](../1-pretraining/data-alignment.md)。
- 跨实现配对见 OpGuard 的 alignment map（`docs/AlignmentMaps.md`）和它的 `rl-alignment` skill
  （`harness/skills/rl-alignment`）。
- 尚未证明：两个训练框架之间的完整迁移；
  [跨引擎案例的报告](../../examples/cross_engine/REPORT.md)展示了 trainer 和引擎之间的舍入点方法。
