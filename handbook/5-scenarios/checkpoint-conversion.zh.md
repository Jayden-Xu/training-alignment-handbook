# 检查点转换与重新分片

## 我们在哪里

在框架、并行布局或数值格式之间迁移，就要转换检查点：Hugging Face 转成 Megatron 风格的布局再转回来，
把分开的 q、k、v 投影合成一个融合的 QKV 张量，把 gate 和 up 合成一个张量，从 TP 2 改到 TP 4，
改变流水线阶段，或者做一次类型转换。转换不做任何计算，它只是复制。所以它的要求是本手册里最严格的：
每个逻辑参数的每一行都必须逐位到达。本页讲怎样检查这一点，以及副本检查看不到什么。
它是[框架迁移](framework-migration.md)的第 2 步，也是[强化学习循环](../2-rl/rl-loop.md)在每次权重同步时做的同一种检查。

## 思路：行在重新打包后保持不变

转换器会把行挪来挪去。这里的一行指权重的一个最后一维向量，比如投影的一个输出通道。
取一个 query group：两个 q 头、一个 k 头、一个 v 头，把它们的行叫作 q0、q1、k0、v0：

| 布局 | 存储的内容 |
|---|---|
| 分开的投影 | `q_proj`：q0 q1 · `k_proj`：k0 · `v_proj`：v0 |
| 融合，按 query group 排列 | `linear_qkv`：q0 q1 k0 v0 |
| 融合，交错方式错误 | `linear_qkv`：q0 k0 q1 v0 |

按布局自己的规则读回来（每组前两行是 q），错误的交错会得到一个由 q0 k0 组成的"q"。
按 TP rank 拆分会把整行挪到别的文件里；行并行层则把每一行切成几段，每个 rank 一段。
这些操作都不能改变任何一行的任何一位。

所以把每个逻辑参数当作一个**精确行的多重集合**来比较：在两边按布局规则把它拼出来，
再按字节匹配行，不管顺序。这样重新打包就不算差异。变了的行、跑到别的参数里的行、丢失的行会剩下来，
检查会给出它的名字：逻辑参数、行号，以及它来自哪个物理张量的哪一行。在上面的例子里，
k0 剩在"q"里，q1 剩在"k"里。

多重集合看不到的，是在正确参数里、但位置错了的行：q 里两个头互换了，或者为了另一种旋转位置编码约定而重排了行。
这是位置问题，需要第二项检查：在目标布局上用固定输入跑一次前向，与源端比较。

规则是：**先逐行证明副本，再用一次前向证明位置。**一般原则是：在变换必须保持的最细单位上比较，
并且使用不受变换允许改变的东西影响的形式。

## 步骤

1. **写映射。**
   - 目标：对每个逻辑参数，写明每种布局怎样存储它。
   - 做法：为 [ckpt_compare.py](../../tools/ckpt_compare.py) 写一个 `ckpt-map-v1` JSON 文件，
     每个逻辑参数、每种布局一条：张量名里用 `{i}` 表示层，用 `{r}` 表示 rank；`cat` 跨 rank 拼接；
     `part` 取融合张量里的分组；`slice` 去掉 padding；`replicated` 声明必须逐位相等的副本；
     `"dim": -1` 把行并行的分段拼回整行。[示例映射](../../examples/ckpt_conversion/mcore_map.json)
     描述了一种 Megatron 风格的布局：

     ```json
     "model.layers.{i}.self_attn.q_proj.weight": {
       "hf": "=",
       "mcore": {"cat": {"over": "r", "each": {
         "part": "mp_rank_{r:02d}/model.decoder.layers.{i}.self_attention.linear_qkv.weight",
         "pattern": ["{qrows}", "{hd}", "{hd}"], "take": 0}}}},
     "model.layers.{i}.self_attn.o_proj.weight": {
       "hf": "=",
       "mcore": {"cat": {"over": "r", "each":
         "mp_rank_{r:02d}/model.decoder.layers.{i}.self_attention.linear_proj.weight"}, "dim": -1}}
     ```
   - 成功标准：加上 `--strict` 时，源端没有任何张量被映射遗漏。
2. **比较行。**
   - 目标：每个逻辑参数的每一行都匹配。
   - 做法：在两个检查点上运行工具，指明映射里的两种布局和目标的 TP 度：

     ```bash
     python tools/ckpt_compare.py SRC_DIR DST_DIR --map map.json --layouts hf mcore --set b:r=2
     ```

     转换后的 QKV 分片里翻转了一位，输出是这样的：

     ```
     model.layers.1.self_attn.k_proj.weight: rows_unmatched
       A 1 of 64 rows unmatched: row 8 <- model.layers.1.self_attn.k_proj.weight[8]
       B 1 of 64 rows unmatched: row 8 <- mp_rank_00/model.decoder.layers.1.self_attention.linear_qkv.weight[40]
     NOT all rows matched
     ```
   - 成功标准："all rows matched"，退出码 0。每一处 dtype 变化都按参数报告：
     拓宽（bf16 到 fp32）是精确的，行在拓宽后比较；收窄会指出每一个丢了位的行。
3. **在目标布局上跑一次前向。**
   - 目标：每一行都在目标代码读取它的位置上。
   - 做法：把转换后的检查点加载进目标的模型代码，使用目标布局，输入固定的 token id，
     把 logits 与**使用相同归约**的源模型比较：TP 2 下的行并行层要加两个部分和，
     即使每个权重都对，位也会和 TP 1 不同（[并行](../2-rl/parallelism.md)）。
     示例 [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py) 在 CPU 上演示了这种做法。
   - 成功标准：logits 逐位相等。
4. **保留下来。**
   - 目标：转换器改动或换了新模型时，用同样的方式检查。
   - 做法：把这对转换连同植入的故障一起作为[回归](../4-operations/regression.md)条目保存。
   - 成功标准：套件报告干净的一对相等，每个植入的故障都被指出。

用哪种行键：

| 行键 | 能看到 | 看不到 | 代价等级 | 何时选择 |
|---|---|---|---|---|
| 按顺序对行字节做摘要（`--rows digest`，默认） | 行的任何变化 | 行内没有盲点 | 小，在 CPU 上离线计算 | 转换和重新分片的检查 |
| XOR 行指纹（`--rows xor`，OpGuard 的格式） | 大多数变化；与 OpGuard 在权重同步时记录的值相同 | 行内字的重排；同一行同一通道上相同的变化（类型转换会产生这种变化） | 在 GPU 上免费 | 在线的权重同步检查，或与 OpGuard 的记录比较 |

目标端的前向与什么比较：

| 参照 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|
| 使用目标归约方式的源模型（相同的切分、相同的部分和顺序） | 位置正确时逐位相等 | 小 | 你能以同样的切分运行源模型的行并行层 |
| 两边都使用逻辑 TP 大小（[并行](../2-rl/parallelism.md)） | 在任何并行度下逐位相等 | 中等（不到 2 倍） | 目标端已经使用与并行度无关的层 |
| 转回源布局，再跑源前向 | 逐位相等，但检查的是反向转换器，而不是目标端的读取方式 | 免费 | 拿不到目标端的模型代码 |
| logits 在容差以内 | 只能发现严重的错位 | 免费 | 最后手段，绝不能当作副本检查 |

已在 CPU 上以已知答案的演示证明：一个 Llama 形状的检查点在两种 TP 度下转换成 Megatron 风格布局再转回来，
是逐位精确的；工具匹配了每个逻辑参数的每一行，两种并行度之间也一样；它按参数和行指出了
一个转置的分片、一个错误的 QKV 交错、一次 bf16 到 fp16 再到 bf16 的类型转换、一个翻转的位和一个丢失的行；
对正确的转换，前向检查逐位相等，并且看到了行检查看不到的两个位置错误。它没有看到那次类型转换
（[报告](../../examples/ckpt_conversion/REPORT.md)）。尚未证明：Megatron-core 自己的加载器和分布式检查点格式、
流水线重新分片、词表 padding、真实模型的权重。

## 可能出错的地方

!!! warning "用前向检查充当副本检查"
    只舍入最小权重的类型转换，在少量输入上可能一个 logit 都不改变（实测：见报告）。
    权重变了，转换却通过了。用行来证明副本；前向只用来检查位置。

!!! warning "离线转换检查用 XOR 指纹"
    XOR 会把同一行同一通道上两个相同的变化抵消，而类型转换恰好会产生这样的成对变化；
    它也忽略行内字的重排，比如以错误顺序拼接的行并行分段。变了的行于是也能匹配。
    离线时用行摘要；XOR 留给与 OpGuard 记录比较的在线检查。

!!! warning "拿分片的前向和不分片的前向比较"
    TP 2 下的行并行层要把两个部分和相加，所以即使转换完美，大多数 logits 也和 TP 1 不同。
    检查会因为错误的原因失败。让两边使用相同的归约。

!!! warning "映射没有提到的张量"
    映射漏掉的逻辑参数永远不会被比较，所以丢掉它的转换器也能通过。
    用 `--strict` 运行，它会在有未使用的张量时失败，并且要读一下未使用张量的列表。

!!! warning "padding 和复制的副本"
    全零的 padding 行能和任何全零行匹配，证明不了位置；如果只读 rank 0，
    某个 rank 上过时的 norm 副本是看不到的。在映射里用切片去掉 padding，并把副本声明为 `replicated`。

## 小结

- 转换就是复制，所以每个逻辑参数的每一行都必须逐位到达。
- 把每个逻辑参数当作精确行的多重集合来比较：重新打包不算差异，剩下的每一行都连同来源被指出。
- 多重集合看不到参数内部挪了位置的行：在目标布局上跑前向检查位置，与使用相同归约的源端比较。
- 离线时用行字节摘要；XOR 指纹在行内有盲点。

## 深入阅读

- [报告：检查点转换演示](../../examples/ckpt_conversion/REPORT.md)：
  布局、映射、每个植入的故障以及每项检查看到了什么。
- 代码：[ckpt_compare.py](../../tools/ckpt_compare.py)、
  [convert.py](../../examples/ckpt_conversion/convert.py)、
  [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py)。
- [强化学习循环](../2-rl/rl-loop.md)：权重同步检查，在运行中的权重上做同样的行匹配。
- [并行](../2-rl/parallelism.md)：为什么 TP 度会改变前向的位，以及消除它的逻辑 TP 大小。
