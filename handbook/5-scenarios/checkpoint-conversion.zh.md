# checkpoint 转换与重新切分

## 我们在哪里

在不同框架、不同并行布局或不同数值格式之间迁移，就免不了要转换 checkpoint：从 Hugging Face
格式转成 Megatron 风格的布局再转回来，把分开的 q、k、v 投影合成一个融合的 QKV 张量，把 gate
和 up 合成一个张量，从 TP 2 改成 TP 4，改变流水线的段数，或者换一种 dtype。转换不做任何计算，
它只是复制。所以它的要求是本手册里最严格的：每个逻辑参数的每一行，都必须逐 bit 原样复制过去。
本页讲怎样检查这一点，以及只检查副本时看不到什么。它是[框架迁移](framework-migration.md)的
第 2 步，也和 [RL 循环](../2-rl/rl-loop.md)在每次权重同步时做的检查是同一种。

## 思路：checkpoint 重新打包后，每一行的值都不变

转换器会把行挪来挪去。这里说的一行，指权重沿最后一维的一个向量，比如投影的一个输出通道。
以一个 query group 为例：两个 q head、一个 k head、一个 v head，把它们的行记作 q0、q1、k0、v0：

| 布局 | 存储的内容 |
|---|---|
| 分开的投影 | `q_proj`：q0 q1 · `k_proj`：k0 · `v_proj`：v0 |
| 融合，按 query group 排列 | `linear_qkv`：q0 q1 k0 v0 |
| 融合，但交错方式错了 | `linear_qkv`：q0 k0 q1 v0 |

如果按布局本身的规则（每组前两行是 q）把最后一种读回来，得到的“q”就是 q0 和 k0。按 TP rank
拆分，会把整行挪到别的文件里；行并行层则把每一行切成几段，每个 rank 拿一段。所有这些操作，
都不允许改变任何一行里的任何一个 bit。

所以要把每个逻辑参数当作一个**由精确的行组成的多重集合**来比较：在两边各自按布局规则把它
拼出来，再按字节给行配对，不管顺序。这样一来，重新打包就不算差异。改变了的行、跑到别的参数
里的行、丢失的行，都会剩下来，检查会把它指出来：是哪个逻辑参数的哪一行，又来自哪个物理张量
的哪一行。在上面的例子里，k0 会剩在“q”里，q1 会剩在“k”里。

多重集合看不到的，是那些参数对了、位置却错了的行：q 里有两个 head 互换了，或者为了另一种
RoPE 约定把行重新排了序。这是位置问题，需要第二项检查：在目标布局上用固定的输入跑
一次前向，再和源端比较。

所以规则是：**先逐行证明副本没错，再用一次前向证明位置没错。**背后的通用原则是：在变换必须
保持不变的最细粒度上比较；比较的形式，要不受变换本来就允许改变的那些东西影响。

## 步骤

1. **写映射。**对每个逻辑参数，写明每种布局是怎样存储它的：为
   [ckpt_compare.py](../../tools/ckpt_compare.py) 写一个 `ckpt-map-v1` 格式的 JSON 文件，
   每个逻辑参数、每种布局各一条。张量名里用 `{i}` 表示层号，用 `{r}` 表示 rank；`cat` 把各 rank
   的分段拼起来，`part` 从融合张量里取出某一组，`slice` 去掉 padding，`replicated` 声明必须逐 bit
   一致的副本，`"dim": -1` 把行并行的分段拼回整行。加上 `--strict` 运行时源端的每个张量都被映射
   用到了，映射才算写全。[示例映射](../../examples/ckpt_conversion/mcore_map.json)描述了一种
   Megatron 风格的布局：

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
2. **比较行。**在两个 checkpoint 上运行这个工具，指明映射里的两种布局和目标的 TP 规模：

     ```bash
     python tools/ckpt_compare.py SRC_DIR DST_DIR --map map.json --layouts hf mcore --set b:r=2
     ```

     如果转换后的某个 QKV shard 里翻转了一个 bit，输出是这样的：

     ```
     model.layers.1.self_attn.k_proj.weight: rows_unmatched
       A 1 of 64 rows unmatched: row 8 <- model.layers.1.self_attn.k_proj.weight[8]
       B 1 of 64 rows unmatched: row 8 <- mp_rank_00/model.decoder.layers.1.self_attention.linear_qkv.weight[40]
     NOT all rows matched
     ```

     工具输出 "all rows matched"、退出码为 0，就说明每个逻辑参数的每一行都配上了对。每一处
     dtype 变化都会按参数报告出来：变宽（bf16 到 fp32）是精确的，行在变宽之后再比较；变窄则会
     指出每一个丢了 bit 的行。
3. **在目标布局上跑一次前向**，确认每一行都放在目标代码读取它的位置上。把转换后的 checkpoint
   加载进目标端的模型代码，使用目标布局，输入固定的 token id，再把 logits 与**使用相同归约**的
   源模型比较，两者必须逐 bit 一致。TP 2 下的行并行层要把两个部分和相加，所以即使每个权重都对，
   结果也不会和 TP 1 逐 bit 一致（[并行](../2-rl/parallelism.md)）。示例
   [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py) 在 CPU 上演示了这种做法。
4. **把检查保留下来**，以后转换器有改动或者换了新模型，都用同样的方式检查。把这一对转换结果连同
   植入的故障，作为一个[回归](../4-operations/regression.md)条目保存；回归套件应当报告正确的那一对
   一致，并指出每一个植入的故障。

用什么作为行的键：

| 行的键 | 能看到什么 | 看不到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|---|
| 按顺序对行的字节做摘要（`--rows digest`，默认） | 行的任何变化 | 行内没有盲点 | 小，在 CPU 上离线计算 | 检查转换和重新切分 |
| 行的 XOR fingerprint（`--rows xor`，OpGuard 的格式） | 大多数变化，而且就是 OpGuard 在权重同步时记录的值 | 行内字的重排；同一行同一通道上成对出现的相同变化（类型转换就会产生这种变化） | 在 GPU 上免费 | 在线的权重同步检查，或者与 OpGuard 的记录比较 |

目标端的前向拿什么来比较：

| 参照 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| 用目标端归约方式运行的源模型（相同的切分、相同的部分和顺序） | 位置正确时逐 bit 一致 | 小 | 你能用同样的切分方式运行源模型的行并行层 |
| 两边都使用逻辑 TP 规模（[并行](../2-rl/parallelism.md)） | 在任何 TP 规模下逐 bit 一致 | 中等（不到 2 倍） | 目标端已经在用不受并行方式影响的层 |
| 转回源布局，再跑源模型的前向 | 逐 bit 一致，但检查的是反向转换器，而不是目标端怎样读取权重 | 免费 | 拿不到目标端的模型代码 |
| logits 落在容差以内 | 只能发现严重的错位 | 免费 | 实在没有办法时才用，绝不能当作副本检查 |

已证明（在 CPU 上，是一次已知答案的演示）：一个 Llama 形状的 checkpoint，在两种 TP 规模下转成
Megatron 风格的布局再转回来，结果逐 bit 一致。工具配对了每个逻辑参数的每一行，两种 TP 规模之间
也一样；它按参数和行指出了一个转置的 shard、一个错误的 QKV 交错、一次 bf16 到 fp16 再到 bf16
的类型转换、一个翻转的 bit 和一个丢失的行。对正确的转换，前向检查逐 bit 一致，并发现了行检查
看不到的两个位置错误，但它没有发现那次类型转换（[报告](../../examples/ckpt_conversion/REPORT.md)）。
尚未证明：Megatron-core 自己的加载器和分布式 checkpoint 格式、流水线的重新切分、词表 padding，
以及真实模型的权重。

## 可能出错的地方

!!! warning "拿前向检查充当副本检查"
    只把最小的那些权重舍入掉的类型转换，在少量输入上可能一个 logit 都不改变（实测：见报告）。
    这样权重变了，转换却照样通过。副本要用逐行比较来证明，前向只用来检查位置。

!!! warning "离线检查转换时用了 XOR fingerprint"
    XOR 会把同一行同一通道上两个相同的变化互相抵消，而类型转换恰好会产生这种成对的变化。
    它也看不到行内字的重排，比如行并行的分段按错误的顺序拼了起来。于是变了的行照样能配上对。
    离线检查要用行的字节摘要；XOR 留给与 OpGuard 记录比较的在线检查。

!!! warning "拿切分过的前向和没切分的前向比较"
    TP 2 下的行并行层要把两个部分和相加，所以即使转换完美无缺，大多数 logits 也会和 TP 1 不同。
    检查就会因为错误的原因失败。要让两边使用相同的归约。

!!! warning "映射里没有提到的张量"
    映射漏掉的逻辑参数永远不会被比较，所以把它弄丢了的转换器也能通过。要用 `--strict` 运行，
    它在有未使用的张量时会失败；还要读一下未使用张量的列表。

!!! warning "padding 和复制出来的副本"
    全零的 padding 行能和任何全零行配上对，证明不了位置是否正确；如果只读 rank 0，某个 rank 上
    过时的 norm 副本就发现不了。要在映射里用 slice 去掉 padding，并把副本声明为 `replicated`。

## 小结

- 转换只是复制，所以每个逻辑参数的每一行都必须逐 bit 原样复制过去。
- 把每个逻辑参数当作由精确的行组成的多重集合来比较：重新打包不算差异，剩下的每一行都会连同
  来源一起被指出来。
- 多重集合看不到在参数内部挪了位置的行，所以要在目标布局上跑前向来检查位置，并与使用相同归约
  的源端比较。
- 离线时用行的字节摘要，因为 XOR fingerprint 在行内有盲点。

## 深入阅读

- [报告：checkpoint 转换演示](../../examples/ckpt_conversion/REPORT.md)：
  布局、映射、每个植入的故障，以及每项检查看到了什么。
- 代码：[ckpt_compare.py](../../tools/ckpt_compare.py)、
  [convert.py](../../examples/ckpt_conversion/convert.py)、
  [ckpt_conversion.py](../../examples/ckpt_conversion/ckpt_conversion.py)。
- [RL 循环](../2-rl/rl-loop.md)：权重同步检查，对运行中的权重做同样的逐行配对。
- [并行](../2-rl/parallelism.md)：为什么 TP 规模会改变前向结果，以及怎样用逻辑 TP 规模
  消除这种变化。
