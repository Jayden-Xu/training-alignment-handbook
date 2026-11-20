# 数据对齐

## 我们在哪里

这一页接着[预训练工作流](README.md)的第 2 步。[预检](determinism-and-preflight.md)已经读回了两次运行的设置，只剩下声明过的那个差异。还没回答的是：两次运行训练的是不是同一份数据。只有每个 rank 在每一步看到的 token 比特都相同，两次运行才能逐位比较。否则之后的每一次比较显示的都是数据的差异，而不是你要找的原因。这一页在 batch 离开 loader 的地方检查 batch 本身，然后才去看[训练 step](training-step.md)。

## 思路：种子相同，batch 不同

一个小例子。八篇文档 A 到 H，batch 为 4，种子相同。运行 1 在一个进程里读。运行 2 用两个 worker 进程，每个 worker 隔一篇取一篇：

| step | 运行 1（无 worker） | 运行 2（两个 worker） |
|---|---|---|
| 0 | A B C D | A C E G |
| 1 | E F G H | B D F H |

两次运行在这个 epoch 里都恰好看到每篇文档一次，所以按 epoch 检查内容会通过。但第 0 步的梯度是另一组样本的和，权重在第一次更新时就出现分歧。种子固定了 shuffle，却没有固定哪个进程交出哪个样本。

从文件到模型之间，有五个地方会发生同样的事：

| 位置 | 改变了什么 | 小例子 |
|---|---|---|
| 顺序 | 样本落在哪一步 | 上面的 worker 划分；用进程级随机状态做种子的 shuffle buffer |
| 划分 | 样本落在哪个 rank | 同一个全局 batch 分给 2 个 rank 还是 4 个 rank |
| 打包 | 文档怎样变成行 | 6 篇文档放进 6 个 padding 行，或打包成 3 行 |
| 分词器 | 文档变成哪些 token id | 一个转小写的 normalizer；多加了一个序列开始 token |
| 状态 | 恢复的运行从哪里开始 | 重启时没有恢复 sampler 的位置，于是重放了最初几步 |

这些都不是数值效应。它们改变的是输入本身。

**规则。** 在 batch 离开 loader 的地方，按 step 和 rank 给每个 batch 取指纹，在看模型之前先对两次运行做逐位比较。loader 的出口是调用顺序上的第一个边界，所以这里的差异就是首个分歧点。这是第 0 部分受控实验的一个实例：输入也是必须保持相同的东西。

只有整个 batch 的一个指纹，说不清哪里出了错。所以保留三个层级：整个字段（有没有变化），每个样本一个（样本是移动了还是变了），每行里每篇文档一个（是不是同样的文档换了打包方式）。XOR 指纹看不出一行里两个 token 互换了位置，所以每个样本的值还会按位置给每个 token 加上标记。

## 步骤

### 2c. 固定数据源和分词器

**目标：** 两次运行读同样的文件，并把它们变成同样的 token id。

**做法：** 对数据文件和分词器做哈希，分词器连同调用它时的参数一起，把两者都存进指纹记录的头部（[data_fingerprint.py](../../tools/data_fingerprint.py) 里的 `source_fingerprint`、`tokenizer_fingerprint`）。像 Megatron-LM 的预处理那样离线分词，就把分词器固定进了文件。

**成功：** 两边的头部一致，或者只在声明过的差异上不同。

### 2d. 在 loader 出口取指纹

**目标：** 记录每个 rank 在每一步消费了什么。

**做法：** 在训练循环里加两行，每个 rank 都加，作用在 loader 刚返回的 batch 上：

```python
import data_fingerprint as dfp

rec = dfp.Recorder("fp/run_a", rank=dp_rank,
                   meta={"tokenizer": dfp.tokenizer_fingerprint(tok, add_special_tokens=True),
                         "source": dfp.source_fingerprint(data_files)})
for step, batch in enumerate(loader):
    rec.record(step, batch)
    ...
```

每个张量字段都会记录：token id、attention mask、position id、labels 或 loss mask。加上 `opguard=` 后，recorder 还会按 OpGuard 的采集格式写出记录，于是 OpGuard 的 `first_diff` 能在模型的边界之前看到数据边界（[用 OpGuard 定位](localize-with-opguard.md)）。恢复训练的运行传 `append=True`。

**成功：** 两次运行在每个 step 和 rank 上各有一条记录。

### 2e. 比较

**目标：** 第一个比特不同的 (step, rank, 字段)。

**做法：**

```bash
python tools/data_fingerprint.py compare fp/run_a fp/run_b
python tools/data_fingerprint.py compare fp/run_a fp/run_b --global
```

比较是精确的，没有容差。第二种写法先把每一步各 rank 的数据拼接起来，用于数据并行规模不同的两次运行。

**成功：** 没有任何记录不同。这时数据已经对齐，比较转到[训练 step](training-step.md)。

### 2f. 分类

**目标：** 判断差异的类型，因为每一类有自己的修法。

**做法：** compare 命令给出首个差异的类别，并统计所有步上各类的数量：

| 类别 | 含义 | 常见原因 |
|---|---|---|
| order | 样本相同，但在 batch 内顺序不同，或落在另一步、另一个 rank | 流式数据集按 worker 划分；数据并行规模；shuffle buffer 的种子；重放式的恢复（它会重复本次运行更早的一步） |
| samples | 有些文档只在一边出现 | 分词器或数据源；另一个种子的 shuffle；重放式恢复的末尾 |
| packing | 每步文档相同，但排进了不同的行 | padding 对比打包；不同的行长 |
| masks | token id 相同，mask、位置或 labels 不同 | position id 没有按文档重置；loss mask 改了 |

头部的差异先报告：分词器改了会在头部显示，除非只改了调用参数而且这些参数没有记录。

**成功：** 一个类别，一个明确的原因。

### 2g. 修复

**目标：** 每个 step、每个 rank 上指纹都相等。

**做法：** 按问题选修法。表里列出的库是按设计解决该问题的库；每个库只在它自己的条件下成立（[调研](../../examples/data_alignment/REPORT.md#existing-work)）。

| 问题 | 选项 | 得到什么 | 代价等级 | 何时选择 |
|---|---|---|---|---|
| worker 数改变顺序 | map-style 数据集，sampler 留在主进程；或者让每个 worker 构造整个 batch，轮流取（Grain 严格轮询各 worker）；MosaicML Streaming 的顺序与 worker 数无关 | 任意 worker 数下，同一步拿到同一个 batch | 免费 | 按样本在 worker 间划分的流式数据集 |
| 数据并行规模改变划分 | 每步一个全局 batch，每个 rank 取一段连续切片（Megatron-LM 的预训练 sampler、OLMo-core 的全局索引）；MosaicML Streaming 固定 canonical node 数 | 合并各 rank 后，每步的 batch 在任意规模下都相同 | 免费 | 比较不同 DP 规模的运行 |
| 恢复时没有恢复位置 | 把已消费的样本数或 loader 状态存进检查点（torchdata 的 StatefulDataLoader、Grain、Megatron-LM 的 consumed samples、Megatron-Energon、Streaming、OLMo-core） | 恢复的运行从停下的地方继续 | 免费 | 每一次恢复训练 |
| shuffle buffer 用进程的种子 | 用 seed、epoch 和 worker id 设种子，或给 loader 一个自己的生成器；Hugging Face datasets 接受 seed 和 epoch | buffer 的顺序不受其他随机抽取影响 | 免费 | 任何流式 shuffle |
| 打包方式不同 | 两边用同样的 collate、行长和打包规则 | 同样的行 | 免费 | 只有一边打包，或换了打包器 |
| 分词器不同 | 固定分词器文件及其调用参数，或离线分词 | 同样的 token id | 免费 | 升级分词器，新增特殊 token |

**成功：** 再比较一次：没有记录不同。

## 可能出错的地方

!!! warning "跨数据并行规模按 rank 比较"
    换了 DP 规模，一个 rank 拿到的是 batch 的另一段，所以按 rank 比较总会不同。你会去追一个设计上就存在的差异。改为合并每一步的各 rank 再比较。

!!! warning "预取之后 sampler 自己的位置"
    worker 会预取，所以 sampler 发出的索引比训练消费的多。恢复这个位置会跳过若干 batch。保存训练循环自己记下的已消费样本数，或使用会跟踪它的 loader（实测：[报告](../../examples/data_alignment/REPORT.md#cases)）。

!!! warning "分词器指纹不含调用参数"
    分词器文件里没有调用参数，例如是否添加特殊 token。于是头部一致，batch 却不同。把调用参数和分词器一起记录下来。

!!! warning "shuffle 种子取自进程"
    用进程随机状态做种子的 buffer，只要之前有别的地方抽过一次随机数就会变，而且只在使用 worker 时才会出现。多初始化一个参数就会改变数据。显式地设种子。

!!! warning "整个 epoch 相同不等于每一步相同"
    样本只是在步之间移动时，按 epoch 检查内容会通过。权重却从第一步就不同了。按步比较。

!!! warning "在 batch 改变之后取指纹"
    在类型转换、带 padding 的设备搬运或模型自己的 mask 之后才记录 batch，会把 loader 和模型的差异混在一起。你分不清是哪一边造成的。记录 loader 刚返回的 batch。

## 小结

- 只有每个 rank 在每一步看到的 token 比特都相同，两次运行才可比较；相同的种子并不保证这一点。
- 顺序、划分、打包、分词器和恢复状态，都会在没有任何数值效应的情况下改变 batch。
- 在 loader 出口按 step 和 rank 给每个字段取指纹，精确比较；DP 规模不同时合并各 rank。
- 读首个差异的类别，用对应的修法，再比较，直到没有记录不同。

已在 CPU 上、于一个进程内模拟多个 rank 证明；尚未在真实的分布式作业上、也尚未用调研中的那些库证明（[报告](../../examples/data_alignment/REPORT.md#limits)）。

## 深入阅读

- [data_fingerprint.py](../../tools/data_fingerprint.py)：工具本身、记录格式和分类器。
- [报告：现有 loader 的调研、每个预置案例及其修复、OpGuard 检查、开销](../../examples/data_alignment/REPORT.md)。
- [data_alignment.py](../../examples/data_alignment/data_alignment.py)：预置的案例。
- [确定性与预检](determinism-and-preflight.md)：在这项检查之前读回的设置和状态。
- [数据并行](data-parallel.md)：同一组样本换个顺序求和，只有在与顺序无关的归约下才逐位相等。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
