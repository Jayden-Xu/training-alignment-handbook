# 数据对齐

## 我们在哪里

这一页接着讲[预训练流程](README.md)的第 2 步。[预检](determinism-and-preflight.md)已经读回了两次运行的设置，只剩下声明过的那个差异。现在还不知道的是：两次运行训练用的是不是同一份数据。只有每个 rank 在每一步拿到的 token 都逐 bit 一致，两次运行才能逐 bit 比较。否则，之后的每一次比较看到的都是数据上的差异，而不是你要找的原因。所以这一页在 batch 离开 loader 的地方直接检查 batch 本身，然后才去采集这一对运行（[用 OpGuard 定位](localize-with-opguard.md)）。

## 思路：种子相同，batch 也可能不同

举个小例子。有八篇文档 A 到 H，batch size 为 4，两次运行用同一个种子。运行 1 在一个进程里读数据；运行 2 用两个 worker 进程，每个 worker 隔一篇取一篇：

| step | 运行 1（不用 worker） | 运行 2（两个 worker） |
|---|---|---|
| 0 | A B C D | A C E G |
| 1 | E F G H | B D F H |

两次运行在这个 epoch 里都恰好把每篇文档看了一遍，所以按 epoch 检查内容是能通过的。可是在第 0 步，两次运行累加的是不同样本的梯度，第一次更新之后权重就已经不同了。种子固定了 shuffle 的结果，却没有固定哪个进程交出哪个样本。

从数据文件到模型之间，有五个地方会出现这种情况：

| 在哪里 | 什么变了 | 小例子 |
|---|---|---|
| 顺序 | 样本在哪一步出现 | 上面的 worker 划分；shuffle buffer 用整个进程共享的随机状态设种子 |
| 划分 | 样本分到哪个 rank | 同一个全局 batch 分给 2 个 rank 还是 4 个 rank |
| packing | 文档怎样排进各行 | 6 篇文档各占一行、共 6 个 padding 行，或者打包进 3 行 |
| tokenizer | 文档被切成哪些 token id | 多了一个把文本转成小写的 normalizer；多加了一个序列开始 token |
| 状态 | 恢复训练后从哪里接着读 | 重启时没有恢复 sampler 的位置，于是把最开始几步又跑了一遍 |

这些都不是数值上的效应，它们改变的是输入本身。

**规则。** 在 batch 离开 loader 的地方，按 step 和 rank 给每个 batch 取 fingerprint；先逐 bit 比较两次运行的数据，再去看模型。loader 的出口是调用顺序上的第一个边界，所以在这里出现的差异，就是首个分歧点。这是第 0 部分对照实验思路的具体应用：输入也属于必须保持相同的东西。

整个 batch 只有一个 fingerprint 的话，只能知道"变了"，说不清是什么变了。所以记录分三个层级：每个字段整体一个 fingerprint（有没有变化），每个样本一个（样本是挪了位置，还是内容变了），每行里的每篇文档一个（是不是同样的文档换了 packing 方式）。XOR fingerprint 看不出一行里两个 token 互换了位置，所以每个样本的值还会把每个 token 的位置一起算进去。

## 步骤

### 2c. 固定数据源和 tokenizer

两次运行要读同样的文件，并且把它们切成同样的 token id。对数据文件和 tokenizer 分别做哈希（tokenizer 要连同调用它时的参数一起），把两个哈希都存进 fingerprint 记录的头部（[data_fingerprint.py](../../tools/data_fingerprint.py) 里的 `source_fingerprint` 和 `tokenizer_fingerprint`）；两次运行的头部应该一致，或者只在声明过的差异上不同。如果像 Megatron-LM 的预处理那样离线做分词，tokenizer 的输出就直接固定在文件里了。

### 2d. 在 batch 离开 loader 时取 fingerprint

为了记下每个 rank 在每一步消费了什么数据，在训练循环里加两行，每个 rank 都加，作用在 loader 刚返回、还没被改动过的 batch 上：

```python
import data_fingerprint as dfp

rec = dfp.Recorder("fp/run_a", rank=dp_rank,
                   meta={"tokenizer": dfp.tokenizer_fingerprint(tok, add_special_tokens=True),
                         "source": dfp.source_fingerprint(data_files)})
for step, batch in enumerate(loader):
    rec.record(step, batch)
    ...
```

每个张量字段都会被记录：token id、attention mask、position id，以及 labels 或 loss mask。加上 `opguard=` 后，recorder 还会按 OpGuard 的采集格式写出记录，这样 OpGuard 的 `first_diff` 会先看到数据这个边界，再看到模型的各个边界（[用 OpGuard 定位](localize-with-opguard.md)）。恢复训练时要传 `append=True`。最后两次运行在每个 step、每个 rank 上都各有一条记录。

### 2e. 比较两次运行

找出第一个结果不同的 (step, rank, 字段)：

```bash
python tools/data_fingerprint.py compare fp/run_a fp/run_b
python tools/data_fingerprint.py compare fp/run_a fp/run_b --global
```

比较是精确的，没有任何容差。第二种写法会先把每一步各个 rank 的数据拼起来再比较；两次运行的 DP 规模不同时，就要用这种写法。如果没有任何记录不同，数据就已经对齐，这一对运行可以去采集了（[用 OpGuard 定位](localize-with-opguard.md)）。

### 2f. 判断差异属于哪一类

每一类差异都有自己的修法。compare 命令会给首个差异标上类别，并统计所有 step 上每一类各出现了多少次；最后要得到一个类别，以及一个明确的原因。

| 类别 | 含义 | 常见原因 |
|---|---|---|
| order | 样本相同，但在 batch 里的顺序不同，或者出现在另一个 step、另一个 rank 上 | 流式数据集按 worker 划分；DP 规模不同；shuffle buffer 的种子；恢复训练时把本次运行更早的某一步又跑了一遍 |
| samples | 有些文档只在其中一次运行里出现 | tokenizer 或数据源不同；shuffle 用了别的种子；恢复训练时重读了一段数据，末尾因此多出来的那部分 |
| packing | 每一步的文档相同，但排进了不同的行 | 一次运行用 padding，另一次用 packing；行长不同 |
| masks | token id 相同，但 mask、position 或 labels 不同 | position id 没有按文档重新计数；loss mask 改过 |

头部的差异会最先报告。tokenizer 换了，会在头部显示出来；除非只改了调用参数，而这些参数又没有被记录。

### 2g. 修复差异

按问题选修法，修完再比较一次：每个 step、每个 rank 上都不能有任何记录不同。表里提到的库在设计上都解决了对应的问题，但每个库只在它自己的前提条件下才成立（[调研](../../examples/data_alignment/REPORT.md#existing-work)）。

| 问题 | 修法 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|---|
| worker 数一变，顺序就变 | 用 map-style 数据集，并把 sampler 留在主进程；或者让每个 worker 组装完整的 batch，再轮流取用（Grain 严格按轮询顺序读各个 worker）；MosaicML Streaming 的顺序本来就与 worker 数无关 | 无论 worker 有多少，同一步拿到的都是同一个 batch | 免费 | 流式数据集按样本分给多个 worker |
| DP 规模一变，划分就变 | 每一步先组一个全局 batch，再让每个 rank 取其中连续的一段（Megatron-LM 的预训练 sampler、OLMo-core 的全局索引）；或者在 MosaicML Streaming 里固定 canonical node 数 | 把各 rank 合起来看，每一步的 batch 在任何 DP 规模下都相同 | 免费 | 比较不同 DP 规模下的运行 |
| 恢复训练时没有恢复读取位置 | 把已经消费的样本数或 loader 的状态和 checkpoint 一起保存（torchdata 的 StatefulDataLoader、Grain、Megatron-LM 的 consumed samples、Megatron-Energon、Streaming、OLMo-core） | 恢复后从停下的地方接着读 | 免费 | 每一次恢复训练 |
| shuffle buffer 用进程的随机状态设种子 | 用 seed、epoch 和 worker id 一起设种子，或者给 loader 一个独立的生成器；Hugging Face datasets 可以传入 seed 和 epoch | buffer 的顺序不再受其他随机抽取的影响 | 免费 | 任何流式 shuffle |
| packing 方式不同 | 两次运行用同样的 collate 函数、行长和 packing 规则 | 同样的行 | 免费 | 只有一次运行做 packing，或者换了 packing 的实现 |
| tokenizer 不同 | 固定 tokenizer 文件和它的调用参数，或者离线分词 | 同样的 token id | 免费 | 升级了 tokenizer，或者新增了特殊 token |

## 可能出错的地方

!!! warning "DP 规模不同时还按 rank 逐个比较"
    换了 DP 规模，每个 rank 拿到的就是 batch 的另一段，所以按 rank 逐个比较一定会不同。你会去追一个按设计本来就存在的差异。应该把每一步的各个 rank 合起来再比较。

!!! warning "预取之后直接恢复 sampler 自己记的位置"
    worker 会预取数据，所以 sampler 发出去的索引比训练实际消费的多。直接恢复 sampler 的位置，会跳过一些 batch。应该保存训练循环自己记下的已消费样本数，或者用一个会跟踪这个数的 loader（实测见[报告](../../examples/data_alignment/REPORT.md#cases)）。

!!! warning "tokenizer 的 fingerprint 没有包含调用参数"
    tokenizer 文件里并不包含调用参数，比如是否添加特殊 token。于是两次运行的头部一致，batch 却不同。要把调用参数和 tokenizer 一起记录下来。

!!! warning "shuffle 的种子取自进程的随机状态"
    用进程随机状态设种子的 buffer，只要在它之前有别的代码抽过一次随机数，顺序就会变，而且只在用了 worker 时才会出现。多初始化一个参数，数据就变了。应该显式地给 buffer 设种子。

!!! warning "整个 epoch 相同，不等于每一步都相同"
    如果样本只是在不同 step 之间挪了位置，按 epoch 检查内容照样能通过，但权重从第一步起就不同了。要逐个 step 比较。

!!! warning "在 batch 被改动之后才取 fingerprint"
    如果在类型转换之后、在搬到设备上并补了 padding 之后，或者在模型自己加 mask 之后才记录 batch，loader 的差异就会和模型的差异混在一起，分不清是哪一边造成的。要记录 loader 刚返回时的 batch。

## 小结

- 只有每个 rank 在每一步拿到的 token 都逐 bit 一致，两次运行才有可比性；种子相同并不能保证这一点。
- 顺序、划分、packing、tokenizer 和恢复训练的状态，都可能在没有任何数值效应的情况下改变 batch。
- 在 batch 离开 loader 的地方，按 step 和 rank 给每个字段取 fingerprint，并做精确比较；DP 规模不同时，先把各 rank 合起来。
- 看首个差异的类别，用对应的修法，再比较一次，直到没有任何记录不同。

以上结果是在 CPU 上、用一个进程模拟多个 rank 得到的；尚未在真实的分布式训练任务上证明，也尚未用调研中的那些库证明（[报告](../../examples/data_alignment/REPORT.md#limits)）。

## 深入阅读

- [data_fingerprint.py](../../tools/data_fingerprint.py)：工具本身，包括记录格式和分类器。
- [报告：现有 loader 的调研、每个植入的案例及其修法、OpGuard 检查、开销](../../examples/data_alignment/REPORT.md)。
- [data_alignment.py](../../examples/data_alignment/data_alignment.py)：植入的各个案例。
- [确定性与预检](determinism-and-preflight.md)讲在这项检查之前读回的设置和状态。
- [数据并行](data-parallel.md)解释为什么同一组样本换个顺序求和，只有在归约与顺序无关时才逐 bit 一致。
- [pretraining-alignment skill](../../skills/pretraining-alignment/SKILL.md)。
