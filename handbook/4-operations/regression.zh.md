# 回归检查

手册里的检查在你现在这套软件栈上都成立；现在它的某一部分要变了：torch、Triton、NCCL、驱动、
CUDA 构建，或者 GPU。本页讲怎样在新的软件栈上把每一项验证过的检查重跑一遍。报告会告诉你哪些
结论仍然成立，哪些坏了，哪些只是原生算子的行为换了地方。

## 思路

设想一个原生的矩阵乘法：batch 超过某个大小之后，同一行的舍入就变了。这个大小是某个库版本在某种
GPU 上的属性。库根据形状挑选 kernel，换一个版本，可能在另一个大小处换 kernel，也可能处处都用
同一个 kernel。所以升级之后，有两种变化看起来很像，意思却正好相反：

| 变了什么 | 例子 | 意味着什么 |
|---|---|---|
| 手册声明成立的某项检查现在失败了 | 手册里不变的 GEMM，在更大的 batch 里给同一行算出了不同的结果 | 一个发现：这条结论在这套软件栈上不再成立 |
| 一个记录在案的原生失败现在通过了，或者换了个地方失败 | 原生 GEMM 从另一个 batch size 开始变化，或者根本不变 | 底下的软件栈变了；更新主题页 |

套件用两个办法把这两种情况分开：每项检查要么标为一条结论（`expect="pass"`），要么标为一个有记录的
原生失败（`expect="fail"`）；同时把判定和为这套软件栈记录的基线比较。每项检查内部的相等都是
逐 bit 的；套件从不按"差不多"来评判一个数。

## 这一步

- 运行器：[tools/bi_suite.py](../../tools/bi_suite.py)
- 条目，也就是要运行的检查列表：[suite/manifest.json](../../suite/manifest.json)
- 基线：[suite/baselines/](../../suite/baselines/)
- 为最早几个示例各自的结果格式准备的 adapter：[suite/adapters/](../../suite/adapters/)
- CI：[.github/workflows/bi-suite.yml](../../.github/workflows/bi-suite.yml)，**未测试**（见[局限](#limits)）

### 什么时候运行

- 升级 torch、Triton、NCCL、驱动或 CUDA 构建之前和之后。
- 换到新硬件时（[见下文](#new-hardware)）。
- 在某个版本发布时声明一致性之前。
- 修改了某个示例、`tools/` 里的某个 kernel 或某个 adapter 之后。

### 怎样运行

把这台机器上能跑的每一项检查都跑一遍，并比较结果。运行器以 0 退出、报告里没有 REGRESSIONS
或 NO RESULT 行，就说明这次运行没有问题。

```bash
python tools/bi_suite.py --list                       # what runs here, what is skipped and why
python tools/bi_suite.py                              # quick mode
python tools/bi_suite.py --full                       # the sizes behind the topic pages
python tools/bi_suite.py --vs-committed               # also compare with the committed result files
python tools/bi_suite.py --lock-file /tmp/gpu.lock    # shared GPUs: one flock per entry
python tools/bi_suite.py --peer node1 --launcher2 ~/launch2.sh   # also the two-node entries
python tools/bi_suite.py --opguard-root ~/OpGuard     # also the entries that need an OpGuard checkout
```

| 选项 | 能得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| quick 模式 | 在缩小的尺寸下检查每条结论和主要的原生失败 | 小 | 每次升级、每次修改之后 |
| `--full` | 用写主题页时的尺寸检查 | 大 | 发布之前、换新硬件时 |
| `--vs-committed` | 另外与主题页背后的结果文件比较 | 免费 | 想看看页面写成以来哪些结果变了 |

- 运行器先识别机器上的 GPU，再运行所有适合这台机器的条目（单 GPU、单节点 N 个 GPU 或两个节点），
  其余条目跳过，并说明原因。
- 每个条目写出一个 `bi-results-v1` 文件（[tools/bi_report.py](../../tools/bi_report.py)），
  运行器再在 `suite/runs/` 下写出一个汇总的 `report.json`，这个文件不提交。
- 条目可以指定自己的解释器（manifest 里的 `"python"`），比如要在某个引擎的环境里运行；
  机器上没有这个解释器时，就跳过这个条目。命令里用到 `{opguard}` 的条目，需要
  `--opguard-root` 或 `$OPGUARD_ROOT`。
- `moe_ep_2node` 要和 `moe_ep` 在同一个 `--out` 下写出的 dump 比较，所以跑两节点那一轮时，
  要用单节点那次运行的输出目录。

### 怎样读报告

| 小节 | 含义 | 退出码 |
|---|---|---|
| REGRESSIONS | 一项 `expect="pass"` 的检查失败了，也就是本手册的一条结论在这里不再成立 | 1 |
| NO RESULT | 某个条目崩溃或超时 | 1 |
| CHANGED | 一个记录在案的原生失败（`expect="fail"`）现在通过了，或者某个判定与基线不同 | 0 |
| AS BASELINE | 一个记录在案的原生失败在这里通过了，在这套软件栈的基线里也是通过的，说明这个失败只出现在某些软件栈上。这类结果会计数，加 `--verbose` 时会列出，也会写进 `report.json` | 0 |
| PATTERN | 判定与基线相同，但失败的形态不同（失败的 batch size 不同、个数不同） | 0 |
| NEW / MISSING | 基线里没有的检查，或者这次运行没有产生的基线检查 | 0 |
| TIMING | 某项测量与基线相差超过 `--timing-ratio`（默认 1.5 倍），不论变快还是变慢 | 0 |
| INCOMPLETE | 条目内部某一节抛出了异常；它的必需检查也会作为回归出现 | — |

- 基线的键由这些组成：GPU 型号和数量（两节点条目还包括第二个节点的）、torch、Triton、NCCL、
  驱动和 CUDA 构建。任何一项变了，键就变了。
- 没有与当前软件栈匹配的基线时，报告会说明。先审查结果，再记录一个基线：
  `python tools/bi_suite.py --from-report <report.json> --record-baseline`。
- 一个原生失败如果通过了，会一直报告为 CHANGED，直到这套软件栈经过审查的基线把它记为通过；
  之后就报告为 AS BASELINE。

### 原生算子的结果变了怎么办

1. 读这个条目的日志和结果文件，确认变化出在原生算子上，而不是出在 harness 上。
2. 更新主题页上的结论和对应的报告，并写明新的软件栈。旧结果对旧软件栈仍然成立。
3. 如果某个原生算子现在通过了当初促使我们写 kernel 的那项检查，这个 kernel 在较旧的软件栈上
   仍然需要。写明是哪些软件栈。
4. 记录一个新的基线。

PATTERN 行不需要处理，除非某个主题页引用了旧的失败形态；那样的话就更新那一页。只出现在某些软件栈
上的原生结果，例子见[测得的覆盖范围](../../suite/COVERAGE.md#stock-results-that-are-stack-specific)。

### 怎样添加一项检查或一个示例

加一个条目，让新结论从此在每套软件栈上都会被重新检查。条目在套件里跑通、它的基线已经提交，
就算完成了。

示例要接受 `--quick` 和 `--json <file>` 两个参数，并通过 `tools/bi_report.py` 以
`bi-results-v1` 格式写出结果：

```python
res = Results("my_example", quick=args.quick)
res.check("a row alone vs inside batches", rep.passed, level="L2", impl="my_kernel")          # our claim
res.check("a row alone vs inside batches", rep2.passed, level="L2", impl="torch.mm", expect="fail")
res.write(args.json); sys.exit(res.exit_code())
```

- 按结论给检查命名，不要按尺寸命名，这样 quick 和 full 模式下的名字相同。尺寸放进 `detail`；
  把失败形态的简短描述放进 `detail["signature"]`，就会得到 PATTERN 行。
- 手册声明成立的用 `expect="pass"`；记录在案的原生失败用 `"fail"`；只记录、不评判的用 `"info"`。

然后在 manifest 里加一个条目：

```json
  {"name": "my_example", "gpus": 1,
   "quick": "{python} examples/my_example/my_bi.py --quick --json {result}",
   "full": "{python} examples/my_example/my_bi.py --json {result}",
   "timeout": {"quick": 900, "full": 3600},
   "committed": "examples/my_example/results.json"},
```

- `gpus` 和 `nodes`（默认 1）说明条目需要什么资源。`quick` 和 `full` 都可以是一个步骤列表。
  `{result}` 是结果文件存放的位置，`{run}` 是条目自己的目录。如果命令自己取锁，就写
  `own_lock: true`。
- `committed` 是可选的，它指向主题页背后的结果文件，`--vs-committed` 会和它比较。如果这个文件
  是示例自己的格式，就加上 `"adapter": "<name>"`（见 `suite/adapters/`）。
- 运行这个条目，审查报告，记录基线，提交基线文件。

### 在新硬件上运行套件

1. 记录软件栈：`python tools/bi_suite.py --env`。
2. 以 full 模式运行：`python tools/bi_suite.py --full --vs-committed`。
3. 审查每一条 REGRESSION 和 CHANGED，以及与已提交结果之间的每一处差别。
4. 用 `--from-report <report.json> --record-baseline` 记录基线，并把文件提交到 `suite/baselines/` 下。
5. 把这套软件栈加到[测得的覆盖范围](../../suite/COVERAGE.md#what-was-measured)里。

## 可能出错的地方

!!! warning "用容差把失败掩盖掉"
    必需检查失败时，很容易想用容差或跳过来"修好"它。这会把一个发现变成这套软件栈上的一条错误
    结论。要把失败报告出来；检查里永远不要对输出取整，也不要加容差。

!!! warning "在已有回归的情况下记录基线"
    基线记录的是一套软件栈的实际行为，包括已经失败的必需检查。记录基线永远不会把失败藏起来：
    这项检查每次运行都仍然是回归。所以要先审查结果，再记录。

!!! warning "两个任务共用一块 GPU"
    两次测量共用一块 GPU，可能会崩溃或显存不足，结果也不可信。每个 GPU 任务都要取所在节点的那一把锁
    （[机器与锁](#machines-and-locks)）。

!!! warning "重复取锁"
    如果套件条目里启动的命令又去取同一把锁，它会永远等下去，因为运行器已经持有这把锁。要么由运行器
    取锁，要么由命令取锁，不要两边都取。

## 目前记录的基线

目前为三套软件栈记录了 quick 模式的基线：单 GPU、单节点八个 GPU，以及两个这样的节点。在每套软件栈
上，所有必需检查都通过了，没有发现回归。基线文件在 [suite/baselines/](../../suite/baselines/) 里，
每套软件栈一个，以它的键命名。检查的数量、运行时间，以及在一种 GPU 上出现、在另一种上不出现的
原生失败，见[测得的覆盖范围](../../suite/COVERAGE.md#baselines-recorded-here-quick-mode)。

## 共用机器：每个节点一把锁

只有一条规则：**每个节点一把锁，节点上的每个 GPU 任务都要取它。** 跨两个节点的任务同时持有两个
节点的锁。锁被占用时不要跳过，要等。任务结束后，不要留下还在运行的进程，也不要在 `/tmp` 里留下
大文件。各台机器、它们的锁文件，以及为什么不再用每个 GPU 一把锁，见
[测得的覆盖范围](../../suite/COVERAGE.md#machines-and-locks)。

## 套件还没覆盖的地方

- CI 工作流**未测试**。这个仓库没有配置远程仓库，所以这个工作流从来没有运行过。工作流里的命令手动
  跑过，但包在外面的工作流本身没有跑过。
- 只有 quick 模式有基线。full 模式还没有通过套件运行过，最近加入的几个条目的 full 模式命令，
  是照着各个示例自己的用法写的，没有运行过。
- 与已提交结果的比较是按检查名匹配的。如果一项检查在 quick 和 full 模式下尺寸不同，比较的就是
  同一条结论在不同尺寸下的结果。

## 小结

- 逐 bit 一致的结果只属于一套软件栈，所以软件栈每次变化都要重跑套件。
- REGRESSIONS 表示一条结论坏了。这是一个发现，绝不能用调参数的办法让它消失。
- CHANGED 表示原生算子的行为换了地方。更新主题页，并记录基线。
- 按结论给检查命名，标明它是结论还是原生失败，永远不加容差。
- 每个 GPU 任务都要取所在节点的那一把锁。

## 深入阅读

- [测得的覆盖范围](../../suite/COVERAGE.md)：基线、机器与锁，以及只出现在某些软件栈上的原生结果。
- [tools/bi_suite.py](../../tools/bi_suite.py)、[tools/bi_report.py](../../tools/bi_report.py)、[suite/manifest.json](../../suite/manifest.json)。
- [硬件覆盖](hardware.md)：哪些因素会让结果在不同软件栈之间变化，以及怎样验收新硬件。
