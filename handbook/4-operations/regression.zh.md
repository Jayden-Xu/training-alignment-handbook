# 回归：重新检查已验证的结果

你已经有一个软件栈，在上面手册的检查都成立；现在它的某一部分要变了：torch、Triton、NCCL、
驱动、CUDA 构建，或者 GPU。本页在新的软件栈上重跑每一项已验证的检查，告诉你哪些结论仍然成立，
哪些坏了，哪些只是原版算子的行为挪了位置。

## 概念

设想一个原版矩阵乘：batch 超过某个大小后，同一行的舍入就变了。这个大小是一个库版本在一种 GPU 上的属性：
库根据形状挑 kernel，另一个版本可能在另一个大小换 kernel，也可能处处用同一个 kernel。
所以升级之后，有两种变化看起来很像，意思却相反：

| 变了什么 | 例子 | 意味着什么 |
|---|---|---|
| 手册声明成立的某项检查现在失败了 | 手册的不变 GEMM 在更大的 batch 里给同一行算出了不同的比特 | 一个发现：这条结论在这个软件栈上不再成立 |
| 一个有记录的原版失败现在通过了，或在别处失败 | 原版 GEMM 从另一个 batch size 开始变化，或者根本不变 | 底下的软件栈变了；更新专题页面 |

套件把两者分开：每项检查标为一条结论（`expect="pass"`）或一个有记录的原版失败（`expect="fail"`），
并把判定和为这个软件栈记录的基线比较。每项检查内部的相等是逐位的；套件从不按"接近"来评判一个数。

## 这一步

- 运行器：[tools/bi_suite.py](../../tools/bi_suite.py)
- 条目：[suite/manifest.json](../../suite/manifest.json)
- 基线：[suite/baselines/](../../suite/baselines/)
- 为最早几个示例自有结果格式准备的适配器：[suite/adapters/](../../suite/adapters/)
- CI：[.github/workflows/bi-suite.yml](../../.github/workflows/bi-suite.yml)，**未测试**（见[局限](#limits)）

### 什么时候运行

- 升级 torch、Triton、NCCL、驱动或 CUDA 构建之前和之后。
- 换到新硬件时（[见下文](#new-hardware)）。
- 在一个版本发布中声明一致性之前。
- 修改了某个示例、`tools/` 里的 kernel 或某个适配器之后。

### 怎样运行

**目标：** 这台机器上能跑的每一项检查都跑过并完成比较。
**成功：** 运行器以 0 退出，报告里没有 REGRESSIONS 或 NO RESULT 行。

```bash
python tools/bi_suite.py --list                       # what runs here, what is skipped and why
python tools/bi_suite.py                              # quick mode
python tools/bi_suite.py --full                       # the sizes behind the topic pages
python tools/bi_suite.py --vs-committed               # also compare with the committed result files
python tools/bi_suite.py --lock-file /tmp/gpu.lock    # shared GPUs: one flock per entry
python tools/bi_suite.py --peer node1 --launcher2 ~/launch2.sh   # also the two-node entries
python tools/bi_suite.py --opguard-root ~/OpGuard     # also the entries that need an OpGuard checkout
```

| 选项 | 得到什么 | 代价等级 | 什么时候选 |
|---|---|---|---|
| quick 模式 | 每条结论和主要的原版失败，尺寸缩小 | 小 | 每次升级、每次修改 |
| `--full` | 写专题页面时所用的尺寸 | 大 | 发布前、换新硬件时 |
| `--vs-committed` | 另外与专题页面背后的结果文件比较 | 免费 | 想看看页面写成以来什么变了 |

- 运行器读取 GPU 信息，运行所有放得下的条目（单 GPU、单节点 N 个 GPU、两个节点），
  跳过其余条目并说明原因。
- 每个条目写出一个 `bi-results-v1` 文件（[tools/bi_report.py](../../tools/bi_report.py)）；
  运行器在 `suite/runs/` 下写出一个汇总的 `report.json`（不提交）。
- 条目可以指定自己的解释器（manifest 里的 `"python"`），用于引擎环境；
  该解释器不存在的地方会跳过这个条目。命令里用到 `{opguard}` 的条目需要 `--opguard-root` 或 `$OPGUARD_ROOT`。
- `moe_ep_2node` 与 `moe_ep` 在同一个 `--out` 下写出的 dump 比较：
  跑两节点这一轮时，用单节点那次运行的输出目录。

### 怎样读报告

| 小节 | 含义 | 退出码 |
|---|---|---|
| REGRESSIONS | 一项 `expect="pass"` 的检查失败了：本手册的一条结论在这里不再成立 | 1 |
| NO RESULT | 某个条目崩溃或超时 | 1 |
| CHANGED | 一个有记录的原版失败（`expect="fail"`）现在通过了，或某个判定与基线不同 | 0 |
| AS BASELINE | 一个有记录的原版失败在这里通过，在这个软件栈的基线里也通过：这个失败只出现在特定软件栈上。会计数，用 `--verbose` 时列出，也写进 `report.json` | 0 |
| PATTERN | 判定与基线相同，但失败的形态不同（失败的 batch size 不同、个数不同） | 0 |
| NEW / MISSING | 基线里没有的检查，或这次运行没有产生的基线检查 | 0 |
| TIMING | 某项测量与基线相差超过 `--timing-ratio`（默认 1.5x），无论快慢 | 0 |
| INCOMPLETE | 条目内部某一节抛出异常；它的必需检查也会作为回归出现 | — |

- 基线的键包括：GPU 型号和数量（两节点条目还包括第二个节点的）、torch、Triton、NCCL、
  驱动和 CUDA 构建。任何一项变了，键就变了。
- 没有与软件栈匹配的基线时，报告会说明。先审查结果，再记录一个：
  `python tools/bi_suite.py --from-report <report.json> --record-baseline`。
- 一个原版失败如果通过了，在为该软件栈记录的、经过审查的基线把它记为通过之前，都是 CHANGED；
  之后就是 AS BASELINE。

### 原版结果变了怎么办

1. 读这个条目的日志和结果文件；确认变化出在原版算子上，而不是测试框架上。
2. 更新专题页面上的结论和它的报告，写明新的软件栈。旧结果对旧软件栈仍然成立。
3. 如果某个原版算子现在通过了当初促使我们写 kernel 的那项检查，这个 kernel 在旧软件栈上仍然需要。写明是哪些。
4. 记录一个新的基线。

PATTERN 行不需要处理，除非某个专题页面引用了旧的形态；那就更新页面。
特定软件栈上的原版结果的例子见[测得的覆盖范围](../../suite/COVERAGE.md#stock-results-that-are-stack-specific)。

### 怎样添加一项检查或一个示例

**目标：** 一条新结论从此在每个软件栈上都会被重新检查。
**成功：** 条目在套件里跑通，它的基线已经提交。

示例接受 `--quick` 和 `--json <file>`，并通过 `tools/bi_report.py` 写出 `bi-results-v1`：

```python
res = Results("my_example", quick=args.quick)
res.check("a row alone vs inside batches", rep.passed, level="L2", impl="my_kernel")          # our claim
res.check("a row alone vs inside batches", rep2.passed, level="L2", impl="torch.mm", expect="fail")
res.write(args.json); sys.exit(res.exit_code())
```

- 按结论给检查命名，不要按尺寸命名，这样 quick 和 full 模式的名字相同。尺寸放进 `detail`；
  把简短的失败形态放进 `detail["signature"]`，就会得到 PATTERN 行。
- 手册声明的用 `expect="pass"`；有记录的原版失败用 `"fail"`；只记录、不评判的用 `"info"`。

然后在 manifest 里加一个条目：

```json
  {"name": "my_example", "gpus": 1,
   "quick": "{python} examples/my_example/my_bi.py --quick --json {result}",
   "full": "{python} examples/my_example/my_bi.py --json {result}",
   "timeout": {"quick": 900, "full": 3600},
   "committed": "examples/my_example/results.json"},
```

- `gpus` 和 `nodes`（默认 1）说明它需要什么。`quick`/`full` 可以是一个步骤列表。
  `{result}` 是结果存放的位置；`{run}` 是条目自己的目录。命令自己取锁时写 `own_lock: true`。
- `committed` 可选：专题页面背后的结果文件，由 `--vs-committed` 比较。
  对于示例自有格式的文件，加上 `"adapter": "<name>"`（见 `suite/adapters/`）。
- 运行它，审查报告，记录基线，提交基线文件。

### 新硬件

1. 记录软件栈：`python tools/bi_suite.py --env`。
2. 运行 full 模式：`python tools/bi_suite.py --full --vs-committed`。
3. 审查每一条 REGRESSION 和 CHANGED，以及与已提交结果的每一处差别。
4. 记录基线：`--from-report <report.json> --record-baseline`，并把文件提交到 `suite/baselines/` 下。
5. 把这个软件栈加到[测得的覆盖范围](../../suite/COVERAGE.md#what-was-measured)。

## 哪里会出错

!!! warning "把失败调没了"
    必需检查失败时，很容易想用容差或跳过来"修"。
    这会把一个发现变成这个软件栈上的一条错误结论。请报告它；检查里永远不要对输出取整，也不要加容差。

!!! warning "在回归之上记录基线"
    基线记录一个软件栈的实际行为，包括已经失败的必需检查。记录基线永远不会把它藏起来：
    这项检查每次运行都仍是回归。所以先审查，再记录。

!!! warning "两个任务共用一块 GPU"
    两次测量共用一块 GPU，可能崩溃或显存不足，结果也不可信。
    每个 GPU 任务都要取所在节点的那一把锁（[机器与锁](#machines-and-locks)）。

!!! warning "重复取锁"
    套件条目里启动的命令如果再取同一把锁，会永远等下去，因为运行器已经持有它。
    要么由运行器取锁，要么由命令取锁，不要两者都取。

## 这里记录的基线

为三个软件栈记录了 quick 模式的基线：单 GPU、单节点八个 GPU、两个这样的节点。
在每个软件栈上，所有必需检查都通过，没有发现回归。文件在
[suite/baselines/](../../suite/baselines/) 中，每个软件栈一个，以它的键命名。
检查数、运行时间，以及在一种 GPU 上通过、在另一种上失败的原版失败项，见
[测得的覆盖范围](../../suite/COVERAGE.md#baselines-recorded-here-quick-mode)。

## 机器与锁

一条规则：**每个节点一把锁，节点上的每个 GPU 任务都要取它。** 跨两个节点的任务同时持有两个节点的锁。
锁被占用时不要跳过：等待。任务结束后不要留下运行中的进程，也不要在 `/tmp` 里留下大文件。
各台机器、它们的锁文件，以及为什么放弃了逐 GPU 的锁，见
[测得的覆盖范围](../../suite/COVERAGE.md#machines-and-locks)。

## 局限

- CI 工作流**未测试**：这个仓库没有远程仓库，所以它从未运行过。它的命令手动跑过，
  外面那层工作流没有。
- 只有 quick 模式有基线。full 模式没有通过套件运行过，最后加入的那些条目的完整命令
  是按各示例自己的用法写的，没有运行过。
- 与已提交结果的比较按检查名匹配；quick 尺寸与 full 尺寸不同的检查，是在另一组尺寸上比较同一条结论。

## 小结

- 一个逐位结果属于一个软件栈；软件栈每次变化都要重跑套件。
- REGRESSIONS 表示一条结论坏了：这是一个发现，绝不能调没。
- CHANGED 表示原版行为挪了位置：更新专题页面，记录基线。
- 按结论给检查命名，标明是结论还是原版失败，永远不加容差。
- 每个节点一把锁，每个 GPU 任务都要取。

## 深入阅读

- [测得的覆盖范围](../../suite/COVERAGE.md)：基线、机器与锁、特定软件栈上的原版结果。
- [tools/bi_suite.py](../../tools/bi_suite.py)、[tools/bi_report.py](../../tools/bi_report.py)、[suite/manifest.json](../../suite/manifest.json)。
- [硬件覆盖](hardware.md)：哪些因素会让比特在软件栈之间变化，以及怎样认定新硬件。
