# 4. 运维：让结果在你的机器上继续成立

你已经有一个软件栈，在上面第 1–3 部分的流程都成立：设置检查过，算子满足不变性，两侧一致。
这一部分在软件栈发生变化时让它继续成立：库升级、驱动更新、换新 GPU。

## 概念

一个逐位结果属于测出它的那个软件栈。同一份源码换一个库版本，可能选中另一个 kernel；
另一个 kernel 可能按另一种顺序把同样的数加起来。所以软件栈每变一次，判定就要重新检查，不能假定。

<figure class="fig">
<div class="fig-chain"><span class="fig-node">软件栈变化</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">运行回归套件</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-decision">与这个软件栈的基线比较</span></div><div class="fig-branch"><div class="fig-chain"><span class="fig-arrow">某条结论不再成立</span><span class="fig-node is-bad">发现：报告并修复</span></div><div class="fig-chain"><span class="fig-arrow">某个原生结果变了</span><span class="fig-node is-key">更新主题页，记录基线</span></div><div class="fig-chain"><span class="fig-arrow">与基线相同</span><span class="fig-node is-ok">完成</span></div></div>
</figure>

## 步骤

| 步骤 | 页面 | 工具 |
|---|---|---|
| 重跑每一项已验证的检查，看哪些回归、哪些变化 | [回归](regression.md) | [tools/bi_suite.py](../../tools/bi_suite.py) |
| 了解在哪个软件栈上测了什么，以及换新硬件时该做什么 | [硬件覆盖](hardware.md) | [tools/bi_suite.py](../../tools/bi_suite.py)、[tools/sdc_scan.py](../../tools/sdc_scan.py) |

所有尚未运行或尚未设计的内容见[待办事项](../open-items.md)。

## 构建网站

本手册也是一个中英双语的书籍式网站，由 MkDocs Material 和 mkdocs-static-i18n
根据 [mkdocs.yml](../../mkdocs.yml) 构建。

- **布局。** Markdown 仍是源文件，在 GitHub 上照常可读。
  网站从 `handbook/`（`docs_dir`）构建；`handbook/index.md` 是网站首页，
  仓库的 `README.md` 仍是 GitHub 首页。
- **中文。** 每个页面 `page.md` 旁边有译文 `page.zh.md`（后缀模式），发布在 `/zh/` 下。
  术语选择见[术语表](../glossary.md)。
  [scripts/check_i18n.py](../../scripts/check_i18n.py) 逐对比较：
  表格和行数、数字、链接目标、代码块、行内代码和标题。
- **Skills。** [scripts/mkdocs_hooks.py](../../scripts/mkdocs_hooks.py) 在构建时把
  `skills/**/*.md` 复制到 `handbook/skills/`（已加入 gitignore），所以 skill 也是网站页面。
- **指向代码的链接。** 同一个 hook 会改写指向仓库文件的链接
  （`tools/`、`examples/`、`suite/`、`AUDIT.md`）。设置了 `HANDBOOK_SOURCE_URL`
  （例如 `https://github.com/ORG/REPO/blob/main`）时，链接指向那里；
  未设置时，这些文件会被复制到网站的 `source/` 下，目录会生成一个文件列表。

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install mkdocs==1.6.1 mkdocs-material==9.7.7 mkdocs-static-i18n==1.3.1 jieba
.venv-docs/bin/mkdocs build --strict              # site/ ; must end with no warnings
python3 scripts/check_i18n.py                     # every page pair
python3 scripts/check_links.py                    # Markdown links in the repo and hrefs in site/
.venv-docs/bin/mkdocs serve -a 127.0.0.1:8765     # preview: / and /zh/
```
