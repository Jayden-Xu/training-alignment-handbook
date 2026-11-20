# 4. Operations: keep the results true on your machines

You have a stack where the workflows of parts 1–3 hold: the settings are
checked, the operators are invariant, the two sides agree. This part keeps it
that way when the stack changes under you: a library upgrade, a driver update,
new GPUs.

## The idea

A bitwise result belongs to the stack it was measured on. The same source code
on another library version can pick another kernel, and another kernel can add
the same numbers in another order. So a verdict is re-checked, not assumed, each
time the stack changes.

<figure class="fig">
<div class="fig-chain"><span class="fig-node">Stack changes</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node">Run the regression suite</span><span class="fig-arrow" aria-hidden="true"></span><span class="fig-node is-decision">Compare with this stack's baseline</span></div><div class="fig-branch"><div class="fig-chain"><span class="fig-arrow">a claim no longer holds</span><span class="fig-node is-bad">Finding: report it, fix it</span></div><div class="fig-chain"><span class="fig-arrow">a stock result changed</span><span class="fig-node is-key">Update the topic page, record a baseline</span></div><div class="fig-chain"><span class="fig-arrow">same as baseline</span><span class="fig-node is-ok">Done</span></div></div>
</figure>

## The steps

| Step | Page | Tool |
|---|---|---|
| Re-run every verified check and read what regressed or changed | [Regression](regression.md) | [tools/bi_suite.py](../../tools/bi_suite.py) |
| Know what was measured on which stack, and what to do on new hardware | [Hardware coverage](hardware.md) | [tools/bi_suite.py](../../tools/bi_suite.py), [tools/sdc_scan.py](../../tools/sdc_scan.py) |

Everything not yet run or designed is in [open items](../open-items.md).

## Building the site

The handbook is also a book-style website in English and Chinese, built with
MkDocs Material and mkdocs-static-i18n from [mkdocs.yml](../../mkdocs.yml).

- **Layout.** The Markdown stays the source and stays readable on GitHub.
  The site is built from `handbook/` (`docs_dir`); `handbook/index.md` is its
  home page and the repository `README.md` stays the GitHub front page.
- **Chinese.** Each page `page.md` has a translation `page.zh.md` next to it
  (suffix mode), served under `/zh/`. Term choices are in the
  [glossary](../glossary.md).
  [scripts/check_i18n.py](../../scripts/check_i18n.py) compares every pair:
  tables and rows, numbers, link targets, code blocks, inline code and headings.
- **Skills.** [scripts/mkdocs_hooks.py](../../scripts/mkdocs_hooks.py) copies
  `skills/**/*.md` into `handbook/skills/` at build time (gitignored), so the
  skills are pages too.
- **Links to code.** The same hook rewrites links to repository files
  (`tools/`, `examples/`, `suite/`, `AUDIT.md`). With `HANDBOOK_SOURCE_URL`
  set (for example `https://github.com/ORG/REPO/blob/main`) they point there;
  unset, the files are copied into the site under `source/` and a directory
  gets a file listing.

```bash
python3 -m venv .venv-docs
.venv-docs/bin/pip install mkdocs==1.6.1 mkdocs-material==9.7.7 mkdocs-static-i18n==1.3.1 jieba
.venv-docs/bin/mkdocs build --strict              # site/ ; must end with no warnings
python3 scripts/check_i18n.py                     # every page pair
python3 scripts/check_links.py                    # Markdown links in the repo and hrefs in site/
.venv-docs/bin/mkdocs serve -a 127.0.0.1:8765     # preview: / and /zh/
```
