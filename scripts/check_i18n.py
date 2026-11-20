#!/usr/bin/env python3
"""Check every English page against its Chinese translation (page.md vs page.zh.md).

For each pair it compares, outside nothing but what is named:
  - tables: number of tables and the row count of each;
  - numbers: the set of numeric tokens in the whole page;
  - link targets: the multiset of Markdown link targets (prose and tables);
  - code blocks: every fenced block, in order, character for character;
  - inline code: the multiset of `inline code` spans;
  - headings: the same sequence of heading levels.
Pages: handbook/**/*.md and skills/**/*.md (the copied handbook/skills/ tree is skipped).
Usage: python scripts/check_i18n.py [files.md ...]   exit 1 on any mismatch or missing pair.
"""
import collections, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^( {0,3}|\s*)(```|~~~).*?^\s*\2\s*$", re.S | re.M)
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
NUM = re.compile(r"(?<![A-Za-z_])\d+(?:[.,]\d+)*")
INLINE = re.compile(r"(`+)(.+?)\1", re.S)


def split(text):
    blocks = [m.group(0).strip() for m in FENCE.finditer(text)]
    prose = FENCE.sub("\n", text)
    return blocks, prose


def tables(prose):
    out, cur = [], 0
    for line in prose.splitlines():
        if line.lstrip().startswith("|"):
            cur += 1
        elif cur:
            out.append(cur); cur = 0
    if cur:
        out.append(cur)
    return out


def facts(text):
    blocks, prose = split(text)
    return {
        "tables": tables(prose),
        "numbers": set(NUM.findall(text)),
        "links": collections.Counter(LINK.findall(prose)),
        "code": blocks,
        "headings": [len(m.group(1)) for m in re.finditer(r"^(#{1,6})\s", prose, re.M)],
        "inline": collections.Counter(m.group(2).strip() for m in INLINE.finditer(prose)),
    }


def compare(en_path, zh_path):
    a, b = facts(en_path.read_text()), facts(zh_path.read_text())
    errs = []
    if a["tables"] != b["tables"]:
        errs.append(f"tables/rows: en {a['tables']} zh {b['tables']}")
    if a["numbers"] != b["numbers"]:
        errs.append(f"numbers only en {sorted(a['numbers'] - b['numbers'])} only zh {sorted(b['numbers'] - a['numbers'])}")
    if a["links"] != b["links"]:
        errs.append(f"links only en {dict(a['links'] - b['links'])} only zh {dict(b['links'] - a['links'])}")
    if a["code"] != b["code"]:
        for i, (x, y) in enumerate(zip(a["code"], b["code"])):
            if x != y:
                errs.append(f"code block {i} differs: {x[:60]!r} vs {y[:60]!r}"); break
        if len(a["code"]) != len(b["code"]):
            errs.append(f"code blocks: en {len(a['code'])} zh {len(b['code'])}")
    if a["headings"] != b["headings"]:
        errs.append(f"heading levels: en {a['headings']} zh {b['headings']}")
    if a["inline"] != b["inline"]:
        errs.append(f"inline code only en {dict(a['inline'] - b['inline'])} only zh {dict(b['inline'] - a['inline'])}")
    return errs


def pages():
    for d in ("handbook", "skills"):
        for p in sorted((ROOT / d).rglob("*.md")):
            rel = p.relative_to(ROOT).as_posix()
            if rel.startswith("handbook/skills/") or p.name.endswith(".zh.md"):
                continue
            yield p


def main(argv):
    targets = [Path(x).resolve() for x in argv] if argv else list(pages())
    bad = 0
    for en in targets:
        if en.name.endswith(".zh.md"):
            en = en.with_name(en.name[:-6] + ".md")
        zh = en.with_name(en.name[:-3] + ".zh.md")
        rel = en.relative_to(ROOT)
        if not zh.exists():
            print(f"MISSING {rel} has no {zh.name}"); bad += 1; continue
        errs = compare(en, zh)
        if errs:
            bad += 1
            print(f"FAIL {rel}")
            for e in errs:
                print("   ", e)
        else:
            print(f"ok   {rel}")
    print(f"{bad} page(s) with problems")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
