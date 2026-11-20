#!/usr/bin/env python3
"""Link checks for the handbook.

1. Markdown: every relative link in the repository's .md files (outside .venv-docs/, site/
   and the build copy handbook/skills/) points at an existing file or directory.
2. Site (if site/ exists, after `mkdocs build`): every internal href/src in site/**/*.html
   resolves to a file in site/, and every #fragment on an internal page link exists there.
Usage: python scripts/check_links.py [--site-only | --md-only]   exit 1 on any broken link.
"""
import os
import re
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SKIP = {".venv-docs", "site", ".git", ".source-listings"}
FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1\s*$", re.S | re.M)
LINK = re.compile(r"\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")


def md_files():
    for dp, dns, fns in os.walk(ROOT):
        rel = Path(dp).relative_to(ROOT).as_posix()
        dns[:] = [d for d in dns if d not in SKIP and not (rel == "handbook" and d == "skills")]
        for f in fns:
            if f.endswith(".md"):
                yield Path(dp) / f


def check_md():
    broken, total = [], 0
    for p in md_files():
        text = FENCE.sub("", p.read_text(encoding="utf-8"))
        for t in LINK.findall(text):
            if re.match(r"^[a-z][a-z0-9+.-]*:", t, re.I) or t.startswith("#"):
                continue
            total += 1
            path = unquote(t.split("#")[0])
            if not (p.parent / path).exists():
                broken.append(f"{p.relative_to(ROOT)}: {t}")
    return total, broken


class Collect(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.ids = [], set()

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if "id" in a:
            self.ids.add(a["id"])
        if tag == "a" and "name" in a:
            self.ids.add(a["name"])
        for k in ("href", "src"):
            if a.get(k):
                self.refs.append(a[k])


def target_file(base, path):
    if path.startswith("/"):
        t = SITE / path.lstrip("/")
    else:
        t = (base.parent / path)
    t = Path(os.path.normpath(t))
    if t.is_dir():
        t = t / "index.html"
    return t


def check_site():
    pages, broken, total = {}, [], 0
    for p in SITE.rglob("*.html"):
        c = Collect()
        c.feed(p.read_text(encoding="utf-8", errors="replace"))
        pages[p] = c
    for p, c in pages.items():
        for ref in c.refs:
            u = urlsplit(ref)
            if u.scheme or u.netloc or ref.startswith(("mailto:", "javascript:", "data:")):
                continue
            total += 1
            path = unquote(u.path)
            t = p if not path else target_file(p, path)
            if not t.exists():
                broken.append(f"{p.relative_to(SITE)}: {ref}")
                continue
            if u.fragment and t.suffix == ".html" and t in pages and unquote(u.fragment) not in pages[t].ids:
                if "source/" not in t.as_posix():
                    broken.append(f"{p.relative_to(SITE)}: {ref} (missing #{u.fragment})")
    return len(pages), total, broken


def main(argv):
    bad = 0
    if "--site-only" not in argv:
        total, broken = check_md()
        print(f"Markdown: {total} relative links, {len(broken)} broken")
        for b in broken:
            print("   ", b)
        bad += len(broken)
    if "--md-only" not in argv and SITE.exists():
        n, total, broken = check_site()
        print(f"Site: {n} HTML files, {total} internal hrefs/srcs, {len(broken)} broken")
        for b in sorted(set(broken))[:200]:
            print("   ", b)
        bad += len(broken)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
