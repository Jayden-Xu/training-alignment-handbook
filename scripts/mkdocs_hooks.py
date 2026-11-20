"""MkDocs hooks for the handbook site (see handbook/4-operations/README.md, "Building the site").

The Markdown stays readable on GitHub: pages link to each other and to repository files
(skills/, tools/, examples/, suite/, AUDIT.md) with plain relative paths. The site is
built from docs_dir: handbook. These hooks bridge the two:

1. on_config: copy skills/**/*.md (English and .zh.md) into handbook/skills/ so the
   skills become pages. The copy is gitignored and rewritten only when a file changed.
2. on_page_markdown: rewrite every relative link by where it points in the repository:
   - a handbook page or a skill page -> a relative link to that page in the site;
   - any other repository file or directory ("source file") ->
       {extra.source_url}/{path}   when extra.source_url is set (e.g. a GitHub
                                   ".../blob/main" URL; env HANDBOOK_SOURCE_URL), or
       source/{path} in the site   otherwise: the file is copied there as plain text,
                                   and a directory gets an index.html listing.
3. on_page_markdown, Chinese pages: give each heading of page.zh.md the anchor id of the
   matching heading of page.md ("## 中文标题 {#english-slug}"), so links with #anchors and
   permalinks are the same in both languages. Headings correspond by order (the check
   scripts/check_i18n.py keeps the heading levels equal).
4. on_files: register those source files with MkDocs so they are copied and so link
   validation (mkdocs build --strict) knows them.
"""

import html
import os
import re
import shutil
from pathlib import Path

import logging

from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.handbook")


class _DropInstantSwitcherNotice(logging.Filter):
    """mkdocs-static-i18n warns that its per-page language-switcher links do not survive
    Material's navigation.instant. The site keeps navigation.instant on purpose; after an
    instant page change the switcher may lead to the other language's version of the page
    first loaded rather than the current one (a full reload fixes it). Known, accepted."""

    def filter(self, record):
        return "not compatible with theme.features = navigation.instant" not in record.getMessage()


logging.getLogger("mkdocs.plugins.mkdocs_static_i18n.reconfigure").addFilter(_DropInstantSwitcherNotice())

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "handbook"
SKILLS_SRC = ROOT / "skills"
SKILLS_DST = DOCS / "skills"
SOURCE_PREFIX = "source"

FENCE = re.compile(r"^\s*(```|~~~).*?^\s*\1\s*$", re.S | re.M)
LINK = re.compile(r"(\]\(\s*<?)([^)\s>]+)(>?(?:\s+\"[^\"]*\")?\s*\))")

_source_url = ""


def _site_path(rel):
    """Site path of a copied repository file; MkDocs skips dot-directories, so .x -> _x."""
    return SOURCE_PREFIX + "/" + "/".join(("_" + p[1:]) if p.startswith(".") else p for p in rel.split("/"))


# --- 1. skills into the docs tree -------------------------------------------------

def _sync_skills():
    wanted = set()
    for src in SKILLS_SRC.rglob("*.md"):
        rel = src.relative_to(SKILLS_SRC)
        dst = SKILLS_DST / rel
        wanted.add(dst)
        data = src.read_bytes()
        if not dst.exists() or dst.read_bytes() != data:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    if SKILLS_DST.exists():
        for old in SKILLS_DST.rglob("*.md"):
            if old not in wanted:
                old.unlink()


def on_config(config, **kwargs):
    global _source_url
    _sync_skills()
    _source_url = (config.extra.get("source_url") or "").rstrip("/")
    return config


# --- link resolution --------------------------------------------------------------

def _origin_dir(src_uri):
    """Repository directory the page's Markdown lives in on GitHub."""
    if src_uri.startswith("skills/"):
        return (ROOT / src_uri).parent  # handbook/skills/x/... is a copy of skills/x/...
    return (DOCS / src_uri).parent


def _classify(src_uri, target):
    """Return (kind, repo_rel, frag) for a relative link, or None to leave it alone."""
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith(("#", "/")):
        return None
    path, _, frag = target.partition("#")
    if not path:
        return None
    abs_t = Path(os.path.normpath(_origin_dir(src_uri) / path))
    try:
        rel = abs_t.relative_to(ROOT).as_posix()
    except ValueError:
        return None
    if rel.startswith("handbook/") and not rel.startswith("handbook/skills/"):
        return "page", rel[len("handbook/"):], frag
    if rel.startswith("skills/") and rel.endswith(".md"):
        return "page", rel, frag
    if abs_t.exists():
        return ("dir" if abs_t.is_dir() else "file"), rel, frag
    return None  # let MkDocs report it


def _page_links(text):
    prose = FENCE.sub("", text)
    return [m.group(2) for m in LINK.finditer(prose)]


# --- 3. source files as static files ----------------------------------------------

_LISTING_DIR = None


def _listing_html(rel):
    d = ROOT / rel
    rows = []
    for p in sorted(d.iterdir()):
        if p.name.startswith("."):
            continue
        name = p.name + ("/" if p.is_dir() else "")
        href = (p.name + "/index.html") if p.is_dir() else p.name
        rows.append(f'<li><a href="{html.escape(href)}">{html.escape(name)}</a></li>')
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html.escape(rel)}/</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:48rem;"
        "padding:0 1rem;line-height:1.6}</style></head><body>"
        f"<h1>{html.escape(rel)}/</h1><ul>{''.join(rows)}</ul></body></html>"
    )


def _collect_sources():
    """Files and directories outside the docs that pages link to."""
    files, dirs = set(), set()
    for page in DOCS.rglob("*.md"):
        src_uri = page.relative_to(DOCS).as_posix()
        for target in _page_links(page.read_text(encoding="utf-8")):
            c = _classify(src_uri, target)
            if not c:
                continue
            kind, rel, _ = c
            if kind == "file":
                files.add(rel)
            elif kind == "dir":
                dirs.add(rel)
    for d in list(dirs):  # a directory listing links to everything below it
        for p in (ROOT / d).rglob("*"):
            if any(part.startswith(".") for part in p.relative_to(ROOT).parts):
                continue
            rel = p.relative_to(ROOT).as_posix()
            (dirs if p.is_dir() else files).add(rel)
    return files, dirs


def on_files(files, config, **kwargs):
    if _source_url:
        return files
    global _LISTING_DIR
    src_files, src_dirs = _collect_sources()
    # Raw copies: give each a non-Markdown source name so MkDocs treats it as a static file.
    for rel in sorted(src_files):
        f = File(rel, str(ROOT), config["site_dir"], config["use_directory_urls"])
        f.src_uri = _site_path(rel) + (".txt" if rel.endswith(".md") else "")
        f.abs_src_path = str(ROOT / rel)
        f.dest_uri = _site_path(rel)
        f.url = f.dest_uri
        f.abs_dest_path = os.path.join(config["site_dir"], f.dest_uri)
        files.append(f)
    # Directory listings, written to a scratch dir inside site_dir's parent build cache.
    _LISTING_DIR = Path(config["site_dir"]).parent / ".source-listings"
    if _LISTING_DIR.exists():
        shutil.rmtree(_LISTING_DIR)
    for rel in sorted(src_dirs):
        out = _LISTING_DIR / rel / "index.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(_listing_html(rel), encoding="utf-8")
        f = File(f"{rel}/index.html", str(_LISTING_DIR), config["site_dir"], config["use_directory_urls"])
        f.src_uri = _site_path(rel) + "/index.html"
        f.abs_src_path = str(out)
        f.dest_uri = f.src_uri
        f.url = f.dest_uri
        f.abs_dest_path = os.path.join(config["site_dir"], f.dest_uri)
        files.append(f)
    return files


# --- 2. link rewriting ------------------------------------------------------------

def _rewrite(text, src_uri):
    page_dir = os.path.dirname(src_uri) or "."

    def sub(m):
        c = _classify(src_uri, m.group(2))
        if not c:
            return m.group(0)
        kind, rel, frag = c
        frag = f"#{frag}" if frag else ""
        if kind == "page":
            target = os.path.relpath(rel, page_dir)
        elif _source_url:
            target = f"{_source_url}/{rel}"
        else:
            dest = _site_path(rel) + ("/index.html" if kind == "dir" else "")
            src = dest + (".txt" if kind == "file" and rel.endswith(".md") else "")
            target = os.path.relpath(src, page_dir)
        return m.group(1) + target + frag + m.group(3)

    out, pos = [], 0
    for fm in FENCE.finditer(text):
        out.append(LINK.sub(sub, text[pos:fm.start()]))
        out.append(fm.group(0))
        pos = fm.end()
    out.append(LINK.sub(sub, text[pos:]))
    return "".join(out)


HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t]*(\{#[^}]*\})?[ \t]*#*[ \t]*$", re.M)


def _english_ids(path, config):
    import markdown as md_lib

    md = md_lib.Markdown(extensions=config["markdown_extensions"], extension_configs=config["mdx_configs"])
    md.convert(path.read_text(encoding="utf-8"))
    out = []

    def walk(tokens):
        for t in tokens:
            out.append((t["level"], t["id"]))
            walk(t["children"])

    walk(md.toc_tokens)
    return out


def _with_english_ids(text, src_uri, config):
    en = DOCS / (src_uri[: -len(".zh.md")] + ".md")
    if not en.exists():
        return text
    ids = _english_ids(en, config)
    fences = [(m.start(), m.end()) for m in FENCE.finditer(text)]
    heads = [m for m in HEADING.finditer(text) if not any(a <= m.start() < b for a, b in fences)]
    if [len(m.group(1)) for m in heads] != [lvl for lvl, _ in ids]:
        log.warning(f"{src_uri}: headings do not match {en.name}; anchors not synced")
        return text
    out, pos = [], 0
    for m, (_, hid) in zip(heads, ids):
        out.append(text[pos:m.start()])
        out.append(f"{m.group(1)} {m.group(2)} {{#{hid}}}")
        pos = m.end()
    out.append(text[pos:])
    return "".join(out)


def on_page_markdown(markdown, page, config, files, **kwargs):
    src_uri = page.file.src_uri
    if src_uri.endswith(".zh.md"):
        markdown = _with_english_ids(markdown, src_uri, config)
    return _rewrite(markdown, src_uri)
