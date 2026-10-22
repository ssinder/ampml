#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a static documentation site for GitHub Pages.

The site is intentionally dependency-free: it converts the Markdown in ``docs/``
to HTML with a small, self-contained converter and copies the figures over.
This keeps the Pages build reproducible and avoids pinning a static site
generator into the CI dependency chain.

Output: ``_site/index.html`` plus one page per document.
"""

from __future__ import annotations

import html
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results"
SITE = ROOT / "_site"

STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans",
    "PingFang SC", "Microsoft YaHei", sans-serif;
  line-height: 1.7;
  color: #1f2328;
  background: #ffffff;
}
.layout { display: flex; min-height: 100vh; }
nav {
  width: 260px; flex: 0 0 260px; padding: 28px 20px;
  border-right: 1px solid #d1d9e0; background: #f6f8fa;
}
nav h2 { font-size: 15px; margin: 0 0 14px; font-weight: 600; }
nav a {
  display: block; padding: 5px 0; color: #0969da;
  text-decoration: none; font-size: 14px;
}
nav a:hover { text-decoration: underline; }
nav a.active { font-weight: 600; color: #1f2328; }
main { flex: 1; padding: 36px 48px; max-width: 980px; }
main h1 { font-size: 28px; margin-top: 0; border-bottom: 1px solid #d1d9e0; padding-bottom: 10px; }
main h2 { font-size: 21px; margin-top: 32px; border-bottom: 1px solid #eaecef; padding-bottom: 6px; }
main h3 { font-size: 17px; margin-top: 26px; }
main p, main li { font-size: 15px; }
code {
  background: #eff1f3; padding: 2px 5px; border-radius: 5px;
  font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, monospace;
  font-size: 87%;
}
pre {
  background: #f6f8fa; padding: 16px; border-radius: 8px; overflow-x: auto;
  border: 1px solid #eaecef;
}
pre code { background: none; padding: 0; font-size: 13px; }
table { border-collapse: collapse; margin: 16px 0; font-size: 14px; display: block; overflow-x: auto; }
th, td { border: 1px solid #d1d9e0; padding: 7px 12px; text-align: left; }
th { background: #f6f8fa; font-weight: 600; }
blockquote {
  margin: 16px 0; padding: 10px 16px; border-left: 4px solid #d1d9e0;
  background: #f6f8fa; color: #59636e;
}
img { max-width: 100%; border-radius: 6px; }
a { color: #0969da; }
footer { padding: 24px 48px; color: #59636e; font-size: 13px; border-top: 1px solid #d1d9e0; }
@media (max-width: 820px) {
  .layout { flex-direction: column; }
  nav { width: 100%; flex: none; border-right: none; border-bottom: 1px solid #d1d9e0; }
  main { padding: 24px 18px; }
}
"""


def md_to_html(markdown: str) -> str:
    """A deliberately small Markdown subset converter."""
    lines = markdown.split("\n")
    out: list[str] = []
    in_code = False
    in_table = False
    in_list = False

    def close_blocks() -> None:
        nonlocal in_table, in_list
        if in_table:
            out.append("</tbody></table>")
            in_table = False
        if in_list:
            out.append("</ul>")
            in_list = False

    for raw in lines:
        line = raw.rstrip()

        if line.startswith("```"):
            close_blocks()
            out.append("<pre><code>" if not in_code else "</code></pre>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(raw))
            continue

        # table row
        if line.startswith("|") and line.count("|") >= 2:
            cells = [c.strip() for c in line.strip("|").split("|")]
            if all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                continue
            if not in_table:
                close_blocks()
                out.append("<table><thead><tr>")
                out.extend(f"<th>{inline(c)}</th>" for c in cells)
                out.append("</tr></thead><tbody>")
                in_table = True
            else:
                out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            out.append("</tbody></table>")
            in_table = False

        # list item
        if re.match(r"^\s*[-*+]\s+", line):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(re.sub(r'^\s*[-*+]\s+', '', line))}</li>")
            continue
        if in_list:
            out.append("</ul>")
            in_list = False

        # headings
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            level = len(m.group(1))
            out.append(f"<h{level}>{inline(m.group(2))}</h{level}>")
            continue

        # blockquote
        if line.startswith(">"):
            out.append(f"<blockquote>{inline(line.lstrip('> ').strip())}</blockquote>")
            continue

        # horizontal rule
        if re.fullmatch(r"-{3,}", line.strip()):
            out.append("<hr>")
            continue

        if not line.strip():
            continue

        out.append(f"<p>{inline(line)}</p>")

    close_blocks()
    if in_code:
        out.append("</code></pre>")
    return "\n".join(out)


def inline(text: str) -> str:
    """Inline Markdown: code spans, bold, italic, links, images."""
    text = html.escape(text)
    # images before links
    text = re.sub(
        r"!\[([^\]]*)\]\(([^)]+)\)",
        lambda m: f'<img src="{m.group(2)}" alt="{m.group(1)}">',
        text,
    )
    text = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r"`([^`]+)`", lambda m: f"<code>{m.group(1)}</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: f"<strong>{m.group(1)}</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", lambda m: f"<em>{m.group(1)}</em>", text)
    return text


def page(title: str, body: str, nav: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)} — AMP-ML</title>
<style>{STYLE}</style>
</head>
<body>
<div class="layout">
<nav>
<h2><a href="index.html" style="color:#1f2328;text-decoration:none">AMP-ML</a></h2>
{nav}
</nav>
<main>
{body}
</main>
</div>
<footer>AMP-ML — MIT License · <a href="https://github.com/ssinder/ampml">GitHub</a></footer>
</body>
</html>
"""


def build() -> None:
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)

    # Copy figures next to the site so relative links resolve.
    shutil.copytree(FIGURES, SITE / "figures")

    documents = sorted(p for p in DOCS.glob("*.md"))
    nav_links = []
    for doc in documents:
        nav_links.append(f'<a href="{doc.stem}.html">{html.escape(doc.stem)}</a>')

    # Index page: the README plus the key results.
    readme = ROOT / "README.md"
    index_body = md_to_html(readme.read_text(encoding="utf-8")) if readme.exists() else "<h1>AMP-ML</h1>"
    (SITE / "index.html").write_text(page("Home", index_body, "\n".join(nav_links)), encoding="utf-8")

    for doc in documents:
        body = md_to_html(doc.read_text(encoding="utf-8"))
        (SITE / f"{doc.stem}.html").write_text(page(doc.stem, body, "\n".join(nav_links)), encoding="utf-8")

    print(f"built {len(documents) + 1} pages into {SITE}")


if __name__ == "__main__":
    build()
