"""Build the static site: content/*.md -> *.html, and refresh the index cards.

Prose lives in `content/<slug>.md` with YAML front matter. Hand-tuned figures
live in `figures/<name>.html` and are pulled in with a `{{figure: <name>}}`
line. Everything else (page chrome, MathJax, analytics) comes from
`templates/post.html`.

    python build.py            # write the HTML files
    python build.py --check    # fail if the HTML on disk is out of date
"""

import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import markdown
import typer
import yaml
from jinja2 import Environment, FileSystemLoader

ROOT = Path(__file__).resolve().parent
CONTENT, FIGURES, TEMPLATES = ROOT / "content", ROOT / "figures", ROOT / "templates"
STYLESHEET = ROOT / "assets" / "style.css"
INDEX = ROOT / "index.html"

PROSE_INDENT = " " * 6
FIGURE_RE = re.compile(r"<p>\{\{figure:\s*([A-Za-z0-9_-]+)\}\}</p>")
CITE_RE = re.compile(r"\{\{cite:\s*([^}]+?)\s*\}\}")
REF_ITEM_RE = re.compile(
    r"^(?P<bullet>[ \t]*(?:\d+\.|[-*]))[ \t]+\{ref:\s*(?P<key>[A-Za-z0-9_-]+)\}[ \t]*"
    r"(?P<text>.*)$",
    re.M,
)
DISPLAY_MATH_RE = re.compile(r"\\\[.*?\\\]", re.S)
INLINE_MATH_RE = re.compile(r"\\\(.*?\\\)", re.S)
EXTERNAL_LINK_RE = re.compile(r'<a href="(https?://[^"]+)">')
INDEX_MARKERS = re.compile(
    r"(<!-- BUILD:WRITING -->\n).*?(\s*<!-- /BUILD:WRITING -->)", re.S
)

app = typer.Typer(add_completion=False, help=__doc__)


def asset_version(path: Path) -> str:
    """Content hash of one asset, used to bust the GitHub Pages cache.

    Args:
        path: The asset to hash, relative to the site root or absolute.

    Returns:
        The first ten hex digits of its SHA-256.
    """
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()[:10]


def stylesheet_version() -> str:
    """Content hash of the shared stylesheet."""
    return asset_version(STYLESHEET)


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Split a `---` delimited YAML header from the Markdown body."""
    if not text.startswith("---\n"):
        raise ValueError("missing front matter")
    _, header, body = text.split("---\n", 2)
    return yaml.safe_load(header), body.strip()


def number_references(body: str) -> tuple[str, dict[str, int]]:
    """Turn `{ref: key}` markers into anchors and number them in list order.

    Args:
        body: Markdown body, whose reference list is an ordered list of
            single-line items each starting with a `{ref: key}` marker.

    Returns:
        The body with every marker replaced by an anchor token that
        `anchor_reference_items` turns into an `id` on the rendered `<li>`, and
        a mapping from reference key to the number the reader will see.
    """
    numbers: dict[str, int] = {}

    def anchor(match: re.Match[str]) -> str:
        bullet, key, text = match.group("bullet", "key", "text")
        if not bullet.strip().endswith("."):
            raise ValueError(
                f"reference {key!r} is a bullet item; the reference list must be "
                "an ordered list so the printed numbers match the citations"
            )
        if key in numbers:
            raise ValueError(f"reference {key!r} is defined twice")
        numbers[key] = len(numbers) + 1
        return f"{bullet} REFZ{key}Z{text}"

    return REF_ITEM_RE.sub(anchor, body), numbers


def link_citations(body: str, numbers: dict[str, int]) -> str:
    """Replace `{{cite: key, ...}}` markers with superscript links to the list.

    Args:
        body: Markdown body, references already numbered.
        numbers: Mapping from reference key to its number, from
            `number_references`.

    Returns:
        The body with every citation marker replaced by a `<sup>` of links.
    """

    def superscript(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",")]
        unknown = [key for key in keys if key not in numbers]
        if unknown:
            raise ValueError(
                f"citation of undefined reference(s): {', '.join(unknown)}"
            )
        links = ", ".join(f'<a href="#ref-{key}">{numbers[key]}</a>' for key in keys)
        return f'<sup class="cite">{links}</sup>'

    return CITE_RE.sub(superscript, body)


def anchor_reference_items(html: str, keys: list[str]) -> str:
    """Move each reference token from the list item text onto the `<li>` itself.

    Args:
        html: Rendered HTML still carrying the `REFZ<key>Z` tokens.
        keys: Every reference key defined in the body.

    Returns:
        The HTML with each token replaced by an `id` on its list item.
    """
    for key in keys:
        token = f"REFZ{key}Z"
        anchored = html.replace(f"<li>{token}", f'<li id="ref-{key}">')
        if anchored == html:
            raise ValueError(f"reference {key!r} did not render as its own list item")
        html = anchored
    return html


def render_body(body: str) -> str:
    """Markdown to HTML, protecting math and splicing in figures and citations."""
    body, numbers = number_references(body)
    body = link_citations(body, numbers)

    math: list[str] = []

    def stash(match: re.Match[str]) -> str:
        math.append(match.group(0))
        return f"MATHZ{len(math) - 1}Z"

    # Markdown would eat the backslashes and read underscores as emphasis.
    body = DISPLAY_MATH_RE.sub(stash, body)
    body = INLINE_MATH_RE.sub(stash, body)

    html = markdown.markdown(body, extensions=["extra"])
    html = anchor_reference_items(html, list(numbers))

    # A display equation is its own block, not a paragraph of prose.
    for index, expression in enumerate(math):
        token = f"MATHZ{index}Z"
        if expression.startswith("\\["):
            html = html.replace(f"<p>{token}</p>", expression)
        html = html.replace(token, expression)

    def figure(match: re.Match[str]) -> str:
        path = FIGURES / f"{match.group(1)}.html"
        if not path.exists():
            raise FileNotFoundError(f"no figure partial at {path}")
        return path.read_text(encoding="utf-8").rstrip("\n")

    html = FIGURE_RE.sub(figure, html)
    html = EXTERNAL_LINK_RE.sub(r'<a href="\1" target="_blank" rel="noopener">', html)
    lines = []
    for line in html.split("\n"):
        if line.startswith("<h2") and lines:  # breathing room between sections
            lines.append("")
        lines.append(PROSE_INDENT + line if line else line)
    return "\n".join(lines)


def load_posts() -> list[dict[str, Any]]:
    """Every post, ordered by the `order` field in its front matter."""
    posts = []
    for path in sorted(CONTENT.glob("*.md")):
        meta, body = split_front_matter(path.read_text(encoding="utf-8"))
        posts.append({**meta, "slug": path.stem, "body": body})
    return sorted(posts, key=lambda p: p.get("order", 0))


def render_post(post: dict[str, Any], env: Environment, version: str) -> str:
    template = env.get_template("post.html")
    return (
        template.render(
            title=post["title"],
            description=post["description"],
            tag=post["tag"],
            math=post.get("math", False),
            unlisted=post.get("unlisted", False),
            script=post.get("script"),
            style=post.get("style"),
            style_version=asset_version(post["style"]) if post.get("style") else None,
            version=version,
            content=render_body(post["body"]),
        ).rstrip("\n")
        + "\n"
    )


def render_index(posts: list[dict[str, Any]], version: str) -> str:
    """Refresh the writing cards and the stylesheet version, leave the rest.

    Posts marked `unlisted: true` are still rendered, just not linked here.
    """
    text = INDEX.read_text(encoding="utf-8")
    cards = "\n".join(
        f'        <a class="card" href="{p["slug"]}.html">\n'
        f'          <div class="tag">{p["tag"]}</div>\n'
        f"          <h3>{p['title']}</h3>\n"
        f"          <p>{p.get('blurb') or p['description']}</p>\n"
        f'          <span class="more">Read →</span>\n'
        f"        </a>"
        for p in posts
    )
    if not INDEX_MARKERS.search(text):
        raise ValueError("index.html is missing the BUILD:WRITING markers")
    text = INDEX_MARKERS.sub(lambda m: m.group(1) + cards + m.group(2), text)
    return re.sub(r'(assets/style\.css\?v=)[^"]*', rf"\g<1>{version}", text)


@app.command()
def build(
    check: bool = typer.Option(
        False, "--check", help="Do not write; exit 1 if any output is stale."
    ),
) -> None:
    """Render every post and refresh index.html."""
    env = Environment(
        loader=FileSystemLoader(TEMPLATES), autoescape=False, keep_trailing_newline=True
    )
    version = stylesheet_version()
    posts = load_posts()

    outputs = {ROOT / f"{p['slug']}.html": render_post(p, env, version) for p in posts}
    listed = [p for p in posts if not p.get("unlisted", False)]
    outputs[INDEX] = render_index(listed, version)

    stale = [
        path
        for path, html in outputs.items()
        if not path.exists() or path.read_text(encoding="utf-8") != html
    ]
    if check:
        for path in stale:
            typer.echo(f"stale: {path.name}")
        if stale:
            raise typer.Exit(1)
        typer.echo(f"up to date ({len(outputs)} files, css v{version})")
        return

    for path, html in outputs.items():
        path.write_text(html, encoding="utf-8")
    changed = ", ".join(p.name for p in stale) or "nothing"
    typer.echo(f"wrote {len(outputs)} files (css v{version}); changed: {changed}")


if __name__ == "__main__":
    sys.exit(app())
