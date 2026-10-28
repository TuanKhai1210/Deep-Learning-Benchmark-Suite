#!/usr/bin/env python3
"""Build and optionally serve a local HTML preview of the docs Jekyll site.

Usage:
    python scripts/build_docs_preview.py          # Builds static HTML in _preview/
    python scripts/build_docs_preview.py --serve  # Builds and serves on http://localhost:8000
"""

import argparse
import os
import re
import shutil
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler


def render_page(md_path: str, layout_html: str, out_dir: str, rel_prefix: str = "./"):
    with open(md_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Parse frontmatter
    title = "DL Benchmark Suite"
    description = ""
    body_class = "article-page"
    page_content = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm = parts[1]
            page_content = parts[2].strip()
            for line in fm.splitlines():
                if line.startswith("title:"):
                    title = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("body_class:"):
                    body_class = line.split(":", 1)[1].strip().strip('"').strip("'")

    # Replace liquid filters in layout and content
    html = layout_html
    html = html.replace("{{ page.title | escape }}", title)
    html = html.replace("{{ page.description | default: site.description | escape }}", description)
    html = html.replace("{{ page.body_class | default: 'article-page' }}", body_class)

    # Insert content
    html = html.replace("{{ content }}", page_content)

    # Resolve {{ '/path' | relative_url }}
    def replace_rel(match):
        url = match.group(1).lstrip("/")
        return rel_prefix + url

    html = re.sub(r"\{\{\s*'([^']+)'\s*\|\s*relative_url\s*\}\}", replace_rel, html)
    html = re.sub(r'\{\{\s*"([^"]+)"\s*\|\s*relative_url\s*\}\}', replace_rel, html)

    base_name = os.path.splitext(os.path.basename(md_path))[0]
    out_file = os.path.join(out_dir, f"{base_name}.html")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Generated: {out_file}")
    return out_file


def build_preview(out_dir: str = "_preview"):
    os.makedirs(out_dir, exist_ok=True)
    layout_path = os.path.join("docs", "_layouts", "default.html")
    with open(layout_path, "r", encoding="utf-8") as f:
        layout_html = f.read()

    # Copy assets
    assets_src = os.path.join("docs", "assets")
    assets_dst = os.path.join(out_dir, "assets")
    if os.path.exists(assets_dst):
        shutil.rmtree(assets_dst)
    shutil.copytree(assets_src, assets_dst)
    print(f"Copied assets to {assets_dst}")

    # Render pages
    docs_dir = "docs"
    for item in os.listdir(docs_dir):
        if item.endswith(".md"):
            md_path = os.path.join(docs_dir, item)
            render_page(md_path, layout_html, out_dir, rel_prefix="./")


def main():
    parser = argparse.ArgumentParser(description="Build and serve local docs preview.")
    parser.add_argument("--out-dir", default="_preview", help="Output directory for generated preview")
    parser.add_argument("--serve", action="store_true", help="Start a local HTTP server")
    parser.add_argument("--port", type=int, default=8000, help="Port for local server")
    args = parser.parse_args()

    build_preview(args.out_dir)
    print(f"\nLocal preview built successfully in '{args.out_dir}/'!")
    print(f"Direct file links:")
    print(f"  Assignment 1: file:///{os.path.abspath(os.path.join(args.out_dir, 'a1.html')).replace('\\', '/')}")
    print(f"  Assignment 2: file:///{os.path.abspath(os.path.join(args.out_dir, 'a2.html')).replace('\\', '/')}")
    print(f"  Home Page:    file:///{os.path.abspath(os.path.join(args.out_dir, 'index.html')).replace('\\', '/')}")

    if args.serve:
        print(f"\nStarting HTTP server at http://localhost:{args.port}/ ...")
        print("Press Ctrl+C to stop.")
        os.chdir(args.out_dir)
        server = HTTPServer(("127.0.0.1", args.port), SimpleHTTPRequestHandler)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
