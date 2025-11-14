from pathlib import Path
import uuid
import re

# Root where all raw .txt files live (any subfolders allowed)
RAW_ROOT = Path("data/raw")

# Where to write markdown files
MD_DIR = Path("data/processed/markdown")
MD_DIR.mkdir(parents=True, exist_ok=True)


# ---------- helpers ----------

def basic_clean(text: str) -> str:
    """Light cleanup of the page body."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = [line.strip() for line in text.split("\n")]
    cleaned_lines = []

    for line in lines:
        # Drop huge ===== lines
        if set(line) == {"="} and len(line) > 5:
            continue

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)
    # Collapse 3+ blank lines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def slugify(text: str) -> str:
    """Make a simple, safe filename slug from a title."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "page"


# ---------- parsing logic ----------

def parse_pages_with_markers(raw_text: str):
    """
    Parse multi-page files using markers like:
    === PAGE: https://... (Depth 11) ===
    TITLE: Some Title ...
    """
    pages = []

    parts = re.split(r"^=== PAGE: (.+?) ===\n", raw_text, flags=re.M)
    # parts: [prefix, url1, content1, url2, content2, ...]

    for i in range(1, len(parts), 2):
        url_and_depth = parts[i].strip()
        content = parts[i + 1] if i + 1 < len(parts) else ""

        # Extract URL and Depth
        url = url_and_depth
        depth_match = re.search(r"\(Depth\s+\d+\)", url_and_depth)
        if depth_match:
            url = url_and_depth[:depth_match.start()].strip()
        depth = None
        if depth_match:
            dnum = re.search(r"\d+", depth_match.group(0))
            if dnum:
                depth = int(dnum.group(0))

        # TITLE: line (optional)
        title_match = re.search(r"^TITLE:\s*(.+)$", content, flags=re.M)
        if title_match:
            title = title_match.group(1).strip()
            body = re.sub(r"^TITLE:.*$", "", content, flags=re.M).strip()
        else:
            title = url or "Untitled Page"
            body = content.strip()

        body = basic_clean(body)
        if not body:
            continue

        pages.append(
            {
                "url": url,
                "depth": depth,
                "title": title,
                "body": body,
            }
        )

    return pages


def parse_single_page(raw_text: str, default_title: str):
    """
    Fallback for plain .txt with no page markers:
    Treat whole file as a single page.
    """
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    # Heuristic: if first line looks like a title (short, no period), use it.
    lines = text.split("\n")
    first_line = lines[0].strip()
    if 0 < len(first_line) < 80 and not re.search(r"[.!?]$", first_line):
        title = first_line
        body = "\n".join(lines[1:]).strip()
    else:
        title = default_title
        body = text

    body = basic_clean(body)
    if not body:
        return []

    return [
        {
            "url": "",
            "depth": None,
            "title": title,
            "body": body,
        }
    ]


# ---------- main conversion per file ----------

def get_source_for_file(txt_path: Path) -> str:
    """
    Source = first directory under RAW_ROOT.
    e.g., data/raw/cci/cci_data_1.txt -> 'cci'
          data/raw/events/oct.txt -> 'events'
          data/raw/whatever.txt -> 'default'
    """
    try:
        rel = txt_path.relative_to(RAW_ROOT)
    except ValueError:
        return "default"

    parts = rel.parts
    if len(parts) > 1:
        return parts[0]
    return "default"


def convert_txt_file(txt_path: Path):
    raw_text = txt_path.read_text(encoding="utf-8", errors="ignore")
    source = get_source_for_file(txt_path)

    if "=== PAGE:" in raw_text:
        pages = parse_pages_with_markers(raw_text)
    else:
        pages = parse_single_page(raw_text, default_title=txt_path.stem)

    if not pages:
        print(f"[SKIP] No content found in {txt_path}")
        return

    print(f"{txt_path}: {len(pages)} page(s)")

    for idx, page in enumerate(pages):
        title = page["title"]
        body = page["body"]
        url = page.get("url", "")
        depth = page.get("depth", None)

        base_stem = txt_path.stem
        title_slug = slugify(title)[:50]
        md_name = f"{base_stem}_{idx:03d}_{title_slug}.md"
        md_path = MD_DIR / md_name

        frontmatter_lines = [
            "---",
            f"id: {uuid.uuid4()}",
            f"title: {title}",
            f"url: {url}",
            "section: ",  # can be filled later (e.g., student-life, events, etc.)
            f"source: {source}",
            f"source_file: {txt_path.relative_to(RAW_ROOT)}",
            f"depth: {depth if depth is not None else ''}",
            "---",
            "",
        ]

        with md_path.open("w", encoding="utf-8") as md_file:
            md_file.write("\n".join(frontmatter_lines))
            md_file.write(body)
            if not body.endswith("\n"):
                md_file.write("\n")

        print(f"  → {md_path.name}")


def main():
    if not RAW_ROOT.exists():
        print(f"RAW_ROOT {RAW_ROOT.resolve()} does not exist.")
        return

    txt_files = sorted(RAW_ROOT.rglob("*.txt"))
    if not txt_files:
        print(f"No .txt files found under {RAW_ROOT.resolve()}")
        return

    print(f"Found {len(txt_files)} .txt files under {RAW_ROOT.resolve()}")

    for txt_path in txt_files:
        convert_txt_file(txt_path)

    print("Done converting all .txt files to markdown pages.")


if __name__ == "__main__":
    main()
