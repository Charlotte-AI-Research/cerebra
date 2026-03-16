#!/usr/bin/env python3
"""
PDF to Markdown Converter
Converts a PDF (e.g., a university course catalog) to Markdown format.

Usage:
    python pdf_to_markdown.py input.pdf [output.md]

Requirements:
    pip install pdfplumber
"""

import sys
import re
import pdfplumber
from pathlib import Path


def clean_text(text: str) -> str:
    """Clean up extracted text artifacts."""
    if not text:
        return ""
    # Normalize whitespace within lines
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove soft hyphens / line-break hyphens
    text = re.sub(r'-\n([a-z])', r'\1', text)
    return text.strip()


def detect_heading(line: str, chars) -> str | None:
    """
    Heuristically detect headings by font size from pdfplumber char data.
    Returns a markdown heading string or None if it's normal body text.
    """
    if not line.strip() or not chars:
        return None

    line_chars = [c for c in chars if c.get('text', '').strip()]
    if not line_chars:
        return None

    sizes = [c.get('size', 0) for c in line_chars]
    avg_size = sum(sizes) / len(sizes) if sizes else 0

    # Adjust these thresholds to match your PDF's font sizes
    if avg_size >= 16:
        return f"# {line.strip()}"
    elif avg_size >= 13:
        return f"## {line.strip()}"
    elif avg_size >= 11.5:
        return f"### {line.strip()}"
    return None


def is_page_number(line: str) -> bool:
    """Skip lines that are just page numbers."""
    return bool(re.fullmatch(r'\s*\d{1,4}\s*', line))


def extract_page_to_md(page, page_num: int) -> str:
    """Extract a single page to Markdown text."""
    lines_md = []

    # Extract words with their positions grouped into lines
    words = page.extract_words(use_text_flow=True, extra_attrs=["size", "fontname"])
    if not words:
        return ""

    # Group words into lines by their top (y) position
    line_groups: dict[float, list] = {}
    for word in words:
        key = round(word["top"], 1)
        line_groups.setdefault(key, []).append(word)

    # Sort lines top-to-bottom
    sorted_lines = sorted(line_groups.items())

    for _, line_words in sorted_lines:
        line_words.sort(key=lambda w: w["x0"])
        line_text = " ".join(w["text"] for w in line_words)
        line_text = clean_text(line_text)

        if not line_text or is_page_number(line_text):
            continue

        # Build fake char list for heading detection
        chars = [{"text": w["text"], "size": w.get("size", 10)} for w in line_words]
        heading = detect_heading(line_text, chars)

        if heading:
            lines_md.append("\n" + heading)
        else:
            lines_md.append(line_text)

    return "\n".join(lines_md)


def pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """
    Convert a PDF file to Markdown.

    Args:
        pdf_path: Path to the input PDF file.
        output_path: Path for the output .md file. Defaults to same name as PDF.

    Returns:
        Path to the written Markdown file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if output_path is None:
        output_path = pdf_path.with_suffix(".md")
    output_path = Path(output_path)

    print(f"Converting: {pdf_path}")
    print(f"Output:     {output_path}")

    all_md = []

    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        print(f"Total pages: {total}")

        for i, page in enumerate(pdf.pages, start=1):
            if i % 50 == 0 or i == 1 or i == total:
                print(f"  Processing page {i}/{total}...")

            page_md = extract_page_to_md(page, i)
            if page_md.strip():
                all_md.append(f"\n\n<!-- Page {i} -->\n")
                all_md.append(page_md)

    markdown_content = "\n".join(all_md)

    # Post-process: collapse excessive blank lines
    markdown_content = re.sub(r'\n{4,}', '\n\n\n', markdown_content)

    output_path.write_text(markdown_content, encoding="utf-8")
    print(f"\nDone! Markdown saved to: {output_path}")
    print(f"Output size: {output_path.stat().st_size / 1024:.1f} KB")
    return str(output_path)


if __name__ == "__main__":
    # Default path — change if needed
    DEFAULT_PDF = r"C:\Users\jayyl\Desktop\cerebra\data\2025-2026-Undergraduate-Catalog.pdf"
    DEFAULT_OUT = r"C:\Users\jayyl\Desktop\cerebra\data\processed\2025-2026-Undergraduate-Catalog.md"

    if len(sys.argv) >= 2:
        input_pdf = sys.argv[1]
        output_md = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        input_pdf = DEFAULT_PDF
        output_md = DEFAULT_OUT

    pdf_to_markdown(input_pdf, output_md)