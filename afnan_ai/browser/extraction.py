"""Page content extraction — clean, normalized page content.

Turns the driver's raw page document (headings, paragraphs,
lists, links, tables) into content a Planner/LLM can actually
use:

* boilerplate and navigation noise are filtered out (short
  fragments, cookie/copyright lines, duplicate paragraphs,
  empty links);
* tables keep their row/column structure;
* long pages are split into bounded chunks with token
  estimates, so nothing downstream is flooded;
* secrets stay out: extracted text and URLs pass through the
  redactor before they can reach AgentState or the model.

Where structured extraction was unavailable, the flat page
text is paragraph-split instead (``structured=False`` says so
honestly).
"""
from __future__ import annotations

import re
from typing import Any

from afnan_ai.redaction import redact_text

_BOILERPLATE = re.compile(
    r"(all rights reserved|cookie (policy|settings|preferences)|"
    r"privacy policy|terms of (service|use)|subscribe to our "
    r"newsletter|sign up for our newsletter|©|copyright \d{4})",
    re.IGNORECASE,
)
_MIN_PARAGRAPH = 25
_MAX_LINKS = 100
_MAX_TABLE_ROWS = 100
_MAX_TABLE_COLS = 20


def _clean_items(items, minimum: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items or []:
        text = re.sub(r"\s+", " ", str(item)).strip()
        if len(text) < minimum or _BOILERPLATE.search(text):
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(redact_text(text))
    return out


def clean_document(
    doc: dict[str, Any], *, max_chars: int = 4000
) -> dict[str, Any]:
    """Normalize a raw page document into bounded content."""
    structured = bool(doc.get("structured", True))
    headings = [
        {"level": h.get("level", 2), "text": redact_text(str(h["text"]))}
        for h in doc.get("headings") or []
        if str(h.get("text", "")).strip()
    ]
    paragraphs = _clean_items(doc.get("paragraphs"), _MIN_PARAGRAPH)
    if not paragraphs and doc.get("text"):
        # flat-text fallback: split visible text into paragraphs
        chunks = re.split(r"\n{2,}|\n", str(doc["text"]))
        paragraphs = _clean_items(chunks, 40)
    lists = []
    for lst in doc.get("lists") or []:
        items = _clean_items(lst, 2)
        if items:
            lists.append(items)
    links = []
    seen_urls: set[str] = set()
    for link in doc.get("links") or []:
        url = str(link.get("url", "")).strip()
        text = re.sub(r"\s+", " ", str(link.get("text", ""))).strip()
        if not url or not text or url in seen_urls or url.startswith("javascript:"):
            continue
        seen_urls.add(url)
        links.append({"text": redact_text(text), "url": redact_text(url)})
        if len(links) >= _MAX_LINKS:
            break
    tables = []
    for table in doc.get("tables") or []:
        rows = [
            [redact_text(re.sub(r"\s+", " ", str(c)).strip())
             for c in row[:_MAX_TABLE_COLS]]
            for row in (table.get("rows") or [])[:_MAX_TABLE_ROWS]
        ]
        rows = [r for r in rows if any(r)]
        if rows:
            entry = {"rows": rows, "row_count": len(rows),
                     "column_count": max(len(r) for r in rows)}
            if table.get("caption"):
                entry["caption"] = redact_text(str(table["caption"]))
            tables.append(entry)

    content_found = bool(paragraphs or headings or lists or tables)
    chunks = _chunk(headings, paragraphs, lists, max_chars)
    total_chars = sum(c["char_count"] for c in chunks)
    return {
        "url": str(doc.get("url", "")),
        "title": redact_text(str(doc.get("title", ""))),
        "structured": structured,
        "content_found": content_found,
        "headings": headings,
        "paragraphs": paragraphs,
        "lists": lists,
        "links": links,
        "tables": tables,
        "chunks": chunks,
        "chunk_count": len(chunks),
        "total_chars": total_chars,
        "token_estimate": total_chars // 4,
        "counts": {
            "headings": len(headings),
            "paragraphs": len(paragraphs),
            "lists": len(lists),
            "links": len(links),
            "tables": len(tables),
        },
    }


def _chunk(headings, paragraphs, lists, max_chars: int) -> list[dict[str, Any]]:
    max_chars = max(200, int(max_chars))
    blocks: list[str] = []
    for heading in headings:
        blocks.append("#" * int(heading.get("level", 2)) + " "
                      + heading["text"])
    blocks.extend(paragraphs)
    for lst in lists:
        blocks.append("\n".join(f"- {item}" for item in lst))
    chunks: list[dict[str, Any]] = []
    current = ""
    for block in blocks:
        if current and len(current) + len(block) + 2 > max_chars:
            chunks.append(_chunk_entry(len(chunks), current))
            current = ""
        if len(block) > max_chars:
            # a single oversize block is hard-split
            for i in range(0, len(block), max_chars):
                chunks.append(_chunk_entry(len(chunks), block[i:i + max_chars]))
            continue
        current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(_chunk_entry(len(chunks), current))
    return chunks


def _chunk_entry(index: int, text: str) -> dict[str, Any]:
    return {
        "index": index,
        "text": text,
        "char_count": len(text),
        "token_estimate": len(text) // 4,
    }
