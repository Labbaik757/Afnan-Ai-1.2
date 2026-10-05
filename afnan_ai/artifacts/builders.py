"""Controlled artifact content builders.

The agent never gets direct filesystem/code execution to
make artifacts.  Instead it calls a builder with *data*;
the builder produces bytes.  Every builder redacts secrets
from the content first, escapes markup, and returns a
fixed, safe format:

* MarkdownBuilder   → .md   (documents, reports)
* HtmlBuilder       → .html (reports, web pages; scripts stripped)
* PdfBuilder        → .pdf  (stdlib-only minimal PDF)
* CsvBuilder        → .csv  (spreadsheets)
* JsonBuilder       → .json (structured data)
* TextBuilder       → .txt  (plain text, code output)
* SlidesBuilder     → .html (presentations as slide decks)
* ImageBuilder      → .png  (title cards / bar charts via the
  screen package's stdlib PNG codec)
"""

from __future__ import annotations

import csv
import html
import io
import json
from typing import Any

from afnan_ai.redaction import redact_text


def _clean(text: Any) -> str:
    return redact_text(str(text or ""))


class ContentBuilder:
    """Base: data in, bytes out."""

    extension: str = ".txt"
    media_type: str = "text/plain"

    def build(self, data: dict[str, Any]) -> bytes:
        raise NotImplementedError


class MarkdownBuilder(ContentBuilder):
    extension = ".md"
    media_type = "text/markdown"

    def build(self, data: dict[str, Any]) -> bytes:
        title = _clean(data.get("title", "Untitled"))
        lines = [f"# {title}", ""]
        intro = _clean(data.get("introduction", ""))
        if intro:
            lines += [intro, ""]
        for section in data.get("sections") or []:
            heading = _clean(section.get("heading", ""))
            body = _clean(section.get("body", ""))
            if heading:
                lines.append(f"## {heading}")
            if body:
                lines.append(body)
            lines.append("")
        sources = data.get("sources") or []
        if sources:
            lines.append("## Sources")
            for source in sources:
                lines.append(f"- {_clean(source)}")
            lines.append("")
        return ("\n".join(lines).strip() + "\n").encode("utf-8")


class HtmlBuilder(ContentBuilder):
    extension = ".html"
    media_type = "text/html"

    def build(self, data: dict[str, Any]) -> bytes:
        title = html.escape(_clean(data.get("title", "Untitled")))
        parts = [
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{title}</title>",
            "<style>body{font-family:sans-serif;max-width:800px;"
            "margin:2em auto;padding:0 1em}h1{border-bottom:1px "
            "solid #ccc}</style></head><body>",
            f"<h1>{title}</h1>",
        ]
        intro = _clean(data.get("introduction", ""))
        if intro:
            parts.append(f"<p>{html.escape(intro)}</p>")
        for section in data.get("sections") or []:
            heading = html.escape(_clean(section.get("heading", "")))
            body = html.escape(_clean(section.get("body", "")))
            if heading:
                parts.append(f"<h2>{heading}</h2>")
            if body:
                # Preserve line breaks, nothing else.
                safe = "<br>".join(body.splitlines())
                parts.append(f"<p>{safe}</p>")
        sources = data.get("sources") or []
        if sources:
            parts.append("<h2>Sources</h2><ul>")
            for source in sources:
                parts.append(f"<li>{html.escape(_clean(source))}</li>")
            parts.append("</ul>")
        # NOTE: no <script> is ever emitted — generated pages
        # are static by construction.
        parts.append("</body></html>")
        return "\n".join(parts).encode("utf-8")


class TextBuilder(ContentBuilder):
    extension = ".txt"

    def build(self, data: dict[str, Any]) -> bytes:
        title = _clean(data.get("title", ""))
        body = _clean(data.get("body", data.get("text", "")))
        text = (f"{title}\n{'=' * len(title)}\n\n{body}\n"
                if title else f"{body}\n")
        return text.encode("utf-8")


class CsvBuilder(ContentBuilder):
    extension = ".csv"
    media_type = "text/csv"

    def build(self, data: dict[str, Any]) -> bytes:
        headers = [
            _clean(h) for h in (data.get("headers") or [])
        ]
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        if headers:
            writer.writerow(headers)
        for row in data.get("rows") or []:
            writer.writerow([_clean(cell) for cell in row])
        return buffer.getvalue().encode("utf-8")


class JsonBuilder(ContentBuilder):
    extension = ".json"
    media_type = "application/json"

    def build(self, data: dict[str, Any]) -> bytes:
        from afnan_ai.redaction import redact_value

        payload = redact_value(data.get("data", data))
        return (
            json.dumps(payload, indent=2, ensure_ascii=False,
                       default=str) + "\n"
        ).encode("utf-8")


class SlidesBuilder(ContentBuilder):
    extension = ".html"
    media_type = "text/html"

    def build(self, data: dict[str, Any]) -> bytes:
        title = html.escape(_clean(data.get("title", "Untitled")))
        slides = []
        for index, slide in enumerate(
            data.get("slides") or [], start=1
        ):
            heading = html.escape(
                _clean(slide.get("heading", f"Slide {index}"))
            )
            bullets = "".join(
                f"<li>{html.escape(_clean(b))}</li>"
                for b in (slide.get("bullets") or [])
            )
            slides.append(
                f'<section class="slide"><h2>{heading}</h2>'
                f"<ul>{bullets}</ul></section>"
            )
        return "\n".join([
            "<!DOCTYPE html>",
            '<html lang="en"><head><meta charset="utf-8">',
            f"<title>{title}</title>",
            "<style>.slide{border:1px solid #ccc;margin:1em 0;"
            "padding:1em}h1{text-align:center}</style></head><body>",
            f"<h1>{title}</h1>",
            *slides,
            "</body></html>",
        ]).encode("utf-8")


def _pdf_escape(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


class PdfBuilder(ContentBuilder):
    """Minimal stdlib PDF: title + wrapped text lines."""

    extension = ".pdf"
    media_type = "application/pdf"

    def build(self, data: dict[str, Any]) -> bytes:
        title = _clean(data.get("title", "Untitled"))
        raw_lines: list[str] = [title, ""]
        for section in data.get("sections") or []:
            heading = _clean(section.get("heading", ""))
            if heading:
                raw_lines.append(heading)
            for line in _clean(
                section.get("body", "")
            ).splitlines():
                raw_lines.append(line)
            raw_lines.append("")
        intro = _clean(data.get("introduction", ""))
        if intro:
            raw_lines.extend([intro, ""])

        # Wrap + paginate (Letter, 1in margins, 12pt).
        wrapped: list[str] = []
        for line in raw_lines:
            while len(line) > 90:
                wrapped.append(line[:90])
                line = line[90:]
            wrapped.append(line)
        pages: list[list[str]] = []
        for index in range(0, len(wrapped) or 1, 45):
            pages.append(wrapped[index:index + 45] or [""])

        objects: list[bytes] = []
        content_ids: list[int] = []
        for page_lines in pages:
            text_ops = ["BT /F1 12 Tf 72 720 Td 14 TL"]
            for line in page_lines:
                safe = _pdf_escape(line[:120])
                text_ops.append(f"({safe}) Tj T*")
            text_ops.append("ET")
            stream = "\n".join(text_ops).encode("latin-1")
            content_ids.append(len(objects) + 1)
            objects.append(
                f"<< /Length {len(stream)} >>\nstream\n".encode(
                    "latin-1"
                )
                + stream + b"\nendstream"
            )
        # Page objects reference their content streams.
        first_page_id = len(objects) + 1
        kids = " ".join(
            f"{first_page_id + i} 0 R"
            for i in range(len(pages))
        )
        for content_id in content_ids:
            objects.append(
                f"<< /Type /Page /Parent {first_page_id + len(pages)} 0 R "
                f"/MediaBox [0 0 612 792] /Contents {content_id} 0 R "
                f"/Resources << /Font << /F1 {first_page_id + len(pages) + 1} 0 R >> >> >>"
                .encode("latin-1")
            )
        pages_id = first_page_id + len(pages)
        font_id = pages_id + 1
        objects.append(
            f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>"
            .encode("latin-1")
        )
        objects.append(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        )
        catalog_id = font_id + 1
        objects.append(
            f"<< /Type /Catalog /Pages {pages_id} 0 R >>"
            .encode("latin-1")
        )

        out = bytearray(b"%PDF-1.4\n")
        offsets: list[int] = []
        for number, body in enumerate(objects, start=1):
            offsets.append(len(out))
            out += f"{number} 0 obj\n".encode("latin-1")
            out += body + b"\nendobj\n"
        xref_at = len(out)
        out += f"xref\n0 {len(objects) + 1}\n".encode("latin-1")
        out += b"0000000000 65535 f \n"
        for offset in offsets:
            out += f"{offset:010d} 00000 n \n".encode("latin-1")
        out += (
            f"trailer\n<< /Size {len(objects) + 1} "
            f"/Root {catalog_id} 0 R >>\nstartxref\n{xref_at}\n"
            f"%%EOF"
        ).encode("latin-1")
        return bytes(out)


class ImageBuilder(ContentBuilder):
    """Simple PNG: title card or bar chart (stdlib codec)."""

    extension = ".png"
    media_type = "image/png"

    def build(self, data: dict[str, Any]) -> bytes:
        from afnan_ai.screen.image import encode_png

        kind = str(data.get("kind", "card"))
        if kind == "chart":
            return self._chart(data, encode_png)
        return self._card(data, encode_png)

    def _card(self, data: dict[str, Any],
              encode_png) -> bytes:
        width, height = 800, 400
        bg = (24, 32, 48)
        fg = (240, 240, 240)
        accent = (90, 160, 255)
        rows = []
        for y in range(height):
            row = []
            for x in range(width):
                if y < 8 or x < 8:
                    row.append(accent)
                else:
                    row.append(bg)
            rows.append(row)
        # Title bar block (pixel-font-less: a bright band whose
        # width encodes the title length).
        title = _clean(data.get("title", "Untitled"))
        band_w = min(width - 40, 40 + len(title) * 6)
        for y in range(120, 150):
            for x in range(40, 40 + band_w):
                rows[y][x] = fg
        return encode_png(width, height, rows)

    def _chart(self, data: dict[str, Any],
               encode_png) -> bytes:
        width, height = 800, 500
        bg = (255, 255, 255)
        bar = (70, 130, 200)
        axis = (60, 60, 60)
        rows = [[bg] * width for _ in range(height)]
        values = [
            max(0.0, float(v))
            for v in (data.get("values") or [1, 2, 3])
        ]
        peak = max(values) or 1.0
        n = len(values)
        slot = width // max(1, n)
        base = height - 40
        for i, value in enumerate(values):
            bar_h = int((value / peak) * (height - 100))
            x0 = i * slot + slot // 4
            x1 = (i + 1) * slot - slot // 4
            for y in range(base - bar_h, base):
                for x in range(x0, x1):
                    rows[y][x] = bar
        for x in range(width):
            rows[base][x] = axis
        return encode_png(width, height, rows)


BUILDERS: dict[str, ContentBuilder] = {
    "document": MarkdownBuilder(),
    "report": MarkdownBuilder(),
    "html": HtmlBuilder(),
    "pdf": PdfBuilder(),
    "spreadsheet": CsvBuilder(),
    "structured_data": JsonBuilder(),
    "code_output": TextBuilder(),
    "presentation": SlidesBuilder(),
    "image": ImageBuilder(),
}


def builder_for(artifact_type: str) -> ContentBuilder:
    """Controlled builder for a type (defaults to text)."""
    return BUILDERS.get(
        str(artifact_type or "").lower(), TextBuilder()
    )
