from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NotebookContext:
    source: str
    text: str


def strip_markdown_images(markdown: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[image omitted]", markdown)
    text = re.sub(r"!\[[^\]]*\]\[[^\]]+\]", "[image omitted]", text)
    text = re.sub(r"<img[^>]*>", "[image omitted]", text, flags=re.IGNORECASE)
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[image omitted]", text)
    return text


def _normalize_cell_types(cell_types: list[str] | tuple[str, ...] | None) -> set[str] | None:
    if not isinstance(cell_types, (list, tuple)):
        return None
    normalized = {str(item).strip().lower() for item in cell_types if str(item).strip()}
    return normalized or None


def _render_output(output: Any) -> str:
    if not isinstance(output, dict):
        return ""
    output_type = str(output.get("output_type") or "").strip().lower()
    if output_type == "stream":
        return str(output.get("text") or "").strip()
    if output_type in {"execute_result", "display_data"}:
        data = output.get("data")
        if isinstance(data, dict):
            for key in ("text/plain", "text/markdown"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, list):
                    joined = "".join(str(item) for item in value).strip()
                    if joined:
                        return joined
    if output_type == "error":
        traceback = output.get("traceback")
        if isinstance(traceback, list):
            joined = "\n".join(str(item) for item in traceback).strip()
            if joined:
                return joined
        return str(output.get("evalue") or "").strip()
    return ""


def load_notebook_text(
    path: Path,
    *,
    cell_types: list[str] | tuple[str, ...] | None = None,
    start_cell: int | None = None,
    end_cell: int | None = None,
    include_outputs: bool = False,
) -> str | None:
    try:
        import nbformat
    except Exception:
        return None

    try:
        notebook = nbformat.read(path, as_version=4)
    except Exception:
        return None

    allowed_types = _normalize_cell_types(cell_types)
    start_index = max(0, int(start_cell or 0))
    end_index = int(end_cell) if isinstance(end_cell, int) else None
    rendered_parts: list[str] = []

    for idx, cell in enumerate(getattr(notebook, "cells", []) or []):
        if idx < start_index:
            continue
        if end_index is not None and idx > end_index:
            break
        cell_type = str(getattr(cell, "cell_type", "unknown") or "unknown").strip().lower()
        if allowed_types is not None and cell_type not in allowed_types:
            continue
        source = str(getattr(cell, "source", "") or "")
        outputs = list(getattr(cell, "outputs", []) or [])

        header = f"### [{cell_type}:{idx}]"
        if cell_type == "markdown":
            body = strip_markdown_images(source).strip()
        elif cell_type == "code":
            body = f"```python\n{source.rstrip()}\n```".strip()
        else:
            body = source.strip()
        if not body and not (include_outputs and outputs):
            continue

        rendered = [header]
        if body:
            rendered.append(body)
        if include_outputs and outputs:
            output_texts = [_render_output(item) for item in outputs]
            output_texts = [text for text in output_texts if text]
            if output_texts:
                rendered.append("Outputs:")
                rendered.extend(output_texts[:5])
        rendered_parts.append("\n".join(rendered).rstrip())

    return "\n\n".join(part for part in rendered_parts if part).strip()


def load_notebook_context(path: Path) -> NotebookContext | None:
    text = load_notebook_text(path)
    if not text:
        return None
    return NotebookContext(source=str(path), text=text)
