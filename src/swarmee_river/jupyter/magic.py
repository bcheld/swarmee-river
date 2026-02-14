from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Agent

from swarmee_river.tools import get_tools
from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.kb_utils import load_system_prompt
from swarmee_river.utils.model_utils import default_model_config, load_model, load_path

_AGENT_SINGLETON: Agent | None = None
_AGENT_FINGERPRINT: str | None = None


@dataclass(frozen=True)
class _NotebookContext:
    source: str
    text: str


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


def _strip_markdown_images(markdown: str) -> str:
    # Markdown image syntax: ![alt](url) or ![alt][ref]
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "[image omitted]", markdown)
    text = re.sub(r"!\[[^\]]*\]\[[^\]]+\]", "[image omitted]", text)
    # HTML images
    text = re.sub(r"<img[^>]*>", "[image omitted]", text, flags=re.IGNORECASE)
    # Inline data URIs (common in markdown/html)
    text = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[image omitted]", text)
    return text


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-(limit // 2) :]
    return f"{head}\n\n… (truncated, {len(text)} chars total) …\n\n{tail}"


def _guess_notebook_path(ipython: Any) -> Path | None:
    env_path = os.getenv("SWARMEE_NOTEBOOK_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser()
        if p.exists() and p.is_file():
            return p

    # Optional: ipynbname can often locate the active notebook in Jupyter.
    try:
        import ipynbname  # type: ignore

        p = Path(str(ipynbname.path())).expanduser()
        if p.exists() and p.is_file():
            return p
    except Exception:
        pass

    # VS Code sometimes exposes a path-like value in the user namespace.
    for key in ("__vsc_ipynb_file__", "__notebook_path__", "NOTEBOOK_PATH"):
        try:
            candidate = ipython.user_ns.get(key)  # type: ignore[attr-defined]
        except Exception:
            candidate = None
        if isinstance(candidate, str) and candidate.endswith(".ipynb"):
            p = Path(candidate).expanduser()
            if p.exists() and p.is_file():
                return p

    return None


def _load_ipynb_context(path: Path) -> _NotebookContext | None:
    try:
        import nbformat  # type: ignore
    except Exception:
        return None

    try:
        nb = nbformat.read(path, as_version=4)
    except Exception:
        return None

    parts: list[str] = []
    for idx, cell in enumerate(getattr(nb, "cells", []) or []):
        cell_type = getattr(cell, "cell_type", "unknown")
        source = getattr(cell, "source", "") or ""
        if not source.strip():
            continue

        if cell_type == "markdown":
            source = _strip_markdown_images(source)
            parts.append(f"### [markdown:{idx}]\n{source}".rstrip())
        elif cell_type == "code":
            # Notebook context should exclude images; outputs are ignored entirely.
            parts.append(f"### [code:{idx}]\n```python\n{source.rstrip()}\n```")
        else:
            parts.append(f"### [{cell_type}:{idx}]\n{source}".rstrip())

    return _NotebookContext(source=str(path), text="\n\n".join(parts).strip())


def _load_ipython_history_context(ipython: Any) -> _NotebookContext:
    # Fallback when we cannot locate a .ipynb file: include executed inputs only.
    inputs: list[str] = []
    try:
        raw = list(getattr(ipython, "user_ns", {}).get("In", []))  # type: ignore[attr-defined]
    except Exception:
        raw = []

    for idx, cell in enumerate(raw):
        if not isinstance(cell, str) or not cell.strip():
            continue
        # Skip the current %%swarmee cell to avoid echoing the prompt back into context.
        if cell.lstrip().startswith("%%swarmee"):
            continue
        inputs.append(f"### [code:{idx}]\n```python\n{cell.rstrip()}\n```")

    return _NotebookContext(source="ipython_history", text="\n\n".join(inputs).strip())


def _collect_notebook_context(ipython: Any) -> _NotebookContext:
    path = _guess_notebook_path(ipython)
    if path is not None:
        ctx = _load_ipynb_context(path)
        if ctx is not None:
            return ctx
    return _load_ipython_history_context(ipython)


def _agent_fingerprint() -> str:
    provider = (os.getenv("SWARMEE_JUPYTER_MODEL_PROVIDER") or os.getenv("SWARMEE_MODEL_PROVIDER") or "bedrock").strip()
    model_id = os.getenv("SWARMEE_OPENAI_MODEL_ID") or os.getenv("STRANDS_MODEL_ID") or ""
    max_tokens = os.getenv("SWARMEE_MAX_TOKENS") or os.getenv("STRANDS_MAX_TOKENS") or ""
    return json.dumps({"provider": provider, "model_id": model_id, "max_tokens": max_tokens}, sort_keys=True)


def _get_or_create_agent() -> Agent:
    global _AGENT_SINGLETON, _AGENT_FINGERPRINT

    fingerprint = _agent_fingerprint()
    if _AGENT_SINGLETON is not None and _AGENT_FINGERPRINT == fingerprint:
        return _AGENT_SINGLETON

    load_env_file()

    provider = (os.getenv("SWARMEE_JUPYTER_MODEL_PROVIDER") or os.getenv("SWARMEE_MODEL_PROVIDER") or "bedrock").strip()
    model_path = load_path(provider)
    model_config = default_model_config(provider)
    model = load_model(model_path, model_config)

    tools = list(get_tools().values())
    system_prompt = load_system_prompt()

    _AGENT_SINGLETON = Agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        messages=[],
        callback_handler=None,  # notebook-friendly: no spinners/streaming callbacks by default
        load_tools_from_directory=True,
    )
    _AGENT_FINGERPRINT = fingerprint
    return _AGENT_SINGLETON


def _format_prompt(*, notebook_context: _NotebookContext, user_prompt: str) -> str:
    context_limit = int(os.getenv("SWARMEE_NOTEBOOK_CONTEXT_CHARS", "120000"))
    ctx_text = _truncate_text(notebook_context.text, context_limit)

    return (
        "You are Swarmee River running inside a Jupyter notebook.\n"
        "You may use tools to explore the project in the current working directory.\n\n"
        f"Notebook context source: {notebook_context.source}\n\n"
        "Notebook context (excluding images):\n"
        f"{ctx_text}\n\n"
        "User prompt:\n"
        f"{user_prompt.strip()}\n"
    ).strip()


def _run_swarmee(ipython: Any, user_prompt: str) -> str:
    agent = _get_or_create_agent()
    notebook_context = _collect_notebook_context(ipython)
    prompt = _format_prompt(notebook_context=notebook_context, user_prompt=user_prompt)
    result = agent(prompt)
    return str(result)


def load_ipython_extension(ipython: Any) -> None:
    try:
        from IPython.core.magic import Magics, cell_magic, magics_class
    except Exception as e:  # pragma: no cover
        raise RuntimeError("IPython is required to use the Swarmee notebook extension.") from e

    @magics_class
    class SwarmeeMagics(Magics):
        @cell_magic
        def swarmee(self, line: str, cell: str) -> str:
            # Allow disabling notebook context for quick one-offs.
            include_context = not _truthy(os.getenv("SWARMEE_NOTEBOOK_NO_CONTEXT"))
            prompt = cell
            if line and line.strip():
                # Treat line as additional instructions / flags.
                prompt = f"{line.strip()}\n\n{cell}"

            if not include_context:
                agent = _get_or_create_agent()
                result = agent(prompt.strip())
                text = str(result)
            else:
                text = _run_swarmee(self.shell, prompt)  # type: ignore[arg-type]

            # Print and also return (so users can assign it).
            print(text)
            return text

    ipython.register_magics(SwarmeeMagics)


def unload_ipython_extension(_ipython: Any) -> None:
    # No-op: IPython does not provide a stable public API for unregistering magics.
    return None
