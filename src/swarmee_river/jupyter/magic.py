from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Agent

from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.packs import enabled_system_prompts, load_enabled_pack_tools
from swarmee_river.planning import WorkPlan, classify_intent, structured_plan_prompt
from swarmee_river.project_map import build_context_snapshot
from swarmee_river.runtime_env import detect_runtime_environment, render_runtime_environment_section
from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import load_settings
from swarmee_river.tools import get_tools
from swarmee_river.utils.env_utils import load_env_file
from swarmee_river.utils.kb_utils import load_system_prompt
from swarmee_river.utils.provider_utils import resolve_model_provider

try:
    from swarmee_river.hooks.jsonl_logger import JSONLLoggerHooks as _JSONLLoggerHooks
    from swarmee_river.hooks.tool_consent import ToolConsentHooks as _ToolConsentHooks
    from swarmee_river.hooks.tool_message_repair import ToolMessageRepairHooks as _ToolMessageRepairHooks
    from swarmee_river.hooks.tool_policy import ToolPolicyHooks as _ToolPolicyHooks
    from swarmee_river.hooks.tool_result_limiter import ToolResultLimiterHooks as _ToolResultLimiterHooks

    _HAS_STRANDS_HOOKS = True
except Exception:
    _JSONLLoggerHooks = None  # type: ignore[misc,assignment]
    _ToolConsentHooks = None  # type: ignore[misc,assignment]
    _ToolMessageRepairHooks = None  # type: ignore[misc,assignment]
    _ToolPolicyHooks = None  # type: ignore[misc,assignment]
    _ToolResultLimiterHooks = None  # type: ignore[misc,assignment]
    _HAS_STRANDS_HOOKS = False
JSONLLoggerHooks: Any = _JSONLLoggerHooks
ToolConsentHooks: Any = _ToolConsentHooks
ToolMessageRepairHooks: Any = _ToolMessageRepairHooks
ToolPolicyHooks: Any = _ToolPolicyHooks
ToolResultLimiterHooks: Any = _ToolResultLimiterHooks

try:
    from strands.types.exceptions import MaxTokensReachedException as _MaxTokensReachedException
except Exception:  # pragma: no cover
    _MaxTokensReachedException = RuntimeError  # type: ignore[misc,assignment]
MaxTokensReachedException: type[Exception] = _MaxTokensReachedException


_RUNTIME_SINGLETON: "_NotebookRuntime | None" = None
_RUNTIME_FINGERPRINT: str | None = None

_TOOL_USAGE_RULES = (
    "Tool usage rules:\n"
    "- Use list/glob/file_list/file_search/file_read for repository exploration and file reading.\n"
    "- Do not use shell for ls/find/sed/cat/grep/rg when file tools can do it.\n"
    "- Reserve shell for real command execution tasks."
)


@dataclass(frozen=True)
class _NotebookContext:
    source: str
    text: str


@dataclass
class _NotebookRuntime:
    agent: Agent
    tools_dict: dict[str, Any]
    settings: Any
    model_manager: SessionModelManager
    runtime_environment: dict[str, Any]
    base_system_prompt: str
    artifact_store: ArtifactStore
    knowledge_base_id: str | None


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
            candidate = ipython.user_ns.get(key)
        except Exception:
            candidate = None
        if isinstance(candidate, str) and candidate.endswith(".ipynb"):
            p = Path(candidate).expanduser()
            if p.exists() and p.is_file():
                return p

    return None


def _load_ipynb_context(path: Path) -> _NotebookContext | None:
    try:
        import nbformat
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
        raw = list(getattr(ipython, "user_ns", {}).get("In", []))
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


def _runtime_fingerprint() -> str:
    cwd = str(Path.cwd().resolve())
    settings_path = Path.cwd() / ".swarmee" / "settings.json"
    try:
        settings_mtime = settings_path.stat().st_mtime if settings_path.exists() else None
    except Exception:
        settings_mtime = None

    env_keys = [
        "SWARMEE_STATE_DIR",
        "SWARMEE_LOG_DIR",
        "SWARMEE_JUPYTER_MODEL_PROVIDER",
        "SWARMEE_MODEL_PROVIDER",
        "SWARMEE_MODEL_TIER",
        "SWARMEE_MAX_TOKENS",
        "OPENAI_API_KEY",
        "SWARMEE_OPENAI_MODEL_ID",
        "STRANDS_MODEL_ID",
        "SWARMEE_OLLAMA_HOST",
        "SWARMEE_OLLAMA_MODEL_ID",
        "SWARMEE_KNOWLEDGE_BASE_ID",
        "STRANDS_KNOWLEDGE_BASE_ID",
    ]
    env = {k: (os.getenv(k) or "") for k in env_keys}
    return json.dumps({"cwd": cwd, "settings_mtime": settings_mtime, "env": env}, sort_keys=True)


def _build_hooks(settings: Any) -> list[Any]:
    if not _HAS_STRANDS_HOOKS:
        return []

    hooks: list[Any] = [
        JSONLLoggerHooks(),
        ToolPolicyHooks(settings.safety),
        ToolConsentHooks(
            settings.safety,
            interactive=False,
            auto_approve=False,
            prompt=lambda _text: "",
        ),
        ToolResultLimiterHooks(),
    ]
    if ToolMessageRepairHooks is not None:
        hooks.insert(2, ToolMessageRepairHooks())
    return hooks


def _create_agent(*, model: Any, tools: list[Any], system_prompt: str, hooks: list[Any]) -> Agent:
    kwargs: dict[str, Any] = {
        "model": model,
        "tools": tools,
        "system_prompt": system_prompt,
        "messages": [],
        "callback_handler": None,  # notebook-friendly: no spinners/streaming callbacks by default
        "load_tools_from_directory": True,
        "hooks": hooks,
    }

    try:
        return Agent(**kwargs)
    except TypeError:
        kwargs.pop("hooks", None)
        return Agent(**kwargs)


def _run_coroutine(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    out: dict[str, Any] = {}
    err: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            out["result"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001
            err["exc"] = e

    t = threading.Thread(target=_worker, daemon=True, name="swarmee-notebook-invoke")
    t.start()
    t.join()
    if "exc" in err:
        raise err["exc"]
    return out.get("result")


def _invoke_agent(
    runtime: _NotebookRuntime,
    query: str,
    *,
    invocation_state: dict[str, Any] | None = None,
    structured_output_model: type[Any] | None = None,
    structured_output_prompt: str | None = None,
) -> Any:
    resolved_state: dict[str, Any] = dict(invocation_state) if isinstance(invocation_state, dict) else {}
    sw_state = resolved_state.setdefault("swarmee", {})
    if isinstance(sw_state, dict):
        sw_state.setdefault("runtime_environment", dict(runtime.runtime_environment))
        sw_state["tier"] = runtime.model_manager.current_tier
        profile = runtime.settings.harness.tier_profiles.get(runtime.model_manager.current_tier)
        if profile is not None:
            sw_state["tool_profile"] = profile.to_dict()
        if sw_state.get("mode") == "plan" and structured_output_model is not None:
            model_tool_name = getattr(structured_output_model, "__name__", "").strip()
            if model_tool_name:
                existing = sw_state.get("plan_allowed_tools")
                merged: set[str] = {model_tool_name}
                if isinstance(existing, (list, tuple, set)):
                    merged.update(str(item).strip() for item in existing if str(item).strip())
                sw_state["plan_allowed_tools"] = sorted(merged)

    invoke_kwargs: dict[str, Any] = {"invocation_state": resolved_state}
    invoke_query = query
    if structured_output_model is not None:
        invoke_kwargs["structured_output_model"] = structured_output_model

    supports_structured_output_prompt = True
    try:
        import inspect

        invoke_sig = inspect.signature(runtime.agent.invoke_async)
        params = invoke_sig.parameters
        supports_structured_output_prompt = "structured_output_prompt" in params or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    except (TypeError, ValueError):
        supports_structured_output_prompt = True

    if structured_output_prompt is not None and supports_structured_output_prompt:
        invoke_kwargs["structured_output_prompt"] = structured_output_prompt
    elif structured_output_prompt is not None:
        prompt_text = structured_output_prompt.strip()
        if prompt_text:
            invoke_query = f"{prompt_text}\n\nUser request:\n{query}"

    return _run_coroutine(runtime.agent.invoke_async(invoke_query, **invoke_kwargs))


def _render_plan(plan: WorkPlan) -> str:
    lines: list[str] = ["Proposed plan:", f"- Summary: {plan.summary}"]
    if plan.assumptions:
        lines.append("- Assumptions:")
        lines.extend([f"  - {a}" for a in plan.assumptions])
    if plan.questions:
        lines.append("- Questions:")
        lines.extend([f"  - {q}" for q in plan.questions])
    if plan.steps:
        lines.append("- Steps:")
        for i, step in enumerate(plan.steps, start=1):
            lines.append(f"  {i}. {step.description}")
            if step.files_to_read:
                lines.append(f"     - read: {', '.join(step.files_to_read)}")
            if step.files_to_edit:
                lines.append(f"     - edit: {', '.join(step.files_to_edit)}")
            if step.tools_expected:
                lines.append(f"     - tools: {', '.join(step.tools_expected)}")
            if step.commands_expected:
                lines.append(f"     - cmds: {', '.join(step.commands_expected)}")
            if step.risks:
                lines.append(f"     - risks: {', '.join(step.risks)}")
    return "\n".join(lines).strip()


def _get_or_create_runtime() -> _NotebookRuntime:
    global _RUNTIME_SINGLETON, _RUNTIME_FINGERPRINT

    fingerprint = _runtime_fingerprint()
    if _RUNTIME_SINGLETON is not None and _RUNTIME_FINGERPRINT == fingerprint:
        return _RUNTIME_SINGLETON

    load_env_file()

    settings = load_settings()
    knowledge_base_id = (
        os.getenv("SWARMEE_KNOWLEDGE_BASE_ID")
        or os.getenv("STRANDS_KNOWLEDGE_BASE_ID")
        or os.getenv("SWARMEE_NOTEBOOK_KNOWLEDGE_BASE_ID")
    )

    selected_provider, _notice = resolve_model_provider(
        cli_provider=None,
        env_provider=os.getenv("SWARMEE_JUPYTER_MODEL_PROVIDER") or os.getenv("SWARMEE_MODEL_PROVIDER"),
        settings_provider=settings.models.provider,
    )

    model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
    model = model_manager.build_model()

    tools_dict = get_tools()
    for name, tool_obj in load_enabled_pack_tools(settings).items():
        tools_dict.setdefault(name, tool_obj)
    tools = list(tools_dict.values())

    artifact_store = ArtifactStore()

    runtime_environment = detect_runtime_environment(cwd=Path.cwd())
    runtime_environment_prompt_section = render_runtime_environment_section(runtime_environment)

    pack_prompt_sections = enabled_system_prompts(settings)
    profile = settings.harness.tier_profiles.get(model_manager.current_tier)
    snapshot = build_context_snapshot(
        artifact_store=artifact_store,
        interactive=False,
        default_preflight_level=profile.preflight_level if profile else None,
    )

    system_prompt = load_system_prompt()
    parts: list[str] = [system_prompt, _TOOL_USAGE_RULES, runtime_environment_prompt_section]
    if pack_prompt_sections:
        parts.extend(pack_prompt_sections)
    if snapshot.project_map_prompt_section:
        parts.append(snapshot.project_map_prompt_section)
    if snapshot.preflight_prompt_section:
        parts.append(snapshot.preflight_prompt_section)
    base_system_prompt = "\n\n".join([p for p in parts if p]).strip()

    hooks = _build_hooks(settings)
    agent = _create_agent(model=model, tools=tools, system_prompt=base_system_prompt, hooks=hooks)

    runtime = _NotebookRuntime(
        agent=agent,
        tools_dict=tools_dict,
        settings=settings,
        model_manager=model_manager,
        runtime_environment=runtime_environment,
        base_system_prompt=base_system_prompt,
        artifact_store=artifact_store,
        knowledge_base_id=knowledge_base_id,
    )
    _RUNTIME_SINGLETON = runtime
    _RUNTIME_FINGERPRINT = fingerprint
    return runtime


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


def _generate_plan(runtime: _NotebookRuntime, prompt: str, *, auto_approve: bool) -> WorkPlan:
    result = _invoke_agent(
        runtime,
        prompt,
        invocation_state={"swarmee": {"mode": "plan", "auto_approve": auto_approve}},
        structured_output_model=WorkPlan,
        structured_output_prompt=structured_plan_prompt(),
    )

    plan = getattr(result, "structured_output", None)
    if not isinstance(plan, WorkPlan):
        raise ValueError("Structured plan parse failed")

    runtime.artifact_store.write_text(
        kind="plan",
        text=plan.model_dump_json(indent=2),
        suffix="json",
        metadata={"request": prompt},
    )
    return plan


def _execute_with_plan(runtime: _NotebookRuntime, prompt: str, plan: WorkPlan, *, auto_approve: bool) -> Any:
    allowed_tools = sorted(tool_name for tool_name in tools_expected_from_plan(plan) if tool_name != "WorkPlan")
    invocation_state = {
        "swarmee": {
            "mode": "execute",
            "enforce_plan": True,
            "allowed_tools": allowed_tools,
            "auto_approve": auto_approve,
        }
    }

    plan_json_for_execution = json.dumps(plan.model_dump(exclude={"confirmation_prompt"}), indent=2, ensure_ascii=False)
    approved_plan_section = (
        "Approved Plan (execute ONLY this plan; if you need changes, regenerate the plan):\n" + plan_json_for_execution
    )

    prev_system_prompt = runtime.agent.system_prompt
    runtime.agent.system_prompt = "\n\n".join([runtime.base_system_prompt, approved_plan_section]).strip()
    try:
        return _invoke_agent(runtime, prompt, invocation_state=invocation_state)
    finally:
        runtime.agent.system_prompt = prev_system_prompt


def _run_swarmee(
    ipython: Any,
    user_prompt: str,
    *,
    include_context: bool,
    force_plan: bool,
    auto_approve: bool,
) -> str:
    runtime = _get_or_create_runtime()

    prompt = user_prompt.strip()
    if include_context:
        notebook_context = _collect_notebook_context(ipython)
        prompt = _format_prompt(notebook_context=notebook_context, user_prompt=prompt)

    intent = "work" if force_plan else classify_intent(user_prompt)
    if intent == "work":
        try:
            plan = _generate_plan(runtime, prompt, auto_approve=auto_approve)
        except MaxTokensReachedException:
            return (
                "Error: Plan generation hit the max output token limit.\n"
                "Fix: increase SWARMEE_MAX_TOKENS / STRANDS_MAX_TOKENS, or ask for a shorter plan."
            )

        rendered = _render_plan(plan)
        if not auto_approve:
            return rendered + "\n\nPlan generated. Re-run with `%%swarmee --yes` to execute."

        try:
            result = _execute_with_plan(runtime, prompt, plan, auto_approve=True)
        except MaxTokensReachedException:
            return (
                "Error: Execution hit the max output token limit.\n"
                "Fix: increase SWARMEE_MAX_TOKENS / STRANDS_MAX_TOKENS, or ask for a shorter response."
            )
        return str(result)

    try:
        result = _invoke_agent(
            runtime,
            prompt,
            invocation_state={"swarmee": {"mode": "execute", "auto_approve": auto_approve}},
        )
    except MaxTokensReachedException:
        return (
            "Error: Response hit the max output token limit.\n"
            "Fix: increase SWARMEE_MAX_TOKENS / STRANDS_MAX_TOKENS, or ask for a shorter response."
        )
    return str(result)


def _parse_magic_line(line: str) -> tuple[bool, bool, bool, str]:
    """
    Parse a `%%swarmee` magic line.

    Supported flags:
    - --yes: auto-approve plan + tool consent for this invocation
    - --plan: force plan mode even for "info" prompts
    - --no-context: do not inject notebook context

    Returns: (auto_approve, force_plan, no_context, extra_text)
    """
    tokens = shlex.split(line or "")
    auto_approve = False
    force_plan = False
    no_context = False
    extra: list[str] = []

    for tok in tokens:
        if tok == "--yes":
            auto_approve = True
            continue
        if tok == "--plan":
            force_plan = True
            continue
        if tok == "--no-context":
            no_context = True
            continue
        extra.append(tok)

    return auto_approve, force_plan, no_context, " ".join(extra).strip()


def load_ipython_extension(ipython: Any) -> None:
    try:
        from IPython.core.magic import Magics, cell_magic, magics_class
    except Exception as e:  # pragma: no cover
        raise RuntimeError("IPython is required to use the Swarmee notebook extension.") from e

    @magics_class
    class SwarmeeMagics(Magics):
        @cell_magic  # type: ignore[untyped-decorator]
        def swarmee(self, line: str, cell: str) -> str:
            # Allow disabling notebook context for quick one-offs.
            auto_approve, force_plan, no_context, extra = _parse_magic_line(line)
            include_context = (not no_context) and (not _truthy(os.getenv("SWARMEE_NOTEBOOK_NO_CONTEXT")))

            prompt = cell
            if extra:
                prompt = f"{extra}\n\n{cell}"

            text = _run_swarmee(
                self.shell,
                prompt,
                include_context=include_context,
                force_plan=force_plan,
                auto_approve=auto_approve,
            )

            # Print and also return (so users can assign it).
            print(text)
            return text

    ipython.register_magics(SwarmeeMagics)


def unload_ipython_extension(_ipython: Any) -> None:
    # No-op: IPython does not provide a stable public API for unregistering magics.
    return None
