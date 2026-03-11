from __future__ import annotations

import asyncio
import json
import re
import shlex
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from strands import Agent

from swarmee_river.artifacts import ArtifactStore, tools_expected_from_plan
from swarmee_river.context.prompt_cache import format_system_reminder, inject_system_reminder
from swarmee_river.packs import enabled_system_prompts, load_enabled_pack_tools
from swarmee_river.planning import WorkPlan, classify_intent, structured_plan_prompt
from swarmee_river.project_map import build_context_snapshot
from swarmee_river.runtime_env import detect_runtime_environment, render_runtime_environment_section
from swarmee_river.runtime_service.client import (
    RuntimeServiceClient,
    default_session_id_for_cwd,
    ensure_runtime_broker,
    runtime_hello_supports_capability,
    shutdown_runtime_broker,
)
from swarmee_river.session.models import SessionModelManager
from swarmee_river.settings import load_settings
from swarmee_river.tools import get_tools
from swarmee_river.utils.agent_runtime_utils import plan_json_for_execution, render_plan_text
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
_RUNTIME_BRANCH_SESSION_ID: str | None = None
_RUNTIME_BRANCH_CWD: str | None = None
_LEGACY_BRANCH_WARNING_CWDS: set[str] = set()
_HTTP_REQUEST_TOOL_WARNING = (
    'Field name "json" in "Http_requestTool" shadows an attribute in parent "BaseModel"'
)


def _install_notebook_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=_HTTP_REQUEST_TOOL_WARNING,
        category=UserWarning,
        module=r"pydantic\.main",
    )


_install_notebook_warning_filters()

_TOOL_USAGE_RULES = (
    "Tool usage rules:\n"
    "- Use list/glob/file_list/file_search/file_read for repository exploration and file reading.\n"
    "- Do not use shell for ls/find/sed/cat/grep/rg when file tools can do it.\n"
    "- Reserve shell for real command execution tasks."
)
_RUNTIME_TEXT_DELTA_EVENTS = {"text_delta", "message_delta", "output_text_delta", "delta"}
_RUNTIME_TEXT_COMPLETE_EVENTS = {"text_complete", "message_complete", "output_text_complete", "complete"}
_NOTEBOOK_READ_ONLY_TOOLS = ("list", "glob", "file_list", "file_search", "file_read")


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
    return json.dumps({"cwd": cwd, "settings_mtime": settings_mtime}, sort_keys=True)


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


def _extract_runtime_text_chunk(event: dict[str, Any]) -> str:
    for key in ("data", "text", "delta", "content", "output_text", "outputText", "textDelta"):
        value = event.get(key)
        if isinstance(value, str):
            return value
    return ""


def _should_use_runtime_broker() -> bool:
    # Notebook mode defaults to using the runtime broker so state can be shared
    # with the CLI/TUI. This is internal wiring behavior, not end-user configuration.
    from swarmee_river.config.env_policy import getenv_internal

    raw = getenv_internal("SWARMEE_NOTEBOOK_USE_RUNTIME")
    if raw is None:
        return True
    return raw.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _runtime_notebook_allowlist() -> list[str]:
    settings = load_settings()
    allow = list(_NOTEBOOK_READ_ONLY_TOOLS)
    if bool(getattr(settings.runtime, "enable_project_context_tool", False)):
        allow.append("project_context")
    return allow


def _warn_legacy_branching_once(*, attach_cwd: Path) -> None:
    cwd_text = str(attach_cwd.resolve())
    if cwd_text in _LEGACY_BRANCH_WARNING_CWDS:
        return
    _LEGACY_BRANCH_WARNING_CWDS.add(cwd_text)
    warnings.warn(
        "Connected runtime broker does not support cache-aware surface branching; "
        "using legacy session attach. Restart the broker to enable shared-prefix reuse.",
        RuntimeWarning,
        stacklevel=2,
    )


def _attach_notebook_runtime_session(
    client: RuntimeServiceClient,
    *,
    attach_cwd: Path,
    hello_event: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None]:
    global _RUNTIME_BRANCH_CWD, _RUNTIME_BRANCH_SESSION_ID

    from swarmee_river.config.env_policy import getenv_internal

    explicit_session_id = str(getenv_internal("SWARMEE_SESSION_ID") or "").strip() or None
    session_id = explicit_session_id
    fork_event: dict[str, Any] | None = None
    supports_surface_fork = runtime_hello_supports_capability(hello_event, "fork_surface_session")

    cwd_str = str(attach_cwd)
    if explicit_session_id is None:
        if _RUNTIME_BRANCH_CWD != cwd_str:
            _RUNTIME_BRANCH_CWD = cwd_str
            _RUNTIME_BRANCH_SESSION_ID = None
        if supports_surface_fork:
            fork_event = client.fork_surface_session(
                cwd=cwd_str,
                surface="jupyter",
                session_id=_RUNTIME_BRANCH_SESSION_ID,
            )
            if fork_event is None:
                raise RuntimeError("runtime broker closed connection during surface fork")
            if str(fork_event.get("event", "")).strip().lower() == "error":
                code = str(fork_event.get("code", "")).strip().lower()
                if code == "no_active_parent_session":
                    _RUNTIME_BRANCH_SESSION_ID = None
                    session_id = default_session_id_for_cwd(attach_cwd)
                elif code == "unknown_cmd":
                    _RUNTIME_BRANCH_SESSION_ID = None
                    session_id = default_session_id_for_cwd(attach_cwd)
                    _warn_legacy_branching_once(attach_cwd=attach_cwd)
                else:
                    raise RuntimeError(
                        str(fork_event.get("message", fork_event)).strip() or "runtime surface fork failed"
                    )
            else:
                session_id = str(fork_event.get("session_id", "")).strip() or None
                if not session_id:
                    raise RuntimeError("runtime broker surface fork did not return a session_id")
                _RUNTIME_BRANCH_SESSION_ID = session_id
        else:
            if explicit_session_id is None:
                _RUNTIME_BRANCH_SESSION_ID = None
                session_id = default_session_id_for_cwd(attach_cwd)
                _warn_legacy_branching_once(attach_cwd=attach_cwd)

    if not session_id:
        session_id = default_session_id_for_cwd(attach_cwd)

    attach = client.attach(session_id=session_id, cwd=cwd_str)
    if attach is None:
        raise RuntimeError("runtime broker closed connection during attach")
    if str(attach.get("event", "")).strip().lower() == "error":
        raise RuntimeError(str(attach.get("message", attach)).strip() or "runtime attach failed")

    if explicit_session_id is None:
        allowlist = _runtime_notebook_allowlist()
        client.send_command(
            {
                "cmd": "set_safety_overrides",
                "overrides": {
                    "tool_consent": "deny",
                    "tool_allowlist": allowlist,
                    "tool_blocklist": [],
                },
            }
        )
    return session_id, fork_event


def _run_swarmee_via_runtime(
    prompt: str,
    *,
    force_plan: bool,
    auto_approve: bool,
) -> str:
    attach_cwd = Path.cwd().resolve()
    discovery = ensure_runtime_broker(cwd=attach_cwd)
    client = RuntimeServiceClient.from_discovery_file(discovery)
    client.connect()

    try:
        hello = client.hello(client_name="swarmee-notebook", surface="jupyter")
        if hello is None:
            raise RuntimeError("runtime broker closed connection during hello")
        if str(hello.get("event", "")).strip().lower() == "error":
            raise RuntimeError(str(hello.get("message", hello)).strip() or "runtime hello failed")

        _session_id, _fork_event = _attach_notebook_runtime_session(
            client,
            attach_cwd=attach_cwd,
            hello_event=hello,
        )

        query_payload: dict[str, Any] = {"cmd": "query", "text": prompt}
        if force_plan:
            query_payload["mode"] = "plan"
        query_payload["auto_approve"] = bool(auto_approve)
        client.send_command(query_payload)

        delta_chunks: list[str] = []
        final_chunks: list[str] = []
        plan_rendered = ""
        errors: list[str] = []
        turn_complete_seen = False

        while True:
            event = client.read_event()
            if event is None:
                break
            etype = str(event.get("event", "")).strip().lower()

            if etype in _RUNTIME_TEXT_DELTA_EVENTS:
                chunk = _extract_runtime_text_chunk(event)
                if chunk:
                    delta_chunks.append(chunk)
                continue

            if etype in _RUNTIME_TEXT_COMPLETE_EVENTS:
                chunk = _extract_runtime_text_chunk(event)
                if chunk:
                    final_chunks.append(chunk)
                continue

            if etype == "replay_turn":
                role = str(event.get("role", "")).strip().lower()
                if role == "assistant":
                    replay_text = str(event.get("text", "")).strip()
                    if replay_text:
                        final_chunks.append(replay_text)
                continue

            if etype == "plan":
                rendered = event.get("rendered")
                if isinstance(rendered, str) and rendered.strip():
                    plan_rendered = rendered.strip()
                continue

            if etype == "consent_prompt":
                errors.append(
                    "Notebook broker sessions are read-only and cannot answer interactive consent prompts. "
                    "Use the TUI/CLI for mutating tools, or rerun with a read-only approach."
                )
                continue

            if etype == "error":
                error_text = str(event.get("message", event.get("text", ""))).strip()
                if error_text:
                    errors.append(error_text)
                continue

            if etype == "turn_complete":
                turn_complete_seen = True
                status = str(event.get("exit_status", "ok")).strip().lower()
                if status not in {"ok", "interrupted"} and not errors:
                    errors.append(f"runtime turn finished with status={status}")
                break

        delta_text = "".join(delta_chunks).strip()
        final_text = "\n".join(
            chunk.strip() for chunk in final_chunks if isinstance(chunk, str) and chunk.strip()
        ).strip()

        output_parts: list[str] = []
        if delta_text:
            output_parts.append(delta_text)
        if final_text and (not delta_text or final_text not in delta_text):
            output_parts.append(final_text)
        if plan_rendered and not output_parts:
            output_parts.append(plan_rendered)
        if errors and not output_parts:
            output_parts.append("\n".join(errors))
        if errors and output_parts:
            output_parts.append("\n".join(errors))
        if not output_parts and not turn_complete_seen:
            raise RuntimeError("runtime broker connection closed before turn_complete")

        return "\n\n".join(part for part in output_parts if part).strip()
    finally:
        client.close()


def _shutdown_runtime_from_notebook() -> str:
    stopped = shutdown_runtime_broker(cwd=Path.cwd().resolve())
    if stopped:
        return "Runtime daemon stopped."
    return "Runtime daemon is not running."


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


def _get_or_create_runtime() -> _NotebookRuntime:
    global _RUNTIME_SINGLETON, _RUNTIME_FINGERPRINT

    fingerprint = _runtime_fingerprint()
    if _RUNTIME_SINGLETON is not None and _RUNTIME_FINGERPRINT == fingerprint:
        return _RUNTIME_SINGLETON

    load_env_file()

    settings = load_settings()
    knowledge_base_id = settings.runtime.knowledge_base_id

    selected_provider, _notice = resolve_model_provider(
        cli_provider=None,
        env_provider=None,
        settings_provider=settings.models.provider,
    )

    model_manager = SessionModelManager(settings, fallback_provider=selected_provider)
    model = model_manager.build_model()

    tools_dict = get_tools()
    for name, tool_obj in load_enabled_pack_tools(settings).items():
        tools_dict.setdefault(name, tool_obj)
    tools = [tools_dict[name] for name in sorted(tools_dict)]

    artifact_store = ArtifactStore()

    runtime_environment = detect_runtime_environment(cwd=Path.cwd())
    runtime_environment_prompt_section = render_runtime_environment_section(runtime_environment)

    pack_prompt_sections = enabled_system_prompts(settings)
    profile = settings.harness.tier_profiles.get(model_manager.current_tier)
    snapshot = build_context_snapshot(
        artifact_store=artifact_store,
        interactive=False,
        runtime=settings.runtime,
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
    context_limit = 120000
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

    plan_json_payload = plan_json_for_execution(plan)
    approved_plan_section = (
        "Approved Plan (execute ONLY this plan; if you need changes, regenerate the plan):\n" + plan_json_payload
    )
    reminder = format_system_reminder([approved_plan_section])
    query = inject_system_reminder(user_query=prompt, reminder=reminder)
    return _invoke_agent(runtime, query, invocation_state=invocation_state)


def _run_swarmee(
    ipython: Any,
    user_prompt: str,
    *,
    include_context: bool,
    force_plan: bool,
    auto_approve: bool,
) -> str:
    prompt = user_prompt.strip()
    if include_context:
        notebook_context = _collect_notebook_context(ipython)
        prompt = _format_prompt(notebook_context=notebook_context, user_prompt=prompt)

    if _should_use_runtime_broker():
        try:
            return _run_swarmee_via_runtime(
                prompt,
                force_plan=force_plan,
                auto_approve=auto_approve,
            )
        except Exception as exc:
            return f"Error: runtime broker invocation failed: {exc}"

    runtime = _get_or_create_runtime()

    intent = "work" if force_plan else classify_intent(user_prompt)
    if intent == "work":
        try:
            plan = _generate_plan(runtime, prompt, auto_approve=auto_approve)
        except MaxTokensReachedException:
            return (
                "Error: Plan generation hit the max output token limit.\n"
                "Fix: increase `models.max_output_tokens` in `.swarmee/settings.json`, or ask for a shorter plan."
            )

        rendered = render_plan_text(plan)
        if not auto_approve:
            return rendered + "\n\nPlan generated. Re-run with `%%swarmee --yes` to execute."

        try:
            result = _execute_with_plan(runtime, prompt, plan, auto_approve=True)
        except MaxTokensReachedException:
            return (
                "Error: Execution hit the max output token limit.\n"
                "Fix: increase `models.max_output_tokens` in `.swarmee/settings.json`, or ask for a shorter response."
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
            "Fix: increase `models.max_output_tokens` in `.swarmee/settings.json`, or ask for a shorter response."
        )
    return str(result)


def _parse_magic_line(line: str) -> tuple[bool, bool, bool | None, bool, str]:
    """
    Parse a `%%swarmee` magic line.

    Supported flags:
    - --yes: auto-approve plan + tool consent for this invocation
    - --plan: force plan mode even for "info" prompts
    - --with-context: inject notebook context even when using the runtime broker
    - --no-context: do not inject notebook context
    - --daemon-stop: stop the shared runtime daemon for this scope

    Returns: (auto_approve, force_plan, include_context_override, daemon_stop, extra_text)
    """
    tokens = shlex.split(line or "")
    auto_approve = False
    force_plan = False
    include_context_override: bool | None = None
    daemon_stop = False
    extra: list[str] = []

    for tok in tokens:
        if tok == "--yes":
            auto_approve = True
            continue
        if tok == "--plan":
            force_plan = True
            continue
        if tok == "--with-context":
            include_context_override = True
            continue
        if tok == "--no-context":
            include_context_override = False
            continue
        if tok == "--daemon-stop":
            daemon_stop = True
            continue
        extra.append(tok)

    return auto_approve, force_plan, include_context_override, daemon_stop, " ".join(extra).strip()


def load_ipython_extension(ipython: Any) -> None:
    try:
        from IPython.core.magic import Magics, cell_magic, magics_class
    except Exception as e:  # pragma: no cover
        raise RuntimeError("IPython is required to use the Swarmee notebook extension.") from e

    @magics_class
    class SwarmeeMagics(Magics):
        @cell_magic  # type: ignore[untyped-decorator]
        def swarmee(self, line: str, cell: str) -> str:
            auto_approve, force_plan, include_context_override, daemon_stop, extra = _parse_magic_line(line)
            if daemon_stop:
                text = _shutdown_runtime_from_notebook()
                print(text)
                return text
            include_context = (
                include_context_override
                if include_context_override is not None
                else (not _should_use_runtime_broker())
            )

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
