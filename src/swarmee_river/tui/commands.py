"""Pure command classification helpers for the TUI."""

from __future__ import annotations

_MODEL_USAGE_TEXT = "Usage: /model show | /model list | /model provider <name> | /model tier <name> | /model reset"
_CONSENT_USAGE_TEXT = "Usage: /consent <y|n|a|v>"
_CONNECT_USAGE_TEXT = "Usage: /connect [github_copilot] | /connect aws [profile]"
_AUTH_USAGE_TEXT = "Usage: /auth list | /auth logout [provider]"
_SEARCH_USAGE_TEXT = "Usage: /search <term>"
_OPEN_USAGE_TEXT = "Usage: /open <number>"
_EXPAND_USAGE_TEXT = "Usage: /expand <tool_use_id>"
_COMPACT_USAGE_TEXT = "Usage: /compact"
_TEXT_USAGE_TEXT = "Usage: /text"
_THINKING_USAGE_TEXT = "Usage: /thinking"
_CONTEXT_USAGE_TEXT = (
    "Usage: /context add file <path> | /context add note <text> | /context add sop <name> | "
    "/context add kb <id> | /context remove <index> | /context list | /context clear"
)
_SOP_USAGE_TEXT = "Usage: /sop list | /sop activate <name> | /sop deactivate <name> | /sop preview <name>"

_COPY_COMMAND_MAP: dict[str, str] = {
    "/copy": "transcript",
    ":copy": "transcript",
    "/copy plan": "plan",
    ":copy plan": "plan",
    "/copy issues": "issues",
    ":copy issues": "issues",
    "/copy artifacts": "artifacts",
    ":copy artifacts": "artifacts",
    "/copy last": "last",
    ":copy last": "last",
    "/copy all": "all",
    ":copy all": "all",
}


def classify_copy_command(normalized: str) -> str | None:
    """Classify copy command variants into action keys."""
    return _COPY_COMMAND_MAP.get(normalized)


def classify_model_command(normalized: str) -> tuple[str, str | None] | None:
    """Classify /model commands into action + optional argument."""
    if normalized == "/model":
        return "help", None
    if normalized == "/model show":
        return "show", None
    if normalized == "/model list":
        return "list", None
    if normalized == "/model reset":
        return "reset", None
    if normalized.startswith("/model provider "):
        return "provider", normalized.split(maxsplit=2)[2].strip()
    if normalized.startswith("/model tier "):
        return "tier", normalized.split(maxsplit=2)[2].strip()
    return None


def classify_pre_run_command(text: str) -> tuple[str, str | None] | None:
    """Classify commands handled before active-run gating."""
    normalized = text.lower()
    if normalized == "/help":
        return "help", None
    if normalized == "/restore":
        return "restore", None
    if normalized == "/new":
        return "new", None
    if normalized == "/context":
        return "context_usage", None
    if normalized.startswith("/context "):
        return "context", text[len("/context ") :]
    if normalized == "/sop":
        return "sop_usage", None
    if normalized.startswith("/sop "):
        return "sop", text[len("/sop ") :]
    if normalized.startswith("/open "):
        return "open", text[len("/open ") :]
    if normalized == "/open":
        return "open_usage", None
    if normalized.startswith("/expand "):
        return "expand", text[len("/expand ") :]
    if normalized == "/expand":
        return "expand_usage", None
    if normalized.startswith("/search "):
        return "search", text[len("/search ") :]
    if normalized == "/search":
        return "search_usage", None
    if normalized == "/text":
        return "text", None
    if normalized.startswith("/text "):
        return "text_usage", None
    if normalized == "/thinking":
        return "thinking", None
    if normalized.startswith("/thinking "):
        return "thinking_usage", None
    if normalized == "/compact":
        return "compact", None
    if normalized.startswith("/compact "):
        return "compact_usage", None
    if normalized in {"/stop", ":stop"}:
        return "stop", None
    if normalized in {"/exit", ":exit"}:
        return "exit", None
    if normalized in {"/daemon restart", "/restart-daemon"}:
        return "daemon_restart", None
    if normalized in {"/daemon stop", "/daemon shutdown", "/broker stop", "/broker shutdown"}:
        return "daemon_stop", None
    if normalized == "/consent":
        return "consent_usage", None
    if normalized.startswith("/consent "):
        return "consent", normalized.split(maxsplit=1)[1].strip()
    if normalized == "/connect":
        return "connect", "github_copilot"
    if normalized.startswith("/connect "):
        return "connect", normalized.split(maxsplit=1)[1].strip()
    if normalized == "/auth":
        return "auth_usage", None
    if normalized.startswith("/auth "):
        return "auth", text[len("/auth ") :].strip()
    model = classify_model_command(normalized)
    if model is not None:
        action, argument = model
        return f"model:{action}", argument
    return None


def classify_post_run_command(text: str) -> tuple[str, str | None] | None:
    """Classify commands handled after active-run gating."""
    normalized = text.lower()
    if normalized == "/approve":
        return "approve", None
    if normalized == "/replan":
        return "replan", None
    if normalized == "/clearplan":
        return "clearplan", None
    if normalized == "/plan":
        return "plan_mode", None
    if text.startswith("/plan "):
        return "plan_prompt", text[len("/plan ") :].strip()
    if normalized == "/run":
        return "run_mode", None
    if text.startswith("/run "):
        return "run_prompt", text[len("/run ") :].strip()
    return None


__all__ = [
    "_AUTH_USAGE_TEXT",
    "_COMPACT_USAGE_TEXT",
    "_CONNECT_USAGE_TEXT",
    "_CONSENT_USAGE_TEXT",
    "_CONTEXT_USAGE_TEXT",
    "_EXPAND_USAGE_TEXT",
    "_MODEL_USAGE_TEXT",
    "_OPEN_USAGE_TEXT",
    "_SEARCH_USAGE_TEXT",
    "_SOP_USAGE_TEXT",
    "_TEXT_USAGE_TEXT",
    "_THINKING_USAGE_TEXT",
    "classify_copy_command",
    "classify_model_command",
    "classify_post_run_command",
    "classify_pre_run_command",
]
