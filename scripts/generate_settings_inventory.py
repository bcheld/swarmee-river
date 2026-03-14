#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
TESTS_DIR = ROOT / "tests"
ENV_EXAMPLE_PATH = ROOT / "env.example"
SETTINGS_JSON_PATH = ROOT / "docs" / "configuration" / "settings_inventory.json"
SETTINGS_MD_PATH = ROOT / "docs" / "configuration" / "settings_inventory.md"

ENV_NAME_RE = r"[A-Z][A-Z0-9_]+"

ENV_READ_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf'os\.getenv\(\s*"(?P<key>{ENV_NAME_RE})"'),
    re.compile(rf'os\.environ\.get\(\s*"(?P<key>{ENV_NAME_RE})"'),
    re.compile(rf'os\.environ\[\s*"(?P<key>{ENV_NAME_RE})"\s*\]'),
    re.compile(rf'\b(?:truthy_env|csv_env|_truthy_env|_env_float|_parse_int_env|_env_int)\(\s*"(?P<key>{ENV_NAME_RE})"'),
    re.compile(rf'\bgetenv_secret\(\s*"(?P<key>{ENV_NAME_RE})"'),
    re.compile(rf'\bgetenv_internal\(\s*"(?P<key>{ENV_NAME_RE})"'),
]
ENV_WRITE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf'os\.environ\[\s*"(?P<key>{ENV_NAME_RE})"\s*\]\s*='),
    re.compile(rf'\benv\[\s*"(?P<key>{ENV_NAME_RE})"\s*\]\s*='),
]

RUNTIME_INTERNAL_ENV_VARS: set[str] = set()
RUNTIME_SECRET_ENV_VARS: set[str] = set()
RUNTIME_INTERNAL_SETTINGS_ENV_OVERRIDE_VARS: set[str] = set()

ALIASES_BY_CANONICAL: dict[str, list[str]] = {
    "SWARMEE_GITHUB_COPILOT_API_KEY": ["GITHUB_TOKEN", "GH_TOKEN"],
}

CLI_FLAG_ENV_HINTS: dict[str, list[str]] = {}

HELPER_TUI_ENV_CONTROLS: dict[str, set[str]] = {}

GENERIC_TUI_SETTING_IDS: dict[str, str] = {
    "settings_env_apply": "Advanced env editor (apply selected env var)",
    "settings_env_default": "Advanced env editor (apply default for selected env var)",
    "settings_env_unset": "Advanced env editor (unset selected env var)",
}

PLATFORM_CONVENTION_ENV_VARS: set[str] = {
    "APPDATA",
    "AWS_PROFILE",
    "CI",
    "COMSPEC",
    "EDITOR",
    "SHELL",
    "TERM",
    "TMUX",
    "XDG_DATA_HOME",
}


@dataclass(frozen=True)
class SettingInventoryEntry:
    id: str
    kind: str
    surface: str
    where_set: list[str]
    precedence: list[str]
    where_read: list[str]
    controls: str
    controls_plain: str
    status: str
    dup_group: str | None
    keep_configurable_justification: str
    notes: str
    usage_count: int
    read_count: int | None = None
    write_count: int | None = None
    classification_reason: str | None = None
    documented: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "surface": self.surface,
            "where_set": self.where_set,
            "precedence": self.precedence,
            "where_read": self.where_read,
            "controls": self.controls,
            "controls_plain": self.controls_plain,
            "status": self.status,
            "dup_group": self.dup_group,
            "keep_configurable_justification": self.keep_configurable_justification,
            "notes": self.notes,
            "usage_count": self.usage_count,
        }
        if self.read_count is not None:
            payload["read_count"] = self.read_count
        if self.write_count is not None:
            payload["write_count"] = self.write_count
        if self.classification_reason is not None:
            payload["classification_reason"] = self.classification_reason
        if self.documented is not None:
            payload["documented"] = self.documented
        return payload


def _all_python_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.py") if path.is_file())


def _line_number_for_offset(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _collect_env_from_example(path: Path) -> tuple[set[str], dict[str, str]]:
    key_set: set[str] = set()
    descriptions: dict[str, str] = {}
    description_buffer: list[str] = []
    for raw_line in _read_text(path).splitlines():
        stripped = raw_line.strip()
        if not stripped:
            description_buffer = []
            continue
        if not stripped.startswith("#"):
            continue
        body = stripped[1:].strip()
        match = re.match(rf"(?P<key>{ENV_NAME_RE})=(?P<value>.*)", body)
        if match:
            key = match.group("key")
            key_set.add(key)
            merged_desc = " ".join(part.strip() for part in description_buffer if part.strip()).strip()
            descriptions[key] = merged_desc
            description_buffer = []
            continue
        if body.startswith("###"):
            description_buffer = []
            continue
        if body:
            description_buffer.append(body)
    return key_set, descriptions


def _scan_env_accesses(root: Path) -> dict[str, dict[str, set[str]]]:
    out: dict[str, dict[str, set[str]]] = {}
    if not root.exists():
        return out
    for py_file in _all_python_files(root):
        text = _read_text(py_file)
        rel = _rel(py_file)
        for pattern in ENV_READ_PATTERNS:
            for match in pattern.finditer(text):
                key = match.group("key")
                line = _line_number_for_offset(text, match.start())
                out.setdefault(key, {"reads": set(), "writes": set()})
                out[key]["reads"].add(f"{rel}:{line}")
        for pattern in ENV_WRITE_PATTERNS:
            for match in pattern.finditer(text):
                key = match.group("key")
                line = _line_number_for_offset(text, match.start())
                out.setdefault(key, {"reads": set(), "writes": set()})
                out[key]["writes"].add(f"{rel}:{line}")
    # Heuristic: `os.environ["X"] = ...` matches both read and write patterns in regex
    # form; de-duplicate by treating any read-site that is also a write-site as write-only.
    for refs in out.values():
        refs["reads"] = set(refs.get("reads", set())) - set(refs.get("writes", set()))
    return out


def _merge_env_refs(
    left: dict[str, dict[str, set[str]]],
    right: dict[str, dict[str, set[str]]],
) -> dict[str, dict[str, set[str]]]:
    merged: dict[str, dict[str, set[str]]] = {}
    for source in (left, right):
        for key, refs in source.items():
            bucket = merged.setdefault(key, {"reads": set(), "writes": set()})
            bucket["reads"].update(refs.get("reads", set()))
            bucket["writes"].update(refs.get("writes", set()))
    return merged


def _add_env_read_ref(
    env_refs: dict[str, dict[str, set[str]]],
    *,
    key: str,
    rel_path: str,
    line: int,
) -> None:
    bucket = env_refs.setdefault(key, {"reads": set(), "writes": set()})
    bucket["reads"].add(f"{rel_path}:{line}")


def _line_for_pattern(text: str, pattern: str) -> int | None:
    match = re.search(pattern, text)
    if not match:
        return None
    return _line_number_for_offset(text, match.start())


def _scan_env_accesses_in_tests(root: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    patterns = [
        re.compile(rf'os\.getenv\(\s*"(?P<key>{ENV_NAME_RE})"'),
        re.compile(rf'os\.environ\[\s*"(?P<key>{ENV_NAME_RE})"\s*\]'),
        re.compile(rf'monkeypatch\.(?:setenv|delenv)\(\s*"(?P<key>{ENV_NAME_RE})"'),
    ]
    for py_file in _all_python_files(root):
        text = _read_text(py_file)
        rel = _rel(py_file)
        for pattern in patterns:
            for match in pattern.finditer(text):
                key = match.group("key")
                line = _line_number_for_offset(text, match.start())
                out.setdefault(key, set()).add(f"{rel}:{line}")
    return out


def _string_literal(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


class _ArgparseCollector(ast.NodeVisitor):
    def __init__(self, path: Path) -> None:
        self.path = path
        self.parser_context: dict[str, str] = {}
        self.subparsers_context: dict[str, str] = {}
        self.entries: list[dict[str, Any]] = []

    def visit_Assign(self, node: ast.Assign) -> Any:
        value = node.value
        target_names = [target.id for target in node.targets if isinstance(target, ast.Name)]
        if not target_names:
            return self.generic_visit(node)
        if isinstance(value, ast.Call):
            # parser = argparse.ArgumentParser(...)
            if isinstance(value.func, ast.Attribute):
                func_obj = value.func.value
                if isinstance(func_obj, ast.Name) and func_obj.id == "argparse" and value.func.attr == "ArgumentParser":
                    prog_value = None
                    for keyword in value.keywords:
                        if keyword.arg == "prog":
                            prog_value = _string_literal(keyword.value)
                    context = prog_value or "swarmee"
                    for name in target_names:
                        self.parser_context[name] = context
                # sub = parser.add_subparsers(...)
                if isinstance(func_obj, ast.Name) and value.func.attr == "add_subparsers":
                    source_context = self.parser_context.get(func_obj.id)
                    if source_context:
                        for name in target_names:
                            self.subparsers_context[name] = source_context
                # list_parser = sub.add_parser("list", ...)
                if isinstance(func_obj, ast.Name) and value.func.attr == "add_parser":
                    base_context = self.subparsers_context.get(func_obj.id)
                    if base_context and value.args:
                        command = _string_literal(value.args[0]) or "subcommand"
                        for name in target_names:
                            self.parser_context[name] = f"{base_context} {command}"
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Attribute) and node.func.attr == "add_argument":
            parser_name = node.func.value.id if isinstance(node.func.value, ast.Name) else None
            if parser_name and parser_name in self.parser_context:
                context = self.parser_context[parser_name]
                raw_flags = [_string_literal(arg) for arg in node.args]
                flags = [flag for flag in raw_flags if isinstance(flag, str) and flag.startswith("--")]
                if flags:
                    primary_flag = flags[0]
                    help_text = ""
                    for keyword in node.keywords:
                        if keyword.arg == "help":
                            help_text = _string_literal(keyword.value) or ""
                    self.entries.append(
                        {
                            "id": f"cli:{context} {primary_flag}",
                            "kind": "cli_arg",
                            "surface": "cli",
                            "flags": flags,
                            "context": context,
                            "help": help_text,
                            "where_read": [f"{_rel(self.path)}:{node.lineno}"],
                            "where_set": ["CLI flag"],
                            "precedence": [],
                            "controls": help_text or f"Optional CLI argument for {context}.",
                            "controls_plain": _sentence(
                                help_text,
                                f"Optional command-line flag for {context}.",
                            ),
                            "status": "active",
                            "dup_group": None,
                            "keep_configurable_justification": (
                                "CLI flags allow per-invocation overrides without mutating shared settings."
                            ),
                            "notes": "",
                            "usage_count": 1,
                        }
                    )
        return self.generic_visit(node)


def _collect_cli_args() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for path in [SRC_DIR / "swarmee_river" / "swarmee.py"]:
        tree = ast.parse(_read_text(path))
        collector = _ArgparseCollector(path)
        collector.visit(tree)
        entries.extend(collector.entries)

    diagnostics_file = SRC_DIR / "swarmee_river" / "cli" / "diagnostics.py"
    diagnostics_text = _read_text(diagnostics_file)
    for pattern, controls in [
        (r'if command == "diff":', "Diff supports optional --staged/--cached flags and optional paths."),
        (r'if "--lines" in sub:', "Tail commands accept optional --lines N."),
    ]:
        match = re.search(pattern, diagnostics_text)
        if not match:
            continue
        line = _line_number_for_offset(diagnostics_text, match.start())
        if "diff" in pattern:
            entries.append(
                {
                    "id": "cli:swarmee diff --staged",
                    "kind": "cli_arg",
                    "surface": "cli",
                    "flags": ["--staged", "--cached"],
                    "context": "swarmee diff",
                    "help": controls,
                    "where_read": [f"{_rel(diagnostics_file)}:{line}"],
                    "where_set": ["CLI flag"],
                    "precedence": [],
                    "controls": controls,
                    "controls_plain": _sentence(
                        "Choose whether diff runs against staged changes.",
                        "Optional diff mode selector.",
                    ),
                    "status": "active",
                    "dup_group": None,
                    "keep_configurable_justification": (
                        "Flag-level control avoids separate subcommands for staged vs unstaged diffs."
                    ),
                    "notes": "",
                    "usage_count": 1,
                }
            )
        else:
            entries.append(
                {
                    "id": "cli:swarmee diagnostics --lines",
                    "kind": "cli_arg",
                    "surface": "cli",
                    "flags": ["--lines"],
                    "context": "swarmee diagnostics",
                    "help": controls,
                    "where_read": [f"{_rel(diagnostics_file)}:{line}"],
                    "where_set": ["CLI flag"],
                    "precedence": [],
                    "controls": controls,
                    "controls_plain": _sentence(
                        "Sets how many lines diagnostics output should show.",
                        "Optional diagnostics line count.",
                    ),
                    "status": "active",
                    "dup_group": None,
                    "keep_configurable_justification": (
                        "Allows users to tune verbosity without changing persistent configuration."
                    ),
                    "notes": "",
                    "usage_count": 1,
                }
            )
    return entries


def _collect_notebook_args(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    entries: list[dict[str, Any]] = []
    for flag in ("--yes", "--plan", "--no-context", "--daemon-stop"):
        match = re.search(rf'tok == "{re.escape(flag)}"', text)
        if not match:
            continue
        line = _line_number_for_offset(text, match.start())
        controls = {
            "--yes": "Auto-approve plan and tool consent for this notebook invocation.",
            "--plan": "Force planning mode for this invocation.",
            "--no-context": "Skip notebook context injection.",
            "--daemon-stop": "Stop shared runtime daemon for current scope.",
        }[flag]
        entries.append(
            {
                "id": f"notebook:%%swarmee {flag}",
                "kind": "notebook_arg",
                "surface": "notebook",
                "where_set": ["Notebook magic flag"],
                "precedence": [
                    "Notebook magic flag",
                    "Notebook env defaults (when applicable)",
                    "Runtime defaults",
                ],
                "where_read": [f"{_rel(path)}:{line}"],
                "controls": controls,
                "controls_plain": _sentence(
                    controls,
                    "Notebook flag that changes behavior for this cell execution.",
                ),
                "status": "active",
                "dup_group": None,
                "keep_configurable_justification": (
                    "Notebook users need per-cell behavior control without changing project-level settings."
                ),
                "notes": "",
                "usage_count": 1,
            }
        )
    return entries


def _collect_tui_settings_controls(app_path: Path) -> list[dict[str, Any]]:
    text = _read_text(app_path)
    lines = text.splitlines()
    button_env_map: dict[str, set[str]] = {}
    select_env_map: dict[str, set[str]] = {}
    control_lines: dict[str, int] = {}

    current_kind: str | None = None
    current_control: str | None = None
    in_persist_call = False
    for idx, line in enumerate(lines, start=1):
        button_match = re.search(r'if button_id == "([^"]+)"', line)
        if button_match:
            current_kind = "button"
            current_control = button_match.group(1)
            control_lines[current_control] = idx
            in_persist_call = False
        select_match = re.search(r'if select_id == "([^"]+)"', line)
        if select_match:
            current_kind = "select"
            current_control = select_match.group(1)
            control_lines[current_control] = idx
            in_persist_call = False

        if current_kind is None or current_control is None:
            continue

        if "_persist_project_setting_env_override(" in line:
            in_persist_call = True

        if in_persist_call:
            for env_key in re.findall(r'"([A-Z][A-Z0-9_]+)"', line):
                if current_kind == "button":
                    button_env_map.setdefault(current_control, set()).add(env_key)
                else:
                    select_env_map.setdefault(current_control, set()).add(env_key)
            if ")" in line:
                in_persist_call = False

    for control_id, env_keys in HELPER_TUI_ENV_CONTROLS.items():
        button_env_map.setdefault(control_id, set()).update(env_keys)

    entries: list[dict[str, Any]] = []
    for control_id, env_keys in sorted(button_env_map.items()):
        line = control_lines.get(control_id)
        entries.append(
            {
                "id": f"tui_setting:{control_id}",
                "kind": "tui_setting",
                "surface": "tui",
                "where_set": ["TUI Settings control", ".swarmee/settings.json"],
                "precedence": [
                    "TUI control value",
                    ".swarmee/settings.json",
                    "built-in defaults",
                ],
                "where_read": [f"{_rel(app_path)}:{line}"] if line else [f"{_rel(app_path)}"],
                "controls": (
                    "Persists workspace settings derived from legacy key identifiers: "
                    + ", ".join(sorted(env_keys))
                    + ". (These identifiers are migrated to structured `.swarmee/settings.json` fields; "
                    "they are no longer applied as generic env overrides.)"
                ),
                "controls_plain": _sentence(
                    "Changes this setting in the TUI and saves it for this workspace.",
                    "TUI setting control.",
                ),
                "status": "active",
                "dup_group": None,
                "keep_configurable_justification": (
                    "Interactive users need in-app control over runtime behavior per workspace."
                ),
                "notes": "",
                "usage_count": len(env_keys),
            }
        )

    for control_id, env_keys in sorted(select_env_map.items()):
        line = control_lines.get(control_id)
        entries.append(
            {
                "id": f"tui_setting:{control_id}",
                "kind": "tui_setting",
                "surface": "tui",
                "where_set": ["TUI Settings select control", ".swarmee/settings.json"],
                "precedence": [
                    "TUI control value",
                    ".swarmee/settings.json",
                    "built-in defaults",
                ],
                "where_read": [f"{_rel(app_path)}:{line}"] if line else [f"{_rel(app_path)}"],
                "controls": (
                    "Persists workspace settings derived from legacy key identifiers: "
                    + ", ".join(sorted(env_keys))
                    + ". (These identifiers are migrated to structured `.swarmee/settings.json` fields; "
                    "they are no longer applied as generic env overrides.)"
                ),
                "controls_plain": _sentence(
                    "Selects a runtime option in the TUI and saves it for this workspace.",
                    "TUI select setting.",
                ),
                "status": "active",
                "dup_group": None,
                "keep_configurable_justification": (
                    "Select controls make high-frequency runtime strategy changes discoverable."
                ),
                "notes": "",
                "usage_count": len(env_keys),
            }
        )

    for control_id, controls_text in GENERIC_TUI_SETTING_IDS.items():
        line = control_lines.get(control_id)
        entries.append(
            {
                "id": f"tui_setting:{control_id}",
                "kind": "tui_setting",
                "surface": "tui",
                "where_set": ["TUI Advanced settings", "process environment (session-only)", ".swarmee/settings.json"],
                "precedence": [
                    "Advanced settings action",
                    ".swarmee/settings.json (structured settings and internal-only env overrides)",
                    "built-in defaults",
                ],
                "where_read": [f"{_rel(app_path)}:{line}"] if line else [f"{_rel(app_path)}"],
                "controls": (
                    controls_text
                    + ". This is migration-oriented: secrets are set for the current process only; "
                    "internal wiring keys may persist under the legacy `env` section (allowlisted); "
                    "non-secret legacy keys are migrated into structured settings fields."
                ),
                "controls_plain": _sentence(
                    (
                        "Advanced TUI control for applying secrets for this session and migrating "
                        "legacy keys into settings."
                    ),
                    "Advanced TUI environment editor control.",
                ),
                "status": "active",
                "dup_group": None,
                "keep_configurable_justification": (
                    "Advanced controls support credential setup and migration without requiring manual file edits."
                ),
                "notes": "",
                "usage_count": 1,
            }
        )
    return entries


def _collect_tui_command_entries(path: Path) -> list[dict[str, Any]]:
    text = _read_text(path)
    usage_entries: list[dict[str, Any]] = []
    for match in re.finditer(r'^_[A-Z_]+_USAGE_TEXT\s*=\s*"Usage:\s*(.+)"', text, flags=re.MULTILINE):
        usage = match.group(1)
        line = _line_number_for_offset(text, match.start())
        for option in [part.strip() for part in usage.split("|") if part.strip()]:
            usage_entries.append(
                {
                    "id": f"tui_command_arg:{option}",
                    "kind": "tui_command_arg",
                    "surface": "tui",
                    "where_set": ["TUI slash command input"],
                    "precedence": [
                        "Slash command argument",
                        "Current session defaults",
                        "Runtime defaults",
                    ],
                    "where_read": [f"{_rel(path)}:{line}"],
                    "controls": f"TUI slash command syntax: {option}",
                    "controls_plain": _sentence(
                        f"Uses `{option}` as an argument form for a TUI slash command.",
                        "TUI slash command argument.",
                    ),
                    "status": "active",
                    "dup_group": None,
                    "keep_configurable_justification": (
                        "Command arguments are required for fast keyboard-driven workflow control."
                    ),
                    "notes": "",
                    "usage_count": 1,
                }
            )
    return usage_entries


def _load_settings_json_paths() -> set[str]:
    # Keep this dependency-free (do not import package modules).
    return {
        # Context management
        "context.manager",
        "context.max_prompt_tokens",
        "context.window_size",
        "context.per_turn",
        "context.truncate_results",
        "context.preserve_recent_messages",
        "context.summary_ratio",
        "context.cache_safe_summary",
        # Runtime behavior
        "runtime.auto_approve",
        "runtime.freeze_tools",
        "runtime.swarm_enabled",
        "runtime.esc_interrupt_enabled",
        "runtime.limit_tool_results",
        "runtime.knowledge_base_id",
        "runtime.session_s3_bucket",
        "runtime.session_s3_prefix",
        "runtime.session_s3_auto_export",
        "runtime.session_kb_promote_on_complete",
        "runtime.tooling_s3_prefix",
        "runtime.enabled_tools[]",
        "runtime.disabled_tools[]",
        "runtime.enable_project_context_tool",
        "runtime.project_map_enabled",
        "runtime.preflight_enabled",
        "runtime.preflight_level",
        "runtime.preflight_max_chars",
        "runtime.preflight_print",
        "runtime.interrupt_timeout_sec",
        "runtime.state_dir",
        # Diagnostics/logging
        "diagnostics.level",
        "diagnostics.redact",
        "diagnostics.log_redact",
        "diagnostics.log_events",
        "diagnostics.log_dir",
        "diagnostics.log_s3_bucket",
        "diagnostics.log_s3_prefix",
        # Pricing overrides
        "pricing.default.input_per_1m",
        "pricing.default.output_per_1m",
        "pricing.default.cached_input_per_1m",
        "pricing.providers.<provider>.input_per_1m",
        "pricing.providers.<provider>.output_per_1m",
        "pricing.providers.<provider>.cached_input_per_1m",
        # Models/provider selection
        "models.provider",
        "models.default_tier",
        "models.default_selection.provider",
        "models.default_selection.tier",
        "models.max_output_tokens",
        "models.tiers.<tier>.provider",
        "models.tiers.<tier>.model_id",
        "models.tiers.<tier>.display_name",
        "models.tiers.<tier>.description",
        "models.tiers.<tier>.client_args",
        "models.tiers.<tier>.params",
        "models.providers.<provider>.display_name",
        "models.providers.<provider>.description",
        "models.providers.<provider>.tiers.<tier>.provider",
        "models.providers.<provider>.tiers.<tier>.model_id",
        "models.providers.<provider>.tiers.<tier>.display_name",
        "models.providers.<provider>.tiers.<tier>.description",
        "models.providers.<provider>.tiers.<tier>.client_args",
        "models.providers.<provider>.tiers.<tier>.params",
        "models.providers.<provider>.extra.<key>",
        "models.auto_escalation.enabled",
        "models.auto_escalation.max_escalations_per_task",
        "models.auto_escalation.triggers",
        "models.auto_escalation.order[]",
        "models.availability",
        "models.hidden_tiers[]",
        # Safety/consent
        "safety.tool_consent",
        "safety.tool_rules[].tool",
        "safety.tool_rules[].default",
        "safety.tool_rules[].remember",
        "safety.permission_rules[].tool",
        "safety.permission_rules[].action",
        "safety.permission_rules[].remember",
        "safety.permission_rules[].when",
        # Packs/harness
        "packs.installed[].type",
        "packs.installed[].name",
        "packs.installed[].path",
        "packs.installed[].enabled",
        "packs.installed[].id",
        "packs.installed[].provider",
        "packs.installed[].tier",
        "packs.installed[].system_prompt_snippets[]",
        "packs.installed[].context_sources[]",
        "packs.installed[].active_sops[]",
        "packs.installed[].knowledge_base_id",
        "packs.installed[].agents[]",
        "packs.installed[].auto_delegate_assistive",
        "packs.installed[].team_presets[]",
        "harness.tier_profiles.<tier>.tool_allowlist[]",
        "harness.tier_profiles.<tier>.tool_blocklist[]",
        "harness.tier_profiles.<tier>.preflight_level",
        # Migration-only legacy payloads
        "env.<ENV_VAR_NAME>",
        "raw",
    }


def _dedupe_group_for_key(key: str) -> tuple[str | None, bool]:
    for canonical, aliases in ALIASES_BY_CANONICAL.items():
        if key == canonical:
            return canonical, False
        if key in aliases:
            return canonical, True
    return None, False


def _derive_env_controls_text(key: str, env_descriptions: dict[str, str], refs: dict[str, set[str]]) -> str:
    desc = (env_descriptions.get(key) or "").strip()
    if desc:
        return desc
    read_refs = sorted(refs.get("reads", set()))
    if read_refs:
        return f"Controls runtime behavior at {read_refs[0]} and related callsites."
    return "Documented environment variable; no runtime reads detected in src/."


def _sentence(text: str, fallback: str) -> str:
    normalized = " ".join(str(text or "").strip().split())
    if not normalized:
        normalized = fallback
    if not normalized.endswith("."):
        normalized += "."
    return normalized


def _plain_for_env(
    *,
    key: str,
    status: str,
    dup_group: str | None,
    controls: str,
    env_descriptions: dict[str, str],
) -> str:
    description = " ".join((env_descriptions.get(key) or "").strip().split())
    if status == "legacy_alias":
        return _sentence(
            f"Legacy compatibility variable; use {dup_group} for new configuration.",
            "Legacy compatibility variable kept for migration.",
        )
    if status == "internal_only":
        return _sentence(
            "Internal runtime wiring variable that is usually set by Swarmee, not by users.",
            "Internal runtime wiring variable.",
        )
    if status == "doc_only":
        return _sentence(
            "Documented setting reserved for compatibility, but runtime code under src/ does not currently read it.",
            "Documented compatibility setting.",
        )
    if key in PLATFORM_CONVENTION_ENV_VARS:
        return _sentence(
            "Platform or shell convention variable that Swarmee reads when available.",
            "Platform convention variable used by runtime.",
        )
    if key.startswith("SWARMEE_PRICE_"):
        return _sentence(
            "Overrides token pricing rates used to estimate cost in usage events.",
            "Overrides pricing used for cost estimation.",
        )
    if key.endswith("_API_KEY") or "TOKEN" in key or "SECRET" in key:
        return _sentence(
            "Supplies credentials for authenticating with an external provider.",
            "Credential variable used for authentication.",
        )
    if key.endswith("_MODEL_ID") or "MODEL_PROVIDER" in key or "MODEL_TIER" in key:
        return _sentence(
            "Selects which model/provider/tier Swarmee uses by default.",
            "Model selection variable.",
        )
    if any(token in key for token in ["TIMEOUT", "RETRY", "MAX_", "BUDGET", "LIMIT"]):
        return _sentence(
            "Tunes runtime limits and reliability behavior for this environment.",
            "Runtime tuning variable.",
        )
    if any(token in key for token in ["MODE", "PROFILE", "CONTEXT", "EVENTS", "LOG", "DEBUG"]):
        return _sentence(
            "Controls runtime mode and diagnostics behavior for this environment.",
            "Runtime mode/diagnostics setting.",
        )
    if "STATE_DIR" in key:
        return _sentence(
            "Sets where Swarmee stores workspace state and session data.",
            "State directory setting.",
        )
    if description and not description.startswith("----"):
        return _sentence(description, "Environment variable that controls runtime behavior.")
    return _sentence(controls, "Environment variable controlling runtime behavior.")


def _classification_reason_for_env(
    *,
    key: str,
    status: str,
    documented: bool,
    read_count: int,
    write_count: int,
    dup_group: str | None,
) -> str:
    if status == "legacy_alias":
        return _sentence(
            f"Alias-chain compatibility key mapped to canonical `{dup_group or key}`.",
            "Alias compatibility key.",
        )
    if status == "internal_only" and key in RUNTIME_INTERNAL_ENV_VARS:
        return _sentence(
            "Marked internal by runtime-internal catalog (`RUNTIME_INTERNAL_ENV_VARS`).",
            "Marked internal by runtime catalog.",
        )
    if status == "internal_only" and read_count == 0 and write_count > 0:
        return _sentence(
            "Write-only in runtime code under src/ and used as internal process wiring.",
            "Write-only runtime wiring variable.",
        )
    if status == "undocumented":
        return _sentence(
            f"Read in runtime code under src/ ({read_count} read sites) but missing from env.example.",
            "Runtime-read variable missing from env.example.",
        )
    if status == "doc_only":
        return _sentence(
            "Documented in env.example but has no runtime reads under src/.",
            "Documented variable with no runtime reads.",
        )
    if status == "active":
        return _sentence(
            (
                "Documented and runtime-read."
                if documented
                else "Runtime-read in src/ and intentionally undocumented."
            ),
            "Active runtime variable.",
        )
    return _sentence(
        "Status derived by deterministic inventory classification rules.",
        "Classified by inventory rules.",
    )


def _env_where_set(
    key: str,
    documented: bool,
    internal_only: bool,
    tui_control_map: dict[str, set[str]],
) -> list[str]:
    _ = tui_control_map
    out: list[str] = []
    if key in RUNTIME_SECRET_ENV_VARS:
        out.extend(["process environment", ".env file"])
    if internal_only:
        out.append("code-only/runtime wiring")
        if key in RUNTIME_INTERNAL_SETTINGS_ENV_OVERRIDE_VARS:
            out.append(".swarmee/settings.json (legacy env overrides, internal-only)")
    if documented and key not in RUNTIME_SECRET_ENV_VARS:
        # Secrets-only policy: documented non-secret env vars should be treated as legacy.
        out.append("legacy documentation (see migration)")
    return out


def _env_precedence(key: str, internal_only: bool) -> list[str]:
    if internal_only:
        return [
            "launcher/runtime broker wiring env",
            "process environment",
            "built-in defaults",
        ]
    if key in RUNTIME_SECRET_ENV_VARS:
        return ["process environment / .env"]
    return ["process environment (unsupported)", "built-in defaults"]


def _justification_for_env(key: str, status: str) -> str:
    if status == "internal_only":
        return "Kept configurable internally to coordinate spawned runtime processes and transport behavior."
    if status == "legacy_alias":
        return "Retained for backward compatibility while canonical variables remain stable."
    if status == "doc_only":
        return "Kept documented for migration/compat context; safe to review for possible deprecation."
    if key in RUNTIME_SECRET_ENV_VARS or key.endswith("_API_KEY") or "TOKEN" in key or "SECRET" in key:
        return "Credentials must remain user-configurable per environment and security boundary."
    return "Non-secret runtime configuration is intentionally not supported via environment variables."


def _build_env_entries(
    env_keys_from_example: set[str],
    env_descriptions: dict[str, str],
    env_refs: dict[str, dict[str, set[str]]],
    tui_control_map: dict[str, set[str]],
    outside_scope_refs: dict[str, dict[str, set[str]]],
) -> list[SettingInventoryEntry]:
    all_env_keys = sorted(set(env_keys_from_example).union(env_refs.keys()))
    entries: list[SettingInventoryEntry] = []
    for key in all_env_keys:
        refs = env_refs.get(key, {"reads": set(), "writes": set()})
        read_refs = sorted(refs.get("reads", set()))
        write_refs = sorted(refs.get("writes", set()))
        where_read = sorted(set(read_refs + write_refs))
        read_count = len(read_refs)
        write_count = len(write_refs)
        documented = key in env_keys_from_example
        explicitly_internal = key in RUNTIME_INTERNAL_ENV_VARS or key in PLATFORM_CONVENTION_ENV_VARS
        write_only = read_count == 0 and write_count > 0

        dup_group, is_alias = _dedupe_group_for_key(key)
        if is_alias:
            status = "legacy_alias"
        elif explicitly_internal or write_only:
            status = "internal_only"
        elif read_count > 0 and not documented:
            status = "undocumented"
        elif documented and read_count == 0:
            status = "doc_only"
        else:
            status = "active"

        note_parts: list[str] = []
        if status == "legacy_alias" and dup_group:
            note_parts.append(f"Prefer canonical variable `{dup_group}` for new configuration.")
        if status == "undocumented":
            note_parts.append("Runtime consumes this variable but it is not documented in env.example.")
        if status == "doc_only":
            note_parts.append("Documented in env.example but no runtime reads found in src/.")
        if status == "internal_only":
            note_parts.append("Internal runtime wiring variable; not intended as primary user-facing knob.")
        outside_refs = outside_scope_refs.get(key, {"reads": set(), "writes": set()})
        outside_sites = sorted(set(outside_refs.get("reads", set())).union(outside_refs.get("writes", set())))
        if outside_sites:
            sample = ", ".join(outside_sites[:3])
            suffix = " ..." if len(outside_sites) > 3 else ""
            note_parts.append(f"Used outside canonical src/ scope at {sample}{suffix}.")
        if key in PLATFORM_CONVENTION_ENV_VARS:
            note_parts.append("Platform convention variable (OS/shell), not Swarmee-specific naming.")
        notes = " ".join(note_parts)

        controls = _derive_env_controls_text(key, env_descriptions, refs)

        entries.append(
            SettingInventoryEntry(
                id=key,
                kind="env",
                surface="runtime_internal" if status == "internal_only" else "cli",
                where_set=_env_where_set(
                    key=key,
                    documented=documented,
                    internal_only=(status == "internal_only"),
                    tui_control_map=tui_control_map,
                ),
                precedence=_env_precedence(key, internal_only=(status == "internal_only")),
                where_read=where_read,
                controls=controls,
                controls_plain=_plain_for_env(
                    key=key,
                    status=status,
                    dup_group=dup_group,
                    controls=controls,
                    env_descriptions=env_descriptions,
                ),
                status=status,
                dup_group=dup_group,
                keep_configurable_justification=_justification_for_env(key, status),
                notes=notes,
                usage_count=read_count,
                read_count=read_count,
                write_count=write_count,
                classification_reason=_classification_reason_for_env(
                    key=key,
                    status=status,
                    documented=documented,
                    read_count=read_count,
                    write_count=write_count,
                    dup_group=dup_group,
                ),
                documented=documented,
            )
        )
    return entries


def _build_settings_json_entries(settings_paths: set[str]) -> list[dict[str, Any]]:
    settings_file = SRC_DIR / "swarmee_river" / "settings.py"
    where_read = [f"{_rel(settings_file)}:1"]
    entries: list[dict[str, Any]] = []
    for path in sorted(settings_paths):
        is_legacy_env = path.startswith("env.")
        is_raw = path == "raw"
        status = "internal_only" if (is_legacy_env or is_raw) else "active"
        set_sources = [".swarmee/settings.json"]
        notes = "Non-secret runtime configuration should be expressed via settings fields and CLI flags, not env vars."
        if is_legacy_env:
            set_sources = [".swarmee/settings.json (legacy `env` section; internal-only allowlist)"]
            notes = (
                "Migration-only legacy surface. New configs must use structured settings fields; "
                "only a small internal wiring allowlist may be persisted here."
            )
        elif is_raw:
            notes = "Internal/debug payload capture; not intended for manual configuration."
        entries.append(
            {
                "id": f"settings_json:{path}",
                "kind": "settings_json_field",
                "surface": "settings_file",
                "where_set": set_sources,
                "precedence": [
                    "CLI flags (where supported)",
                    ".swarmee/settings.json",
                    "built-in defaults",
                ],
                "where_read": where_read,
                "controls": f"Project settings file field `{path}`.",
                "controls_plain": _sentence(
                    f"Sets `{path}` in `.swarmee/settings.json` for workspace defaults.",
                    "Project settings.json field.",
                ),
                "status": status,
                "dup_group": None,
                "keep_configurable_justification": (
                    "Workspace-scoped JSON config enables repeatable team defaults without shell setup."
                ),
                "notes": notes,
                "usage_count": 1,
            }
        )
    return entries


def _augment_precedence_for_non_env(entries: list[dict[str, Any]]) -> None:
    for entry in entries:
        kind = entry.get("kind")
        if kind == "cli_arg":
            entry["precedence"] = [
                "CLI flag argument",
                ".swarmee/settings.json",
                "built-in defaults",
            ]
        elif kind == "tui_setting":
            entry["precedence"] = [
                "TUI control value",
                ".swarmee/settings.json",
                "built-in defaults",
            ]
        elif kind == "tui_command_arg":
            entry["precedence"] = [
                "TUI slash command argument",
                "current session state",
                "runtime defaults",
            ]
        elif kind == "notebook_arg":
            entry["precedence"] = [
                "Notebook magic flag",
                ".swarmee/settings.json",
                "built-in defaults",
            ]


def _to_inventory_dict(entries: list[SettingInventoryEntry], non_env_entries: list[dict[str, Any]]) -> dict[str, Any]:
    env_entries = [entry.to_dict() for entry in entries]
    merged_entries = sorted(env_entries + non_env_entries, key=lambda item: str(item["id"]))
    return {
        "schema_version": 2,
        "generated_from_scope": "src",
        "precedence_model": {
            "global": [
                "CLI flags",
                ".swarmee/settings.json",
                "built-in defaults",
            ],
            "exceptions": [
                "Secrets/credentials are read from process environment (including `.env` for local dev).",
                "Alias fallback chains are preserved for backward compatibility.",
            ],
        },
        "entries": merged_entries,
    }


def _render_table(entries: list[dict[str, Any]]) -> str:
    headers = [
        "ID",
        "Surface",
        "Status",
        "Where Set",
        "Where Read",
        "Controls",
        "Plain Language",
        "Dup Group",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for entry in entries:
        row = [
            str(entry.get("id", "")),
            str(entry.get("surface", "")),
            str(entry.get("status", "")),
            "<br>".join(entry.get("where_set", [])),
            "<br>".join(entry.get("where_read", [])),
            str(entry.get("controls", "")),
            str(entry.get("controls_plain", "")),
            str(entry.get("dup_group") or ""),
        ]
        row = [cell.replace("\n", " ").replace("|", "\\|") for cell in row]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _render_markdown(
    inventory_payload: dict[str, Any],
    test_only_vars: dict[str, set[str]],
) -> str:
    entries = inventory_payload["entries"]
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        by_kind.setdefault(entry["kind"], []).append(entry)

    runtime_used_undocumented = [
        entry for entry in entries if entry["kind"] == "env" and entry["status"] == "undocumented"
    ]
    documented_unused = [entry for entry in entries if entry["kind"] == "env" and entry["status"] == "doc_only"]

    alias_rows: list[dict[str, Any]] = []
    for canonical, aliases in sorted(ALIASES_BY_CANONICAL.items()):
        alias_rows.append(
            {
                "canonical": canonical,
                "aliases": ", ".join(aliases),
                "note": f"Prefer `{canonical}` for new configs.",
            }
        )

    lines: list[str] = [
        "# Settings / Args / Env Inventory",
        "",
        "Generated by `scripts/generate_settings_inventory.py`.",
        "",
        "## How To Use This Index",
        "- Lookup by `id` in `settings_inventory.json` for exact machine-readable metadata.",
        "- Use this Markdown report for grouped browsing and quick references.",
        "- Scope is runtime code under `src/`; test-only variables are listed in an appendix.",
        "- If a key is also referenced outside canonical `src/` scope, that appears in the entry notes.",
        "",
        "## Precedence Model",
        "Global precedence (high to low):",
    ]
    for layer in inventory_payload["precedence_model"]["global"]:
        lines.append(f"- {layer}")
    lines.append("")
    lines.append("Exceptions:")
    for item in inventory_payload["precedence_model"]["exceptions"]:
        lines.append(f"- {item}")

    lines.extend(["", "## Duplicate / Alias Matrix", "", "| Canonical | Aliases | Note |", "| --- | --- | --- |"])
    for row in alias_rows:
        lines.append(f"| {row['canonical']} | {row['aliases']} | {row['note']} |")

    for kind in [
        "env",
        "cli_arg",
        "tui_setting",
        "tui_command_arg",
        "notebook_arg",
        "settings_json_field",
    ]:
        kind_entries = by_kind.get(kind, [])
        lines.extend(["", f"## {kind}", ""])
        if not kind_entries:
            lines.append("_None_")
            continue
        lines.append(_render_table(kind_entries))

    lines.extend(["", "## Gaps", ""])
    lines.append("### Runtime-used but undocumented env vars")
    if runtime_used_undocumented:
        for entry in runtime_used_undocumented:
            lines.append(f"- `{entry['id']}` ({', '.join(entry.get('where_read', []))})")
    else:
        lines.append("- None")

    lines.extend(["", "### Documented but runtime-unused env vars"])
    if documented_unused:
        for entry in documented_unused:
            lines.append(f"- `{entry['id']}`")
    else:
        lines.append("- None")

    lines.extend(["", "## Test-only Variables Appendix", ""])
    src_env_ids = {entry["id"] for entry in entries if entry["kind"] == "env"}
    test_only_keys = sorted(key for key in test_only_vars.keys() if key not in src_env_ids)
    if not test_only_keys:
        lines.append("- None")
    else:
        for key in test_only_keys:
            refs = ", ".join(sorted(test_only_vars[key]))
            lines.append(f"- `{key}`: {refs}")

    return "\n".join(lines).strip() + "\n"


def _load_env_policy_sets() -> tuple[set[str], set[str], set[str]]:
    """
    Load env allowlists from `src/swarmee_river/config/env_policy.py` without importing runtime code.

    Returns: (secret_allowlist, internal_allowlist, internal_settings_override_allowlist)
    """
    path = SRC_DIR / "swarmee_river" / "config" / "env_policy.py"
    text = _read_text(path)
    tree = ast.parse(text or "", filename=str(path))

    def _extract(name: str) -> set[str]:
        for node in tree.body:
            target: ast.Name | None = None
            value: ast.AST | None = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                target = node.target
                value = node.value
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name) and t.id == name:
                        target = t
                        value = node.value
                        break
            if target is None or target.id != name or value is None:
                continue
            if isinstance(value, ast.Set):
                out: set[str] = set()
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
                return out
        return set()

    secret = _extract("SECRET_ENV_ALLOWLIST")
    internal = _extract("INTERNAL_ENV_ALLOWLIST")
    internal_settings = _extract("INTERNAL_SETTINGS_ENV_OVERRIDE_ALLOWLIST")
    return secret, internal, internal_settings


def build_inventory_payload() -> tuple[dict[str, Any], str]:
    global RUNTIME_INTERNAL_ENV_VARS, RUNTIME_SECRET_ENV_VARS, RUNTIME_INTERNAL_SETTINGS_ENV_OVERRIDE_VARS

    (
        RUNTIME_SECRET_ENV_VARS,
        RUNTIME_INTERNAL_ENV_VARS,
        RUNTIME_INTERNAL_SETTINGS_ENV_OVERRIDE_VARS,
    ) = _load_env_policy_sets()

    env_keys_from_example, env_descriptions = _collect_env_from_example(ENV_EXAMPLE_PATH)
    env_refs = _scan_env_accesses(SRC_DIR)
    outside_scope_refs = _scan_env_accesses(ROOT / "tools")
    cli_entries = _collect_cli_args()
    notebook_entries = _collect_notebook_args(SRC_DIR / "swarmee_river" / "jupyter" / "magic.py")
    tui_setting_entries = _collect_tui_settings_controls(SRC_DIR / "swarmee_river" / "tui" / "app.py")
    tui_command_entries = _collect_tui_command_entries(SRC_DIR / "swarmee_river" / "tui" / "commands.py")

    tui_control_map: dict[str, set[str]] = {}
    for item in tui_setting_entries:
        if item["id"].startswith("tui_setting:"):
            mapped = set(re.findall(rf"\b({ENV_NAME_RE})\b", item["controls"]))
            if mapped:
                tui_control_map[item["id"]] = mapped

    env_entries = _build_env_entries(
        env_keys_from_example=env_keys_from_example,
        env_descriptions=env_descriptions,
        env_refs=env_refs,
        tui_control_map=tui_control_map,
        outside_scope_refs=outside_scope_refs,
    )
    settings_json_entries = _build_settings_json_entries(_load_settings_json_paths())

    non_env_entries: list[dict[str, Any]] = []
    non_env_entries.extend(cli_entries)
    non_env_entries.extend(tui_setting_entries)
    non_env_entries.extend(tui_command_entries)
    non_env_entries.extend(notebook_entries)
    non_env_entries.extend(settings_json_entries)

    _augment_precedence_for_non_env(non_env_entries)

    payload = _to_inventory_dict(env_entries, non_env_entries)
    test_env_refs = _scan_env_accesses_in_tests(TESTS_DIR)
    markdown = _render_markdown(payload, test_env_refs)
    return payload, markdown


def _write_if_changed(path: Path, content: str) -> bool:
    current = path.read_text(encoding="utf-8") if path.exists() else None
    if current == content:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate runtime settings/args/env inventory docs.")
    parser.add_argument("--output-json", type=Path, default=SETTINGS_JSON_PATH)
    parser.add_argument("--output-md", type=Path, default=SETTINGS_MD_PATH)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if generated output differs from committed files.",
    )
    args = parser.parse_args()

    payload, markdown = build_inventory_payload()
    json_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.check:
        changed = False
        json_current = args.output_json.read_text(encoding="utf-8") if args.output_json.exists() else ""
        md_current = args.output_md.read_text(encoding="utf-8") if args.output_md.exists() else ""
        if json_current != json_text:
            changed = True
            print(f"Outdated: {args.output_json}")
        if md_current != markdown:
            changed = True
            print(f"Outdated: {args.output_md}")
        return 1 if changed else 0

    _write_if_changed(args.output_json, json_text)
    _write_if_changed(args.output_md, markdown)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
