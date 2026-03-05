from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

ENV_NAME_RE = r"[A-Z][A-Z0-9_]+"

REQUIRED_ENTRY_KEYS = {
    "id",
    "kind",
    "surface",
    "where_set",
    "precedence",
    "where_read",
    "controls",
    "controls_plain",
    "status",
    "dup_group",
    "keep_configurable_justification",
    "notes",
    "usage_count",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _inventory_json_path() -> Path:
    return _repo_root() / "docs" / "configuration" / "settings_inventory.json"


def _load_inventory() -> dict[str, Any]:
    return json.loads(_inventory_json_path().read_text(encoding="utf-8"))


def _env_entries(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in payload["entries"] if entry.get("kind") == "env"}


def _env_vars_from_example() -> set[str]:
    path = _repo_root() / "env.example"
    keys: set[str] = set()
    pattern = re.compile(rf"^#\s*({ENV_NAME_RE})=", flags=re.MULTILINE)
    text = path.read_text(encoding="utf-8")
    for match in pattern.finditer(text):
        keys.add(match.group(1))
    return keys


def _env_vars_read_in_src() -> set[str]:
    src_root = _repo_root() / "src"
    patterns = [
        re.compile(rf'os\.getenv\(\s*"({ENV_NAME_RE})"'),
        re.compile(rf'os\.environ\.get\(\s*"({ENV_NAME_RE})"'),
        re.compile(rf'os\.environ\[\s*"({ENV_NAME_RE})"\s*\]'),
        re.compile(
            rf'\b(?:truthy_env|csv_env|_truthy_env|_env_float|_parse_int_env|_env_int)\(\s*"({ENV_NAME_RE})"'
        ),
        re.compile(rf'\bgetenv_secret\(\s*"({ENV_NAME_RE})"'),
        re.compile(rf'\bgetenv_internal\(\s*"({ENV_NAME_RE})"'),
    ]
    out: set[str] = set()
    for py_file in sorted(src_root.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        for pattern in patterns:
            for match in pattern.finditer(text):
                out.add(match.group(1))
    return out


def _load_env_policy_sets() -> tuple[set[str], set[str]]:
    """
    Parse `src/swarmee_river/config/env_policy.py` without importing runtime code.

    Returns: (secret_allowlist, internal_allowlist)
    """
    path = _repo_root() / "src" / "swarmee_river" / "config" / "env_policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def _extract(name: str) -> set[str]:
        for node in tree.body:
            value: ast.AST | None = None
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
                value = node.value
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == name:
                        value = node.value
                        break
            if value is None:
                continue
            if isinstance(value, ast.Set):
                out: set[str] = set()
                for elt in value.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        out.add(elt.value)
                return out
        return set()

    return _extract("SECRET_ENV_ALLOWLIST"), _extract("INTERNAL_ENV_ALLOWLIST")


def test_inventory_schema_has_required_fields() -> None:
    payload = _load_inventory()
    entries = payload.get("entries")
    assert isinstance(entries, list)
    assert entries

    for entry in entries:
        assert isinstance(entry, dict)
        missing = REQUIRED_ENTRY_KEYS - set(entry.keys())
        assert not missing, f"missing required keys: {sorted(missing)}"
        assert isinstance(entry["where_set"], list)
        assert isinstance(entry["precedence"], list)
        assert isinstance(entry["where_read"], list)


def test_env_reads_in_src_are_represented_in_inventory() -> None:
    payload = _load_inventory()
    inventory_env_ids = {entry["id"] for entry in payload["entries"] if entry.get("kind") == "env"}
    src_env_ids = _env_vars_read_in_src()
    missing = sorted(src_env_ids - inventory_env_ids)
    assert not missing, f"missing env vars from inventory: {missing}"


def test_env_example_variables_are_documented_in_inventory() -> None:
    payload = _load_inventory()
    env_entries = _env_entries(payload)
    expected = _env_vars_from_example()
    missing = sorted(key for key in expected if key not in env_entries)
    assert not missing, f"env.example vars missing from inventory: {missing}"
    undocumented = sorted(key for key in expected if env_entries[key].get("documented") is not True)
    assert not undocumented, f"env.example vars not marked documented=true: {undocumented}"


def test_inventory_contains_required_anchor_entries() -> None:
    payload = _load_inventory()
    ids = {entry["id"] for entry in payload["entries"]}
    required_ids = {
        # Secrets-only env policy anchors
        "OPENAI_API_KEY",
        "SWARMEE_GITHUB_COPILOT_API_KEY",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "notebook:%%swarmee --yes",
        "notebook:%%swarmee --plan",
        "notebook:%%swarmee --no-context",
        "notebook:%%swarmee --daemon-stop",
        "tui_setting:settings_toggle_auto_approve",
        "tui_setting:settings_toggle_swarm",
        "tui_setting:settings_general_context_manager",
        "cli:swarmee --model-provider",
        "cli:swarmee --context-manager",
        "cli:swarmee diagnostics --lines",
        "cli:swarmee diff --staged",
        # Structured settings anchors
        "settings_json:context.manager",
        "settings_json:runtime.auto_approve",
        "settings_json:models.provider",
        "settings_json:diagnostics.redact",
        "settings_json:pricing.providers.<provider>.input_per_1m",
    }
    missing = sorted(required_ids - ids)
    assert not missing, f"required inventory anchors missing: {missing}"


def test_inventory_precedence_fields_are_non_empty() -> None:
    payload = _load_inventory()
    for entry in payload["entries"]:
        precedence = entry.get("precedence")
        assert isinstance(precedence, list)
        assert precedence, f"empty precedence for {entry.get('id')}"
        assert all(isinstance(item, str) and item.strip() for item in precedence)


def test_controls_plain_present_and_non_empty() -> None:
    payload = _load_inventory()
    for entry in payload["entries"]:
        value = entry.get("controls_plain")
        assert isinstance(value, str) and value.strip(), f"missing controls_plain for {entry.get('id')}"


def test_env_entries_have_classification_details() -> None:
    payload = _load_inventory()
    for env in _env_entries(payload).values():
        assert isinstance(env.get("read_count"), int), f"missing read_count for {env.get('id')}"
        assert isinstance(env.get("write_count"), int), f"missing write_count for {env.get('id')}"
        reason = env.get("classification_reason")
        assert isinstance(reason, str) and reason.strip(), f"missing classification_reason for {env.get('id')}"


def test_no_active_env_entry_has_zero_reads() -> None:
    payload = _load_inventory()
    bad = [
        env["id"]
        for env in _env_entries(payload).values()
        if env.get("status") == "active" and int(env.get("read_count", 0)) == 0
    ]
    assert not bad, f"active env entries with zero reads: {bad}"


def test_runtime_does_not_read_non_secret_env_config() -> None:
    """
    Secrets-only env policy: runtime may read secrets and internal wiring env vars only.

    Everything else must be configured via `.swarmee/settings.json` fields and/or CLI flags.
    """
    payload = _load_inventory()
    secret_allowlist, internal_allowlist = _load_env_policy_sets()

    bad: list[str] = []
    for env in _env_entries(payload).values():
        key = str(env.get("id") or "")
        read_count = int(env.get("read_count") or 0)
        status = str(env.get("status") or "")
        if read_count <= 0:
            continue
        if key in secret_allowlist:
            continue
        if key in internal_allowlist:
            continue
        if status == "internal_only":
            # Platform convention env vars (CI, SHELL, etc.) and write-only wiring keys can land here.
            continue
        bad.append(f"{key} (status={status}, reads={read_count})")

    assert not bad, "unsupported env reads found:\n" + "\n".join(sorted(bad))


def test_env_example_contains_only_secret_env_vars() -> None:
    secret_allowlist, _internal_allowlist = _load_env_policy_sets()
    declared = _env_vars_from_example()
    extra = sorted(key for key in declared if key not in secret_allowlist)
    assert not extra, f"env.example should document secrets only; found non-secret keys: {extra}"


def test_generator_check_mode_is_clean() -> None:
    repo = _repo_root()
    result = subprocess.run(
        [sys.executable, "scripts/generate_settings_inventory.py", "--check"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
