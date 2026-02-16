import importlib
from pathlib import Path

from swarmee_river.tools import get_tools


def test_tools_includes_shell_and_python_repl():
    tools = get_tools()

    assert "python_repl" in tools
    assert "shell" in tools


def test_tools_include_core_coding_primitives():
    tools = get_tools()

    assert "git" in tools
    assert "patch_apply" in tools
    assert "run_checks" in tools
    assert "file_list" in tools
    assert "file_search" in tools
    assert "file_read" in tools
    assert "todoread" in tools
    assert "todowrite" in tools


def test_tools_include_opencode_aliases():
    tools = get_tools()

    for alias_name in ["grep", "read", "bash", "patch", "write", "edit"]:
        assert alias_name in tools


def test_project_context_tool_disabled_by_default(monkeypatch):
    monkeypatch.delenv("SWARMEE_ENABLE_PROJECT_CONTEXT_TOOL", raising=False)
    tools = get_tools()
    assert "project_context" not in tools


def test_project_context_tool_can_be_enabled(monkeypatch):
    monkeypatch.setenv("SWARMEE_ENABLE_PROJECT_CONTEXT_TOOL", "true")
    tools = get_tools()
    assert "project_context" in tools


def _result_text(result: dict[str, object]) -> str:
    content = result.get("content")
    if isinstance(content, list) and content:
        first = content[0]
        if isinstance(first, dict):
            value = first.get("text")
            if isinstance(value, str):
                return value
    return ""


def _simulate_missing_strands_tools(monkeypatch) -> None:
    import swarmee_river.tools as tools_module

    real_import_module = importlib.import_module

    def _import_module(name: str, package: str | None = None):
        if name == "strands_tools":
            raise ImportError("simulated missing strands_tools")
        return real_import_module(name, package)

    monkeypatch.setattr(tools_module.importlib, "import_module", _import_module)


def test_get_tools_includes_editor_and_file_write_fallbacks_without_strands_tools(monkeypatch) -> None:
    _simulate_missing_strands_tools(monkeypatch)

    tools = get_tools()

    assert "file_write" in tools
    assert "editor" in tools
    assert "retrieve" in tools
    assert "http_request" in tools
    assert "calculator" in tools
    assert "current_time" in tools
    assert "environment" in tools
    assert "use_agent" in tools
    assert "use_llm" in tools


def test_opencode_write_and_edit_aliases_work_without_strands_tools(tmp_path: Path, monkeypatch) -> None:
    _simulate_missing_strands_tools(monkeypatch)
    monkeypatch.chdir(tmp_path)

    tools = get_tools()

    write_result = tools["write"](path="notes.txt", content="alpha\n", cwd=str(tmp_path))
    assert write_result.get("status") == "success"
    assert "unavailable" not in _result_text(write_result).lower()
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "alpha\n"

    view_result = tools["edit"](command="view", path="notes.txt")
    assert view_result.get("status") == "success"
    assert "alpha" in _result_text(view_result)

    replace_result = tools["edit"](
        command="replace",
        path="notes.txt",
        old_str="alpha",
        new_str="beta",
    )
    assert replace_result.get("status") == "success"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "beta\n"

    insert_result = tools["edit"](
        command="insert",
        path="notes.txt",
        insert_line=2,
        file_text="gamma\n",
    )
    assert insert_result.get("status") == "success"
    assert (tmp_path / "notes.txt").read_text(encoding="utf-8") == "beta\ngamma\n"


def test_opencode_write_and_edit_aliases_block_parent_traversal_without_strands_tools(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _simulate_missing_strands_tools(monkeypatch)
    workspace = tmp_path / "workspace"
    subdir = workspace / "subdir"
    subdir.mkdir(parents=True)

    tools = get_tools()

    write_result = tools["write"](path="../escape.txt", content="x", cwd=str(subdir))
    assert write_result.get("status") == "error"
    assert "outside cwd" in _result_text(write_result).lower()
    assert not (workspace / "escape.txt").exists()

    monkeypatch.chdir(subdir)
    edit_result = tools["edit"](command="write", path="../escape.txt", file_text="x")
    assert edit_result.get("status") == "error"
    assert "outside cwd" in _result_text(edit_result).lower()
    assert not (workspace / "escape.txt").exists()
