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
