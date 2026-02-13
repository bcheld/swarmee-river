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
