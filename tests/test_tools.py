from swarmee_river.tools import get_tools


def test_tools_includes_shell_and_python_repl():
    tools = get_tools()

    assert "python_repl" in tools
    assert "shell" in tools
