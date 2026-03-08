from pathlib import Path

import tomllib


def test_pyproject_requires_openai_sdk_2x() -> None:
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    payload = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = payload.get("project", {}).get("dependencies", [])

    assert "openai>=2.0.0,<3.0.0" in dependencies
