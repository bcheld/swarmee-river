from __future__ import annotations

from swarmee_river.tui.views import settings as settings_view


def test_env_var_specs_include_model_provider() -> None:
    specs = settings_view.env_var_specs()
    provider_spec = next((spec for spec in specs if spec.key == "SWARMEE_MODEL_PROVIDER"), None)
    assert provider_spec is not None
    assert provider_spec.category
    assert "openai" in provider_spec.choices
    assert "bedrock" in provider_spec.choices


def test_build_env_sidebar_items_includes_default_and_current(monkeypatch) -> None:
    monkeypatch.setenv("SWARMEE_MODEL_PROVIDER", "openai")
    items = settings_view.build_env_sidebar_items()
    target = next((item for item in items if item["id"] == "SWARMEE_MODEL_PROVIDER"), None)
    assert target is not None
    assert "current: openai" in target["subtitle"]
    assert "default:" in target["subtitle"]
