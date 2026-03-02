from __future__ import annotations

import json
from pathlib import Path

from swarmee_river.prompt_assets import (
    PromptAsset,
    ensure_prompt_assets_bootstrapped,
    load_prompt_assets,
    resolve_agent_prompt_text,
    resolve_orchestrator_prompt_from_agent,
    save_prompt_assets,
)


def test_prompt_assets_bootstrap_migrates_legacy_files(tmp_path: Path, monkeypatch):
    state_path = tmp_path / ".swarmee"
    state_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(state_path))
    monkeypatch.chdir(tmp_path)

    (tmp_path / ".prompt").write_text("Legacy orchestrator prompt", encoding="utf-8")
    legacy_templates = [
        {
            "id": "legacy_template",
            "name": "Legacy Template",
            "content": "Template content",
            "tags": ["legacy"],
        }
    ]
    (state_path / "prompt_templates.json").write_text(json.dumps(legacy_templates), encoding="utf-8")

    first = ensure_prompt_assets_bootstrapped()
    assert first.migrated is True
    assets = {asset.id: asset for asset in load_prompt_assets()}
    assert "orchestrator_base" in assets
    assert assets["orchestrator_base"].content == "Legacy orchestrator prompt"
    assert "legacy_template" in assets
    assert assets["legacy_template"].source == "migrated_template"

    second = ensure_prompt_assets_bootstrapped()
    assert second.migrated is False
    assert (state_path / "prompt_templates.json").exists()
    assert (tmp_path / ".prompt").exists()


def test_resolve_orchestrator_prompt_from_agent_composes_refs_and_fallback(tmp_path: Path, monkeypatch):
    state_path = tmp_path / ".swarmee"
    state_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(state_path))
    monkeypatch.chdir(tmp_path)

    save_prompt_assets(
        [
            PromptAsset(id="orchestrator_base", name="Base", content="Base prompt"),
            PromptAsset(id="custom_orchestrator", name="Custom", content="Custom prompt"),
            PromptAsset(id="inline_extra", name="Extra", content="Extra prompt"),
        ]
    )

    selected = resolve_orchestrator_prompt_from_agent(
        {"id": "orchestrator", "prompt_refs": ["custom_orchestrator"], "prompt": "Inline tail"}
    )
    fallback = resolve_orchestrator_prompt_from_agent({"id": "orchestrator", "prompt_refs": ["missing"]})
    assert selected == "Custom prompt\n\nInline tail"
    assert fallback == "Base prompt"


def test_resolve_agent_prompt_text_composes_refs_then_inline():
    assets = {
        "one": PromptAsset(id="one", name="One", content="Prompt One"),
        "two": PromptAsset(id="two", name="Two", content="Prompt Two"),
    }
    text = resolve_agent_prompt_text(
        {
            "prompt_refs": ["two", "one", "missing"],
            "prompt": "Inline prompt",
        },
        assets,
    )
    assert text == "Prompt Two\n\nPrompt One\n\nInline prompt"
