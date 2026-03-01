from __future__ import annotations

from pathlib import Path

import pytest

from swarmee_river.profiles import AgentProfile, delete_profile, list_profiles, load_profile, save_profile


def test_profiles_store_roundtrip(tmp_path: Path) -> None:
    root_dir = tmp_path / "profiles"

    saved = save_profile(
        {
            "id": "researcher",
            "name": "Researcher",
            "provider": "OpenAI",
            "tier": "Deep",
            "system_prompt_snippets": ["Be concise", "Cite assumptions"],
            "context_sources": [{"type": "file", "path": "notes/today.md", "id": "notes-today-md"}],
            "active_sops": ["investigation"],
            "knowledge_base_id": "KB-001",
            "agents": [
                {
                    "id": "triage-research",
                    "name": "Triage Research",
                    "summary": "Investigates incoming issues",
                    "prompt": "You triage and categorize incidents.",
                    "provider": "OpenAI",
                    "tier": "Balanced",
                    "tool_names": ["file_read", "shell", "shell"],
                    "sop_names": ["incident-triage"],
                    "knowledge_base_id": "kb-123",
                    "activated": True,
                }
            ],
            "auto_delegate_assistive": True,
            "team_presets": [
                {
                    "id": "triage-team",
                    "name": "Triage Team",
                    "description": "Default incident triage lineup",
                    "spec": {
                        "mode": "swarm",
                        "agents": [{"id": "lead"}, {"id": "reviewer"}],
                    },
                }
            ],
        },
        root_dir=root_dir,
    )

    assert saved.id == "researcher"
    assert saved.provider == "openai"
    assert saved.tier == "deep"
    assert saved.agents == [
        {
            "id": "orchestrator",
            "name": "Orchestrator",
            "summary": "",
            "prompt": "",
            "prompt_refs": ["orchestrator_base"],
            "provider": None,
            "tier": None,
            "tool_names": [],
            "sop_names": [],
            "knowledge_base_id": None,
            "activated": False,
        },
        {
            "id": "triage-research",
            "name": "Triage Research",
            "summary": "Investigates incoming issues",
            "prompt": "You triage and categorize incidents.",
            "prompt_refs": [],
            "provider": "openai",
            "tier": "balanced",
            "tool_names": ["file_read", "shell"],
            "sop_names": ["incident-triage"],
            "knowledge_base_id": "kb-123",
            "activated": True,
        },
    ]
    assert saved.auto_delegate_assistive is True
    assert saved.team_presets[0]["id"] == "triage-team"

    listed = list_profiles(root_dir=root_dir)
    assert [item.id for item in listed] == ["researcher"]

    loaded = load_profile("researcher", root_dir=root_dir)
    assert loaded.to_dict() == saved.to_dict()


def test_save_profile_replaces_existing_profile_id(tmp_path: Path) -> None:
    root_dir = tmp_path / "profiles"

    save_profile({"id": "writer", "name": "Writer"}, root_dir=root_dir)
    save_profile({"id": "writer", "name": "Writer 2", "tier": "balanced"}, root_dir=root_dir)

    listed = list_profiles(root_dir=root_dir)
    assert len(listed) == 1
    assert listed[0].id == "writer"
    assert listed[0].name == "Writer 2"
    assert listed[0].tier == "balanced"


def test_delete_profile_and_missing_profile_behavior(tmp_path: Path) -> None:
    root_dir = tmp_path / "profiles"

    save_profile({"id": "ops", "name": "Ops"}, root_dir=root_dir)

    assert delete_profile("ops", root_dir=root_dir) is True
    assert delete_profile("ops", root_dir=root_dir) is False
    with pytest.raises(FileNotFoundError):
        load_profile("ops", root_dir=root_dir)


def test_list_profiles_ignores_corrupt_or_invalid_entries(tmp_path: Path) -> None:
    root_dir = tmp_path / "profiles"
    root_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = root_dir / "profiles.json"
    catalog_path.write_text(
        '{"version":1,"profiles":[{"id":"ok","name":"OK"},{"name":"missing-id"},"bad-entry"]}',
        encoding="utf-8",
    )

    listed = list_profiles(root_dir=root_dir)
    assert [item.id for item in listed] == ["ok"]

    catalog_path.write_text("{not-json", encoding="utf-8")
    assert list_profiles(root_dir=root_dir) == []


def test_profile_schema_normalizes_context_sources_and_lists() -> None:
    profile = AgentProfile.from_dict(
        {
            "id": "qa",
            "name": "QA",
            "system_prompt_snippets": ["  Use checklists  ", "", "Use checklists"],
            "context_sources": [
                {"type": "file", "path": " /tmp/plan.md "},
                {"type": "file", "path": "/tmp/plan.md", "id": "duplicate"},
                {"type": "note", "text": "  release criteria  "},
                {"type": "kb", "kb_id": "kb-55"},
                {"type": "url", "path": "https://example.com/brief"},
                {"type": "invalid", "foo": "bar"},
            ],
            "active_sops": ["  verify  ", "", "verify", "triage"],
        }
    )

    assert profile.system_prompt_snippets == ["Use checklists"]
    assert profile.active_sops == ["verify", "triage"]
    assert profile.context_sources[0] == {"type": "file", "path": "/tmp/plan.md", "id": "tmp-plan.md"}
    assert profile.context_sources[1]["type"] == "note"
    assert profile.context_sources[1]["text"] == "release criteria"
    assert profile.context_sources[2] == {"type": "kb", "id": "kb-55"}
    assert profile.context_sources[3]["type"] == "url"
    assert profile.context_sources[3]["url"] == "https://example.com/brief"


def test_profile_schema_normalizes_team_presets() -> None:
    profile = AgentProfile.from_dict(
        {
            "id": "ops",
            "name": "Ops",
            "team_presets": [
                {"id": "  alpha  ", "name": "Alpha", "description": "  Primary ", "spec": {"graph": {"id": "a"}}},
                {"name": "Bravo Team", "spec": {"swarm": {"workers": 2}}},
                {"id": "alpha", "name": "Duplicate", "spec": {"ignored": True}},
                {"id": "bad-spec", "name": "Bad", "spec": ["not", "object"]},
            ],
        }
    )

    assert [item["id"] for item in profile.team_presets] == ["alpha", "Bravo-Team"]
    assert profile.team_presets[0]["description"] == "Primary"
    assert profile.team_presets[1]["name"] == "Bravo Team"
    assert profile.team_presets[1]["spec"] == {"swarm": {"workers": 2}}


def test_profile_schema_normalizes_agents_and_auto_delegate_assistive() -> None:
    profile = AgentProfile.from_dict(
        {
            "id": "ops",
            "name": "Ops",
            "auto_delegate_assistive": "false",
            "agents": [
                {
                    "id": "  triage-research  ",
                    "name": "Triage Research",
                    "summary": "  Investigates incoming issues  ",
                    "prompt": "  You triage and categorize incidents.  ",
                    "provider": "OpenAI",
                    "tier": "Balanced",
                    "tool_names": ["file_read", "shell", "SHELL", ""],
                    "sop_names": ["incident-triage", "incident-triage"],
                    "knowledge_base_id": "  kb-123  ",
                    "activated": "yes",
                },
                {"id": "triage-research", "name": "Duplicate", "activated": True},
                {"id": "missing-name", "prompt": "ignored"},
                {"name": "No Spec Fields"},
            ],
        }
    )

    assert profile.auto_delegate_assistive is False
    assert profile.agents == [
        {
            "id": "orchestrator",
            "name": "Orchestrator",
            "summary": "",
            "prompt": "",
            "prompt_refs": ["orchestrator_base"],
            "provider": None,
            "tier": None,
            "tool_names": [],
            "sop_names": [],
            "knowledge_base_id": None,
            "activated": False,
        },
        {
            "id": "triage-research",
            "name": "Triage Research",
            "summary": "Investigates incoming issues",
            "prompt": "You triage and categorize incidents.",
            "prompt_refs": [],
            "provider": "openai",
            "tier": "balanced",
            "tool_names": ["file_read", "shell", "SHELL"],
            "sop_names": ["incident-triage"],
            "knowledge_base_id": "kb-123",
            "activated": True,
        },
        {
            "id": "No-Spec-Fields",
            "name": "No Spec Fields",
            "summary": "",
            "prompt": "",
            "prompt_refs": [],
            "provider": None,
            "tier": None,
            "tool_names": [],
            "sop_names": [],
            "knowledge_base_id": None,
            "activated": False,
        },
    ]


def test_default_state_dir_profiles_location(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_root = tmp_path / "state-root"
    monkeypatch.setenv("SWARMEE_STATE_DIR", str(state_root))

    save_profile({"id": "stateful", "name": "Stateful"})

    catalog = state_root / "profiles" / "profiles.json"
    assert catalog.exists()
    listed = list_profiles()
    assert [item.id for item in listed] == ["stateful"]
