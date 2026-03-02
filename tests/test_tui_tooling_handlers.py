from __future__ import annotations

from swarmee_river.tui.tooling_handlers import (
    build_kb_table_rows,
    build_prompt_table_rows,
    build_sop_table_rows,
    render_kb_detail,
    render_sop_detail,
)


def test_build_prompt_table_rows_formats_fields_and_truncates_preview() -> None:
    rows = build_prompt_table_rows(
        [
            {
                "id": "tmpl-1",
                "name": "Review Prompt",
                "content": "A" * 120,
                "tags": ["code", "review"],
                "source": "s3",
            }
        ]
    )

    assert len(rows) == 1
    template_id, name, tags, source, preview = rows[0]
    assert template_id == "tmpl-1"
    assert name == "Review Prompt"
    assert tags == "code, review"
    assert source == "s3"
    assert preview.endswith("…")
    assert len(preview) == 80


def test_build_sop_table_rows_marks_active_and_uses_fallback_preview() -> None:
    rows = build_sop_table_rows(
        [
            {"name": "deploy", "source": "pack:ops", "first_paragraph_preview": ""},
            {"name": "incident", "source": "local", "first_paragraph_preview": "Do this first."},
        ],
        {"incident"},
    )

    assert rows == [
        ("deploy", "no", "pack:ops", "(no preview available)"),
        ("incident", "yes", "local", "Do this first."),
    ]


def test_build_kb_table_rows_applies_id_name_fallbacks() -> None:
    rows = build_kb_table_rows(
        [
            {"description": "No id or name provided."},
            {"id": "kb-alpha", "name": "Alpha", "description": "D" * 150},
        ]
    )

    assert rows[0] == ("kb-1", "KB 1", "No id or name provided.")
    assert rows[1][0] == "kb-alpha"
    assert rows[1][1] == "Alpha"
    assert rows[1][2].endswith("…")
    assert len(rows[1][2]) == 100


def test_render_detail_helpers_include_key_metadata() -> None:
    sop_detail = render_sop_detail(
        {
            "name": "deploy",
            "source": "local",
            "path": "/tmp/deploy.sop.md",
            "first_paragraph_preview": "Preview",
            "content": "Step 1",
        },
        active=True,
    )
    assert "Status: active" in sop_detail
    assert "Press Enter to activate/deactivate." in sop_detail
    assert "Path: /tmp/deploy.sop.md" in sop_detail

    kb_detail = render_kb_detail({"id": "kb-alpha", "name": "Alpha", "description": "Useful context"})
    assert "# Alpha" in kb_detail
    assert "ID: kb-alpha" in kb_detail
    assert "Useful context" in kb_detail
