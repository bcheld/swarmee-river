"""Command palette visibility tests (doc 13 F3)."""

from __future__ import annotations

from swarmee_river.tui.widgets import CommandPalette


def test_palette_stays_visible_on_no_match() -> None:
    palette = CommandPalette()
    palette.filter("/definitely-not-a-command")
    assert palette.is_visible, "palette must not vanish silently on a typo"
    assert palette.get_selected() is None


def test_palette_falls_back_to_substring_match() -> None:
    palette = CommandPalette()
    palette.filter("/plan")  # prefix match
    prefix_matches = list(palette._filtered)
    assert ("/plan", "Generate a plan") in prefix_matches

    palette.filter("/lan")  # typo: no prefix match, substring should recover
    substring_matches = [cmd for cmd, _ in palette._filtered]
    assert "/plan" in substring_matches


def test_palette_bare_slash_shows_all_commands() -> None:
    palette = CommandPalette()
    palette.filter("/")
    assert len(palette._filtered) == len(CommandPalette.TUI_COMMANDS)
