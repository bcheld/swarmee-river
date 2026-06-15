"""Guards for footer keybinding visibility.

Textual's Footer renders one chip per visible binding, in BINDINGS order,
and silently clips whatever exceeds the terminal width. Duplicate visible
bindings for the same action therefore push real commands off-screen
("menu items are not visible"). These tests keep that from regressing.
"""

from __future__ import annotations

from collections import Counter

from textual.binding import Binding

from swarmee_river.tui.app import get_swarmee_tui_class


def _normalized_bindings() -> list[Binding]:
    cls = get_swarmee_tui_class()
    bindings: list[Binding] = []
    for entry in cls.BINDINGS:
        if isinstance(entry, Binding):
            bindings.append(entry)
        else:
            key, action, description = entry
            bindings.append(Binding(key, action, description))
    return bindings


def test_at_most_one_visible_binding_per_action() -> None:
    visible_actions = Counter(b.action for b in _normalized_bindings() if b.show)
    duplicates = {action: count for action, count in visible_actions.items() if count > 1}
    assert not duplicates, (
        f"Actions with multiple visible footer bindings (clips the footer): {duplicates}"
    )


def test_hidden_alternates_remain_registered() -> None:
    """Power-user alternates must stay bound even though they're hidden."""
    keys = {b.key for b in _normalized_bindings()}
    for expected in ("ctrl+k", "ctrl+space", "ctrl+h", "ctrl+l", "f5", "f6", "f7", "ctrl+c"):
        assert expected in keys, f"alternate binding {expected!r} was removed"


def test_primary_actions_lead_the_footer() -> None:
    """The most important chips must come first so they survive narrow widths."""
    visible_actions = [b.action for b in _normalized_bindings() if b.show]
    assert visible_actions[:2] == ["open_action_sheet", "interrupt_run"]


def test_help_binding_is_visible() -> None:
    """F1 must advertise the keybinding reference in the footer."""
    visible = {b.action for b in _normalized_bindings() if b.show}
    assert "show_keys" in visible


def test_keys_reference_lists_hidden_alternates() -> None:
    """The /keys output is generated from BINDINGS and must include alternates."""
    cls = get_swarmee_tui_class()

    class _Host:
        BINDINGS = cls.BINDINGS

        def __init__(self) -> None:
            self.lines: list[str] = []

        _write_keybindings_reference = cls._write_keybindings_reference

        def _write_transcript_line(self, text: str) -> None:
            self.lines.append(text)

    host = _Host()
    host._write_keybindings_reference()
    output = "\n".join(host.lines)
    assert "ctrl+p" in output
    assert "ctrl+h" in output, "hidden alternates must be discoverable via /keys"
    assert "Widen side" in output
    assert "f5" in output
