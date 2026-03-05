from __future__ import annotations

import contextlib
import time

from swarmee_river.tui.text_sanitize import sanitize_output_text

_THINKING_DISPLAY_DEBOUNCE_S = 0.2
_THINKING_ANIMATION_INTERVAL_S = 0.5
_THINKING_MIN_VISIBLE_S = 0.35


class ThinkingMixin:
    def _should_emit_reasoning_unavailable_notice(self) -> bool:
        provider = str(getattr(self.state.daemon, "provider", "") or "").strip().lower()
        tier = str(getattr(self.state.daemon, "tier", "") or "").strip().lower()
        tiers = getattr(self.state.daemon, "tiers", None)
        if not provider or not tier or not isinstance(tiers, list):
            return False
        current = next(
            (
                item
                for item in tiers
                if isinstance(item, dict)
                and str(item.get("provider", "") or "").strip().lower() == provider
                and str(item.get("name", "") or "").strip().lower() == tier
            ),
            None,
        )
        if not isinstance(current, dict):
            return False
        reasoning_mode = str(current.get("reasoning_mode", "") or "").strip().lower()
        reasoning_effort = str(current.get("reasoning_effort", "") or "").strip().lower()
        transport = str(current.get("transport", "") or "").strip().lower()
        if provider == "bedrock":
            return reasoning_mode in {"extended", "adaptive"} and reasoning_effort in {"low", "medium", "high"}
        if provider == "openai":
            return transport == "responses"
        return reasoning_effort in {"low", "medium", "high"}

    def _maybe_emit_reasoning_unavailable_notice(self) -> None:
        if bool(getattr(self, "_thinking_seen_turn", False)):
            return
        if bool(getattr(self, "_thinking_unavailable_notice_emitted_turn", False)):
            return
        if not self._should_emit_reasoning_unavailable_notice():
            return
        self._thinking_unavailable_notice_emitted_turn = True
        self._write_transcript_line("[thinking] no reasoning stream was emitted by the model for this turn.")

    def _show_thinking_indicator(self) -> None:
        from swarmee_river.tui.widgets import ThinkingIndicator

        if self._active_thinking_indicator is not None:
            return
        indicator = ThinkingIndicator()
        self._active_thinking_indicator = indicator
        self._mount_transcript_widget(indicator)

    def _clear_thinking_indicator(self) -> None:
        indicator = self._active_thinking_indicator
        self._active_thinking_indicator = None
        if indicator is None:
            return
        with contextlib.suppress(Exception):
            indicator.remove()

    def _cancel_thinking_min_visible_timer(self) -> None:
        timer = getattr(self, "_thinking_min_visible_timer", None)
        self._thinking_min_visible_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _ensure_thinking_state_active(self) -> None:
        if self._current_thinking:
            return
        self._current_thinking = True
        if not isinstance(self._thinking_started_mono, float):
            self._thinking_started_mono = time.monotonic()
        self._thinking_frame_index = 0
        self._ensure_thinking_animation_timer()

    def _cancel_thinking_display_timer(self) -> None:
        timer = self._thinking_display_timer
        self._thinking_display_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _cancel_thinking_animation_timer(self) -> None:
        timer = self._thinking_animation_timer
        self._thinking_animation_timer = None
        if timer is not None:
            with contextlib.suppress(RuntimeError):
                timer.stop()

    def _thinking_elapsed_s(self) -> float:
        started = self._thinking_started_mono
        if not isinstance(started, float):
            return 0.0
        return max(0.0, time.monotonic() - started)

    def _thinking_preview(self) -> str:
        for chunk in reversed(self._thinking_buffer):
            text = sanitize_output_text(str(chunk or "")).strip()
            if text:
                return text
        return ""

    def _render_thinking_bar(self) -> None:
        bar = self._thinking_bar
        if bar is None:
            return
        bar.show_thinking(
            char_count=max(0, int(self._thinking_char_count)),
            elapsed_s=self._thinking_elapsed_s(),
            preview=self._thinking_preview(),
            frame_index=self._thinking_frame_index,
        )

    def _on_thinking_display_timer(self) -> None:
        self._thinking_display_timer = None
        if self._current_thinking:
            self._render_thinking_bar()

    def _schedule_thinking_display_update(self) -> None:
        self._cancel_thinking_display_timer()
        self._thinking_display_timer = self.set_timer(_THINKING_DISPLAY_DEBOUNCE_S, self._on_thinking_display_timer)

    def _on_thinking_animation_tick(self) -> None:
        if not self._current_thinking:
            return
        self._thinking_frame_index = (self._thinking_frame_index + 1) % 3
        self._render_thinking_bar()

    def _ensure_thinking_animation_timer(self) -> None:
        if self._thinking_animation_timer is not None:
            return
        self._thinking_animation_timer = self.set_interval(
            _THINKING_ANIMATION_INTERVAL_S,
            self._on_thinking_animation_tick,
        )

    def _reset_thinking_state(self) -> None:
        self._cancel_thinking_display_timer()
        self._cancel_thinking_animation_timer()
        self._cancel_thinking_min_visible_timer()
        self._clear_thinking_indicator()
        self._current_thinking = False
        self._thinking_buffer = []
        self._thinking_char_count = 0
        self._thinking_started_mono = None
        self._thinking_frame_index = 0
        bar = self._thinking_bar
        if bar is not None:
            with contextlib.suppress(Exception):
                bar.hide_thinking()

    def _record_thinking_event(self, thinking_text: str) -> None:
        from swarmee_river.tui.widgets import ReasoningBlock

        chunk = sanitize_output_text(str(thinking_text or ""))
        if not self._current_thinking:
            self._ensure_thinking_state_active()
        else:
            self._current_thinking = True
        if not chunk:
            self._show_thinking_indicator()
        if chunk and self._active_reasoning_block is None:
            self._clear_thinking_indicator()
            self._active_reasoning_block = ReasoningBlock(timestamp=self._turn_timestamp())
            self._mount_transcript_widget(self._active_reasoning_block)
        if chunk:
            self._thinking_seen_turn = True
            self._thinking_buffer.append(chunk)
            self._thinking_char_count += len(chunk)
            block = self._active_reasoning_block
            if block is not None:
                with contextlib.suppress(Exception):
                    block.append_delta(chunk)
        self._render_thinking_bar()
        self._schedule_thinking_display_update()

    def _dismiss_thinking(self, *, emit_summary: bool = False) -> None:
        """Hide live thinking UI and finalize the reasoning activity block."""
        had_thinking = bool(
            self._current_thinking
            or self._thinking_started_mono is not None
            or self._thinking_char_count > 0
            or self._thinking_buffer
        )
        if not had_thinking:
            self._clear_thinking_indicator()
            bar = self._thinking_bar
            if bar is not None:
                with contextlib.suppress(Exception):
                    bar.hide_thinking()
            return

        # Keep the indicator visible briefly so fast first-delta/tool-start turns
        # still show clear "LLM in progress" feedback.
        char_count = self._thinking_char_count
        elapsed_s = self._thinking_elapsed_s()
        if char_count <= 0 and elapsed_s < _THINKING_MIN_VISIBLE_S:
            if getattr(self, "_thinking_min_visible_timer", None) is None:
                remaining = max(0.0, _THINKING_MIN_VISIBLE_S - elapsed_s)
                self._thinking_min_visible_timer = self.set_timer(
                    remaining,
                    lambda: self._dismiss_thinking(emit_summary=emit_summary),
                )
            return

        self._cancel_thinking_min_visible_timer()
        elapsed_s = self._thinking_elapsed_s()
        full_thinking = "".join(self._thinking_buffer)
        self._last_thinking_text = full_thinking
        char_count = self._thinking_char_count
        if emit_summary and char_count > 0:
            elapsed_label = max(0, int(round(elapsed_s)))
            summary_line = f"💭 Reasoning ({elapsed_label}s, {char_count:,} chars)"
            self._record_transcript_fallback(summary_line)
        block = self._active_reasoning_block
        if block is not None:
            with contextlib.suppress(Exception):
                block.finalize(elapsed_s=elapsed_s)
            self._active_reasoning_block = None

        self._reset_thinking_state()
