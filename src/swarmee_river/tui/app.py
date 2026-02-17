"""Optional Textual app scaffold for `swarmee tui`."""

from __future__ import annotations

import contextlib
import importlib
import os
import signal
import subprocess
import sys
import threading
from typing import Any

_CONSENT_CHOICES = {"y", "n", "a", "v"}


def build_swarmee_cmd(prompt: str, *, auto_approve: bool) -> list[str]:
    """Build a subprocess command for a one-shot Swarmee run."""
    command = [sys.executable, "-u", "-m", "swarmee_river.swarmee"]
    if auto_approve:
        command.append("--yes")
    command.append(prompt)
    return command


def extract_plan_section(output: str) -> str | None:
    """Extract the one-shot plan block beginning at 'Proposed plan:' if present."""
    marker = "Proposed plan:"
    marker_index = output.find(marker)
    if marker_index < 0:
        return None

    trailing_hint_prefix = "Plan generated. Re-run with --yes"
    candidate = output[marker_index:]
    extracted_lines: list[str] = []
    for line in candidate.splitlines():
        if not extracted_lines:
            local_index = line.find(marker)
            if local_index >= 0:
                line = line[local_index:]
        if line.strip().startswith(trailing_hint_prefix):
            break
        extracted_lines.append(line.rstrip())

    while extracted_lines and not extracted_lines[-1].strip():
        extracted_lines.pop()

    if not extracted_lines:
        return None

    extracted = "\n".join(extracted_lines).strip()
    return extracted or None


def looks_like_plan_output(text: str) -> bool:
    """Detect whether one-shot output likely contains a generated plan."""
    return extract_plan_section(text) is not None


def render_tui_hint_after_plan() -> str:
    """Hint shown when a plan-only run is detected."""
    return "Plan detected. Type /approve to execute, /replan to regenerate, /clearplan to clear."


def detect_consent_prompt(line: str) -> str | None:
    """Detect consent-related subprocess output lines."""
    normalized = line.strip().lower()
    if "~ consent>" in normalized:
        return "prompt"
    if "allow tool '" in normalized:
        return "header"
    return None


def update_consent_capture(
    consent_active: bool,
    consent_buffer: list[str],
    line: str,
    *,
    max_lines: int = 20,
) -> tuple[bool, list[str]]:
    """Update consent capture state from a single output line."""
    kind = detect_consent_prompt(line)
    if kind is None and not consent_active:
        return consent_active, consent_buffer

    updated = list(consent_buffer)
    updated.append(line.rstrip("\n"))
    if len(updated) > max_lines:
        updated = updated[-max_lines:]
    return True, updated


def write_to_proc(proc: subprocess.Popen[str], text: str) -> bool:
    """Write a response line to a subprocess stdin."""
    if proc.stdin is None:
        return False

    payload = text if text.endswith("\n") else f"{text}\n"
    try:
        proc.stdin.write(payload)
        proc.stdin.flush()
    except Exception:
        return False
    return True


def spawn_swarmee(prompt: str, *, auto_approve: bool) -> subprocess.Popen[str]:
    """Spawn Swarmee as a subprocess with line-buffered merged output."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        build_swarmee_cmd(prompt, auto_approve=auto_approve),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        env=env,
    )


def stop_process(proc: subprocess.Popen[str], *, timeout_s: float = 2.0) -> None:
    """Stop a running subprocess, escalating from interrupt to terminate to kill."""
    if proc.poll() is not None:
        return

    if os.name == "posix" and hasattr(signal, "SIGINT"):
        with contextlib.suppress(Exception):
            proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=timeout_s)
            return
        except subprocess.TimeoutExpired:
            pass

    with contextlib.suppress(Exception):
        proc.terminate()
    try:
        proc.wait(timeout=timeout_s)
        return
    except subprocess.TimeoutExpired:
        pass
    except Exception:
        return

    with contextlib.suppress(Exception):
        proc.kill()
    with contextlib.suppress(Exception):
        proc.wait(timeout=timeout_s)


def run_tui() -> int:
    """Run the full-screen TUI if Textual is installed."""
    try:
        textual_app = importlib.import_module("textual.app")
        textual_containers = importlib.import_module("textual.containers")
        textual_widgets = importlib.import_module("textual.widgets")
    except ImportError:
        print(
            "Textual is required for `swarmee tui`.\n"
            'Install with: pip install "swarmee-river[tui]"\n'
            'For editable installs in this repo: pip install -e ".[tui]"',
            file=sys.stderr,
        )
        return 1

    AppBase = textual_app.App
    Horizontal = textual_containers.Horizontal
    Vertical = textual_containers.Vertical
    Header = textual_widgets.Header
    Footer = textual_widgets.Footer
    Input = textual_widgets.Input
    RichLog = textual_widgets.RichLog

    class SwarmeeTUI(AppBase):
        CSS = """
        Screen {
            layout: vertical;
        }

        #panes {
            layout: horizontal;
            height: 1fr;
            width: 100%;
        }

        #transcript {
            width: 2fr;
            height: 1fr;
            border: round $accent;
            padding: 0 1;
        }

        #plan {
            height: 1fr;
            border: round $accent;
            padding: 0 1;
        }

        #consent {
            height: 1fr;
            border: round $warning;
            padding: 0 1;
        }

        #side {
            width: 1fr;
            height: 1fr;
            layout: vertical;
        }

        #prompt {
            dock: bottom;
        }
        """

        _proc: subprocess.Popen[str] | None = None
        _runner_thread: threading.Thread | None = None
        _last_prompt: str | None = None
        _pending_plan_prompt: str | None = None
        _last_run_auto_approve: bool = False
        _consent_active: bool = False
        _consent_buffer: list[str] = []

        def compose(self) -> Any:
            yield Header()
            with Horizontal(id="panes"):
                yield RichLog(id="transcript", auto_scroll=True, wrap=True)
                with Vertical(id="side"):
                    yield RichLog(id="plan", auto_scroll=True, wrap=True)
                    yield RichLog(id="consent", auto_scroll=True, wrap=True)
            yield Input(placeholder="Type prompt (default plan-first; /run to execute)", id="prompt")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#prompt", Input).focus()
            self._reset_plan_panel()
            self._reset_consent_panel()
            self._write_transcript("Swarmee TUI ready. Enter a prompt to run Swarmee.")
            self._write_transcript(
                "Commands: /plan, /run, /approve, /replan, /clearplan, /consent <y|n|a|v>, /stop, /exit."
            )

        def _write_transcript(self, line: str) -> None:
            self.query_one("#transcript", RichLog).write(line)

        def _set_plan_panel(self, content: str) -> None:
            plan_panel = self.query_one("#plan", RichLog)
            plan_panel.clear()
            for line in content.splitlines():
                plan_panel.write(line)
            if not content.strip():
                plan_panel.write("(no plan)")

        def _reset_plan_panel(self) -> None:
            self._set_plan_panel("(no plan)")

        def _render_consent_panel(self) -> None:
            consent_panel = self.query_one("#consent", RichLog)
            consent_panel.clear()
            if not self._consent_active:
                consent_panel.write("(no active consent prompt)")
                return
            for line in self._consent_buffer[-10:]:
                consent_panel.write(line)
            consent_panel.write("")
            consent_panel.write("[y] yes  [n] no  [a] always(session)  [v] never(session)")

        def _reset_consent_panel(self) -> None:
            self._consent_active = False
            self._consent_buffer = []
            self._render_consent_panel()

        def _apply_consent_capture(self, line: str) -> None:
            next_active, next_buffer = update_consent_capture(
                self._consent_active,
                self._consent_buffer,
                line,
                max_lines=20,
            )
            if next_active != self._consent_active or next_buffer != self._consent_buffer:
                self._consent_active = next_active
                self._consent_buffer = next_buffer
                self._render_consent_panel()

        def _submit_consent_choice(self, choice: str) -> None:
            normalized_choice = choice.strip().lower()
            if normalized_choice not in _CONSENT_CHOICES:
                self._write_transcript("Usage: /consent <y|n|a|v>")
                return
            if not self._consent_active:
                self._write_transcript("[consent] no active prompt.")
                return
            if self._proc is None or self._proc.poll() is not None:
                self._write_transcript("[consent] process is not running.")
                self._reset_consent_panel()
                return
            if not write_to_proc(self._proc, normalized_choice):
                self._write_transcript("[consent] failed to send response (stdin unavailable).")
                self._reset_consent_panel()
                return
            self._write_transcript(f"[consent] sent '{normalized_choice}'.")
            self._reset_consent_panel()
            self.query_one("#prompt", Input).focus()

        def _handle_output_line(self, line: str) -> None:
            self._write_transcript(line)
            self._apply_consent_capture(line)

        def _finalize_run(
            self,
            proc: subprocess.Popen[str],
            *,
            return_code: int,
            prompt: str,
            output_text: str,
        ) -> None:
            if self._proc is not proc:
                return
            self._write_transcript(f"[run] exited with code {return_code}.")
            extracted_plan = extract_plan_section(output_text)
            if extracted_plan:
                self._pending_plan_prompt = prompt
                self._set_plan_panel(extracted_plan)
                self._write_transcript(render_tui_hint_after_plan())
            self._reset_consent_panel()
            self._proc = None
            self._runner_thread = None

        def _stream_output(self, proc: subprocess.Popen[str], prompt: str) -> None:
            output_chunks: list[str] = []
            if proc.stdout is None:
                self.call_from_thread(self._write_transcript, "[run] error: subprocess stdout unavailable.")
                return_code = proc.poll()
                self.call_from_thread(
                    self._finalize_run,
                    proc,
                    return_code=(return_code if return_code is not None else 1),
                    prompt=prompt,
                    output_text="",
                )
                return

            try:
                for raw_line in proc.stdout:
                    output_chunks.append(raw_line)
                    self.call_from_thread(self._handle_output_line, raw_line.rstrip("\n"))
            except Exception as exc:
                self.call_from_thread(self._write_transcript, f"[run] output stream error: {exc}")
            finally:
                with contextlib.suppress(Exception):
                    proc.stdout.close()
                return_code = proc.wait()
                self.call_from_thread(
                    self._finalize_run,
                    proc,
                    return_code=return_code,
                    prompt=prompt,
                    output_text="".join(output_chunks),
                )

        def _start_run(self, prompt: str, *, auto_approve: bool) -> None:
            self._pending_plan_prompt = None
            self._reset_consent_panel()
            try:
                proc = spawn_swarmee(prompt, auto_approve=auto_approve)
            except Exception as exc:
                self._write_transcript(f"[run] failed to start: {exc}")
                return

            self._proc = proc
            self._last_prompt = prompt
            self._last_run_auto_approve = auto_approve
            pid = proc.pid if proc.pid is not None else "unknown"
            mode = "execute" if auto_approve else "plan"
            self._write_transcript(f"[run] started ({mode}) pid={pid}.")
            self._runner_thread = threading.Thread(
                target=self._stream_output,
                args=(proc, prompt),
                daemon=True,
                name="swarmee-tui-runner",
            )
            self._runner_thread.start()

        def _stop_run(self) -> None:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                self._write_transcript("[run] no active run.")
                if proc is not None and proc.poll() is not None:
                    self._proc = None
                return
            stop_process(proc)
            self._write_transcript("[run] stopped.")

        def on_key(self, event: Any) -> None:
            if not self._consent_active:
                return
            key = str(getattr(event, "key", "")).lower()
            if key in _CONSENT_CHOICES:
                event.stop()
                event.prevent_default()
                self._submit_consent_choice(key)

        def on_input_submitted(self, event: Any) -> None:
            text = event.value.strip()
            event.input.value = ""
            if not text:
                return

            self._write_transcript(f"> {text}")
            normalized = text.lower()

            if normalized in {"/stop", ":stop"}:
                self._stop_run()
                return

            if normalized in {"/exit", ":exit"}:
                if self._proc is not None and self._proc.poll() is None:
                    stop_process(self._proc)
                    self._write_transcript("[run] stopped.")
                self.exit(return_code=0)
                return

            if normalized == "/consent":
                self._write_transcript("Usage: /consent <y|n|a|v>")
                return

            if normalized.startswith("/consent "):
                choice = normalized.split(maxsplit=1)[1].strip()
                self._submit_consent_choice(choice)
                return

            if self._proc is not None and self._proc.poll() is None:
                self._write_transcript("[run] already running; use /stop.")
                return

            if normalized == "/approve":
                if not self._pending_plan_prompt:
                    self._write_transcript("[run] no pending plan.")
                    return
                self._start_run(self._pending_plan_prompt, auto_approve=True)
                return

            if normalized == "/replan":
                if not self._last_prompt:
                    self._write_transcript("[run] no previous prompt to replan.")
                    return
                self._start_run(self._last_prompt, auto_approve=False)
                return

            if normalized == "/clearplan":
                self._pending_plan_prompt = None
                self._reset_plan_panel()
                self._write_transcript("[run] plan cleared.")
                return

            if normalized == "/plan":
                self._write_transcript("Usage: /plan <prompt>")
                return

            if text.startswith("/plan "):
                prompt = text[len("/plan ") :].strip()
                if not prompt:
                    self._write_transcript("Usage: /plan <prompt>")
                    return
                self._start_run(prompt, auto_approve=False)
                return

            if normalized == "/run":
                self._write_transcript("Usage: /run <prompt>")
                return

            if text.startswith("/run "):
                prompt = text[len("/run ") :].strip()
                if not prompt:
                    self._write_transcript("Usage: /run <prompt>")
                    return
                self._start_run(prompt, auto_approve=True)
                return

            if text.startswith("/") or text.startswith(":"):
                self._write_transcript(f"[run] unknown command: {text}")
                return

            self._start_run(text, auto_approve=False)

    try:
        SwarmeeTUI().run()
    except KeyboardInterrupt:
        return 130
    return 0
