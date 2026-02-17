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


def spawn_swarmee(prompt: str, *, auto_approve: bool) -> subprocess.Popen[str]:
    """Spawn Swarmee as a subprocess with line-buffered merged output."""
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        build_swarmee_cmd(prompt, auto_approve=auto_approve),
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
            width: 1fr;
            height: 1fr;
            border: round $accent;
            padding: 0 1;
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

        def compose(self) -> Any:
            yield Header()
            with Horizontal(id="panes"):
                yield RichLog(id="transcript", auto_scroll=True, wrap=True)
                yield RichLog(id="plan", auto_scroll=True, wrap=True)
            yield Input(placeholder="Type prompt (default plan-first; /run to execute)", id="prompt")
            yield Footer()

        def on_mount(self) -> None:
            self.query_one("#prompt", Input).focus()
            self._reset_plan_panel()
            self._write_transcript("Swarmee TUI ready. Enter a prompt to run Swarmee.")
            self._write_transcript("Commands: /plan, /run, /approve, /replan, /clearplan, /stop, /exit.")

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
                    self.call_from_thread(self._write_transcript, raw_line.rstrip("\n"))
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
