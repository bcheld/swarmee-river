from __future__ import annotations

import contextlib
from typing import Any


class PlanMixin:
    def _set_plan_panel(self, content: str) -> None:
        from textual.widgets import TextArea

        self.state.plan.text = content
        plan_panel = self.query_one("#plan", TextArea)
        text = content if content.strip() else "(no plan)"
        plan_panel.load_text(text)
        plan_panel.scroll_end(animate=False)

    def _extract_plan_step_descriptions(self, plan_json: dict[str, Any]) -> list[str]:
        steps_raw = plan_json.get("steps", [])
        if not isinstance(steps_raw, list):
            return []
        steps: list[str] = []
        for step in steps_raw:
            if isinstance(step, str):
                desc = step.strip()
            elif isinstance(step, dict):
                desc = str(step.get("description", step.get("title", step))).strip()
            else:
                desc = str(step).strip()
            if desc:
                steps.append(desc)
        return steps

    def _refresh_plan_status_bar(self) -> None:
        if self._status_bar is None:
            return
        if not self.state.daemon.query_active:
            self._status_bar.set_plan_step(current=None, total=None)
            return
        total = self.state.plan.current_steps_total
        if total <= 0:
            self._status_bar.set_plan_step(current=None, total=None)
            return
        current: int | None = None
        if isinstance(self.state.plan.current_active_step, int) and self.state.plan.current_active_step >= 0:
            current = self.state.plan.current_active_step + 1
        else:
            completed = sum(1 for item in self.state.plan.current_step_statuses if item == "completed")
            if completed >= total:
                current = total
            elif completed > 0:
                current = completed
        self._status_bar.set_plan_step(current=current, total=total)

    def _render_plan_panel_from_status(self) -> None:
        if self.state.plan.current_steps_total <= 0 or not self.state.plan.current_steps:
            return
        text_lines: list[str] = []
        if self.state.plan.current_summary:
            text_lines.append(self.state.plan.current_summary)
            text_lines.append("")
        for index, desc in enumerate(self.state.plan.current_steps, start=1):
            status = (
                self.state.plan.current_step_statuses[index - 1]
                if index - 1 < len(self.state.plan.current_step_statuses)
                else "pending"
            )
            marker = "☐"
            if status == "in_progress":
                marker = "▶"
            elif status == "completed":
                marker = "☑"
            text_lines.append(f"{marker} {index}. {desc}")
        text_lines.append("")
        text_lines.append("/approve  /replan  /clearplan")
        self._set_plan_panel("\n".join(text_lines))
        self._refresh_plan_status_bar()

    def _populate_planning_view(self, plan_json: dict[str, Any]) -> None:
        """Populate the interactive planning view with PlanStepRow widgets."""
        import contextlib as _ctx

        from swarmee_river.tui.widgets import PlanStepRow

        self.state.plan.plan_json = dict(plan_json)

        # Render summary + assumptions + questions
        summary_widget = self._engage_plan_summary
        if summary_widget is not None:
            summary_lines: list[str] = []
            summary_text = str(plan_json.get("summary", "")).strip()
            if summary_text:
                summary_lines.append(f"[bold]Summary:[/bold] {summary_text}")
            assumptions = plan_json.get("assumptions", [])
            if isinstance(assumptions, list) and assumptions:
                summary_lines.append("")
                summary_lines.append("[bold]Assumptions:[/bold]")
                for assumption in assumptions[:5]:
                    summary_lines.append(f"  - {assumption}")
            questions = plan_json.get("questions", [])
            if isinstance(questions, list) and questions:
                summary_lines.append("")
                summary_lines.append("[bold yellow]Questions:[/bold yellow]")
                for question in questions[:5]:
                    summary_lines.append(f"  ? {question}")
            summary_widget.update("\n".join(summary_lines) if summary_lines else "")

        # Clear existing plan step rows
        container = self._engage_plan_items
        if container is None:
            with _ctx.suppress(Exception):
                from textual.containers import VerticalScroll
                container = self.query_one("#engage_plan_items", VerticalScroll)
                self._engage_plan_items = container
        if container is None:
            return
        for child in list(container.children):
            with _ctx.suppress(Exception):
                child.remove()

        # Mount PlanStepRow widgets
        steps = plan_json.get("steps", [])
        if not isinstance(steps, list) or not steps:
            from textual.widgets import Static as _Static
            container.mount(_Static("[dim](no steps in plan)[/dim]"))
            return
        for index, step in enumerate(steps):
            if isinstance(step, dict):
                desc = str(step.get("description", step.get("title", str(step)))).strip()
                files_to_edit = step.get("files_to_edit", [])
                files_to_read = step.get("files_to_read", [])
                tools_expected = step.get("tools_expected", [])
                risks = step.get("risks", [])
            else:
                desc = str(step).strip()
                files_to_edit = []
                files_to_read = []
                tools_expected = []
                risks = []
            row = PlanStepRow(
                step_index=index,
                description=desc,
                files_to_edit=files_to_edit,
                files_to_read=files_to_read,
                tools_expected=tools_expected,
                risks=risks,
                id=f"plan_step_row_{index}",
            )
            container.mount(row)

        # Toggle button visibility: hide "Start Plan", show "Continue"
        with _ctx.suppress(Exception):
            from textual.widgets import Button
            self.query_one("#engage_start_plan", Button).styles.display = "none"
        with _ctx.suppress(Exception):
            from textual.widgets import Button
            self.query_one("#engage_continue_plan", Button).styles.display = "block"
        # Update header
        with _ctx.suppress(Exception):
            from textual.widgets import Static as _Static
            self.query_one("#engage_planning_header", _Static).update(
                "Review the plan below. Uncheck steps to exclude,\n"
                "add comments to request changes, then press Continue."
            )

    def _handle_planning_continue(self) -> None:
        """Process the Continue action in the Planning view."""
        import contextlib as _ctx

        from swarmee_river.tui.widgets import PlanStepRow

        container = self._engage_plan_items
        if container is None:
            return

        rows: list[PlanStepRow] = [
            child for child in container.children if isinstance(child, PlanStepRow)
        ]
        if not rows:
            self._write_transcript_line("[plan] no plan steps to process.")
            return

        all_included = True
        has_comments = False
        feedback_parts: list[str] = []

        for row in rows:
            included = row.is_included
            comment = row.comment
            if not included:
                all_included = False
                feedback_parts.append(
                    f"- Step {row.step_index + 1}: EXCLUDED"
                    + (f" (reason: {comment})" if comment else "")
                )
            elif comment:
                has_comments = True
                feedback_parts.append(
                    f"- Step {row.step_index + 1}: MODIFY ({comment})"
                )

        if all_included and not has_comments:
            # All steps approved — approve and execute
            self._restore_planning_sidebar_width()
            self._set_engage_view_mode("execution")
            if self.state.plan.pending_prompt:
                self._dispatch_plan_action("approve")
            else:
                self._write_transcript_line("[plan] plan finalized. Enter a prompt to execute.")
            return

        # Build annotated feedback prompt for refinement
        plan_json = self.state.plan.plan_json
        original_summary = (
            str((plan_json or {}).get("summary", "")).strip() if plan_json else ""
        )
        feedback_prompt = (
            f"Revise the previous plan"
            + (f" ({original_summary})" if original_summary else "")
            + " based on user feedback:\n"
            + "\n".join(feedback_parts)
        )
        self._write_transcript_line("[plan] Sending refinement feedback...")
        self._start_run(feedback_prompt, auto_approve=False, mode="plan")

    def _restore_planning_sidebar_width(self) -> None:
        """Restore sidebar width saved before planning expansion."""
        saved = self.state.plan.pre_planning_split_ratio
        if saved is not None:
            self._split_ratio = max(1, min(4, int(saved)))
            self.state.plan.pre_planning_split_ratio = None
            self._apply_split_ratio()
        else:
            while self._split_ratio < 2:
                self.action_widen_transcript()

    def _clear_planning_view(self) -> None:
        """Reset the interactive planning view to its empty state."""
        import contextlib as _ctx
        if self._engage_plan_summary is not None:
            with _ctx.suppress(Exception):
                self._engage_plan_summary.update("")
        container = self._engage_plan_items
        if container is not None:
            for child in list(container.children):
                with _ctx.suppress(Exception):
                    child.remove()
        with _ctx.suppress(Exception):
            from textual.widgets import Button
            self.query_one("#engage_start_plan", Button).styles.display = "block"
        with _ctx.suppress(Exception):
            from textual.widgets import Button
            self.query_one("#engage_continue_plan", Button).styles.display = "none"
        with _ctx.suppress(Exception):
            from textual.widgets import Static as _Static
            self.query_one("#engage_planning_header", _Static).update(
                "Describe what you want to build. The orchestrator will\n"
                "develop a plan you can review and refine."
            )

    def action_copy_plan(self) -> None:
        self._copy_text((self.state.plan.text or "").rstrip() + "\n", label="plan")

    def _dispatch_plan_action(self, action: str) -> None:
        normalized = action.strip().lower()
        if normalized == "approve":
            if not self.state.plan.pending_prompt:
                self._write_transcript_line("[run] no pending plan.")
                return
            self._start_run(self.state.plan.pending_prompt, auto_approve=True, mode="execute")
            return
        if normalized == "replan":
            if not self._last_prompt:
                self._write_transcript_line("[run] no previous prompt to replan.")
                return
            self._start_run(self._last_prompt, auto_approve=False, mode="plan")
            return
        if normalized == "clearplan":
            self.state.plan.pending_prompt = None
            self._reset_plan_panel()
            self._write_transcript_line("[run] plan cleared.")
            return
