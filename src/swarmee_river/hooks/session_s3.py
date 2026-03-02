from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

from strands.hooks import HookProvider, HookRegistry
from strands.hooks.events import AfterInvocationEvent

from swarmee_river.hooks._compat import register_hook_callback
from tools.session_s3 import export_session_to_s3, promote_session_to_kb

logger = logging.getLogger(__name__)


def _truthy_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "t", "yes", "y", "on", "enabled", "enable"}


class SessionS3Hooks(HookProvider):
    """Best-effort background export/promotion hook for persisted sessions."""

    def __init__(self, *, debounce_seconds: int = 30, promote_debounce_seconds: int = 300) -> None:
        self.bucket = (os.getenv("SWARMEE_SESSION_S3_BUCKET") or "").strip()
        self.prefix = (os.getenv("SWARMEE_SESSION_S3_PREFIX") or "swarmee/sessions/").strip()
        self.auto_export = _truthy_env("SWARMEE_SESSION_S3_AUTO_EXPORT", False)
        self.promote_on_complete = _truthy_env("SWARMEE_SESSION_KB_PROMOTE_ON_COMPLETE", False)

        self.enabled = bool(self.bucket) and (self.auto_export or self.promote_on_complete)
        self.debounce_seconds = max(1, int(debounce_seconds))
        self.promote_debounce_seconds = max(10, int(promote_debounce_seconds))

        self._lock = threading.Lock()
        self._last_export_at = 0.0
        self._last_promote_at = 0.0

    def register_hooks(self, registry: HookRegistry, **_: Any) -> None:
        register_hook_callback(registry, AfterInvocationEvent, self.after_invocation)

    def _resolve_session_id(self, event: AfterInvocationEvent) -> str:
        sid = (os.getenv("SWARMEE_SESSION_ID") or "").strip()
        if sid:
            return sid

        invocation_state = getattr(event, "invocation_state", None)
        if isinstance(invocation_state, dict):
            sw = invocation_state.get("swarmee")
            if isinstance(sw, dict):
                fallback = str(sw.get("session_id") or "").strip()
                if fallback:
                    return fallback
        return ""

    def after_invocation(self, event: AfterInvocationEvent) -> None:
        if not self.enabled:
            return

        session_id = self._resolve_session_id(event)
        if not session_id:
            return

        now = time.time()
        should_export = False
        should_promote = False

        with self._lock:
            if self.auto_export and (now - self._last_export_at) >= float(self.debounce_seconds):
                self._last_export_at = now
                should_export = True

            if self.promote_on_complete and (now - self._last_promote_at) >= float(self.promote_debounce_seconds):
                self._last_promote_at = now
                should_promote = True

        if not should_export and not should_promote:
            return

        thread = threading.Thread(
            target=self._run_background,
            kwargs={
                "session_id": session_id,
                "run_export": should_export,
                "run_promote": should_promote,
            },
            daemon=True,
            name="swarmee-session-s3-hook",
        )
        thread.start()

    def _run_background(self, *, session_id: str, run_export: bool, run_promote: bool) -> None:
        if run_export:
            try:
                export_session_to_s3(
                    session_id=session_id,
                    s3_bucket=self.bucket,
                    s3_prefix=self.prefix,
                )
            except Exception as exc:
                logger.debug("SessionS3Hooks export failed for %s: %s", session_id, exc)

        if run_promote:
            kb_id = (os.getenv("SWARMEE_KNOWLEDGE_BASE_ID") or os.getenv("STRANDS_KNOWLEDGE_BASE_ID") or "").strip()
            if not kb_id:
                return
            try:
                promote_session_to_kb(
                    session_id=session_id,
                    knowledge_base_id=kb_id,
                    content_filter="all",
                )
            except Exception as exc:
                logger.debug("SessionS3Hooks promote failed for %s: %s", session_id, exc)
