"""Ingest-cursor reconciliation and replay-identity for the LCM engine (WS5 Seam 4).

The ``ReconcileMixin`` holds the machinery that reconciles the persisted store
tail against the active message list after a process restart, plus the stable
replay-identity primitives it relies on. These methods were lifted verbatim out
of ``LCMEngine`` and continue to run bound to the engine instance (``self`` is
the ``LCMEngine``), so they read the engine's runtime state (``_store``,
``_session_id``, ``_config``, ``_ingest_cursor`` is written by the engine from
the value these return) and call back into engine helpers through normal
attribute lookup. ``LCMEngine`` mixes this in, so no call site and no test
changes.

``_PRESERVED_OBJECTIVE_CONTEXT_PREFIX`` lives here (used by the reconciliation
scan) and is re-exported to ``engine.py``; the two tool-call-identity
staticmethods reference the mixin class directly rather than ``LCMEngine`` to
avoid an import cycle (staticmethod resolution is identical).
"""

from __future__ import annotations

import json
import re
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Optional

from .externalize import (
    extract_externalized_ref,
    externalized_tool_result_has_persisted_output_marker,
    find_externalized_tool_result_content_for_call,
    load_externalized_payload,
)
from .ingest_protection import (
    _add_inline_persisted_output_generation_metadata,
    _add_inline_persisted_output_identity_metadata,
    _expected_persisted_output_chars,
    _has_inline_persisted_output_generation_metadata,
    _has_lossy_sensitive_redaction,
    _is_hermes_persisted_output_marker,
    _json_has_duplicate_object_keys,
    _persisted_output_marker_identity_digest,
    _persisted_output_saved_path,
    recover_hermes_persisted_output_with_file_stat,
    redact_sensitive_value,
)
from .message_content import normalize_content_value, text_content_for_pattern_matching
from .sanitize import _clean_active_assistant_message

import logging

logger = logging.getLogger(__name__)

_PRESERVED_OBJECTIVE_CONTEXT_PREFIX = "[Current user objective preserved from compacted history]"


class ReconcileMixin:
    @staticmethod
    def _canonicalize_tool_call_identity_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: ReconcileMixin._canonicalize_tool_call_identity_value(val)
                for key, val in value.items()
            }
        if isinstance(value, list):
            return [ReconcileMixin._canonicalize_tool_call_identity_value(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if stripped and stripped[0] in "[{":
                if _json_has_duplicate_object_keys(value):
                    return value
                try:
                    parsed = json.loads(value)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return value
                if isinstance(parsed, (dict, list)):
                    canonical = ReconcileMixin._canonicalize_tool_call_identity_value(parsed)
                    return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return value
        return value

    @staticmethod
    def _stable_tool_calls_identity(tool_calls: Any) -> str:
        if not tool_calls:
            return ""
        try:
            canonical = ReconcileMixin._canonicalize_tool_call_identity_value(tool_calls)
            return json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        except (TypeError, ValueError):
            return str(tool_calls)

    def _has_durable_persisted_output_replay_identity(
        self,
        msg: Dict[str, Any],
        *,
        session_id: str | None = None,
        conversation_id: str | None = None,
    ) -> bool:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        resolved_session_id = str(
            session_id or msg.get("session_id") or getattr(self, "_session_id", None) or ""
        )
        resolved_conversation_id = (
            conversation_id
            if conversation_id is not None
            else getattr(self, "_conversation_id", None)
        )
        if role != "tool" or not _is_hermes_persisted_output_marker(content):
            return False
        expected_chars = _expected_persisted_output_chars(content)
        persisted_output_source_path = _persisted_output_saved_path(content)
        persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
        if (
            expected_chars is None
            or not persisted_output_source_path
            or not persisted_output_preview_sha256
        ):
            return False
        recovered_with_stat = recover_hermes_persisted_output_with_file_stat(content)
        if recovered_with_stat is None:
            return False
        require_live_file_freshness = True
        durable_content = find_externalized_tool_result_content_for_call(
            tool_call_id=str(msg.get("tool_call_id") or ""),
            session_id=resolved_session_id,
            expected_chars=expected_chars,
            persisted_output_source_path=persisted_output_source_path,
            persisted_output_preview_sha256=persisted_output_preview_sha256,
            require_persisted_output_file_not_newer=require_live_file_freshness,
            allow_redacted_preview_match=allow_redacted_preview_match,
            config=self._config,
            hermes_home=self._hermes_home,
        )
        recovered_content, recovered_generation = recovered_with_stat
        recovered_identity_content = normalize_content_value(
            redact_sensitive_value(
                recovered_content,
                self._config,
                parse_json_strings=False,
            )
        )
        if _has_lossy_sensitive_redaction(recovered_identity_content):
            # Lossy content alone cannot distinguish legitimate retries. An
            # exact live-file generation is sufficient provenance without
            # retaining a potentially sensitive raw-preview digest.
            exact_generation_content = find_externalized_tool_result_content_for_call(
                tool_call_id=str(msg.get("tool_call_id") or ""),
                session_id=resolved_session_id,
                expected_chars=expected_chars,
                persisted_output_source_path=persisted_output_source_path,
                persisted_output_file_size=recovered_generation["size"],
                persisted_output_file_mtime_ns=recovered_generation["mtime_ns"],
                persisted_output_file_ctime_ns=recovered_generation["ctime_ns"],
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if (
                exact_generation_content is not None
                and _has_lossy_sensitive_redaction(exact_generation_content)
                and self._recovered_content_matches_durable_identity(
                    recovered_content,
                    exact_generation_content,
                )
            ):
                return True
        if durable_content is not None:
            if _has_lossy_sensitive_redaction(durable_content):
                return False
            if self._recovered_content_matches_durable_identity(
                recovered_content,
                durable_content,
            ):
                return True

        # Older payloads created from raw tool content have no persisted-source
        # provenance. Resolve those rows by the durable session and call ID,
        # then require exact recovered-content identity.
        call_id = str(msg.get("tool_call_id") or "").strip()
        if not call_id:
            return False
        durable_rows = self._store.get_tool_call_replay_neighborhoods(
            resolved_session_id,
            {call_id},
            conversation_id=resolved_conversation_id,
        )
        for durable_row in durable_rows:
            if (
                str(durable_row.get("role") or "") != "tool"
                or str(durable_row.get("tool_call_id") or "").strip() != call_id
            ):
                continue
            durable_identity_content = self._message_replay_identity(
                durable_row,
                stored_row=True,
            )[1]
            if _has_lossy_sensitive_redaction(durable_identity_content):
                continue
            if self._recovered_content_matches_durable_identity(
                recovered_content,
                durable_identity_content,
            ):
                return True
        return False

    def _message_replay_identity(
        self,
        msg: Dict[str, Any],
        *,
        stored_row: bool = False,
        inferred_tool_name: str | None = None,
    ) -> tuple[str, str, str, str, str]:
        role = str(msg.get("role") or "unknown")
        content = normalize_content_value(msg.get("content")) or ""
        if (
            role == "tool"
            and _is_hermes_persisted_output_marker(content)
            and bool(getattr(self._config, "large_output_externalization_enabled", True))
        ):
            expected_chars = _expected_persisted_output_chars(content)
            persisted_output_source_path = _persisted_output_saved_path(content)
            persisted_output_preview_sha256, allow_redacted_preview_match = self._persisted_output_marker_replay_proof(content)
            durable_content = None
            recovered_with_stat = recover_hermes_persisted_output_with_file_stat(content) if not stored_row else None
            recovered_content = recovered_with_stat[0] if recovered_with_stat is not None else None
            recovered_identity_content = None
            if recovered_content is not None:
                recovered_identity_content = normalize_content_value(
                    redact_sensitive_value(
                        recovered_content,
                        self._config,
                        parse_json_strings=False,
                    )
                )
            require_live_file_freshness = recovered_with_stat is not None

            def live_file_generation_identity() -> str:
                try:
                    live_stat = Path(str(persisted_output_source_path)).stat()
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"mtime_ns={live_stat.st_mtime_ns}; "
                        f"chars={expected_chars}]"
                    )
                except OSError:
                    return (
                        "[LCM persisted-output live file: "
                        f"path={persisted_output_source_path}; "
                        f"chars={expected_chars}]"
                    )

            if (
                not stored_row
                and expected_chars is not None
                and persisted_output_source_path
                and persisted_output_preview_sha256
                and recovered_with_stat is not None
            ):
                durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    require_persisted_output_file_not_newer=require_live_file_freshness,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
            exact_generation_content = None
            if (
                not stored_row
                and expected_chars is not None
                and persisted_output_source_path
                and recovered_with_stat is not None
                and _has_lossy_sensitive_redaction(recovered_identity_content)
            ):
                recovered_generation = recovered_with_stat[1]
                exact_generation_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_file_size=recovered_generation["size"],
                    persisted_output_file_mtime_ns=recovered_generation["mtime_ns"],
                    persisted_output_file_ctime_ns=recovered_generation["ctime_ns"],
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
            if (
                exact_generation_content is not None
                and _has_lossy_sensitive_redaction(exact_generation_content)
                and recovered_content is not None
                and self._recovered_content_matches_durable_identity(
                    recovered_content,
                    exact_generation_content,
                )
            ):
                content = exact_generation_content
            elif (
                durable_content is not None
                and not _has_lossy_sensitive_redaction(durable_content)
                and not _has_lossy_sensitive_redaction(recovered_identity_content)
                and (
                    recovered_content is None
                    or self._recovered_content_matches_durable_identity(recovered_content, durable_content)
                )
            ):
                assert durable_content is not None
                content = durable_content
            elif recovered_content is not None:
                stale_durable_content = find_externalized_tool_result_content_for_call(
                    tool_call_id=str(msg.get("tool_call_id") or ""),
                    session_id=str(msg.get("session_id") or self._session_id or ""),
                    expected_chars=expected_chars,
                    persisted_output_source_path=persisted_output_source_path,
                    persisted_output_preview_sha256=persisted_output_preview_sha256,
                    allow_redacted_preview_match=allow_redacted_preview_match,
                    config=self._config,
                    hermes_home=self._hermes_home,
                )
                if (
                    stale_durable_content is not None
                    and self._recovered_content_matches_durable_identity(recovered_content, stale_durable_content)
                    and not _has_lossy_sensitive_redaction(stale_durable_content)
                    and not _has_lossy_sensitive_redaction(recovered_identity_content)
                ):
                    content = stale_durable_content
                elif stale_durable_content is not None:
                    content = live_file_generation_identity()
                elif recovered_with_stat is not None:
                    content = _add_inline_persisted_output_generation_metadata(
                        _add_inline_persisted_output_identity_metadata(
                            content,
                            _persisted_output_marker_identity_digest(content),
                        ),
                        recovered_with_stat[1],
                    )
                elif recovered_identity_content is not None:
                    content = recovered_identity_content
        tool_calls = msg.get("tool_calls")
        if stored_row:
            session_id = str(msg.get("session_id") or self._session_id or "")
            content = self._restore_ingest_payload_placeholders_in_content_identity(
                content,
                session_id=session_id,
            )
            tool_calls = self._restore_ingest_payload_placeholders_in_value(tool_calls, session_id=session_id)
        ref = extract_externalized_ref(content)
        if ref and "quarantined_assistant_output" not in content:
            payload = load_externalized_payload(
                ref,
                config=self._config,
                hermes_home=self._hermes_home,
            )
            if payload is not None and isinstance(payload.get("content"), str):
                content = payload["content"]
        tool_calls_identity = self._stable_tool_calls_identity(tool_calls)
        tool_name = str(msg.get("tool_name") or "") if role == "tool" else ""
        if role == "tool" and not tool_name and inferred_tool_name:
            tool_name = inferred_tool_name
        return (
            role,
            content,
            str(msg.get("tool_call_id") or ""),
            tool_calls_identity,
            tool_name,
        )

    @staticmethod
    def _matches_store_tail_suffix(
        stored_tail: list[tuple[str, str, str, str, str]],
        candidate_prefix: list[tuple[str, str, str, str, str]],
    ) -> bool:
        if not candidate_prefix:
            return True
        if len(candidate_prefix) > len(stored_tail):
            return False
        return stored_tail[-len(candidate_prefix) :] == candidate_prefix

    @staticmethod
    def _strip_inline_persisted_output_generation_identity(
        identity: tuple[str, str, str, str, str],
    ) -> tuple[str, str, str, str, str]:
        role, content, tool_call_id, tool_calls, tool_name = identity
        if role != "tool" or not isinstance(content, str):
            return identity
        stripped = re.sub(
            r"\n?\[LCM persisted-output file generation: "
            r"size=\d+; mtime_ns=\d+; ctime_ns=\d+\]\n?(?=</persisted-output>)",
            "\n",
            content,
        )
        return (role, stripped, tool_call_id, tool_calls, tool_name)

    def _stored_row_has_durable_persisted_output_marker(self, row: Dict[str, Any]) -> bool:
        if str(row.get("role") or "") != "tool":
            return False
        content = normalize_content_value(row.get("content")) or ""
        ref = extract_externalized_ref(content)
        if not ref:
            return False
        return externalized_tool_result_has_persisted_output_marker(
            ref,
            config=self._config,
            hermes_home=self._hermes_home,
        )

    @staticmethod
    def _persisted_output_durable_wildcard_identity(
        identity: tuple[str, str, str, str, str],
    ) -> tuple[str, str, str, str, str]:
        role, _content, tool_call_id, tool_calls, tool_name = identity
        return (
            role,
            "[LCM persisted-output durable replay]",
            tool_call_id,
            tool_calls,
            tool_name,
        )

    def _matches_persisted_output_durable_full_replay(
        self,
        candidate_messages: list[Dict[str, Any]],
        candidate_prefix: list[tuple[str, str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str, str]],
        stored_tail_rows: list[Dict[str, Any]] | None,
    ) -> bool:
        if not stored_tail_rows or len(candidate_prefix) != len(stored_tail) or len(candidate_messages) != len(candidate_prefix):
            return False
        transformed_candidate: list[tuple[str, str, str, str, str]] = []
        transformed_stored: list[tuple[str, str, str, str, str]] = []
        saw_persisted_output = False
        for candidate_msg, candidate_identity, stored_identity, stored_row in zip(
            candidate_messages,
            candidate_prefix,
            stored_tail,
            stored_tail_rows,
        ):
            candidate_content = normalize_content_value(candidate_msg.get("content")) or ""
            candidate_is_persisted_marker = (
                str(candidate_msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(candidate_content)
            )
            stored_is_persisted_output = self._stored_row_has_durable_persisted_output_marker(stored_row)
            if candidate_is_persisted_marker or stored_is_persisted_output:
                if (
                    not candidate_is_persisted_marker
                    or not stored_is_persisted_output
                    or not self._has_durable_persisted_output_replay_identity(candidate_msg)
                ):
                    return False
                saw_persisted_output = True
                transformed_candidate.append(self._persisted_output_durable_wildcard_identity(candidate_identity))
                transformed_stored.append(self._persisted_output_durable_wildcard_identity(stored_identity))
                continue
            transformed_candidate.append(candidate_identity)
            transformed_stored.append(stored_identity)
        return saw_persisted_output and transformed_candidate == transformed_stored

    @classmethod
    def _identity_content_for_active_cleanup(cls, content: str) -> Any:
        """Decode canonical stored JSON content before active-cleanup checks.

        Structured assistant content is persisted as deterministic JSON. Active
        replay cleanup sees the original list/dict shape, so restart
        reconciliation has to decode the stored identity before deciding whether
        a durable assistant row could be absent from sanitized active context.
        """
        if not isinstance(content, str):
            return content
        try:
            decoded = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError):
            return content
        if isinstance(decoded, (list, dict)) and normalize_content_value(decoded) == content:
            return decoded
        return content

    @classmethod
    def _active_cleanup_replay_identity(
        cls,
        identity: tuple[str, str, str, str, str],
    ) -> tuple[str, str, str, str, str] | None:
        role, content, tool_call_id, tool_calls, tool_name = identity
        if role != "assistant":
            return identity
        msg: dict[str, Any] = {
            "role": role,
            "content": cls._identity_content_for_active_cleanup(content),
        }
        if tool_calls:
            try:
                decoded_tool_calls = json.loads(tool_calls)
            except (TypeError, ValueError, json.JSONDecodeError):
                decoded_tool_calls = tool_calls
            msg["tool_calls"] = decoded_tool_calls
        cleaned = _clean_active_assistant_message(msg)
        if cleaned is None:
            return None
        return (
            role,
            normalize_content_value(cleaned.get("content")) or "",
            tool_call_id,
            tool_calls,
            tool_name,
        )

    @staticmethod
    def _is_quarantined_assistant_replay_identity(identity: tuple[str, str, str, str, str]) -> bool:
        role, content, _tool_call_id, _tool_calls, _tool_name = identity
        if role != "assistant":
            return False
        text = str(content or "").strip()
        return bool(
            re.fullmatch(
                r"\[Externalized LCM ingest payload: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"field=[A-Za-z0-9_.:/<>\[\]-]+; "
                r"chars=\d+; bytes=\d+; "
                r"ref=[^\]\s]+\]",
                text,
            )
            or re.fullmatch(
                r"\[LCM active replay placeholder: assistant output quarantined; "
                r"kind=quarantined_assistant_output; "
                r"reason=[A-Za-z0-9_.:/-]+; "
                r"scope=ignored_message_pattern; field=content; "
                r"chars=\d+; bytes=\d+; "
                r"sha256=[0-9a-f]{16}\]",
                text,
            )
        )

    def _stored_tail_for_sanitized_active_replay(
        self,
        stored_tail: list[tuple[str, str, str, str, str]],
    ) -> list[tuple[str, str, str, str, str]]:
        """Mirror active-context cleanup for restart replay reconciliation.

        Raw storage remains lossless. This view is used only to reconcile a
        restarted process when the host replays sanitized active context where
        assistant rows may be removed or have internal content stripped.
        """
        sanitized_tail: list[tuple[str, str, str, str, str]] = []
        for identity in stored_tail:
            cleaned_identity = self._active_cleanup_replay_identity(identity)
            if cleaned_identity is not None:
                sanitized_tail.append(cleaned_identity)
        return sanitized_tail

    def _find_reconciled_cursor_for_store_tail(
        self,
        messages: List[Dict[str, Any]],
        stored_tail: list[tuple[str, str, str, str, str]],
        *,
        stored_tail_rows: list[Dict[str, Any]] | None = None,
        allow_empty_prefix: bool,
        session_count: int,
        raw_session_count: int,
    ) -> int | None:
        sanitized_replay_tail = self._stored_tail_for_sanitized_active_replay(stored_tail)
        effective_session_count = len(sanitized_replay_tail)
        sanitized_tail_collapsed = len(sanitized_replay_tail) < len(stored_tail)
        boundary_messages = list(stored_tail_rows or [])
        if not boundary_messages:
            for role, content, tool_call_id, tool_calls, tool_name in stored_tail:
                try:
                    decoded_tool_calls = json.loads(tool_calls) if tool_calls else []
                except (TypeError, ValueError, json.JSONDecodeError):
                    decoded_tool_calls = []
                boundary_messages.append({
                    "role": role,
                    "content": content,
                    "tool_call_id": tool_call_id,
                    "tool_calls": decoded_tool_calls,
                    "tool_name": tool_name,
                })
        effective_fresh_tail_count = self._fresh_tail_boundary(boundary_messages).count
        empty_prefix_cursor: int | None = None
        for cursor in range(len(messages), -1, -1):
            candidate_messages = messages[:cursor]
            candidate_visible_messages = [
                msg
                for msg in candidate_messages
                if not self._is_replayed_context_scaffold_message(msg)
                and not self._matches_ignore_message_patterns(msg)
            ]
            candidate_non_placeholder_messages = [
                msg
                for msg in candidate_visible_messages
                if not self._is_volatile_ignored_quarantine_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                and not self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                and not (
                    self._compiled_ignore_message_patterns
                    and self._is_quarantined_assistant_replay_identity(
                        self._message_replay_identity(msg)
                    )
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
            ]
            filtered_candidate_placeholders = len(candidate_non_placeholder_messages) < len(candidate_visible_messages)
            candidate_has_scaffold_evidence = any(
                self._is_replayed_context_scaffold_message(msg) for msg in candidate_messages
            )
            candidate_has_quarantined_replay_evidence = any(
                self._is_quarantined_assistant_replay_identity(self._message_replay_identity(msg))
                for msg in candidate_messages
            )
            candidate_identity_messages = (
                candidate_non_placeholder_messages
                if candidate_non_placeholder_messages or filtered_candidate_placeholders
                else candidate_visible_messages
            )
            candidate_visible_prefix = [
                self._message_replay_identity(msg)
                for msg in candidate_visible_messages
            ]
            candidate_prefix = [
                self._message_replay_identity(msg)
                for msg in candidate_identity_messages
            ]
            if not candidate_prefix:
                empty_prefix_cursor = cursor
                if allow_empty_prefix and (
                    not filtered_candidate_placeholders
                    or candidate_has_scaffold_evidence
                    or candidate_has_quarantined_replay_evidence
                ):
                    return cursor
                continue

            matches_sanitized_tail = (
                len(candidate_prefix) <= len(sanitized_replay_tail)
                and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_prefix)
            )
            matches_raw_tail = self._matches_store_tail_suffix(stored_tail, candidate_prefix)
            matches_visible_sanitized_tail = (
                filtered_candidate_placeholders
                and bool(candidate_visible_prefix)
                and len(candidate_visible_prefix) <= len(sanitized_replay_tail)
                and self._matches_store_tail_suffix(sanitized_replay_tail, candidate_visible_prefix)
            )
            matches_visible_raw_tail = (
                filtered_candidate_placeholders
                and bool(candidate_visible_prefix)
                and self._matches_store_tail_suffix(stored_tail, candidate_visible_prefix)
            )
            early_candidate_has_unrecoverable_persisted_marker = any(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                and recover_hermes_persisted_output_with_file_stat(
                    normalize_content_value(msg.get("content")) or ""
                )
                is None
                for msg in candidate_identity_messages
            )
            if (matches_visible_sanitized_tail or matches_visible_raw_tail) and not early_candidate_has_unrecoverable_persisted_marker:
                return cursor
            candidate_has_persisted_marker = any(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                for msg in candidate_identity_messages
            )
            matches_durable_persisted_output_full_replay = self._matches_persisted_output_durable_full_replay(
                candidate_identity_messages,
                candidate_prefix,
                stored_tail,
                stored_tail_rows,
            )
            candidate_has_unrecoverable_persisted_marker = any(
                str(msg.get("role") or "") == "tool"
                and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                and recover_hermes_persisted_output_with_file_stat(
                    normalize_content_value(msg.get("content")) or ""
                )
                is None
                for msg in candidate_identity_messages
            )
            matches_inline_generation_cleanup_tail = False
            if candidate_has_unrecoverable_persisted_marker:
                generationless_sanitized_tail = [
                    self._strip_inline_persisted_output_generation_identity(identity)
                    for identity in sanitized_replay_tail
                ]
                generationless_candidate_prefix = [
                    self._strip_inline_persisted_output_generation_identity(identity)
                    for identity in candidate_prefix
                ]
                matches_inline_generation_cleanup_tail = self._matches_store_tail_suffix(
                    generationless_sanitized_tail,
                    generationless_candidate_prefix,
                )
            raw_tail_suffix = stored_tail[-len(candidate_prefix) :] if matches_raw_tail else []
            raw_suffix_needs_cleanup_equivalence = any(
                self._active_cleanup_replay_identity(identity) != identity
                for identity in raw_tail_suffix
            )
            if (
                not matches_sanitized_tail
                and not matches_raw_tail
                and not matches_inline_generation_cleanup_tail
                and not matches_durable_persisted_output_full_replay
            ):
                continue

            # Matching a stored suffix is not enough evidence by itself.  A
            # gateway restart may provide only newly arrived delta messages; if
            # the first delta happens to repeat the durable tail, treating that
            # row as replay silently loses it.  Only advance the cursor when the
            # incoming prefix proves replay by covering the full durable session.
            # A system prompt is a strong anchor. Older/minimal transcripts can
            # start directly with user/assistant turns, so multi-row full replay
            # is accepted only when active cleanup did not collapse the durable
            # tail; otherwise a fresh delta can repeat the remaining visible
            # suffix and must be preserved.
            candidate_has_system = any(identity[0] == "system" for identity in candidate_prefix)
            candidate_dropped_quarantine_replay_placeholder = any(
                self._is_volatile_ignored_quarantine_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                or self._is_ignored_active_replay_placeholder(
                    msg,
                    text_content_for_pattern_matching(msg.get("content")) or "",
                )
                or (
                    self._compiled_ignore_message_patterns
                    and self._is_quarantined_assistant_replay_identity(
                        self._message_replay_identity(msg)
                    )
                    and self._matches_ignore_message_patterns(msg, stored_row=True)
                )
                for msg in candidate_messages
            )
            has_quarantined_singleton_replay = (
                matches_sanitized_tail
                and len(candidate_prefix) == 1
                and effective_session_count == 1
                and self._is_quarantined_assistant_replay_identity(candidate_prefix[0])
                and self._is_quarantined_assistant_replay_identity(sanitized_replay_tail[0])
            )
            candidate_singleton_original_content = (
                normalize_content_value(candidate_identity_messages[0].get("content")) or ""
                if len(candidate_identity_messages) == 1
                else ""
            )
            has_externalized_singleton_replay = (
                matches_raw_tail
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and bool(extract_externalized_ref(candidate_singleton_original_content))
                and candidate_prefix == stored_tail
            )
            has_persisted_marker_singleton_replay = (
                matches_raw_tail
                and not candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) == 1
                and raw_session_count == 1
                and candidate_prefix == stored_tail
                and candidate_prefix[0][0] == "tool"
                and _is_hermes_persisted_output_marker(candidate_singleton_original_content)
            )
            has_durable_persisted_marker_suffix_replay = (
                (matches_sanitized_tail or matches_raw_tail)
                and any(
                    str(msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                    and self._has_durable_persisted_output_replay_identity(msg)
                    for msg in candidate_messages
                )
            )
            has_filtered_full_replay = (
                matches_sanitized_tail
                and candidate_dropped_quarantine_replay_placeholder
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_generation_cleanup_replay = (
                matches_inline_generation_cleanup_tail
                and candidate_has_unrecoverable_persisted_marker
                and len(candidate_prefix) >= effective_session_count
                and effective_session_count > 0
            )
            has_inline_persisted_generation_suffix_replay = (
                matches_sanitized_tail
                and any(
                    str(msg.get("role") or "") == "tool"
                    and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
                    and _has_inline_persisted_output_generation_metadata(normalize_content_value(msg.get("content")) or "")
                    for msg in candidate_identity_messages
                )
            )
            if candidate_has_unrecoverable_persisted_marker:
                continue
            has_raw_persisted_marker_exact_replay = (
                candidate_has_persisted_marker
                and not candidate_has_unrecoverable_persisted_marker
                and matches_raw_tail
                and candidate_prefix == stored_tail[-len(candidate_prefix) :]
            )
            has_persisted_marker_specific_replay_evidence = (
                not candidate_has_persisted_marker
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_persisted_marker_singleton_replay
                or has_raw_persisted_marker_exact_replay
            )
            has_effective_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and len(candidate_prefix) >= effective_session_count
                and (
                    candidate_has_system
                    or (effective_session_count > 1 and not sanitized_tail_collapsed)
                    or has_quarantined_singleton_replay
                    or has_filtered_full_replay
                )
            )

            has_scaffold_evidence = any(
                self._is_replayed_context_scaffold_message(msg) for msg in candidate_messages
            )
            has_raw_full_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and not has_scaffold_evidence
                and len(candidate_messages) >= raw_session_count
                and raw_session_count > 1
            )
            has_preserved_objective_scaffold = any(
                str(msg.get("role") or "") != "system"
                and (normalize_content_value(msg.get("content")) or "").lstrip().startswith(
                    _PRESERVED_OBJECTIVE_CONTEXT_PREFIX
                )
                for msg in candidate_messages
            )
            candidate_suffix_has_user_turn = any(identity[0] == "user" for identity in candidate_prefix)
            has_scaffold_suffix_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_sanitized_tail
                and has_preserved_objective_scaffold
                and not candidate_suffix_has_user_turn
            )
            has_raw_cleanup_replay = (
                has_persisted_marker_specific_replay_evidence
                and matches_raw_tail
                and has_scaffold_evidence
                and cursor < len(messages)
                and len(candidate_prefix) >= max(1, effective_fresh_tail_count)
                and raw_suffix_needs_cleanup_equivalence
            )
            if (
                has_effective_full_replay
                or has_externalized_singleton_replay
                or has_persisted_marker_singleton_replay
                or has_durable_persisted_marker_suffix_replay
                or matches_durable_persisted_output_full_replay
                or has_inline_generation_cleanup_replay
                or has_inline_persisted_generation_suffix_replay
                or has_raw_full_replay
                or has_scaffold_suffix_replay
                or has_raw_cleanup_replay
            ):
                return cursor
        return empty_prefix_cursor if allow_empty_prefix else None

    def _record_ingest_reconciliation(
        self,
        *,
        action: str,
        reason: str,
        cursor: int,
        incoming: int,
        session_count: int,
        stored_tail_count: int,
        effective_incoming: int | None = None,
    ) -> None:
        self._last_ingest_reconciliation = {
            "action": action,
            "reason": reason,
            "cursor": cursor,
            "incoming": incoming,
            "session_count": session_count,
            "stored_tail_count": stored_tail_count,
        }
        if effective_incoming is not None:
            self._last_ingest_reconciliation["effective_incoming"] = effective_incoming

    def _effective_replay_identities(
        self,
        messages: List[Dict[str, Any]],
    ) -> list[tuple[str, str, str, str, str]]:
        return [
            self._message_replay_identity(msg)
            for msg in messages
            if not self._is_replayed_context_scaffold_message(msg)
            and not self._matches_ignore_message_patterns(msg)
        ]

    def _find_tool_anchored_replay_cursor(self, messages: List[Dict[str, Any]]) -> int | None:
        """Find a replayed leading snapshot proven by an exact durable tool result.

        Gateway resumes can replay an older compacted prefix that no longer
        overlaps the durable tail. Ordinary repeated text remains ambiguous, but
        a tool result's call id plus its full replay identity is a stable anchor:
        the host must not execute a new tool call with an old call id. Only a
        contiguous leading visible sequence containing such an anchor is
        skipped; the first unmatched row and everything after it remain a delta.
        """
        visible_messages = [
            (raw_index, msg)
            for raw_index, msg in enumerate(messages)
            if not self._is_replayed_context_scaffold_message(msg)
            and not self._matches_ignore_message_patterns(msg)
        ]
        if not any(
            str(msg.get("role") or "") == "tool"
            and bool(str(msg.get("tool_call_id") or ""))
            for _raw_index, msg in visible_messages
        ):
            return None

        stored_rows: list[Dict[str, Any]] = []
        after_store_id = 0
        while True:
            page = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=after_store_id,
            )
            if not page:
                break
            stored_rows.extend(
                row
                for row in page
                if not self._matches_ignore_message_patterns(row, stored_row=True)
            )
            after_store_id = page[-1]["store_id"]
        if not stored_rows or not visible_messages:
            return None

        incoming_identities = [
            self._message_replay_identity(msg)
            for _raw_index, msg in visible_messages
        ]
        stored_identities = [
            self._message_replay_identity(row, stored_row=True)
            for row in stored_rows
        ]

        def identities_match(
            incoming_identity: tuple[str, str, str, str, str],
            stored_identity: tuple[str, str, str, str, str],
        ) -> bool:
            if incoming_identity == stored_identity:
                return True
            return self._active_cleanup_replay_identity(stored_identity) == incoming_identity

        best_visible_count = 0
        for stored_start, stored_identity in enumerate(stored_identities):
            if not identities_match(incoming_identities[0], stored_identity):
                continue
            matched_tool_anchor = False
            visible_count = 0
            for incoming_offset, incoming_identity in enumerate(incoming_identities):
                stored_offset = stored_start + incoming_offset
                if stored_offset >= len(stored_identities):
                    break
                if not identities_match(incoming_identity, stored_identities[stored_offset]):
                    break
                _raw_index, incoming_message = visible_messages[incoming_offset]
                if (
                    str(incoming_message.get("role") or "") == "tool"
                    and bool(str(incoming_message.get("tool_call_id") or ""))
                ):
                    matched_tool_anchor = True
                visible_count += 1
            if matched_tool_anchor and visible_count > best_visible_count:
                best_visible_count = visible_count

        if best_visible_count <= 0:
            return None
        return visible_messages[best_visible_count - 1][0] + 1

    def _find_tool_anchored_replay_indexes(
        self,
        messages: List[Dict[str, Any]],
        *,
        suppress_tool_less_duplicates: bool = False,
        durable_key_lookup: bool = False,
        preignored_indexes: set[int] | None = None,
    ) -> tuple[set[int], int]:
        """Find stable tool rows replayed after a changed active-context prefix.

        A cursor cannot represent a mixed batch with unmatched rows before a
        replayed durable tool result.  Exact tool-result identities are safe to
        filter because the host must not execute a new call with an old call ID.
        The immediately preceding assistant tool-call row is also safe when both
        durable and incoming rows carry that same call ID.  Plain neighboring
        user/assistant rows remain ambiguous and are deliberately preserved.

        ``suppress_tool_less_duplicates`` extends suppression to tool-less rows
        byte-matching durable rows once the batch is tool-anchor-proven.  It is
        only warranted on the restart/rebind path, where the batch is a
        redelivered context snapshot; in a live session the same bytes can be a
        legitimate repeated turn and stay preserved.
        """
        visible_messages = [
            (raw_index, msg)
            for raw_index, msg in enumerate(messages)
            if (
                not self._is_replayed_context_scaffold_message(msg)
                or (
                    str(msg.get("role") or "") == "tool"
                    and bool(str(msg.get("tool_call_id") or "").strip())
                )
            )
            and (
                raw_index not in preignored_indexes
                if preignored_indexes is not None
                else not self._matches_ignore_message_patterns(msg)
            )
        ]
        incoming_tool_offsets = [
            offset
            for offset, (_raw_index, msg) in enumerate(visible_messages)
            if str(msg.get("role") or "") == "tool"
            and bool(str(msg.get("tool_call_id") or ""))
        ]
        if not incoming_tool_offsets:
            return set(), 0

        if durable_key_lookup:
            incoming_call_ids = {
                str(msg.get("tool_call_id") or "").strip()
                for offset in incoming_tool_offsets
                for _raw_index, msg in [visible_messages[offset]]
                if str(msg.get("tool_call_id") or "").strip()
            }
            scanned_rows = self._store.get_tool_call_replay_neighborhoods(
                self._session_id,
                incoming_call_ids,
                conversation_id=getattr(self, "_conversation_id", None),
            )
            if suppress_tool_less_duplicates:
                # Durable-key neighborhoods prove old tool anchors even when they
                # are outside the bounded tail. Preserve the tail scan as well so
                # restart/rebind filtering can still suppress byte-identical
                # tool-less snapshot rows that are not adjacent to those anchors.
                scan_limit = min(max(len(visible_messages) * 4, 256), 4096)
                tail_rows = self._store.get_session_tail(
                    self._session_id,
                    limit=scan_limit,
                )
                rows_by_store_id = {
                    int(row["store_id"]): row for row in (*tail_rows, *scanned_rows)
                }
                scanned_rows = [rows_by_store_id[key] for key in sorted(rows_by_store_id)]
        else:
            scan_limit = min(max(len(visible_messages) * 4, 256), 4096)
            scanned_rows = self._store.get_session_tail(
                self._session_id,
                limit=scan_limit,
            )
        scanned_row_count = len(scanned_rows)
        stored_rows = [
            row
            for row in scanned_rows
            if not self._matches_ignore_message_patterns(row, stored_row=True)
        ]
        if not stored_rows:
            return set(), scanned_row_count

        def inferred_tool_names(rows: List[Dict[str, Any]]) -> list[str | None]:
            inferred: list[str | None] = [None] * len(rows)
            for offset, row in enumerate(rows):
                if (
                    offset == 0
                    or str(row.get("role") or "") != "tool"
                    or bool(str(row.get("tool_name") or "").strip())
                ):
                    continue
                call_id = str(row.get("tool_call_id") or "").strip()
                previous = rows[offset - 1]
                if not call_id or str(previous.get("role") or "") != "assistant":
                    continue
                tool_calls = previous.get("tool_calls") or []
                if isinstance(tool_calls, str):
                    try:
                        tool_calls = json.loads(tool_calls)
                    except (TypeError, ValueError, json.JSONDecodeError):
                        continue
                if isinstance(tool_calls, dict):
                    tool_calls = [tool_calls]
                if not isinstance(tool_calls, list):
                    continue
                names: set[str] = set()
                for tool_call in tool_calls:
                    if not isinstance(tool_call, dict):
                        continue
                    candidate_call_id = str(
                        tool_call.get("id") or tool_call.get("tool_call_id") or ""
                    ).strip()
                    function = tool_call.get("function") or {}
                    tool_name = (
                        str(function.get("name") or "").strip()
                        if isinstance(function, dict)
                        else ""
                    )
                    if candidate_call_id == call_id and tool_name:
                        names.add(tool_name)
                if len(names) == 1:
                    inferred[offset] = next(iter(names))
            return inferred

        incoming_rows = [msg for _raw_index, msg in visible_messages]
        incoming_name_map = inferred_tool_names(incoming_rows)
        stored_name_map = inferred_tool_names(stored_rows)
        incoming_identities = [
            self._message_replay_identity(
                msg,
                inferred_tool_name=incoming_name_map[offset],
            )
            for offset, msg in enumerate(incoming_rows)
        ]
        stored_identities = [
            self._message_replay_identity(
                row,
                stored_row=True,
                inferred_tool_name=stored_name_map[offset],
            )
            for offset, row in enumerate(stored_rows)
        ]

        def identities_match(
            incoming_identity: tuple[str, str, str, str, str],
            stored_identity: tuple[str, str, str, str, str],
        ) -> bool:
            if incoming_identity == stored_identity:
                return True
            incoming_clean = self._active_cleanup_replay_identity(incoming_identity)
            stored_clean = self._active_cleanup_replay_identity(stored_identity)
            return (
                incoming_clean is not None
                and stored_clean is not None
                and incoming_clean == stored_clean
            )

        def assistant_tool_call_ids(msg: Dict[str, Any]) -> set[str]:
            tool_calls = msg.get("tool_calls") or []
            if isinstance(tool_calls, str):
                try:
                    tool_calls = json.loads(tool_calls)
                except (TypeError, ValueError, json.JSONDecodeError):
                    return set()
            if isinstance(tool_calls, dict):
                tool_calls = [tool_calls]
            if not isinstance(tool_calls, list):
                return set()
            call_ids: set[str] = set()
            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue
                value = tool_call.get("id") or tool_call.get("tool_call_id")
                if value:
                    call_ids.add(str(value).strip())
            return call_ids

        def is_orphan_recovery_result(msg: Dict[str, Any]) -> bool:
            if str(msg.get("role") or "") != "tool":
                return False
            content = normalize_content_value(msg.get("content")) or ""
            return content.lstrip().startswith("[Orphan recovery:")

        durable_result_call_ids = {
            str(row.get("tool_call_id") or "").strip()
            for row in stored_rows
            if str(row.get("role") or "") == "tool"
            and str(row.get("tool_call_id") or "").strip()
        }
        durable_assistant_call_ids: set[str] = set()
        for row in stored_rows:
            if str(row.get("role") or "") == "assistant":
                durable_assistant_call_ids.update(assistant_tool_call_ids(row))

        stored_tool_anchors: dict[tuple[str, str, str, str, str], list[int]] = {}
        for stored_offset, (stored_row, stored_identity) in enumerate(
            zip(stored_rows, stored_identities)
        ):
            if (
                str(stored_row.get("role") or "") != "tool"
                or not bool(str(stored_row.get("tool_call_id") or ""))
            ):
                continue
            keys = {stored_identity}
            cleaned_identity = self._active_cleanup_replay_identity(stored_identity)
            if cleaned_identity is not None:
                keys.add(cleaned_identity)
            for key in keys:
                stored_tool_anchors.setdefault(key, []).append(stored_offset)


        replayed_raw_indexes: set[int] = set()
        matched_tool_anchor_pairs: list[tuple[int, int]] = []
        claimed_stored_tool_anchors: set[int] = set()
        # Align repeated identical anchors as an ordered durable subsequence.
        # Walking both sides newest-to-oldest keeps the latest-match bias for a
        # singleton replay without allowing multiple incoming anchors to bind
        # to the same durable occurrence.
        stored_anchor_ceiling = len(stored_rows)
        for incoming_anchor in reversed(incoming_tool_offsets):
            incoming_raw_index, incoming_tool = visible_messages[incoming_anchor]
            incoming_identity = incoming_identities[incoming_anchor]
            # Missing persisted-output source files do not make an exact stored
            # marker identity new. Re-appending the same
            # (session_id, tool_call_id, content) pointer cannot recover its
            # vanished payload; it only creates a duplicate row. The durable
            # identity lookup below still preserves changed markers and markers
            # whose call id/content have never been stored.
            candidates = stored_tool_anchors.get(incoming_identity, [])
            call_id = str(incoming_tool.get("tool_call_id") or "").strip()
            persisted_output_anchor_proven = False
            has_adjacent_incoming_call = False
            if incoming_anchor > 0 and call_id:
                incoming_previous = visible_messages[incoming_anchor - 1][1]
                has_adjacent_incoming_call = (
                    str(incoming_previous.get("role") or "") == "assistant"
                    and call_id in assistant_tool_call_ids(incoming_previous)
                )
            if not candidates and call_id and has_adjacent_incoming_call:
                persisted_output_anchor_proven = (
                    self._has_durable_persisted_output_replay_identity(incoming_tool)
                )
                if persisted_output_anchor_proven:
                    candidates = [
                        stored_offset
                        for stored_offset, stored_row in enumerate(stored_rows)
                        if str(stored_row.get("role") or "") == "tool"
                        and str(stored_row.get("tool_call_id") or "").strip() == call_id
                    ]

            def anchor_identity_matches(stored_anchor: int) -> bool:
                return identities_match(
                    incoming_identity,
                    stored_identities[stored_anchor],
                ) or (
                    persisted_output_anchor_proven
                    and previous_assistant_matches(stored_anchor)
                )

            def previous_assistant_matches(stored_anchor: int) -> bool:
                if incoming_anchor <= 0 or stored_anchor <= 0:
                    return False
                call_id = str(incoming_tool.get("tool_call_id") or "").strip()
                if not call_id:
                    return False
                incoming_previous = visible_messages[incoming_anchor - 1][1]
                stored_previous = stored_rows[stored_anchor - 1]
                return (
                    str(incoming_previous.get("role") or "") == "assistant"
                    and str(stored_previous.get("role") or "") == "assistant"
                    and call_id in assistant_tool_call_ids(incoming_previous)
                    and call_id in assistant_tool_call_ids(stored_previous)
                    and identities_match(
                        incoming_identities[incoming_anchor - 1],
                        stored_identities[stored_anchor - 1],
                    )
                )

            matching_candidates = [
                stored_anchor
                for stored_anchor in reversed(candidates)
                if stored_anchor not in claimed_stored_tool_anchors
                and stored_anchor < stored_anchor_ceiling
                and anchor_identity_matches(stored_anchor)
            ]
            matched_out_of_order_unique = False
            if not matching_candidates:
                unique_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate not in claimed_stored_tool_anchors
                    and anchor_identity_matches(candidate)
                ]
                if len(unique_candidates) != 1:
                    # Exact duplicate tool rows are still replay-safe to drop,
                    # but reusing a claimed durable occurrence must not create
                    # another anchor pair: that would turn singleton evidence
                    # into false multi-anchor proof across an unmatched gap.
                    if (
                        candidates
                        and not unique_candidates
                        and not persisted_output_anchor_proven
                    ):
                        replayed_raw_indexes.add(incoming_raw_index)
                        if len(candidates) == 1:
                            claimed_anchor = candidates[0]
                            for pair_index, (
                                paired_incoming,
                                paired_stored,
                            ) in enumerate(matched_tool_anchor_pairs):
                                if (
                                    paired_stored == claimed_anchor
                                    and incoming_anchor < paired_incoming
                                ):
                                    # Keep singleton proof at the oldest replay
                                    # occurrence. A later duplicate remains
                                    # suppressible as a tool row, but cannot
                                    # move the boundary past an unmatched gap.
                                    matched_tool_anchor_pairs[pair_index] = (
                                        incoming_anchor,
                                        claimed_anchor,
                                    )
                                    break
                        if incoming_anchor > 0:
                            incoming_previous_raw, _ = visible_messages[
                                incoming_anchor - 1
                            ]
                            if any(
                                previous_assistant_matches(candidate)
                                for candidate in candidates
                            ):
                                replayed_raw_indexes.add(incoming_previous_raw)
                    # A resumed Hermes session can replace an already-durable
                    # result with an orphan-recovery placeholder. The content
                    # differs, but the placeholder proves no new execution
                    # completed. Suppress its assistant row only when every
                    # call in that row is already durable.
                    call_id = str(incoming_tool.get("tool_call_id") or "").strip()
                    if (
                        incoming_raw_index not in replayed_raw_indexes
                        and call_id in durable_result_call_ids
                        and is_orphan_recovery_result(incoming_tool)
                    ):
                        replayed_raw_indexes.add(incoming_raw_index)
                        if incoming_anchor > 0:
                            incoming_previous_raw, incoming_previous = visible_messages[
                                incoming_anchor - 1
                            ]
                            previous_call_ids = assistant_tool_call_ids(incoming_previous)
                            if (
                                str(incoming_previous.get("role") or "") == "assistant"
                                and call_id in previous_call_ids
                                and previous_call_ids
                                and previous_call_ids.issubset(durable_assistant_call_ids)
                            ):
                                replayed_raw_indexes.add(incoming_previous_raw)
                    continue
                # A unique exact tool identity is safe even when a restarted
                # snapshot wraps from the durable tail back to older context.
                # Only repeated identities need the monotonic ceiling to bind
                # each incoming anchor to a distinct durable occurrence.
                matching_candidates = unique_candidates
                matched_out_of_order_unique = True
            stored_anchor = next(
                (
                    candidate
                    for candidate in matching_candidates
                    if previous_assistant_matches(candidate)
                ),
                matching_candidates[0],
            )
            if not matched_out_of_order_unique:
                stored_anchor_ceiling = stored_anchor
            claimed_stored_tool_anchors.add(stored_anchor)
            replayed_raw_indexes.add(incoming_raw_index)
            matched_tool_anchor_pairs.append((incoming_anchor, stored_anchor))
            call_id = str(incoming_tool.get("tool_call_id") or "").strip()
            if incoming_anchor <= 0 or stored_anchor <= 0 or not call_id:
                continue
            incoming_previous_raw, incoming_previous = visible_messages[incoming_anchor - 1]
            stored_previous = stored_rows[stored_anchor - 1]
            if (
                str(incoming_previous.get("role") or "") == "assistant"
                and str(stored_previous.get("role") or "") == "assistant"
                and call_id in assistant_tool_call_ids(incoming_previous)
                and call_id in assistant_tool_call_ids(stored_previous)
                and identities_match(
                    incoming_identities[incoming_anchor - 1],
                    stored_identities[stored_anchor - 1],
                )
            ):
                replayed_raw_indexes.add(incoming_previous_raw)

        if matched_tool_anchor_pairs and suppress_tool_less_duplicates:
            # Multiple ordered anchors prove that the batch is a replayed
            # snapshot even when fresh compacted/filler rows are interleaved.
            # A singleton anchor does not provide that boundary: after the
            # first unmatched row, later equal identities may be a legitimate
            # repeated suffix and must remain appendable.
            allow_interleaved_gaps = len(matched_tool_anchor_pairs) > 1
            # Extend each proven tool anchor only through ordered rows on the
            # same side of the corresponding durable anchor. Incoming scaffold
            # or compacted filler may be interleaved, so exact adjacency is not
            # required, but matching cannot cross another tool-result anchor or
            # reverse durable order. This keeps suppression bound to the proven
            # tool exchange instead of using global identity membership.
            segment_positions_cache: dict[
                tuple[int, int],
                tuple[int, dict[tuple[str, str, str, str, str], list[int]]],
            ] = {}
            for incoming_anchor, stored_anchor in matched_tool_anchor_pairs:
                for step in (-1, 1):
                    cache_key = (stored_anchor, step)
                    cached_segment = segment_positions_cache.get(cache_key)
                    if cached_segment is None:
                        stored_segment: list[int] = []
                        stored_offset = stored_anchor + step
                        while 0 <= stored_offset < len(stored_rows):
                            stored_message = stored_rows[stored_offset]
                            if str(stored_message.get("role") or "") == "tool" and bool(
                                str(stored_message.get("tool_call_id") or "")
                            ):
                                break
                            stored_segment.append(stored_offset)
                            stored_offset += step
                        positions_by_identity: dict[
                            tuple[str, str, str, str, str], list[int]
                        ] = {}
                        for cursor_position, candidate_offset in enumerate(stored_segment):
                            candidate_identity = stored_identities[candidate_offset]
                            keys = {candidate_identity}
                            cleaned_identity = self._active_cleanup_replay_identity(
                                candidate_identity
                            )
                            if cleaned_identity is not None:
                                keys.add(cleaned_identity)
                            for key in keys:
                                positions_by_identity.setdefault(key, []).append(
                                    cursor_position
                                )
                        cached_segment = (len(stored_segment), positions_by_identity)
                        segment_positions_cache[cache_key] = cached_segment

                    stored_segment_length, positions_by_identity = cached_segment

                    stored_cursor = 0
                    incoming_offset = incoming_anchor + step
                    while (
                        0 <= incoming_offset < len(visible_messages)
                        and stored_cursor < stored_segment_length
                    ):
                        raw_index, message = visible_messages[incoming_offset]
                        if str(message.get("role") or "") == "tool" and bool(
                            str(message.get("tool_call_id") or "")
                        ):
                            break
                        incoming_identity = incoming_identities[incoming_offset]
                        matched_at = None
                        incoming_keys = {incoming_identity}
                        incoming_cleaned = self._active_cleanup_replay_identity(
                            incoming_identity
                        )
                        if incoming_cleaned is not None:
                            incoming_keys.add(incoming_cleaned)
                        for key in incoming_keys:
                            positions = positions_by_identity.get(key, [])
                            position_index = bisect_left(positions, stored_cursor)
                            if position_index < len(positions):
                                candidate_position = positions[position_index]
                                if matched_at is None or candidate_position < matched_at:
                                    matched_at = candidate_position
                        if matched_at is None:
                            if not allow_interleaved_gaps:
                                # Once a visible non-tool row is not
                                # durable-proven, a singleton anchor cannot
                                # distinguish farther replay from a legitimate
                                # repeated suffix.
                                break
                            incoming_offset += step
                            continue
                        replayed_raw_indexes.add(raw_index)
                        stored_cursor = matched_at + 1
                        incoming_offset += step

        return replayed_raw_indexes, scanned_row_count

    def _is_suspicious_stale_no_overlap_snapshot(
        self,
        incoming_identities: list[tuple[str, str, str, str, str]],
        stored_tail: list[tuple[str, str, str, str, str]],
        stored_head: list[tuple[str, str, str, str, str]],
    ) -> bool:
        """Return true for short stale snapshots with no durable-tail overlap.

        A restarted gateway can hand LCM a stale, short in-memory snapshot from
        the beginning of a longer session.  When that snapshot has no overlap
        with the durable tail, appending it as a delta creates duplicate rows.
        Fail closed only when the short batch is proven stale by matching the
        contiguous durable-store prefix; singleton no-overlap deltas remain
        ambiguous and are preserved.
        """
        if len(incoming_identities) <= 1:
            return False
        if incoming_identities[0][0] != "system":
            return False
        if not stored_tail or len(incoming_identities) >= len(stored_tail):
            return False
        if set(incoming_identities).intersection(stored_tail):
            return False
        if len(incoming_identities) > len(stored_head):
            return False
        return stored_head[: len(incoming_identities)] == incoming_identities

    def _reconcile_ingest_cursor_from_store(self, messages: List[Dict[str, Any]]) -> int:
        """Infer the in-memory cursor for an existing session after process restart."""
        if not self._session_id or not messages:
            return 0

        try:
            session_count = self._store.get_session_count(self._session_id)
        except Exception as exc:  # pragma: no cover - defensive only
            logger.debug("LCM ingest cursor reconciliation count failed: %s", exc)
            return 0
        if session_count <= 0:
            placeholder_budget = self._load_generated_ignored_placeholder_hash_counts()
            placeholder_ordinals = self._load_generated_ignored_placeholder_hash_ordinals()
            if placeholder_budget and placeholder_ordinals:
                consumed: dict[str, int] = {}
                cursor = 0
                for msg in messages:
                    text = text_content_for_pattern_matching(msg.get("content")) or ""
                    digest = self._active_replay_placeholder_digest(text)
                    if not digest:
                        break
                    consumed[digest] = consumed.get(digest, 0) + 1
                    ordinal = consumed[digest]
                    remaining = int(placeholder_budget.get(digest, 0) or 0)
                    if remaining <= 0 or ordinal not in placeholder_ordinals.get(digest, set()):
                        break
                    cursor += 1
                if cursor > 0:
                    self._record_ingest_reconciliation(
                        action="advanced cursor",
                        reason="replayed generated placeholders in empty session",
                        cursor=cursor,
                        incoming=len(messages),
                        session_count=session_count,
                        stored_tail_count=0,
                        effective_incoming=cursor,
                    )
                    return cursor
            return 0

        tail_limit = min(max(len(messages) * 4, 64), session_count)
        stored_rows = self._store.get_session_tail(self._session_id, limit=tail_limit)
        if not stored_rows:
            return 0
        stored_tail_rows = [
            row
            for row in stored_rows
            if not self._matches_ignore_message_patterns(row, stored_row=True)
        ]
        stored_tail = [
            self._message_replay_identity(row, stored_row=True)
            for row in stored_tail_rows
        ]
        incoming_has_raw_persisted_marker = any(
            str(msg.get("role") or "") == "tool"
            and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
            for msg in messages
        )
        cursor = self._find_reconciled_cursor_for_store_tail(
            messages,
            stored_tail,
            stored_tail_rows=stored_tail_rows,
            allow_empty_prefix=True,
            session_count=len(stored_tail),
            raw_session_count=session_count,
        )
        if cursor is not None and cursor > 0:
            reason = (
                "skipped scaffold-only prefix"
                if not self._effective_replay_identities(messages[:cursor])
                else "replayed durable tail"
            )
            if reason == "skipped scaffold-only prefix" and not incoming_has_raw_persisted_marker:
                tool_anchored_cursor = self._find_tool_anchored_replay_cursor(messages)
                if tool_anchored_cursor is not None and tool_anchored_cursor > cursor:
                    cursor = tool_anchored_cursor
                    reason = "replayed durable tool-anchored prefix"
            self._record_ingest_reconciliation(
                action="advanced cursor",
                reason=reason,
                cursor=cursor,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(self._effective_replay_identities(messages)),
            )
            logger.debug(
                "LCM reconciled ingest cursor after existing-session bind: session=%s cursor=%d incoming=%d stored_tail=%d session_count=%d reason=%s",
                self._session_id,
                cursor,
                len(messages),
                len(stored_tail),
                session_count,
                reason,
            )
            return cursor

        incoming_identities = self._effective_replay_identities(messages)
        tool_anchored_cursor = (
            None
            if incoming_has_raw_persisted_marker
            else self._find_tool_anchored_replay_cursor(messages)
        )
        if tool_anchored_cursor is not None and tool_anchored_cursor > 0:
            self._record_ingest_reconciliation(
                action="advanced cursor",
                reason="replayed durable tool-anchored prefix",
                cursor=tool_anchored_cursor,
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            logger.info(
                "LCM reconciled tool-anchored replay prefix: session=%s cursor=%d incoming=%d session_count=%d",
                self._session_id,
                tool_anchored_cursor,
                len(messages),
                session_count,
            )
            return tool_anchored_cursor

        stored_head_rows = self._store.get_session_messages(
            self._session_id,
            limit=tail_limit,
        )
        stored_head = [self._message_replay_identity(row, stored_row=True) for row in stored_head_rows]
        # Stale-snapshot proof uses the raw durable prefix.  Ignore-message
        # filters may suppress noisy rows for tail reconciliation, but filtered
        # history alone must not create replay evidence for skipping a batch.
        incoming_has_unproofed_raw_persisted_marker = any(
            str(msg.get("role") or "") == "tool"
            and _is_hermes_persisted_output_marker(normalize_content_value(msg.get("content")) or "")
            and recover_hermes_persisted_output_with_file_stat(
                normalize_content_value(msg.get("content")) or ""
            )
            is None
            for msg in messages
        )
        if (
            not incoming_has_unproofed_raw_persisted_marker
            and self._is_suspicious_stale_no_overlap_snapshot(
                incoming_identities,
                stored_tail,
                stored_head,
            )
        ):
            self._record_ingest_reconciliation(
                action="skipped batch",
                reason="skipped stale no-overlap snapshot",
                cursor=len(messages),
                incoming=len(messages),
                session_count=session_count,
                stored_tail_count=len(stored_tail),
                effective_incoming=len(incoming_identities),
            )
            logger.warning(
                "LCM skipped stale no-overlap snapshot after existing-session bind: session=%s incoming=%d effective_incoming=%d stored_tail=%d session_count=%d",
                self._session_id,
                len(messages),
                len(incoming_identities),
                len(stored_tail),
                session_count,
            )
            return len(messages)

        self._record_ingest_reconciliation(
            action="persisted batch",
            reason="persisted ambiguous delta",
            cursor=0,
            incoming=len(messages),
            session_count=session_count,
            stored_tail_count=len(stored_tail),
            effective_incoming=len(incoming_identities),
        )
        return 0

    def _raw_externalized_placeholder_replay_identity(self, msg: Dict[str, Any]) -> tuple[str, str, str, str, str]:
        return (
            str(msg.get("role") or "unknown"),
            normalize_content_value(msg.get("content")) or "",
            self._stable_tool_calls_identity(msg.get("tool_calls")),
            str(msg.get("tool_call_id") or ""),
            str(msg.get("tool_name") or ""),
        )

    def _get_store_id_map_for_messages(self, messages: List[Dict[str, Any]]) -> dict[int, int]:
        """Map current raw message objects back to store_ids in stable order.

        Matching starts strictly after ``_last_compacted_store_id`` so repeated
        content from older already-compacted history cannot hijack the mapping.
        Synthetic summary messages simply fail to match and are skipped.  When
        active context has more occurrences of an identical replay identity than
        the store has, the surplus earliest active occurrences are treated as
        synthetic/carry-over and left unmapped so they cannot steal later stored
        literal copies with the same content.
        """
        candidates: list[Dict[str, Any]] = []
        next_candidate_after = self._last_compacted_store_id
        while True:
            page = self._store.get_session_messages_after(
                self._session_id,
                after_store_id=next_candidate_after,
            )
            if not page:
                break
            candidates.extend(page)
            next_candidate_after = page[-1]["store_id"]
        active_identity_counts: dict[tuple[Any, ...], int] = {}
        for msg in messages:
            identity = self._message_replay_identity(msg)
            active_identity_counts[identity] = active_identity_counts.get(identity, 0) + 1
        stored_identity_counts: dict[tuple[Any, ...], int] = {}
        stored_cleanup_identity_counts: dict[tuple[Any, ...], int] = {}
        # Capture each candidate's identity (and its cleanup variant) here - both
        # are already computed for the counts below, so this adds no work. The
        # match-probe loops reuse them instead of recomputing
        # _message_replay_identity(stored_row=True) for every (message, probe)
        # pair. That call is expensive when a stored row carries an externalized
        # payload (JSON canonicalization + a payload-file read), so eliminating
        # the O(candidates^2) recomputes removes repeated disk reads on
        # tool-output-heavy histories. Raw-placeholder identities stay lazy (see
        # the memo below) since most rows never need them.
        stored_identities: list[tuple[Any, ...]] = []
        stored_cleanup_identities: list[Optional[tuple[Any, ...]]] = []
        for stored in candidates:
            identity = self._message_replay_identity(stored, stored_row=True)
            stored_identities.append(identity)
            cleanup_identity = self._active_cleanup_replay_identity(identity)
            stored_cleanup_identities.append(cleanup_identity)
            stored_identity_counts[identity] = stored_identity_counts.get(identity, 0) + 1
            if cleanup_identity is not None:
                stored_cleanup_identity_counts[cleanup_identity] = (
                    stored_cleanup_identity_counts.get(cleanup_identity, 0) + 1
                )

        # Lazily memoize raw-placeholder identities: only the placeholder-ref
        # paths need them, and most histories have few (or none), so computing
        # them on demand keeps the common case free.
        _raw_placeholder_identity_cache: dict[int, tuple[str, str, str, str, str]] = {}

        def stored_raw_placeholder_identity(probe_idx: int) -> tuple[str, str, str, str, str]:
            cached = _raw_placeholder_identity_cache.get(probe_idx)
            if cached is None:
                cached = self._raw_externalized_placeholder_replay_identity(candidates[probe_idx])
                _raw_placeholder_identity_cache[probe_idx] = cached
            return cached
        active_surplus_skips: dict[tuple[Any, ...], int] = {}
        generated_surplus_skip_message_ids: set[int] = set()
        generated_placeholder_message_ids = getattr(
            self,
            "_generated_ignored_active_replay_placeholder_message_ids",
            set(),
        )
        for identity, active_count in active_identity_counts.items():
            wanted_cleanup_identity = self._active_cleanup_replay_identity(identity)
            stored_exact = stored_identity_counts.get(identity, 0)
            stored_cleanup = 0
            if wanted_cleanup_identity is not None:
                stored_cleanup = stored_cleanup_identity_counts.get(wanted_cleanup_identity, 0)
            stored_available = max(stored_exact, stored_cleanup)
            if active_count > stored_available:
                surplus_count = active_count - stored_available
                for msg in messages:
                    if surplus_count <= 0:
                        break
                    if id(msg) not in generated_placeholder_message_ids:
                        continue
                    if self._message_replay_identity(msg) != identity:
                        continue
                    generated_surplus_skip_message_ids.add(id(msg))
                    surplus_count -= 1
                if surplus_count > 0:
                    active_surplus_skips[identity] = surplus_count

        placeholder_identity_counts: dict[tuple[str, str, str, str, str], int] = {}
        for msg in messages:
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                placeholder_identity_counts[raw_identity] = placeholder_identity_counts.get(raw_identity, 0) + 1
        self._current_compress_placeholder_identity_counts = placeholder_identity_counts

        def find_raw_placeholder_match_index(
            raw_identity: tuple[str, str, str, str, str],
            start_idx: int,
        ) -> int | None:
            probe_idx = start_idx
            while probe_idx < len(candidates):
                if stored_raw_placeholder_identity(probe_idx) == raw_identity:
                    return probe_idx
                probe_idx += 1
            return None

        def find_message_match_index(msg: Dict[str, Any], start_idx: int) -> int | None:
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                raw_match_idx = find_raw_placeholder_match_index(raw_identity, start_idx)
                if raw_match_idx is not None:
                    return raw_match_idx

            message_identity = self._message_replay_identity(msg)
            wanted_cleanup_identity = self._active_cleanup_replay_identity(message_identity)
            probe_idx = start_idx
            while probe_idx < len(candidates):
                stored_identity = stored_identities[probe_idx]
                if stored_identity == message_identity:
                    return probe_idx
                if (
                    wanted_cleanup_identity is not None
                    and stored_cleanup_identities[probe_idx] == wanted_cleanup_identity
                ):
                    return probe_idx
                probe_idx += 1
            return None

        def matched_remaining_message_ids(
            message_start_idx: int,
            start_store_idx: int,
            surplus_skips: dict[tuple[Any, ...], int],
        ) -> set[int]:
            matched_message_ids: set[int] = set()
            local_surplus_skips = dict(surplus_skips)
            probe_idx = start_store_idx
            for remaining_msg in messages[message_start_idx:]:
                msg_content = normalize_content_value(remaining_msg.get("content")) or ""
                if (
                    remaining_msg.get("store_id") is None
                    and self._content_has_externalized_placeholder_ref(msg_content)
                ):
                    raw_identity = self._raw_externalized_placeholder_replay_identity(remaining_msg)
                    raw_match_idx = find_raw_placeholder_match_index(raw_identity, probe_idx)
                    if raw_match_idx is not None:
                        matched_message_ids.add(id(remaining_msg))
                        probe_idx = raw_match_idx + 1
                        continue
                message_identity = self._message_replay_identity(remaining_msg)
                if id(remaining_msg) in generated_surplus_skip_message_ids:
                    continue
                surplus = local_surplus_skips.get(message_identity, 0)
                if surplus > 0:
                    local_surplus_skips[message_identity] = surplus - 1
                    continue
                match_idx = find_message_match_index(remaining_msg, probe_idx)
                if match_idx is None:
                    continue
                matched_message_ids.add(id(remaining_msg))
                probe_idx = match_idx + 1
            return matched_message_ids

        ids_by_message_id: dict[int, int] = {}
        store_idx = 0
        for msg_idx, msg in enumerate(messages):
            msg_content = normalize_content_value(msg.get("content")) or ""
            if msg.get("store_id") is None and self._content_has_externalized_placeholder_ref(msg_content):
                raw_identity = self._raw_externalized_placeholder_replay_identity(msg)
                if placeholder_identity_counts.get(raw_identity, 0) > 1:
                    match_idx = find_raw_placeholder_match_index(raw_identity, store_idx)
                    if match_idx is not None:
                        ids_by_message_id[id(msg)] = candidates[match_idx]["store_id"]
                        store_idx = match_idx + 1
                else:
                    # Prefer a later duplicate only when it does not orphan
                    # later active messages that still need monotonic mapping.
                    first_match_idx = find_raw_placeholder_match_index(raw_identity, store_idx)
                    if first_match_idx is not None:
                        baseline_suffix_ids = matched_remaining_message_ids(
                            msg_idx + 1,
                            first_match_idx + 1,
                            active_surplus_skips,
                        )
                    else:
                        baseline_suffix_ids = set()
                    probe_idx = len(candidates) - 1
                    while first_match_idx is not None and probe_idx >= first_match_idx:
                        stored = candidates[probe_idx]
                        if stored_raw_placeholder_identity(probe_idx) == raw_identity:
                            candidate_suffix_ids = matched_remaining_message_ids(
                                msg_idx + 1,
                                probe_idx + 1,
                                active_surplus_skips,
                            )
                            if not baseline_suffix_ids.issubset(candidate_suffix_ids):
                                probe_idx -= 1
                                continue
                            ids_by_message_id[id(msg)] = stored["store_id"]
                            store_idx = probe_idx + 1
                            break
                        probe_idx -= 1
                if id(msg) in ids_by_message_id:
                    continue
            message_identity = self._message_replay_identity(msg)
            if id(msg) in generated_surplus_skip_message_ids:
                continue
            surplus = active_surplus_skips.get(message_identity, 0)
            if surplus > 0:
                active_surplus_skips[message_identity] = surplus - 1
                continue
            match_idx = find_message_match_index(msg, store_idx)
            if match_idx is not None:
                ids_by_message_id[id(msg)] = candidates[match_idx]["store_id"]
                store_idx = match_idx + 1

        return ids_by_message_id
