"""Caption history persistence using SQLite.

Stores every captioning result for review, export, and comparison.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llama_video.types import CaptionResult

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS captions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT NOT NULL,
    mode        TEXT NOT NULL,
    source_path TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    template    TEXT,
    variables   TEXT,
    preset      TEXT NOT NULL,
    caption     TEXT NOT NULL,
    settings    TEXT NOT NULL,
    token_usage TEXT,
    duration_ms REAL
)
"""


class CaptionHistory:
    """SQLite-backed caption history."""

    def __init__(self, db_path: str = "~/.llama-video/captions.db") -> None:
        expanded = Path(db_path).expanduser()
        expanded.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(expanded))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_CREATE_TABLE)
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def save(self, result: CaptionResult) -> int:
        """Save a caption result. Returns the row ID."""
        cursor = self._conn.execute(
            """INSERT INTO captions
               (created_at, mode, source_path, prompt, template, variables,
                preset, caption, settings, token_usage, duration_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now(UTC).isoformat(),
                result.mode,
                result.source_path,
                result.prompt_rendered,
                result.template_name,
                json.dumps(result.variables),
                result.preset_name,
                result.caption,
                json.dumps(result.settings),
                None,  # token_usage — future
                result.duration_ms,
            ),
        )
        self._conn.commit()
        assert cursor.lastrowid is not None
        return cursor.lastrowid

    def get(self, row_id: int) -> dict[str, Any]:
        """Get a single caption by ID. Raises KeyError if not found."""
        row = self._conn.execute("SELECT * FROM captions WHERE id = ?", (row_id,)).fetchone()
        if row is None:
            raise KeyError(f"Caption {row_id} not found")
        return dict(row)

    def list_captions(
        self,
        mode: str | None = None,
        template: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List captions with optional filters."""
        query = "SELECT * FROM captions"
        params: list[Any] = []
        conditions: list[str] = []

        if mode is not None:
            conditions.append("mode = ?")
            params.append(mode)
        if template is not None:
            conditions.append("template = ?")
            params.append(template)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def export(
        self,
        ids: list[int] | None = None,
        format: str = "json",
    ) -> str:
        """Export captions as JSON or CSV string."""
        if ids is not None:
            placeholders = ",".join("?" for _ in ids)
            rows = self._conn.execute(
                f"SELECT * FROM captions WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM captions ORDER BY id").fetchall()

        records = [dict(row) for row in rows]

        if format == "json":
            return json.dumps(records, indent=2)
        elif format == "csv":
            if not records:
                return ""
            output = io.StringIO()
            writer = csv.DictWriter(output, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported format: {format!r}. Use 'json' or 'csv'.")

    def scrub(self) -> int:
        """Delete all captions. Returns count deleted."""
        cursor = self._conn.execute("SELECT COUNT(*) FROM captions")
        count = cursor.fetchone()[0]
        self._conn.execute("DELETE FROM captions")
        self._conn.commit()
        return int(count)
