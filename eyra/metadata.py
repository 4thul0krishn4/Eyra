"""SQLite sidecar database — fast metadata storage, filtering, and full-text search.

Works alongside ChromaDB: ChromaDB stores vectors, SQLite stores metadata.
This gives us fast filtering by tags/format/dimensions/dates without loading
all vectors from ChromaDB.
"""

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from .config import EYRA_DIR

DB_PATH = EYRA_DIR / "metadata.db"


class MetadataStore:
    """SQLite-backed metadata store with FTS5 full-text search."""

    def __init__(self, db_path: Optional[str | Path] = None):
        self.db_path = str(db_path or DB_PATH)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get a thread-local database connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _transaction(self):
        """Context manager for transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                path TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                width INTEGER DEFAULT 0,
                height INTEGER DEFAULT 0,
                format TEXT DEFAULT '',
                caption TEXT DEFAULT '',
                tags_json TEXT DEFAULT '[]',
                modified TEXT DEFAULT '',
                indexed_at TEXT DEFAULT (datetime('now')),
                has_caption INTEGER DEFAULT 0,
                ocr_text TEXT DEFAULT '',
                date_taken TEXT DEFAULT '',
                exif_json TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_images_format ON images(format);
            CREATE INDEX IF NOT EXISTS idx_images_has_caption ON images(has_caption);
            CREATE INDEX IF NOT EXISTS idx_images_size ON images(size_bytes);
            CREATE INDEX IF NOT EXISTS idx_images_width ON images(width);
            CREATE INDEX IF NOT EXISTS idx_images_height ON images(height);

            -- FTS5 virtual table for caption full-text search
            CREATE VIRTUAL TABLE IF NOT EXISTS images_fts USING fts5(
                path,
                caption,
                tags,
                ocr_text,
                content='images',
                content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS images_ai AFTER INSERT ON images BEGIN
                INSERT INTO images_fts(rowid, path, caption, tags, ocr_text)
                VALUES (new.rowid, new.path, new.caption, new.tags_json, new.ocr_text);
            END;

            CREATE TRIGGER IF NOT EXISTS images_ad AFTER DELETE ON images BEGIN
                INSERT INTO images_fts(images_fts, rowid, path, caption, tags, ocr_text)
                VALUES ('delete', old.rowid, old.path, old.caption, old.tags_json, old.ocr_text);
            END;

            CREATE TRIGGER IF NOT EXISTS images_au AFTER UPDATE ON images BEGIN
                INSERT INTO images_fts(images_fts, rowid, path, caption, tags, ocr_text)
                VALUES ('delete', old.rowid, old.path, old.caption, old.tags_json, old.ocr_text);
                INSERT INTO images_fts(rowid, path, caption, tags, ocr_text)
                VALUES (new.rowid, new.path, new.caption, new.tags_json, new.ocr_text);
            END;
        """)
        conn.commit()

    def add_image(self, path: str, filename: str, size_bytes: int = 0,
                  width: int = 0, height: int = 0, fmt: str = "",
                  caption: str = "", tags: list[str] = None,
                  modified: str = "", ocr_text: str = "",
                  date_taken: str = "", exif: dict = None):
        """Add or update an image's metadata."""
        tags_json = json.dumps(tags or [])
        has_caption = 1 if caption.strip() else 0
        exif_json = json.dumps(exif or {})

        with self._transaction() as conn:
            conn.execute("""
                INSERT INTO images (path, filename, size_bytes, width, height,
                                    format, caption, tags_json, modified, has_caption,
                                    ocr_text, date_taken, exif_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    filename = excluded.filename,
                    size_bytes = excluded.size_bytes,
                    width = excluded.width,
                    height = excluded.height,
                    format = excluded.format,
                    caption = excluded.caption,
                    tags_json = excluded.tags_json,
                    modified = excluded.modified,
                    has_caption = excluded.has_caption,
                    ocr_text = excluded.ocr_text,
                    date_taken = excluded.date_taken,
                    exif_json = excluded.exif_json,
                    indexed_at = datetime('now')
            """, (path, filename, size_bytes, width, height, fmt,
                  caption, tags_json, modified, has_caption,
                  ocr_text, date_taken, exif_json))

    def add_batch(self, records: list[dict]):
        """Add/update multiple images at once.

        Each record: {path, filename, size_bytes, width, height, format, caption, tags, modified}
        """
        with self._transaction() as conn:
            for r in records:
                tags = r.get("tags", [])
                tags_json = json.dumps(tags) if isinstance(tags, list) else tags
                caption = r.get("caption", "")
                has_caption = 1 if caption.strip() else 0
                exif = r.get("exif", {})
                exif_json = json.dumps(exif) if isinstance(exif, dict) else exif

                conn.execute("""
                    INSERT INTO images (path, filename, size_bytes, width, height,
                                        format, caption, tags_json, modified, has_caption,
                                        ocr_text, date_taken, exif_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(path) DO UPDATE SET
                        filename = excluded.filename,
                        size_bytes = excluded.size_bytes,
                        width = excluded.width,
                        height = excluded.height,
                        format = excluded.format,
                        caption = excluded.caption,
                        tags_json = excluded.tags_json,
                        modified = excluded.modified,
                        has_caption = excluded.has_caption,
                        ocr_text = excluded.ocr_text,
                        date_taken = excluded.date_taken,
                        exif_json = excluded.exif_json,
                        indexed_at = datetime('now')
                """, (r["path"], r["filename"], r.get("size_bytes", 0),
                      r.get("width", 0), r.get("height", 0), r.get("format", ""),
                      caption, tags_json, r.get("modified", ""), has_caption,
                      r.get("ocr_text", ""), r.get("date_taken", ""), exif_json))

    def delete_image(self, path: str):
        """Remove an image from the metadata store."""
        with self._transaction() as conn:
            conn.execute("DELETE FROM images WHERE path = ?", (path,))

    def exists(self, path: str) -> bool:
        """Check if an image exists in the metadata store."""
        conn = self._get_conn()
        row = conn.execute("SELECT 1 FROM images WHERE path = ?", (path,)).fetchone()
        return row is not None

    def count(self) -> int:
        """Total number of indexed images."""
        conn = self._get_conn()
        return conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

    def get_image(self, path: str) -> Optional[dict]:
        """Get metadata for a single image."""
        conn = self._get_conn()
        row = conn.execute("SELECT * FROM images WHERE path = ?", (path,)).fetchone()
        if row:
            return self._row_to_dict(row)
        return None

    def list_images(self, offset: int = 0, limit: int = 50,
                    order_by: str = "indexed_at", order_dir: str = "DESC",
                    fmt: str = None, min_width: int = None,
                    min_height: int = None, has_caption: bool = None,
                    tag: str = None) -> dict:
        """List images with filtering and pagination.

        Returns: {images: [...], total: int, offset: int, limit: int}
        """
        conn = self._get_conn()

        # Build WHERE clause
        conditions = []
        params = []

        if fmt:
            conditions.append("format = ?")
            params.append(fmt.upper())
        if min_width is not None:
            conditions.append("width >= ?")
            params.append(min_width)
        if min_height is not None:
            conditions.append("height >= ?")
            params.append(min_height)
        if has_caption is not None:
            conditions.append("has_caption = ?")
            params.append(1 if has_caption else 0)
        if tag:
            # JSON array contains check
            conditions.append("tags_json LIKE ?")
            params.append(f'%"{tag}"%')

        where = " AND ".join(conditions) if conditions else "1=1"

        # Validate order_by to prevent injection
        valid_order = {"indexed_at", "filename", "size_bytes", "width", "height", "modified"}
        if order_by not in valid_order:
            order_by = "indexed_at"
        order_dir = "DESC" if order_dir.upper() == "DESC" else "ASC"

        # Get total count
        total = conn.execute(
            f"SELECT COUNT(*) FROM images WHERE {where}", params
        ).fetchone()[0]

        # Get page
        rows = conn.execute(
            f"SELECT * FROM images WHERE {where} ORDER BY {order_by} {order_dir} LIMIT ? OFFSET ?",
            params + [limit, offset]
        ).fetchall()

        return {
            "images": [self._row_to_dict(r) for r in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def search(self, query: str, limit: int = 50, offset: int = 0) -> dict:
        """Full-text search on captions and tags using FTS5.

        Returns: {images: [...], total: int, offset: int, limit: int}
        """
        conn = self._get_conn()

        # FTS5 search
        try:
            total_row = conn.execute(
                "SELECT COUNT(*) FROM images_fts WHERE images_fts MATCH ?",
                (query,)
            ).fetchone()
            total = total_row[0] if total_row else 0

            rows = conn.execute("""
                SELECT i.* FROM images i
                JOIN images_fts f ON i.rowid = f.rowid
                WHERE images_fts MATCH ?
                ORDER BY rank
                LIMIT ? OFFSET ?
            """, (query, limit, offset)).fetchall()
        except sqlite3.OperationalError:
            # FTS query syntax error — fall back to LIKE
            like_q = f"%{query}%"
            total_row = conn.execute(
                "SELECT COUNT(*) FROM images WHERE caption LIKE ? OR tags_json LIKE ?",
                (like_q, like_q)
            ).fetchone()
            total = total_row[0] if total_row else 0

            rows = conn.execute(
                "SELECT * FROM images WHERE caption LIKE ? OR tags_json LIKE ? LIMIT ? OFFSET ?",
                (like_q, like_q, limit, offset)
            ).fetchall()

        return {
            "images": [self._row_to_dict(r) for r in rows],
            "total": total,
            "offset": offset,
            "limit": limit,
        }

    def all_tags(self, min_count: int = 1, limit: int = 100) -> list[dict]:
        """Get all tags with their counts.

        Returns: [{tag: str, count: int}, ...]
        """
        conn = self._get_conn()
        rows = conn.execute("SELECT tags_json FROM images WHERE tags_json != '[]'").fetchall()

        tag_counts = {}
        for row in rows:
            try:
                tags = json.loads(row["tags_json"])
                for tag in tags:
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                continue

        result = [
            {"tag": tag, "count": count}
            for tag, count in tag_counts.items()
            if count >= min_count
        ]
        result.sort(key=lambda x: x["count"], reverse=True)
        return result[:limit]

    def stats(self) -> dict:
        """Get index statistics from SQLite (fast, no vector loading)."""
        conn = self._get_conn()

        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]

        # Format breakdown
        fmt_rows = conn.execute(
            "SELECT format, COUNT(*) as cnt FROM images WHERE format != '' GROUP BY format ORDER BY cnt DESC"
        ).fetchall()
        formats = {r["format"]: r["cnt"] for r in fmt_rows}

        # Caption stats
        captioned = conn.execute(
            "SELECT COUNT(*) FROM images WHERE has_caption = 1"
        ).fetchone()[0]

        # Top tags
        top_tags = self.all_tags(limit=20)

        # Size stats
        size_row = conn.execute(
            "SELECT AVG(size_bytes) as avg_size, SUM(size_bytes) as total_size FROM images"
        ).fetchone()

        return {
            "total_images": total,
            "captioned": captioned,
            "uncaptioned": total - captioned,
            "formats": formats,
            "top_tags": top_tags,
            "avg_size_bytes": int(size_row["avg_size"] or 0),
            "total_size_bytes": int(size_row["total_size"] or 0),
        }

    def get_all_paths(self) -> list[str]:
        """Get all indexed image paths (for sync operations)."""
        conn = self._get_conn()
        rows = conn.execute("SELECT path FROM images").fetchall()
        return [r["path"] for r in rows]

    def get_uncaptioned(self, limit: int = 0) -> list[dict]:
        """Get images that don't have captions yet."""
        conn = self._get_conn()
        query = "SELECT * FROM images WHERE has_caption = 0"
        if limit > 0:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def update_ocr(self, path: str, ocr_text: str):
        """Update the OCR text for a specific image."""
        with self._transaction() as conn:
            conn.execute("UPDATE images SET ocr_text = ? WHERE path = ?", (ocr_text, path))

    def get_unocr(self, limit: int = 0) -> list[dict]:
        """Get images that haven't been OCR'd yet."""
        conn = self._get_conn()
        query = "SELECT * FROM images WHERE ocr_text = ''"
        if limit > 0:
            query += f" LIMIT {limit}"
        rows = conn.execute(query).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def ocr_stats(self) -> dict:
        """Get OCR statistics."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM images").fetchone()[0]
        ocred = conn.execute("SELECT COUNT(*) FROM images WHERE ocr_text != ''").fetchone()[0]
        return {"total": total, "ocr_done": ocred, "ocr_pending": total - ocred}

    def timeline(self, group_by: str = "month", limit: int = 100) -> list[dict]:
        """Get timeline of images grouped by date.

        Args:
            group_by: "day", "month", or "year"
            limit: max number of groups to return

        Returns: [{date, count, images: [{path, filename, caption, tags}]}]
        """
        conn = self._get_conn()

        # Use date_taken if available, fall back to modified, then indexed_at
        date_expr = "COALESCE(NULLIF(date_taken, ''), NULLIF(modified, ''), indexed_at)"

        if group_by == "day":
            trunc = f"substr({date_expr}, 1, 10)"
        elif group_by == "year":
            trunc = f"substr({date_expr}, 1, 4)"
        else:  # month
            trunc = f"substr({date_expr}, 1, 7)"

        groups = conn.execute(f"""
            SELECT {trunc} as period, COUNT(*) as cnt
            FROM images
            WHERE {date_expr} != ''
            GROUP BY period
            ORDER BY period DESC
            LIMIT ?
        """, (limit,)).fetchall()

        result = []
        for g in groups:
            period = g["period"]
            # Get sample images from this period
            if group_by == "day":
                date_filter = f"{date_expr} LIKE '{period}%'"
            elif group_by == "year":
                date_filter = f"{date_expr} LIKE '{period}%'"
            else:
                date_filter = f"{date_expr} LIKE '{period}%'"

            imgs = conn.execute(f"""
                SELECT path, filename, caption, tags_json
                FROM images
                WHERE {date_filter}
                ORDER BY {date_expr}
                LIMIT 20
            """).fetchall()

            images = []
            for ir in imgs:
                tags = []
                try:
                    tags = json.loads(ir["tags_json"]) if ir["tags_json"] else []
                except (json.JSONDecodeError, TypeError):
                    pass
                images.append({
                    "path": ir["path"],
                    "filename": ir["filename"],
                    "caption": ir["caption"] or "",
                    "tags": tags,
                })

            result.append({
                "date": period,
                "count": g["cnt"],
                "images": images,
            })

        return result

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """Convert a SQLite row to a dict, parsing JSON fields."""
        d = dict(row)
        # Parse tags
        if "tags_json" in d:
            try:
                d["tags"] = json.loads(d["tags_json"])
            except (json.JSONDecodeError, TypeError):
                d["tags"] = []
        return d
