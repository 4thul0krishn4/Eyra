"""Web server — FastAPI backend for the Eyra search UI."""

import json
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_RESULTS, THUMBNAILS_DIR, ensure_dirs
from .embedder import Embedder
from .indexer import Indexer
from .metadata import MetadataStore
from .search import SearchEngine
from .tasks import get_task_queue
from .thumbnails import generate_thumbnail


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    ensure_dirs()

    app = FastAPI(title="Eyra", version="0.3.0")
    embedder = Embedder()
    indexer = Indexer()
    meta = indexer.metadata_store
    engine = SearchEngine(indexer, embedder)

    # Web UI HTML
    WEB_DIR = Path(__file__).parent.parent / "web"
    if not WEB_DIR.exists():
        WEB_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/", response_class=HTMLResponse)
    async def home():
        index_html = WEB_DIR / "index.html"
        if index_html.exists():
            return index_html.read_text()
        return "<html><body><h1>Eyra</h1><p>Web UI not found. Run from project root.</p></body></html>"

    @app.get("/api/search")
    async def api_search(
        q: str = Query(..., description="Search query"),
        limit: int = Query(DEFAULT_RESULTS, ge=1, le=100),
        min_similarity: float = Query(0.1, ge=0.0, le=1.0),
        mode: str = Query("hybrid", description="Search mode: hybrid, vector, keyword"),
    ):
        """Search images by natural language."""
        results = engine.search(query=q, n_results=limit, min_similarity=min_similarity, mode=mode)

        # Add thumbnail URLs and extract caption/tags
        for r in results:
            thumb = generate_thumbnail(r["path"])
            r["thumbnail"] = f"/api/thumbnail?path={r['path']}" if thumb else None
            r["similarity"] = round(r["similarity"], 4)
            r["filename"] = r["metadata"].get("filename", "")
            # Surface caption and tags at top level
            r["caption"] = r["metadata"].get("caption", "")
            tags_str = r["metadata"].get("tags", "[]")
            try:
                r["tags"] = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            except (json.JSONDecodeError, TypeError):
                r["tags"] = []

        return {"query": q, "results": results, "count": len(results)}

    @app.get("/api/similar")
    async def api_similar(
        path: str = Query(..., description="Path to reference image"),
        limit: int = Query(10, ge=1, le=50),
    ):
        """Find similar images."""
        results = engine.find_similar(image_path=path, n_results=limit)

        for r in results:
            thumb = generate_thumbnail(r["path"])
            r["thumbnail"] = f"/api/thumbnail?path={r['path']}" if thumb else None
            r["similarity"] = round(r["similarity"], 4)
            r["filename"] = r["metadata"].get("filename", "")
            r["caption"] = r["metadata"].get("caption", "")
            tags_str = r["metadata"].get("tags", "[]")
            try:
                r["tags"] = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            except (json.JSONDecodeError, TypeError):
                r["tags"] = []

        return {"reference": path, "results": results, "count": len(results)}

    @app.get("/api/stats")
    async def api_stats():
        """Get index statistics (fast, from SQLite)."""
        return meta.stats()

    @app.get("/api/tags")
    async def api_tags(
        min_count: int = Query(1, ge=1, description="Minimum tag occurrence count"),
        limit: int = Query(100, ge=1, le=500),
    ):
        """Get all tags with their counts."""
        return {"tags": meta.all_tags(min_count=min_count, limit=limit)}

    @app.get("/api/search/text")
    async def api_fts_search(
        q: str = Query(..., description="Full-text search query"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        """Full-text search on captions and tags (SQLite FTS5)."""
        result = meta.search(q, limit=limit, offset=offset)

        for img in result["images"]:
            thumb = generate_thumbnail(img["path"])
            img["thumbnail"] = f"/api/thumbnail?path={img['path']}" if thumb else None

        return result

    @app.get("/api/images")
    async def api_images(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
        order_by: str = Query("indexed_at", description="Sort field: indexed_at, filename, size_bytes, width, height, modified"),
        order_dir: str = Query("DESC", description="Sort direction: ASC or DESC"),
        fmt: str = Query(None, description="Filter by image format (JPEG, PNG, etc.)"),
        min_width: int = Query(None, description="Minimum image width in pixels"),
        min_height: int = Query(None, description="Minimum image height in pixels"),
        has_caption: bool = Query(None, description="Filter to captioned/uncaptioned images"),
        tag: str = Query(None, description="Filter by tag"),
    ):
        """Get indexed images with filtering and pagination (powered by SQLite)."""
        result = meta.list_images(
            offset=offset,
            limit=limit,
            order_by=order_by,
            order_dir=order_dir,
            fmt=fmt,
            min_width=min_width,
            min_height=min_height,
            has_caption=has_caption,
            tag=tag,
        )

        for img in result["images"]:
            thumb = generate_thumbnail(img["path"])
            img["thumbnail"] = f"/api/thumbnail?path={img['path']}" if thumb else None

        return result

    @app.get("/api/thumbnail")
    async def api_thumbnail(path: str = Query(...)):
        """Get a thumbnail for an image."""
        thumb_path = generate_thumbnail(path)
        if thumb_path and thumb_path.exists():
            return FileResponse(str(thumb_path), media_type="image/jpeg")
        # Fallback: return the original image
        original = Path(path)
        if original.exists():
            return FileResponse(str(original))
        raise HTTPException(status_code=404, detail="Image not found")

    @app.get("/api/caption")
    async def api_caption(
        path: str = Query(..., description="Path to image to caption"),
        backend: str = Query("florence2", description="Captioning backend"),
    ):
        """Generate caption and tags for a single image on-demand."""
        img_path = Path(path).resolve()
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        from .captioner import Captioner
        captioner = Captioner(backend=backend)
        desc = captioner.describe(str(img_path))

        return {
            "path": str(img_path),
            "caption": desc["caption"],
            "tags": desc["tags"],
            "backend": desc["backend"],
            "model": desc["model"],
        }

    @app.get("/api/image")
    async def api_image(path: str = Query(...)):
        """Serve the full-resolution image."""
        img_path = Path(path).resolve()
        if not img_path.exists():
            raise HTTPException(status_code=404, detail="Image not found")

        media_types = {
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".gif": "image/gif",
            ".webp": "image/webp", ".bmp": "image/bmp",
        }
        media_type = media_types.get(img_path.suffix.lower(), "image/jpeg")
        return FileResponse(str(img_path), media_type=media_type)

    # --- Background Task Endpoints ---

    @app.post("/api/tasks/index")
    async def api_task_index(
        folder: str = Query(..., description="Folder to index"),
        batch_size: int = Query(32, ge=1, le=128),
        auto_caption: bool = Query(False, description="Also generate captions"),
        backend: str = Query("florence2", description="Captioning backend"),
    ):
        """Submit a background indexing task."""
        queue = get_task_queue()
        task_id = queue.submit("index", {
            "folder": folder,
            "batch_size": batch_size,
            "auto_caption": auto_caption,
            "backend": backend,
        })
        return {"task_id": task_id, "status": "submitted"}

    @app.post("/api/tasks/caption")
    async def api_task_caption(
        folder: str = Query("", description="Folder to caption (empty = all)"),
        backend: str = Query("florence2"),
        reindex: bool = Query(False, description="Re-caption existing"),
        limit: int = Query(0, ge=0, description="Max images (0 = all)"),
    ):
        """Submit a background captioning task."""
        queue = get_task_queue()
        task_id = queue.submit("caption", {
            "folder": folder,
            "backend": backend,
            "reindex": reindex,
            "limit": limit,
        })
        return {"task_id": task_id, "status": "submitted"}

    @app.get("/api/tasks")
    async def api_tasks():
        """Get all active tasks."""
        queue = get_task_queue()
        return {"tasks": queue.get_active_tasks()}

    @app.get("/api/tasks/history")
    async def api_task_history(limit: int = Query(20, ge=1, le=100)):
        """Get recent task history."""
        queue = get_task_queue()
        return {"tasks": queue.get_recent_tasks(limit=limit)}

    @app.get("/api/tasks/{task_id}")
    async def api_task_status(task_id: str):
        """Get status of a specific task."""
        queue = get_task_queue()
        task = queue.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task

    @app.delete("/api/tasks/{task_id}")
    async def api_task_cancel(task_id: str):
        """Cancel a pending task."""
        queue = get_task_queue()
        if queue.cancel(task_id):
            return {"cancelled": True}
        raise HTTPException(status_code=400, detail="Task not found or already running")

    return app
