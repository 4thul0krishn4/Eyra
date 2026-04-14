"""Web server — FastAPI backend for the Eyra search UI."""

import json
from pathlib import Path

from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_RESULTS, THUMBNAILS_DIR, ensure_dirs
from .embedder import Embedder
from .indexer import Indexer
from .search import SearchEngine
from .thumbnails import generate_thumbnail


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    ensure_dirs()

    app = FastAPI(title="Eyra", version="0.1.0")
    embedder = Embedder()
    indexer = Indexer()
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
    ):
        """Search images by natural language."""
        results = engine.search(query=q, n_results=limit, min_similarity=min_similarity)

        # Add thumbnail URLs
        for r in results:
            thumb = generate_thumbnail(r["path"])
            r["thumbnail"] = f"/api/thumbnail?path={r['path']}" if thumb else None
            r["similarity"] = round(r["similarity"], 4)
            r["filename"] = r["metadata"].get("filename", "")

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

        return {"reference": path, "results": results, "count": len(results)}

    @app.get("/api/stats")
    async def api_stats():
        """Get index statistics."""
        return engine.stats()

    @app.get("/api/images")
    async def api_images(
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        """Get all indexed images with pagination."""
        all_images = indexer.get_all()
        page = all_images[offset:offset + limit]

        for img in page:
            thumb = generate_thumbnail(img["path"])
            img["thumbnail"] = f"/api/thumbnail?path={img['path']}" if thumb else None
            img["filename"] = img["metadata"].get("filename", "")

        return {"images": page, "total": len(all_images), "offset": offset, "limit": limit}

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

    return app
