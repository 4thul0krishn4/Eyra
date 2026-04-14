"""Thumbnail generator — creates small preview images for the web UI."""

from pathlib import Path

from PIL import Image

from .config import THUMBNAIL_SIZE, THUMBNAILS_DIR


def generate_thumbnail(image_path: str | Path, size: tuple[int, int] = THUMBNAIL_SIZE) -> Path:
    """Generate a thumbnail for an image and return its path.

    Thumbnails are cached in ~/.eyra/thumbnails/ with a hash-based filename.
    """
    image_path = Path(image_path).resolve()

    # Create cache filename from path hash
    import hashlib
    path_hash = hashlib.md5(str(image_path).encode()).hexdigest()[:12]
    thumb_path = THUMBNAILS_DIR / f"{path_hash}.jpg"

    # Return cached if exists and source hasn't changed
    if thumb_path.exists():
        if thumb_path.stat().st_mtime >= image_path.stat().st_mtime:
            return thumb_path

    # Generate thumbnail
    try:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(thumb_path, "JPEG", quality=85)
        return thumb_path
    except Exception as e:
        print(f"  ⚠️ Thumbnail failed for {image_path}: {e}")
        return None
