"""Configuration and constants for Eyra."""

import os
from pathlib import Path

# Directories
EYRA_DIR = Path.home() / ".eyra"
DB_DIR = EYRA_DIR / "db"
MODELS_DIR = EYRA_DIR / "models"
THUMBNAILS_DIR = EYRA_DIR / "thumbnails"

# Supported image formats
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif",
    ".webp", ".heic", ".heif", ".avif", ".svg",
}

# Model settings
CLIP_MODEL = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"

# ChromaDB settings
COLLECTION_NAME = "eyra_images"

# Thumbnail settings
THUMBNAIL_SIZE = (256, 256)

# Search defaults
DEFAULT_RESULTS = 20

# Web server
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080


def ensure_dirs():
    """Create all required directories."""
    for d in [EYRA_DIR, DB_DIR, MODELS_DIR, THUMBNAILS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
