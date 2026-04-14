"""Image folder scanner — finds and extracts metadata from images."""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Generator

from PIL import Image
from PIL.ExifTags import TAGS

from .config import SUPPORTED_EXTENSIONS


def is_image(path: Path) -> bool:
    """Check if a file is a supported image format."""
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def file_hash(path: Path) -> str:
    """Generate a hash for a file to detect duplicates."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def scan_folder(folder: str | Path) -> Generator[dict, None, None]:
    """Recursively scan a folder and yield metadata for each image.

    Yields dicts with:
        path: absolute path to the image
        filename: just the filename
        size_bytes: file size
        modified: last modified timestamp
        dimensions: (width, height) or None
        format: image format (JPEG, PNG, etc.)
        exif: dict of EXIF data or empty dict
        hash: MD5 hash of file contents
    """
    folder = Path(folder).expanduser().resolve()

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    for root, _dirs, files in os.walk(folder):
        for fname in sorted(files):
            filepath = Path(root) / fname

            if not is_image(filepath):
                continue

            try:
                stat = filepath.stat()
                img = Image.open(filepath)

                # Extract EXIF data
                exif = {}
                if hasattr(img, "_getexif") and img._getexif():
                    for tag_id, value in img._getexif().items():
                        tag = TAGS.get(tag_id, tag_id)
                        # Skip binary data
                        if isinstance(value, (str, int, float)):
                            exif[tag] = str(value)

                yield {
                    "path": str(filepath),
                    "filename": fname,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    "dimensions": img.size,
                    "format": img.format,
                    "exif": exif,
                    "hash": file_hash(filepath),
                }

                img.close()

            except Exception as e:
                # Skip corrupted or unreadable images
                print(f"  ⚠️ Skipping {filepath}: {e}")
                continue


def count_images(folder: str | Path) -> int:
    """Count total images in a folder."""
    folder = Path(folder).expanduser().resolve()
    count = 0
    for root, _dirs, files in os.walk(folder):
        for fname in files:
            if is_image(Path(root) / fname):
                count += 1
    return count
