"""Obsidian export — export indexed images and metadata to Markdown notes.

Creates a structured Obsidian vault with:
- One note per image with YAML frontmatter
- Organized by tags or date
- Embedded image references
- Dataview-compatible metadata
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .metadata import MetadataStore


class ObsidianExporter:
    """Export Eyra image data to an Obsidian vault."""

    def __init__(self, meta: Optional[MetadataStore] = None):
        self.meta = meta or MetadataStore()

    def export(
        self,
        output_dir: str | Path,
        copy_images: bool = False,
        organize_by: str = "tags",
        include_ocr: bool = True,
        tag_filter: Optional[str] = None,
        limit: int = 0,
    ) -> dict:
        """Export all indexed images as Obsidian markdown notes.

        Args:
            output_dir: path to the Obsidian vault folder
            copy_images: if True, copy images into the vault; if False, use absolute paths
            organize_by: "tags" (folder per top tag), "date" (folder per month), or "flat"
            include_ocr: include OCR text in notes
            tag_filter: only export images with this tag
            limit: max images to export (0 = all)

        Returns:
            {exported: int, skipped: int, output_dir: str, tags_used: list}
        """
        output_dir = Path(output_dir).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create attachments folder
        attachments_dir = output_dir / "attachments"
        attachments_dir.mkdir(exist_ok=True)

        # Get images
        result = self.meta.list_images(limit=0, offset=0)
        images = result["images"]

        if tag_filter:
            images = [img for img in images if self._has_tag(img, tag_filter)]

        if limit > 0:
            images = images[:limit]

        exported = 0
        skipped = 0
        tags_used = set()

        for img in images:
            try:
                path = Path(img["path"])
                if not path.exists():
                    skipped += 1
                    continue

                tags = img.get("tags", [])
                if isinstance(tags, str):
                    try:
                        tags = json.loads(tags)
                    except (json.JSONDecodeError, TypeError):
                        tags = []

                # Determine subfolder
                if organize_by == "tags" and tags:
                    subfolder = tags[0].replace("/", "-")
                elif organize_by == "date" and img.get("date_taken"):
                    subfolder = img["date_taken"][:7]  # YYYY-MM
                elif organize_by == "date" and img.get("modified"):
                    subfolder = img["modified"][:7]
                else:
                    subfolder = ""

                note_dir = output_dir / subfolder if subfolder else output_dir
                note_dir.mkdir(parents=True, exist_ok=True)

                # Copy or reference image
                if copy_images:
                    dest = attachments_dir / path.name
                    if not dest.exists():
                        shutil.copy2(path, dest)
                    image_ref = f"![[attachments/{path.name}]]"
                else:
                    image_ref = f"![]({path})"

                # Build frontmatter
                frontmatter = self._build_frontmatter(img, tags)

                # Build note content
                note_name = path.stem + ".md"
                note_path = note_dir / note_name

                ocr_text = img.get("ocr_text", "")
                ocr_section = ""
                if include_ocr and ocr_text and ocr_text.strip():
                    ocr_section = f"\n## Extracted Text\n\n```\n{ocr_text.strip()}\n```\n"

                content = f"""---
{frontmatter}
---

# {path.name}

{image_ref}
{ocr_section}
## Metadata

- **Format:** {img.get('format', 'unknown')}
- **Dimensions:** {img.get('width', 0)}×{img.get('height', 0)}
- **Size:** {self._human_size(img.get('size_bytes', 0))}
- **Indexed:** {img.get('indexed_at', '')}
"""

                if img.get("caption"):
                    content += f"- **Caption:** {img['caption']}\n"

                if tags:
                    tag_strs = [f"#{t.replace(' ', '-')}" for t in tags]
                    content += f"\n## Tags\n\n{' '.join(tag_strs)}\n"
                    tags_used.update(tags)

                note_path.write_text(content, encoding="utf-8")
                exported += 1

            except Exception as e:
                print(f"  ⚠️ Skipping {img.get('path', '?')}: {e}")
                skipped += 1

        # Create index note
        self._write_index(output_dir, tags_used, organize_by)

        return {
            "exported": exported,
            "skipped": skipped,
            "output_dir": str(output_dir),
            "tags_used": sorted(tags_used),
        }

    def _build_frontmatter(self, img: dict, tags: list[str]) -> str:
        """Build YAML frontmatter for an Obsidian note."""
        lines = []
        lines.append(f'title: "{Path(img["path"]).stem}"')
        lines.append(f'source: "{img["path"]}"')

        if img.get("caption"):
            # Escape quotes in caption
            caption = img["caption"].replace('"', '\\"')
            lines.append(f'caption: "{caption}"')

        if tags:
            tag_list = ", ".join(f'"{t}"' for t in tags)
            lines.append(f"tags: [{tag_list}]")

        if img.get("date_taken"):
            lines.append(f"date_taken: {img['date_taken']}")

        if img.get("modified"):
            lines.append(f"modified: {img['modified']}")

        lines.append(f"indexed_at: {img.get('indexed_at', '')}")
        lines.append(f"format: {img.get('format', '')}")
        lines.append(f"width: {img.get('width', 0)}")
        lines.append(f"height: {img.get('height', 0)}")
        lines.append(f"size_bytes: {img.get('size_bytes', 0)}")

        return "\n".join(lines)

    def _write_index(self, output_dir: Path, tags: set, organize_by: str):
        """Create an index/MOC note."""
        content = f"""---
title: "Eyra Export Index"
created: {datetime.now().isoformat()}
---

# 📸 Eyra Image Library

Exported from Eyra on {datetime.now().strftime('%Y-%m-%d %H:%M')}

## By Tags

"""
        for tag in sorted(tags):
            safe_tag = tag.replace("/", "-")
            if organize_by == "tags":
                content += f"- [[{safe_tag}/{safe_tag}|{tag}]]\n"
            else:
                content += f"- `#{safe_tag}`\n"

        content += f"""
## Stats

- **Total tags:** {len(tags)}
- **Organized by:** {organize_by}

---

Generated by [Eyra](https://github.com/4thul0krishn4/Eyra)
"""
        (output_dir / "INDEX.md").write_text(content, encoding="utf-8")

    def _has_tag(self, img: dict, tag: str) -> bool:
        """Check if image has a specific tag."""
        tags = img.get("tags", [])
        if isinstance(tags, str):
            try:
                tags = json.loads(tags)
            except (json.JSONDecodeError, TypeError):
                tags = []
        return any(tag.lower() in t.lower() for t in tags)

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human readable size."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f} TB"
