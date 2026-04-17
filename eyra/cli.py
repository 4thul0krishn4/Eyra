"""Eyra CLI — command line interface for the image memory system."""

import json
import sys
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from .config import DEFAULT_RESULTS, DEFAULT_HOST, DEFAULT_PORT, ensure_dirs, CAPTION_BACKEND
from .embedder import Embedder
from .indexer import Indexer
from .scanner import scan_folder, count_images
from .search import SearchEngine

app = typer.Typer(
    name="eyra",
    help="Local-first AI image memory — turns any folder into a searchable visual knowledge base.",
    no_args_is_help=True,
)
console = Console()


def get_components():
    """Initialize and return the core components."""
    ensure_dirs()
    embedder = Embedder()
    indexer = Indexer()
    engine = SearchEngine(indexer, embedder)
    return embedder, indexer, engine


@app.command()
def index(
    folder: str = typer.Argument(..., help="Path to the folder containing images"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size for embedding generation"),
    auto_caption: bool = typer.Option(False, "--auto-caption", "-c", help="Generate captions and tags during indexing"),
    backend: str = typer.Option(CAPTION_BACKEND, "--backend", help="Captioning backend: florence2 or blip2"),
    background: bool = typer.Option(False, "--background", "-g", help="Run indexing in background (server must be running)"),
):
    """Index all images in a folder for search."""
    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        console.print(f"[red]Error:[/red] Folder not found: {folder_path}")
        raise typer.Exit(1)

    if background:
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({
            "folder": str(folder_path),
            "batch_size": batch_size,
            "auto_caption": auto_caption,
            "backend": backend,
        })
        url = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/tasks/index?{params}"

        try:
            req = urllib.request.Request(url, method="POST")
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
            console.print(f"\n[bold]Eyra Index (Background)[/bold]")
            console.print(f"  Task submitted: {data['task_id']}")
            console.print(f"  Check status: curl http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/tasks/{data['task_id']}")
            console.print(f"  Or visit the web UI\n")
        except Exception as e:
            console.print(f"[red]Failed to submit task:[/red] {e}")
            console.print(f"[dim]Is the server running? Start with: eyra serve[/dim]")
        return

    # Synchronous indexing (existing logic)
    embedder, indexer, _ = get_components()

    # Count images first
    total = count_images(folder_path)
    if total == 0:
        console.print("[yellow]No images found in folder.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Eyra Indexing[/bold]")
    console.print(f"  Folder: {folder_path}")
    console.print(f"  Images: {total}")
    console.print(f"  Batch size: {batch_size}\n")

    # Load model
    embedder.load()

    # Load captioner if auto-captioning
    captioner = None
    if auto_caption:
        from .captioner import Captioner
        captioner = Captioner(backend=backend)
        captioner.load()
        console.print(f"  Captioner: {backend} ✅\n")

    # Scan and index
    indexed = 0
    skipped = 0
    batch_paths = []
    batch_metadatas = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Indexing images...", total=total)

        for img_data in scan_folder(folder_path):
            img_path = img_data["path"]

            # Skip if already indexed
            if indexer.exists(img_path):
                skipped += 1
                progress.advance(task)
                continue

            batch_paths.append(img_path)

            # Generate tags and caption if auto-captioning
            img_tags = []
            img_caption = ""
            if captioner:
                try:
                    desc = captioner.describe(img_path)
                    img_tags = desc["tags"]
                    img_caption = desc["caption"]
                except Exception:
                    pass  # Fallback to empty if captioning fails

            batch_metadatas.append({
                "filename": img_data["filename"],
                "size_bytes": img_data["size_bytes"],
                "modified": img_data["modified"],
                "width": img_data["dimensions"][0] if img_data["dimensions"] else 0,
                "height": img_data["dimensions"][1] if img_data["dimensions"] else 0,
                "format": img_data["format"] or "",
                "tags": json.dumps(img_tags),
                "caption": img_caption,
            })

            # Process batch when full
            if len(batch_paths) >= batch_size:
                embeddings = embedder.embed_batch(batch_paths)
                indexer.add_batch(
                    image_ids=batch_paths,
                    embeddings=embeddings,
                    metadatas=batch_metadatas,
                )
                indexed += len(batch_paths)
                progress.advance(task, len(batch_paths))
                batch_paths = []
                batch_metadatas = []

        # Process remaining
        if batch_paths:
            embeddings = embedder.embed_batch(batch_paths)
            indexer.add_batch(
                image_ids=batch_paths,
                embeddings=embeddings,
                metadatas=batch_metadatas,
            )
            indexed += len(batch_paths)
            progress.advance(task, len(batch_paths))

    console.print(f"\n[green]✅ Done![/green] Indexed: {indexed}, Skipped (already indexed): {skipped}")
    console.print(f"  Total in index: {indexer.count()}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (natural language)"),
    limit: int = typer.Option(DEFAULT_RESULTS, "--limit", "-n", help="Max results to show"),
    min_similarity: float = typer.Option(0.1, "--min-sim", help="Minimum similarity score (0-1)"),
    mode: str = typer.Option("hybrid", "--mode", "-m", help="Search mode: hybrid, vector, keyword"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search images by describing what you're looking for."""
    _, _, engine = get_components()

    results = engine.search(
        query=query,
        n_results=limit,
        min_similarity=min_similarity,
        mode=mode,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    if json_output:
        # Clean up for JSON output
        clean = []
        for r in results:
            meta = r.get("metadata", {})
            tags_str = meta.get("tags", "[]")
            try:
                tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            except (json.JSONDecodeError, TypeError):
                tags = []
            clean.append({
                "path": r["path"],
                "similarity": round(r["similarity"], 4),
                "filename": meta.get("filename", ""),
                "caption": meta.get("caption", ""),
                "tags": tags,
            })
        print(json.dumps(clean, indent=2))
    else:
        console.print(f"\n[bold]Search results for:[/bold] \"{query}\" [dim]({mode})[/dim]\n")
        table = Table(show_header=True, header_style="bold")
        table.add_column("#", style="dim", width=3)
        table.add_column("Score", width=8)
        table.add_column("File", style="cyan")
        table.add_column("Caption", style="dim", max_width=50)
        table.add_column("Tags", style="magenta", max_width=30)

        for i, r in enumerate(results, 1):
            sim = f"{r['similarity']:.1%}"
            meta = r.get("metadata", {})
            filename = meta.get("filename", "")
            caption = meta.get("caption", "")
            if len(caption) > 48:
                caption = caption[:48] + "…"

            tags_str = meta.get("tags", "[]")
            try:
                tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            except (json.JSONDecodeError, TypeError):
                tags = []
            tags_display = ", ".join(tags[:5]) if tags else ""

            table.add_row(str(i), sim, filename, caption, tags_display)

        console.print(table)


@app.command()
def tags(
    tag_query: str = typer.Argument(..., help="Tag or keyword to search for"),
    limit: int = typer.Option(DEFAULT_RESULTS, "--limit", "-n", help="Max results"),
):
    """Search images by tag or keyword."""
    _, _, engine = get_components()

    results = engine.search(
        query=tag_query,
        n_results=limit,
        mode="keyword",
    )

    if not results:
        console.print(f"[yellow]No images tagged with '{tag_query}'.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Images matching tag:[/bold] \"{tag_query}\"\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", width=8)
    table.add_column("File", style="cyan")
    table.add_column("Caption", style="dim", max_width=50)
    table.add_column("Tags", style="magenta", max_width=30)

    for i, r in enumerate(results, 1):
        score = f"{r.get('keyword_score', r['similarity']):.1%}"
        meta = r.get("metadata", {})
        filename = meta.get("filename", "")
        caption = meta.get("caption", "")
        if len(caption) > 48:
            caption = caption[:48] + "…"

        tags_str = meta.get("tags", "[]")
        try:
            img_tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
        except (json.JSONDecodeError, TypeError):
            img_tags = []
        tags_display = ", ".join(img_tags[:5]) if img_tags else ""

        table.add_row(str(i), score, filename, caption, tags_display)

    console.print(table)


@app.command()
def stats():
    """Show index statistics."""
    from .metadata import MetadataStore

    meta = MetadataStore()
    info = meta.stats()

    console.print(f"\n[bold]Eyra Stats[/bold]\n")
    console.print(f"  Total indexed images: {info['total_images']}")
    console.print(f"  Captioned: {info['captioned']}, Uncaptioned: {info['uncaptioned']}")

    if info["total_size_bytes"] > 0:
        size_mb = info["total_size_bytes"] / (1024 * 1024)
        avg_kb = info["avg_size_bytes"] / 1024
        console.print(f"  Total size: {size_mb:.1f} MB (avg {avg_kb:.0f} KB/image)")

    if info["formats"]:
        console.print(f"\n  Formats:")
        for fmt, count in info["formats"].items():
            console.print(f"    {fmt}: {count}")

    if info["top_tags"]:
        console.print(f"\n  Top tags:")
        for t in info["top_tags"][:10]:
            console.print(f"    {t['tag']}: {t['count']}")


@app.command()
def sync():
    """Sync SQLite metadata from ChromaDB (backfill for upgrades)."""
    from .metadata import MetadataStore

    _, indexer, _ = get_components()
    meta = MetadataStore()

    console.print(f"\n[bold]Eyra Sync[/bold]")
    console.print(f"  Syncing SQLite metadata from ChromaDB...\n")

    all_images = indexer.get_all()
    total = len(all_images)

    if total == 0:
        console.print("[yellow]No images in ChromaDB index.[/yellow]")
        raise typer.Exit(0)

    records = []
    for img in all_images:
        meta_d = img.get("metadata", {})
        tags_str = meta_d.get("tags", "[]")
        try:
            tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
        except (json.JSONDecodeError, TypeError):
            tags = []

        records.append({
            "path": img["path"],
            "filename": meta_d.get("filename", Path(img["path"]).name),
            "size_bytes": meta_d.get("size_bytes", 0),
            "width": meta_d.get("width", 0),
            "height": meta_d.get("height", 0),
            "format": meta_d.get("format", ""),
            "caption": meta_d.get("caption", ""),
            "tags": tags,
            "modified": meta_d.get("modified", ""),
        })

    meta.add_batch(records)

    console.print(f"[green]✅ Synced {total} images to SQLite metadata store.[/green]")


@app.command()
def tasks(
    task_id: str = typer.Argument(None, help="Task ID to check (omit to list active)"),
):
    """Check background task status (requires running server)."""
    import urllib.request

    base = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/tasks"

    try:
        if task_id:
            url = f"{base}/{task_id}"
        else:
            url = base

        with urllib.request.urlopen(url) as resp:
            data = json.loads(resp.read())

        if task_id:
            # Single task
            t = data
            console.print(f"\n[bold]Task: {t['task_id']}[/bold]")
            console.print(f"  Type: {t['type']}")
            console.print(f"  Status: {t['status']}")
            if t['total'] > 0:
                console.print(f"  Progress: {t['progress']}/{t['total']} ({t['percent']}%)")
            if t['message']:
                console.print(f"  Message: {t['message']}")
            if t['error']:
                console.print(f"  [red]Error: {t['error']}[/red]")
        else:
            tasks_list = data.get("tasks", [])
            if not tasks_list:
                console.print("[dim]No active tasks.[/dim]")
            else:
                console.print(f"\n[bold]Active Tasks[/bold]\n")
                for t in tasks_list:
                    pct = f" ({t['percent']}%)" if t['total'] > 0 else ""
                    console.print(f"  {t['task_id']}: {t['status']}{pct} — {t.get('message', '')}")

    except Exception as e:
        console.print(f"[red]Failed to connect:[/red] {e}")
        console.print(f"[dim]Is the server running? Start with: eyra serve[/dim]")


@app.command()
def similar(
    image: str = typer.Argument(..., help="Path to the reference image"),
    limit: int = typer.Option(10, "--limit", "-n", help="Max results"),
):
    """Find images visually similar to a given image."""
    _, _, engine = get_components()

    results = engine.find_similar(image_path=image, n_results=limit)

    if not results:
        console.print("[yellow]No similar images found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Images similar to:[/bold] {image}\n")
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", style="dim", width=3)
    table.add_column("Similarity", width=10)
    table.add_column("File", style="cyan")
    table.add_column("Path", style="dim")

    for i, r in enumerate(results, 1):
        sim = f"{r['similarity']:.1%}"
        filename = r["metadata"].get("filename", "")
        table.add_row(str(i), sim, filename, r["path"])

    console.print(table)


@app.command()
def watch(
    folder: str = typer.Argument(..., help="Path to the folder to watch"),
    auto_caption: bool = typer.Option(False, "--auto-caption", "-c", help="Generate captions and tags for new images"),
    backend: str = typer.Option(CAPTION_BACKEND, "--backend", help="Captioning backend: florence2 or blip2"),
):
    """Watch a folder and automatically index new images."""
    from .watcher import watch_folder

    embedder, indexer, _ = get_components()

    # Load captioner if auto-captioning
    captioner = None
    if auto_caption:
        from .captioner import Captioner
        captioner = Captioner(backend=backend)
        captioner.load()

    def on_new_image(path: str):
        """Called when a new image is detected."""
        try:
            embedding = embedder.embed_image(path)

            # Generate tags and caption if enabled
            tags = []
            caption = ""
            if captioner:
                try:
                    desc = captioner.describe(path)
                    tags = desc["tags"]
                    caption = desc["caption"]
                except Exception:
                    pass

            import json
            meta = {
                "filename": Path(path).name,
                "size_bytes": Path(path).stat().st_size,
                "modified": "",
                "width": 0,
                "height": 0,
                "format": "",
                "tags": json.dumps(tags),
                "caption": caption,
            }
            indexer.add_image(path, embedding, meta)

            tag_info = f" ({len(tags)} tags)" if tags else ""
            console.print(f"  [green]✅ Indexed:[/green] {Path(path).name}{tag_info}")
        except Exception as e:
            console.print(f"  [red]❌ Failed:[/red] {Path(path).name}: {e}")

    # Load model first
    embedder.load()
    console.print(f"\n[bold]Eyra Watch[/bold]")
    watch_folder(folder, on_new_image)


@app.command()
def caption(
    folder: str = typer.Argument(..., help="Path to the folder with indexed images"),
    backend: str = typer.Option(CAPTION_BACKEND, "--backend", help="Captioning backend: florence2 or blip2"),
    reindex: bool = typer.Option(False, "--reindex", "-r", help="Re-index images that already have captions"),
    limit: int = typer.Option(0, "--limit", "-n", help="Max images to caption (0 = all)"),
):
    """Auto-tag and caption indexed images using a vision model."""
    from .captioner import Captioner

    _, indexer, engine = get_components()

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        console.print(f"[red]Error:[/red] Folder not found: {folder_path}")
        raise typer.Exit(1)

    # Get all indexed images
    all_images = indexer.get_all()
    if not all_images:
        console.print("[yellow]No indexed images found. Run 'eyra index <folder>' first.[/yellow]")
        raise typer.Exit(0)

    # Filter to images in this folder that need captioning
    folder_str = str(folder_path)
    to_caption = []
    for img in all_images:
        img_path = img["id"]
        if not img_path.startswith(folder_str):
            continue
        meta = img.get("metadata", {})
        has_caption = bool(meta.get("caption", "").strip())
        if reindex or not has_caption:
            to_caption.append(img)

    if limit > 0:
        to_caption = to_caption[:limit]

    if not to_caption:
        console.print("[green]All images already have captions. Use --reindex to regenerate.[/green]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Eyra Caption[/bold]")
    console.print(f"  Backend: {backend}")
    console.print(f"  Images to caption: {len(to_caption)}\n")

    # Load captioner
    captioner = Captioner(backend=backend)
    captioner.load()

    # Load embedder once
    embedder = Embedder()
    embedder.load()

    captioned = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Captioning images...", total=len(to_caption))

        for img in to_caption:
            img_path = img["id"]
            meta = dict(img.get("metadata", {}))

            try:
                desc = captioner.describe(img_path)
                meta["tags"] = desc["tags_json"]
                meta["caption"] = desc["caption"]

                # Re-add to index with updated metadata
                embedding = embedder.embed_image(img_path)
                indexer.add_image(img_path, embedding, meta)

                captioned += 1
            except Exception as e:
                console.print(f"\n  [red]❌ {Path(img_path).name}: {e}[/red]")
                failed += 1

            progress.advance(task)

    console.print(f"\n[green]✅ Done![/green] Captioned: {captioned}, Failed: {failed}")
    console.print(f"  Total in index: {indexer.count()}")


@app.command()
def cluster(
    n_clusters: int = typer.Option(None, "--clusters", "-k", help="Number of clusters (auto if omitted)"),
    show: bool = typer.Option(False, "--show", "-s", help="Show saved clusters without re-clustering"),
):
    """Auto-group images by visual similarity using K-Means clustering."""
    from .clusterer import Clusterer

    _, indexer, _ = get_components()
    clusterer = Clusterer(indexer)

    if show:
        result = clusterer.get_clusters()
        if not result["clusters"]:
            console.print("[yellow]No clusters found. Run 'eyra cluster' first.[/yellow]")
            raise typer.Exit(0)

        console.print(f"\n[bold]Eyra Clusters[/bold] ({result['n_clusters']} groups)\n")
        for c in result["clusters"]:
            console.print(f"  [cyan]{c['label']}[/cyan] — {c['size']} images")
            for img in c["images"][:3]:
                caption = img.get("caption", "")
                if len(caption) > 50:
                    caption = caption[:50] + "…"
                console.print(f"    • {img['filename']} {f'— {caption}' if caption else ''}")
            if c["size"] > 3:
                console.print(f"    … and {c['size'] - 3} more")
            console.print()
        raise typer.Exit(0)

    console.print(f"\n[bold]Eyra Clustering[/bold]")
    console.print(f"  Clustering {indexer.count()} images...")

    auto = n_clusters is None
    result = clusterer.cluster(n_clusters=n_clusters, auto=auto)

    if result.get("error"):
        console.print(f"[red]Error:[/red] {result['error']}")
        raise typer.Exit(1)

    console.print(f"\n  Method: {result['method']}")
    console.print(f"  Found {result['n_clusters']} clusters ({result['total_images']} images)\n")

    for c in result["clusters"]:
        console.print(f"  [cyan]{c['label']}[/cyan] — {c['size']} images")
        for path in c["images"][:3]:
            console.print(f"    • {Path(path).name}")
        if c["size"] > 3:
            console.print(f"    … and {c['size'] - 3} more")
        console.print()

    console.print(f"[green]✅ Clusters saved! View in Web UI or with 'eyra cluster --show'[/green]")


@app.command()
def ocr(
    reindex: bool = typer.Option(False, "--reindex", "-r", help="Re-OCR images that already have text"),
    limit: int = typer.Option(0, "--limit", "-n", help="Max images to process (0 = all)"),
    backend: str = typer.Option("auto", "--backend", help="OCR backend: vision, tesseract, or auto"),
):
    """Extract text from images using OCR."""
    from .ocr import OCREngine

    _, indexer, _ = get_components()
    meta = indexer.metadata_store

    # Get images needing OCR
    if reindex:
        images = meta.list_images(limit=0, offset=0)["images"]
    else:
        images = meta.get_unocr()

    if limit > 0:
        images = images[:limit]

    if not images:
        console.print("[green]All images already have OCR text. Use --reindex to re-process.[/green]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Eyra OCR[/bold]")
    console.print(f"  Backend: {backend}")
    console.print(f"  Images to process: {len(images)}\n")

    engine = OCREngine(backend=backend)
    engine.load()

    processed = 0
    with_text = 0
    failed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Extracting text...", total=len(images))

        for img in images:
            img_path = img["path"]
            try:
                result = engine.extract_structured(img_path)
                meta.update_ocr(img_path, result["text"])
                processed += 1
                if result["has_text"]:
                    with_text += 1
            except Exception as e:
                console.print(f"\n  [red]❌ {Path(img_path).name}: {e}[/red]")
                failed += 1

            progress.advance(task)

    console.print(f"\n[green]✅ Done![/green] Processed: {processed}, With text: {with_text}, Failed: {failed}")


@app.command()
def export(
    output: str = typer.Argument(..., help="Path to Obsidian vault folder"),
    copy: bool = typer.Option(False, "--copy", "-c", help="Copy images into vault (default: absolute paths)"),
    organize: str = typer.Option("tags", "--organize", "-o", help="Organize by: tags, date, or flat"),
    tag: str = typer.Option(None, "--tag", "-t", help="Only export images with this tag"),
    no_ocr: bool = typer.Option(False, "--no-ocr", help="Exclude OCR text from notes"),
    limit: int = typer.Option(0, "--limit", "-n", help="Max images to export (0 = all)"),
):
    """Export indexed images to Obsidian-compatible Markdown notes."""
    from .exporter import ObsidianExporter

    _, indexer, _ = get_components()
    exporter = ObsidianExporter(indexer.metadata_store)

    console.print(f"\n[bold]Eyra Export[/bold]")
    console.print(f"  Output: {output}")
    console.print(f"  Organize by: {organize}")
    if copy:
        console.print(f"  Copy images: yes")
    if tag:
        console.print(f"  Tag filter: {tag}")

    result = exporter.export(
        output_dir=output,
        copy_images=copy,
        organize_by=organize,
        include_ocr=not no_ocr,
        tag_filter=tag,
        limit=limit,
    )

    console.print(f"\n[green]✅ Exported {result['exported']} notes ({result['skipped']} skipped)[/green]")
    console.print(f"  Location: {result['output_dir']}")
    if result["tags_used"]:
        console.print(f"  Tags: {', '.join(result['tags_used'][:10])}")


@app.command()
def timeline(
    group_by: str = typer.Option("month", "--group", "-g", help="Group by: day, month, or year"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max groups to show"),
):
    """Show a timeline of images by date (EXIF/file dates)."""
    from .metadata import MetadataStore

    meta = MetadataStore()
    groups = meta.timeline(group_by=group_by, limit=limit)

    if not groups:
        console.print("[yellow]No dated images found.[/yellow]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Eyra Timeline[/bold] (by {group_by})\n")

    for g in groups:
        console.print(f"  [cyan]{g['date']}[/cyan] — {g['count']} images")
        for img in g["images"][:3]:
            caption = img.get("caption", "")
            if len(caption) > 45:
                caption = caption[:45] + "…"
            console.print(f"    • {img['filename']} {f'— {caption}' if caption else ''}")
        if g["count"] > 3:
            console.print(f"    … and {g['count'] - 3} more")
        console.print()


@app.command()
def serve(
    host: str = typer.Option(DEFAULT_HOST, "--host", "-h"),
    port: int = typer.Option(DEFAULT_PORT, "--port", "-p"),
):
    """Start the web UI."""
    import uvicorn
    from .server import create_app

    console.print(f"\n[bold]Eyra Web UI[/bold]")
    console.print(f"  Starting at http://{host}:{port}")
    console.print(f"  Press Ctrl+C to stop\n")

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    app()
