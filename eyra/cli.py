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
):
    """Index all images in a folder for search."""
    embedder, indexer, _ = get_components()

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        console.print(f"[red]Error:[/red] Folder not found: {folder_path}")
        raise typer.Exit(1)

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
    _, _, engine = get_components()

    info = engine.stats()

    console.print(f"\n[bold]Eyra Stats[/bold]\n")
    console.print(f"  Total indexed images: {info['total_images']}")

    if info["top_tags"]:
        console.print(f"\n  Top tags:")
        for tag, count in info["top_tags"][:10]:
            console.print(f"    {tag}: {count}")


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
