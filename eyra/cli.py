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

from .config import DEFAULT_RESULTS, DEFAULT_HOST, DEFAULT_PORT, ensure_dirs
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
            batch_metadatas.append({
                "filename": img_data["filename"],
                "size_bytes": img_data["size_bytes"],
                "modified": img_data["modified"],
                "width": img_data["dimensions"][0] if img_data["dimensions"] else 0,
                "height": img_data["dimensions"][1] if img_data["dimensions"] else 0,
                "format": img_data["format"] or "",
                "tags": "[]",
                "caption": "",
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
    json_output: bool = typer.Option(False, "--json", "-j", help="Output as JSON"),
):
    """Search images by describing what you're looking for."""
    _, _, engine = get_components()

    results = engine.search(
        query=query,
        n_results=limit,
        min_similarity=min_similarity,
    )

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit(0)

    if json_output:
        # Clean up for JSON output
        clean = []
        for r in results:
            clean.append({
                "path": r["path"],
                "similarity": round(r["similarity"], 4),
                "filename": r["metadata"].get("filename", ""),
            })
        print(json.dumps(clean, indent=2))
    else:
        console.print(f"\n[bold]Search results for:[/bold] \"{query}\"\n")
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
):
    """Watch a folder and automatically index new images."""
    from .watcher import watch_folder

    embedder, indexer, _ = get_components()

    def on_new_image(path: str):
        """Called when a new image is detected."""
        try:
            embedding = embedder.embed_image(path)
            meta = {
                "filename": Path(path).name,
                "size_bytes": Path(path).stat().st_size,
                "modified": "",
                "width": 0,
                "height": 0,
                "format": "",
                "tags": "[]",
                "caption": "",
            }
            indexer.add_image(path, embedding, meta)
            console.print(f"  [green]✅ Indexed:[/green] {Path(path).name}")
        except Exception as e:
            console.print(f"  [red]❌ Failed:[/red] {Path(path).name}: {e}")

    # Load model first
    embedder.load()
    console.print(f"\n[bold]Eyra Watch[/bold]")
    watch_folder(folder, on_new_image)


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
