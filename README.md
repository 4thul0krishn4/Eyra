# 👁️ Eyra

Local-first AI image memory — turns any folder into a searchable visual knowledge base.

Instead of relying on filenames or manual organization, Eyra uses vision models to automatically understand each image, generate tags and captions, and create embeddings for semantic search. Run `eyra search "cat on a roof"` and it finds the image — even if the file is called `IMG_4892.jpg`.

## Features

- 🔍 **Semantic search** — find images by describing what you're looking for
- 🏷️ **Auto-tagging** — AI generates tags and captions for every image (BLIP)
- 🔀 **Hybrid search** — combines vector similarity + keyword matching for best results
- 📂 **Folder indexing** — point it at any folder, it handles the rest
- 👁️ **File watching** — automatically indexes new images as they appear
- 🖥️ **Web UI** — beautiful dark-mode search interface with mode toggle
- 🔒 **100% local** — no cloud, no accounts, no data leaves your machine
- ⚡ **Fast** — optimized for Apple Silicon (M-series Macs)

## Quick Start

```bash
git clone https://github.com/4thul0krishn4/Eyra.git
cd Eyra
./setup.sh
```

Then index your images (with auto-captioning):

```bash
source venv/bin/activate
eyra index ~/Pictures --auto-caption
```

Search:

```bash
eyra search "sunset at the beach"
```

Start the web UI:

```bash
eyra serve
```

Open http://localhost:8080 in your browser.

## Commands

```
eyra index <folder>                    # Index all images in a folder
eyra index <folder> --auto-caption     # Index with auto-captioning (BLIP)
eyra index <folder> --background       # Index in background (server must be running)
eyra search <query>                    # Search (hybrid by default)
eyra search <query> --mode vector      # Vector-only search
eyra search <query> --mode keyword     # Tag/caption keyword search
eyra tags <keyword>                    # Search by tag or keyword
eyra similar <image>                   # Find visually similar images
eyra caption <folder>                  # Auto-tag/caption already-indexed images
eyra watch <folder>                    # Watch folder for new images
eyra watch <folder> --auto-caption     # Watch with auto-captioning
eyra stats                             # Show index statistics (from SQLite)
eyra sync                              # Sync SQLite metadata from ChromaDB
eyra tasks                             # Check background task status
eyra serve                             # Start web UI at localhost:8080
```

## Auto-Tagging

Eyra uses vision-language models to generate natural language captions and descriptive tags for every image:

- **BLIP** (default) — Salesforce's lightweight captioning model, fast on Apple Silicon
- **Florence-2** — Microsoft's multi-task vision model (requires compatible transformers version)
- **BLIP-2** — heavier model, higher quality captions (requires more RAM)

### Indexing with captions

```bash
# Index and caption at the same time
eyra index ~/Pictures --auto-caption

# Caption already-indexed images
eyra caption ~/Pictures

# Use a different backend
eyra caption ~/Pictures --backend blip2
```

### Hybrid Search

By default, search combines:
- **Vector similarity** (60%) — CLIP embeddings understand visual concepts
- **Keyword matching** (40%) — matches against generated tags and captions

This means searching for "robot" will find images tagged with "robot" even if the visual embedding alone might miss them. Use `--mode vector` or `--mode keyword` to search with only one method.

## Performance & Scale

Eyra is built to handle thousands of images without breaking a sweat.

### SQLite Sidecar Database

Metadata is stored in SQLite alongside ChromaDB's vectors. This enables:
- **Fast filtering** by format, dimensions, tags, caption status
- **Full-text search** (FTS5) on captions and tags
- **Instant stats** without loading all vectors
- **Proper pagination** with accurate total counts

```bash
# View detailed stats (from SQLite)
eyra stats

# Backfill SQLite from existing ChromaDB index (one-time upgrade)
eyra sync
```

### Background Indexing

Index large folders without blocking the UI:

```bash
# Submit a background task (server must be running)
eyra index ~/Pictures --background

# Check task status
eyra tasks
```

Or use the web API directly:
```bash
# Start indexing
curl -X POST "http://localhost:8080/api/tasks/index?folder=~/Pictures&auto_caption=true"

# Check status
curl http://localhost:8080/api/tasks
```

### Web UI Features

The web UI supports:
- **Infinite scroll** — loads 50 images at a time, no lag on large collections
- **Format/dimensions sorting** — sort by newest, filename, file size, width, height
- **Filters** — show only captioned/uncaptioned, specific formats
- **Task monitoring** — live progress bars for background indexing/captioning
- **Keyboard navigation** — arrow keys in the lightbox modal

## How It Works

1. **Scan** — finds all images in the folder (JPG, PNG, WebP, HEIC, etc.)
2. **Embed** — generates CLIP embeddings (vector representations) for each image
3. **Caption** (optional) — BLIP generates captions and tags for each image
4. **Store** — saves embeddings + metadata in a local ChromaDB vector database
5. **Search** — hybrid search: combines vector similarity with keyword matching

Everything runs on your machine. No API calls, no cloud, no tracking.

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended) or Linux
- ~2GB disk space for models (downloaded once on first run)

## Tech Stack

- OpenCLIP — open-source CLIP model for image/text embeddings
- BLIP — lightweight vision model for captioning and tagging
- ChromaDB — local vector database
- FastAPI — web backend
- Typer + Rich — beautiful CLI
- Watchdog — file system monitoring
- Pillow — image processing

## License

MIT
