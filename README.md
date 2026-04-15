# 👁️ Eyra

Local-first AI image memory — turns any folder into a searchable visual knowledge base.

Instead of relying on filenames or manual organization, Eyra uses vision models to automatically understand each image, generate tags and captions, and create embeddings for semantic search. Run `eyra search "cat on a roof"` and it finds the image — even if the file is called `IMG_4892.jpg`.

## Features

- 🔍 **Semantic search** — find images by describing what you're looking for
- 🏷️ **Auto-tagging** — AI generates tags and captions for every image (Florence-2 / BLIP-2)
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

Then index your images:

```bash
eyra index ~/Pictures
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
eyra index <folder> --auto-caption     # Index with auto-captioning (Florence-2)
eyra search <query>                    # Search (hybrid by default)
eyra search <query> --mode vector      # Vector-only search
eyra search <query> --mode keyword     # Tag/caption keyword search
eyra tags <keyword>                    # Search by tag or keyword
eyra similar <image>                   # Find visually similar images
eyra caption <folder>                  # Auto-tag/caption already-indexed images
eyra watch <folder>                    # Watch folder for new images
eyra watch <folder> --auto-caption     # Watch with auto-captioning
eyra stats                             # Show index statistics
eyra serve                             # Start web UI at localhost:8080
```

## Auto-Tagging

Eyra uses vision-language models to generate natural language captions and descriptive tags for every image:

- **Florence-2** (default) — Microsoft's lightweight multi-task vision model, fast on CPU
- **BLIP-2** — Salesforce's heavier model, higher quality captions

### Indexing with captions

```bash
# Index and caption at the same time
eyra index ~/Pictures --auto-caption

# Caption already-indexed images
eyra caption ~/Pictures

# Use BLIP-2 instead of Florence-2
eyra caption ~/Pictures --backend blip2
```

### Hybrid Search

By default, search combines:
- **Vector similarity** (60%) — CLIP embeddings understand visual concepts
- **Keyword matching** (40%) — matches against generated tags and captions

This means searching for "robot" will find images tagged with "robot" even if the visual embedding alone might miss them. Use `--mode vector` or `--mode keyword` to search with only one method.

## How It Works

1. **Scan** — finds all images in the folder (JPG, PNG, WebP, HEIC, etc.)
2. **Embed** — generates CLIP embeddings (vector representations) for each image
3. **Caption** (optional) — vision model generates captions and tags for each image
4. **Store** — saves embeddings + metadata in a local ChromaDB vector database
5. **Search** — hybrid search: combines vector similarity with keyword matching

Everything runs on your machine. No API calls, no cloud, no tracking.

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended) or Linux
- ~2GB disk space for models (downloaded once on first run)

## Tech Stack

- OpenCLIP — open-source CLIP model for image/text embeddings
- ChromaDB — local vector database
- FastAPI — web backend
- Typer + Rich — beautiful CLI
- Watchdog — file system monitoring
- Pillow — image processing

## License

MIT
