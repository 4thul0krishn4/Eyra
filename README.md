# 👁️ Eyra

Local-first AI image memory — turns any folder into a searchable visual knowledge base.

Instead of relying on filenames or manual organization, Eyra uses vision models to automatically understand each image, generate tags and captions, and create embeddings for semantic search. Run `eyra search "cat on a roof"` and it finds the image — even if the file is called `IMG_4892.jpg`.

## Features

- 🔍 Semantic search — find images by describing what you're looking for
- 🏷️ Auto-tagging — AI generates tags and captions for every image
- 📂 Folder indexing — point it at any folder, it handles the rest
- 👁️ File watching — automatically indexes new images as they appear
- 🖥️ Web UI — beautiful dark-mode search interface
- 🔒 100% local — no cloud, no accounts, no data leaves your machine
- ⚡ Fast — optimized for Apple Silicon (M-series Macs)

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
eyra index <folder>        # Index all images in a folder
eyra search <query>        # Search by natural language
eyra similar <image>       # Find visually similar images
eyra watch <folder>        # Watch folder for new images (auto-index)
eyra stats                 # Show index statistics
eyra serve                 # Start web UI at localhost:8080
```

## How It Works

1. **Scan** — finds all images in the folder (JPG, PNG, WebP, HEIC, etc.)
2. **Embed** — generates CLIP embeddings (vector representations) for each image
3. **Store** — saves embeddings in a local ChromaDB vector database
4. **Search** — converts your text query to an embedding and finds the closest matches

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
