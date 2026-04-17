# 👁️ Eyra

Local-first AI image memory — turns any folder into a searchable visual knowledge base.

Instead of relying on filenames or manual organization, Eyra uses vision models to understand each image, generate tags and captions, and create embeddings for semantic search. Run `eyra search "cat on a roof"` and it finds the image — even if the file is called `IMG_4892.jpg`.

## Features

- 🔍 **Semantic search** — find images by describing what you're looking for
- 🏷️ **Auto-tagging** — vision models generate tags and captions (Florence-2 or BLIP)
- 🔀 **Hybrid search** — combines vector similarity + keyword matching
- 🧩 **Visual clustering** — auto-group similar images using K-Means on CLIP embeddings
- 📝 **OCR** — extract text from images (Apple Vision or Tesseract)
- 💬 **Chat UI** — conversational search with natural language summaries
- 📅 **Timeline** — browse images by date from EXIF data
- 📓 **Obsidian export** — export to Markdown notes with YAML frontmatter
- 📂 **Folder indexing** — point it at any folder, it handles the rest
- 👁️ **File watching** — automatically indexes new images as they appear
- 🖥️ **Web UI** — dark-mode interface with Browse, Clusters, Chat, and Timeline tabs
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
eyra index <folder> --auto-caption     # Index with auto-captioning
eyra index <folder> --background       # Index in background (server must be running)
eyra search <query>                    # Search (hybrid by default)
eyra search <query> --mode vector      # Vector-only search
eyra search <query> --mode keyword     # Tag/caption keyword search
eyra tags <keyword>                    # Search by tag or keyword
eyra similar <image>                   # Find visually similar images
eyra caption <folder>                  # Auto-tag/caption already-indexed images
eyra cluster                           # Group images by visual similarity
eyra cluster --show                    # View existing clusters
eyra ocr                               # Extract text from images
eyra export <vault-path>               # Export to Obsidian Markdown
eyra timeline                          # Browse images by date
eyra watch <folder>                    # Watch folder for new images
eyra stats                             # Show index statistics
eyra serve                             # Start web UI at localhost:8080
```

## Auto-Tagging

Eyra uses Florence-2 (default) or BLIP to generate natural language captions and descriptive tags for every image.

```bash
# Index and caption at the same time
eyra index ~/Pictures --auto-caption

# Caption already-indexed images
eyra caption ~/Pictures

# Use Florence-2 (default) or BLIP
eyra caption ~/Pictures --backend florence2
```

## Hybrid Search

By default, search combines:
- **Vector similarity** (60%) — CLIP embeddings understand visual concepts
- **Keyword matching** (40%) — matches against generated tags and captions

Use `--mode vector` or `--mode keyword` to search with only one method.

## Visual Clustering

Automatically groups your images by visual similarity using K-Means on CLIP embeddings. The number of clusters is auto-detected via silhouette score.

```bash
# Run clustering (auto-detects optimal number of groups)
eyra cluster

# View saved clusters
eyra cluster --show
```

Clusters are also visible in the Web UI under the Clusters tab.

## OCR

Extract text from images using Apple Vision (macOS) or Tesseract.

```bash
# Run OCR on all unprocessed images
eyra ocr

# Re-process all images
eyra ocr --reindex

# Limit to first 100
eyra ocr --limit 100
```

Extracted text is stored in SQLite and searchable via full-text search.

**Requirements:** Install one of:
- `pip install pytesseract` + `brew install tesseract` (cross-platform)
- `pip install pyobjc-framework-Vision` (macOS, uses built-in Apple Vision)

## Obsidian Export

Export your indexed images as Obsidian-compatible Markdown notes with YAML frontmatter, Dataview-compatible metadata, and embedded image references.

```bash
# Export with absolute image paths (default)
eyra export ~/MyVault

# Copy images into the vault
eyra export ~/MyVault --copy

# Organize by date instead of tags
eyra export ~/MyVault --organize date

# Only export images with a specific tag
eyra export ~/MyVault --tag screenshot
```

Each note includes:
- YAML frontmatter (title, tags, caption, dates, dimensions)
- Embedded image reference
- OCR text (if available)
- Auto-generated index/MOC note

## Timeline

Browse your images by date, extracted from EXIF metadata or file modification dates.

```bash
# Group by month (default)
eyra timeline

# Group by year
eyra timeline --group year

# Group by day
eyra timeline --group day
```

Also available in the Web UI Timeline tab.

## Web UI

The web UI (`eyra serve`) has four tabs:

- **Browse** — search, filter, and browse all images with infinite scroll
- **Clusters** — run and view visual clusters
- **Chat** — conversational search with natural language summaries
- **Timeline** — browse images by date

Features:
- Infinite scroll for large collections
- Sort by newest, filename, size, dimensions
- Filter by caption status, image format
- Live task monitoring with progress bars
- Keyboard navigation in the lightbox modal
- Full-text search (FTS5) on captions, tags, and OCR text

## Performance & Scale

Eyra handles thousands of images without breaking a sweat.

### SQLite Sidecar Database

Metadata is stored in SQLite alongside ChromaDB vectors:
- **Fast filtering** by format, dimensions, tags, caption status
- **Full-text search** (FTS5) on captions, tags, and OCR text
- **Instant stats** without loading all vectors
- **Proper pagination** with accurate total counts

### Background Indexing

Index large folders without blocking the UI:

```bash
eyra index ~/Pictures --background
eyra tasks
```

## How It Works

1. **Scan** — finds all images in the folder (JPG, PNG, WebP, HEIC, etc.)
2. **Embed** — generates CLIP embeddings for each image
3. **Caption** — Florence-2 generates captions and tags (optional)
4. **Store** — embeddings in ChromaDB, metadata in SQLite
5. **Search** — hybrid vector + keyword search
6. **Discover** — clustering, OCR, timeline, export

Everything runs on your machine. No API calls, no cloud, no tracking.

## Requirements

- Python 3.10+
- macOS (Apple Silicon recommended) or Linux
- ~2GB disk space for models (downloaded once on first run)

## Tech Stack

- OpenCLIP — CLIP embeddings for images and text
- Florence-2 / BLIP — vision models for captioning
- ChromaDB — local vector database
- SQLite — metadata storage with FTS5 full-text search
- scikit-learn — K-Means clustering
- FastAPI — web backend
- Typer + Rich — CLI
- Watchdog — file system monitoring
- Pillow — image processing

## License

MIT
