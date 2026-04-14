"""File system watcher — automatically indexes new images in real time."""

import time
from pathlib import Path
from typing import Optional

from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileMovedEvent
from watchdog.observers import Observer

from .config import SUPPORTED_EXTENSIONS


class ImageWatcher(FileSystemEventHandler):
    """Watches a folder for new images and triggers indexing."""

    def __init__(self, callback, folder: str):
        self.callback = callback
        self.folder = Path(folder).resolve()
        self.observer = None

    def on_created(self, event):
        if not event.is_directory:
            path = Path(event.src_path)
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                print(f"  📸 New image detected: {path.name}")
                self.callback(str(path))

    def on_moved(self, event):
        if not event.is_directory:
            path = Path(event.dest_path)
            if path.suffix.lower() in SUPPORTED_EXTENSIONS:
                print(f"  📸 Image moved: {path.name}")
                self.callback(str(path))

    def start(self):
        """Start watching the folder."""
        self.observer = Observer()
        self.observer.schedule(self, str(self.folder), recursive=True)
        self.observer.start()
        print(f"  👁️ Watching: {self.folder}")
        print("  Press Ctrl+C to stop")

    def stop(self):
        """Stop watching."""
        if self.observer:
            self.observer.stop()
            self.observer.join()


def watch_folder(folder: str, callback):
    """Watch a folder for new images and call callback for each one.

    Args:
        folder: path to watch
        callback: function called with the new image path
    """
    handler = ImageWatcher(callback, folder)
    handler.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n  Stopping watcher...")
        handler.stop()
