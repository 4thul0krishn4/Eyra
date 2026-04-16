"""Background task queue — non-blocking indexing and captioning.

Runs tasks in background threads so the web UI stays responsive.
No external dependencies (no Celery/Redis) — fully local-first.
"""

import json
import threading
import time
import traceback
from collections import deque
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task:
    """A single background task."""

    def __init__(self, task_id: str, task_type: str, params: dict):
        self.task_id = task_id
        self.task_type = task_type
        self.params = params
        self.status = TaskStatus.PENDING
        self.progress = 0
        self.total = 0
        self.message = ""
        self.error = ""
        self.result = {}
        self.created_at = datetime.now().isoformat()
        self.started_at = None
        self.finished_at = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "type": self.task_type,
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "percent": round(self.progress / self.total * 100) if self.total > 0 else 0,
            "message": self.message,
            "error": self.error,
            "result": self.result,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


class TaskQueue:
    """Simple background task queue with progress tracking."""

    def __init__(self, max_workers: int = 2):
        self.max_workers = max_workers
        self._tasks: dict[str, Task] = {}
        self._queue: deque[str] = deque()
        self._lock = threading.Lock()
        self._workers: list[threading.Thread] = []
        self._handlers: dict[str, Callable] = {}
        self._running = True
        self._counter = 0

        # Start worker threads
        for i in range(max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"eyra-worker-{i}", daemon=True)
            t.start()
            self._workers.append(t)

    def register_handler(self, task_type: str, handler: Callable):
        """Register a handler function for a task type.

        Handler signature: handler(task: Task, params: dict) -> dict
        Should update task.progress/task.total during work.
        Returns result dict on success.
        """
        self._handlers[task_type] = handler

    def submit(self, task_type: str, params: dict = None) -> str:
        """Submit a new task to the queue.

        Returns the task_id.
        """
        with self._lock:
            self._counter += 1
            task_id = f"{task_type}-{self._counter}-{int(time.time())}"
            task = Task(task_id, task_type, params or {})
            self._tasks[task_id] = task
            self._queue.append(task_id)

        return task_id

    def get_task(self, task_id: str) -> Optional[dict]:
        """Get task status by ID."""
        task = self._tasks.get(task_id)
        return task.to_dict() if task else None

    def get_active_tasks(self) -> list[dict]:
        """Get all non-completed tasks."""
        return [
            t.to_dict() for t in self._tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.RUNNING)
        ]

    def get_recent_tasks(self, limit: int = 20) -> list[dict]:
        """Get recent tasks (all statuses)."""
        tasks = sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task (not running ones)."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task and task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                # Remove from queue
                try:
                    self._queue.remove(task_id)
                except ValueError:
                    pass
                return True
        return False

    def _worker_loop(self):
        """Worker thread that processes tasks from the queue."""
        while self._running:
            task_id = None

            with self._lock:
                if self._queue:
                    task_id = self._queue.popleft()

            if task_id is None:
                time.sleep(0.5)
                continue

            task = self._tasks.get(task_id)
            if not task or task.status == TaskStatus.CANCELLED:
                continue

            handler = self._handlers.get(task.task_type)
            if not handler:
                task.status = TaskStatus.FAILED
                task.error = f"No handler registered for task type: {task.task_type}"
                task.finished_at = datetime.now().isoformat()
                continue

            # Run the task
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()

            try:
                result = handler(task, task.params)
                task.status = TaskStatus.COMPLETED
                task.result = result or {}
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = f"{type(e).__name__}: {e}"
                task.result = {"traceback": traceback.format_exc()}

            task.finished_at = datetime.now().isoformat()

    def shutdown(self):
        """Shut down all workers."""
        self._running = False
        for t in self._workers:
            t.join(timeout=5)


# --- Task Handlers ---

def handle_index(task: Task, params: dict) -> dict:
    """Background indexing handler."""
    from .embedder import Embedder
    from .indexer import Indexer
    from .scanner import scan_folder, count_images

    folder = params.get("folder", "")
    batch_size = params.get("batch_size", 32)
    auto_caption = params.get("auto_caption", False)
    backend = params.get("backend", "florence2")

    folder_path = Path(folder).expanduser().resolve()
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    # Count images
    total = count_images(folder_path)
    task.total = total
    task.message = f"Scanning {folder_path.name} ({total} images)"

    if total == 0:
        return {"indexed": 0, "skipped": 0}

    # Load models
    embedder = Embedder()
    embedder.load()

    captioner = None
    if auto_caption:
        from .captioner import Captioner
        captioner = Captioner(backend=backend)
        captioner.load()

    indexer = Indexer()

    indexed = 0
    skipped = 0
    batch_paths = []
    batch_metadatas = []

    for img_data in scan_folder(folder_path):
        # Check if cancelled
        if task.status == TaskStatus.CANCELLED:
            break

        img_path = img_data["path"]

        if indexer.exists(img_path):
            skipped += 1
            task.progress += 1
            continue

        batch_paths.append(img_path)

        img_tags = []
        img_caption = ""
        if captioner:
            try:
                desc = captioner.describe(img_path)
                img_tags = desc["tags"]
                img_caption = desc["caption"]
            except Exception:
                pass

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

        if len(batch_paths) >= batch_size:
            embeddings = embedder.embed_batch(batch_paths)
            indexer.add_batch(
                image_ids=batch_paths,
                embeddings=embeddings,
                metadatas=batch_metadatas,
            )
            indexed += len(batch_paths)
            task.progress += len(batch_paths)
            task.message = f"Indexed {indexed}/{total}"
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
        task.progress += len(batch_paths)

    task.message = f"Done — indexed {indexed}, skipped {skipped}"
    return {"indexed": indexed, "skipped": skipped, "total": total}


def handle_caption(task: Task, params: dict) -> dict:
    """Background captioning handler."""
    from .captioner import Captioner
    from .embedder import Embedder
    from .indexer import Indexer

    folder = params.get("folder", "")
    backend = params.get("backend", "florence2")
    reindex = params.get("reindex", False)
    limit = params.get("limit", 0)

    indexer = Indexer()
    meta = indexer.metadata_store

    # Get images that need captioning
    if reindex:
        result = meta.list_images(limit=0, offset=0)
        all_images = result["images"]
        if folder:
            folder_str = str(Path(folder).resolve())
            all_images = [img for img in all_images if img["path"].startswith(folder_str)]
    else:
        if folder:
            folder_str = str(Path(folder).resolve())
            result = meta.list_images(limit=0, offset=0)
            all_images = [img for img in result["images"]
                         if img["path"].startswith(folder_str) and not img.get("has_caption")]
        else:
            all_images = meta.get_uncaptioned()

    if limit > 0:
        all_images = all_images[:limit]

    task.total = len(all_images)
    if task.total == 0:
        task.message = "No images need captioning"
        return {"captioned": 0, "failed": 0}

    captioner = Captioner(backend=backend)
    captioner.load()

    embedder = Embedder()
    embedder.load()

    captioned = 0
    failed = 0

    for img in all_images:
        if task.status == TaskStatus.CANCELLED:
            break

        img_path = img["path"]
        task.message = f"Captioning {Path(img_path).name} ({captioned + 1}/{task.total})"

        try:
            desc = captioner.describe(img_path)
            meta_d = dict(img)
            meta_d["tags"] = desc["tags_json"]
            meta_d["caption"] = desc["caption"]

            embedding = embedder.embed_image(img_path)
            indexer.add_image(img_path, embedding, meta_d)

            captioned += 1
        except Exception as e:
            failed += 1

        task.progress += 1

    task.message = f"Done — captioned {captioned}, failed {failed}"
    return {"captioned": captioned, "failed": failed, "total": task.total}


# --- Singleton ---

_queue: Optional[TaskQueue] = None


def get_task_queue() -> TaskQueue:
    """Get or create the singleton task queue."""
    global _queue
    if _queue is None:
        _queue = TaskQueue(max_workers=2)
        _queue.register_handler("index", handle_index)
        _queue.register_handler("caption", handle_caption)
    return _queue
