"""Vector database indexer — stores and retrieves image embeddings using ChromaDB."""

import json
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np

from .config import COLLECTION_NAME, DB_DIR


class Indexer:
    """Manages the ChromaDB vector store for image embeddings."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DB_DIR)
        self.client = None
        self.collection = None

    def connect(self):
        """Connect to the vector database."""
        if self.client is not None:
            return

        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add_image(
        self,
        image_id: str,
        embedding: np.ndarray,
        metadata: dict,
    ):
        """Add a single image to the index.

        Args:
            image_id: unique identifier (usually the file path)
            embedding: numpy array from the embedder
            metadata: dict with filename, tags, caption, dimensions, etc.
        """
        self.connect()

        # ChromaDB expects lists, not numpy arrays
        self.collection.upsert(
            ids=[image_id],
            embeddings=[embedding.tolist()],
            metadatas=[self._clean_metadata(metadata)],
        )

    def add_batch(
        self,
        image_ids: list[str],
        embeddings: list[np.ndarray],
        metadatas: list[dict],
    ):
        """Add multiple images to the index at once."""
        self.connect()

        self.collection.upsert(
            ids=image_ids,
            embeddings=[e.tolist() for e in embeddings],
            metadatas=[self._clean_metadata(m) for m in metadatas],
        )

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 20,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Search for similar images.

        Returns list of dicts with id, metadata, and distance.
        """
        self.connect()

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where,
        )

        # Format results
        output = []
        if results["ids"] and results["ids"][0]:
            for i, img_id in enumerate(results["ids"][0]):
                output.append({
                    "id": img_id,
                    "path": img_id,
                    "distance": results["distances"][0][i],
                    "similarity": 1 - results["distances"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                })

        return output

    def get_all(self) -> list[dict]:
        """Get all indexed images."""
        self.connect()

        results = self.collection.get()
        output = []
        if results["ids"]:
            for i, img_id in enumerate(results["ids"]):
                output.append({
                    "id": img_id,
                    "path": img_id,
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return output

    def count(self) -> int:
        """Count indexed images."""
        self.connect()
        return self.collection.count()

    def delete(self, image_id: str):
        """Remove an image from the index."""
        self.connect()
        self.collection.delete(ids=[image_id])

    def exists(self, image_id: str) -> bool:
        """Check if an image is already indexed."""
        self.connect()
        result = self.collection.get(ids=[image_id])
        return len(result["ids"]) > 0

    def _clean_metadata(self, metadata: dict) -> dict:
        """Ensure all metadata values are ChromaDB-compatible types."""
        clean = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                clean[key] = value
            elif isinstance(value, list):
                clean[key] = json.dumps(value)
            elif isinstance(value, dict):
                clean[key] = json.dumps(value)
            elif value is None:
                clean[key] = ""
            else:
                clean[key] = str(value)
        return clean
