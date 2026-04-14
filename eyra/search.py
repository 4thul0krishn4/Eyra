"""Search engine — combines semantic search with tag/metadata filtering."""

from pathlib import Path
from typing import Optional

from .embedder import Embedder
from .indexer import Indexer


class SearchEngine:
    """High-level search interface for Eyra."""

    def __init__(self, indexer: Indexer, embedder: Embedder):
        self.indexer = indexer
        self.embedder = embedder

    def search(
        self,
        query: str,
        n_results: int = 20,
        tags: Optional[list[str]] = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """Search images by natural language query.

        Args:
            query: text description to search for
            n_results: max number of results
            tags: optional list of tags to filter by
            min_similarity: minimum similarity score (0-1)

        Returns:
            List of result dicts sorted by relevance.
        """
        # Generate embedding for the text query
        query_embedding = self.embedder.embed_text(query)

        # Build filter
        where = None
        if tags:
            where = {"tags": {"$contains": tags[0]}}  # ChromaDB basic filter

        # Search
        results = self.indexer.search(
            query_embedding=query_embedding,
            n_results=n_results * 2,  # fetch extra, filter later
            where=where,
        )

        # Filter by minimum similarity
        results = [r for r in results if r["similarity"] >= min_similarity]

        return results[:n_results]

    def find_similar(
        self,
        image_path: str,
        n_results: int = 10,
    ) -> list[dict]:
        """Find images visually similar to a given image.

        Args:
            image_path: path to the reference image
            n_results: max number of results

        Returns:
            List of similar images sorted by similarity.
        """
        query_embedding = self.embedder.embed_image(image_path)

        results = self.indexer.search(
            query_embedding=query_embedding,
            n_results=n_results + 1,  # +1 because the image itself might be in the index
        )

        # Exclude the reference image itself
        results = [r for r in results if r["path"] != str(Path(image_path).resolve())]

        return results[:n_results]

    def stats(self) -> dict:
        """Get index statistics."""
        count = self.indexer.count()
        all_images = self.indexer.get_all()

        # Gather tag stats
        tag_counts = {}
        for img in all_images:
            tags_str = img["metadata"].get("tags", "[]")
            try:
                import json
                tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
                if isinstance(tags, list):
                    for tag in tags:
                        tag_counts[tag] = tag_counts.get(tag, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        # Top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "total_images": count,
            "top_tags": top_tags,
        }
