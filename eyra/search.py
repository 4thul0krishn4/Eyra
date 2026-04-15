"""Search engine — combines semantic search with tag/metadata filtering."""

import json
import re
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
        mode: str = "hybrid",
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> list[dict]:
        """Search images by natural language query.

        Args:
            query: text description to search for
            n_results: max number of results
            tags: optional list of tags to filter by
            min_similarity: minimum similarity score (0-1)
            mode: "hybrid" (vector + keyword), "vector" (embeddings only),
                  or "keyword" (tag/caption text matching only)
            vector_weight: weight for vector similarity in hybrid mode (0-1)
            keyword_weight: weight for keyword match in hybrid mode (0-1)

        Returns:
            List of result dicts sorted by relevance.
        """
        if mode == "keyword":
            return self._keyword_search(query, n_results, tags)
        elif mode == "vector":
            return self._vector_search(query, n_results, tags, min_similarity)
        else:  # hybrid
            return self._hybrid_search(
                query, n_results, tags, min_similarity,
                vector_weight, keyword_weight,
            )

    def _vector_search(
        self,
        query: str,
        n_results: int = 20,
        tags: Optional[list[str]] = None,
        min_similarity: float = 0.0,
    ) -> list[dict]:
        """Pure vector similarity search."""
        query_embedding = self.embedder.embed_text(query)

        where = None
        if tags:
            where = {"tags": {"$contains": tags[0]}}

        results = self.indexer.search(
            query_embedding=query_embedding,
            n_results=n_results * 2,
            where=where,
        )

        results = [r for r in results if r["similarity"] >= min_similarity]
        return results[:n_results]

    def _keyword_search(
        self,
        query: str,
        n_results: int = 20,
        tags: Optional[list[str]] = None,
    ) -> list[dict]:
        """Search by matching query words against tags and captions."""
        all_images = self.indexer.get_all()
        query_words = set(self._tokenize(query))

        scored = []
        for img in all_images:
            meta = img.get("metadata", {})

            # Parse tags
            tags_str = meta.get("tags", "[]")
            try:
                img_tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str
            except (json.JSONDecodeError, TypeError):
                img_tags = []

            # Parse caption
            caption = meta.get("caption", "")

            # Build searchable text
            searchable = " ".join(img_tags) + " " + caption
            searchable_words = set(self._tokenize(searchable))

            # Score: how many query words appear in the image metadata
            if not searchable_words:
                continue

            matches = query_words & searchable_words
            if not matches:
                continue

            # Jaccard-like score with bonus for exact tag matches
            score = len(matches) / len(query_words)

            # Bonus if tags match directly
            tag_match_count = sum(1 for t in img_tags if t.lower() in query.lower())
            score += tag_match_count * 0.15

            # Tag filter
            if tags:
                tag_set = {t.lower() for t in img_tags}
                if not any(t.lower() in tag_set for t in tags):
                    continue

            scored.append({
                "id": img["id"],
                "path": img["path"],
                "metadata": meta,
                "keyword_score": min(score, 1.0),
                "similarity": 0.0,
            })

        scored.sort(key=lambda x: x["keyword_score"], reverse=True)
        return scored[:n_results]

    def _hybrid_search(
        self,
        query: str,
        n_results: int = 20,
        tags: Optional[list[str]] = None,
        min_similarity: float = 0.0,
        vector_weight: float = 0.6,
        keyword_weight: float = 0.4,
    ) -> list[dict]:
        """Combine vector similarity with keyword matching."""
        # Get vector results (more than needed to merge)
        vector_results = self._vector_search(query, n_results * 3, tags, min_similarity=0.0)

        # Get keyword results
        keyword_results = self._keyword_search(query, n_results * 3, tags)

        # Build lookup maps
        vector_map = {r["path"]: r for r in vector_results}
        keyword_map = {r["path"]: r for r in keyword_results}

        # Merge scores
        all_paths = set(vector_map.keys()) | set(keyword_map.keys())
        merged = []

        for path in all_paths:
            v = vector_map.get(path, {"similarity": 0.0, "metadata": {}})
            k = keyword_map.get(path, {"keyword_score": 0.0, "metadata": v.get("metadata", {})})

            combined_score = (
                v["similarity"] * vector_weight +
                k.get("keyword_score", 0.0) * keyword_weight
            )

            if combined_score < min_similarity and v["similarity"] < min_similarity:
                continue

            merged.append({
                "id": path,
                "path": path,
                "metadata": v.get("metadata") or k.get("metadata", {}),
                "distance": v.get("distance", 1.0),
                "similarity": v["similarity"],
                "keyword_score": k.get("keyword_score", 0.0),
                "combined_score": combined_score,
            })

        # Sort by combined score
        merged.sort(key=lambda x: x["combined_score"], reverse=True)

        # Overwrite similarity with combined_score for consistent display
        for r in merged:
            r["similarity"] = r["combined_score"]

        return merged[:n_results]

    def _tokenize(self, text: str) -> list[str]:
        """Split text into lowercase tokens, removing stop words."""
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "and", "or", "but", "not", "it",
            "its", "this", "that", "show", "shows", "showing", "image",
            "photo", "picture", "find", "search", "look",
        }
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)
        words = [w.strip('-').strip() for w in text.split()]
        return [w for w in words if w and w not in stop_words and len(w) > 1]

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
