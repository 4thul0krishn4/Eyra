"""Visual clustering — auto-group images by visual similarity using CLIP embeddings.

Uses K-Means on the stored CLIP embeddings to discover visual clusters.
Works entirely with existing indexed data — no re-processing needed.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

from .indexer import Indexer
from .metadata import MetadataStore


class Clusterer:
    """Discovers visual clusters from indexed CLIP embeddings."""

    def __init__(self, indexer: Optional[Indexer] = None):
        self.indexer = indexer or Indexer()
        self.meta = self.indexer.metadata_store

    def cluster(
        self,
        n_clusters: Optional[int] = None,
        auto: bool = True,
        max_clusters: int = 20,
        min_clusters: int = 3,
        sample_size: int = 5000,
    ) -> dict:
        """Cluster all indexed images by visual similarity.

        Args:
            n_clusters: fixed number of clusters (overrides auto detection)
            auto: if True and n_clusters is None, auto-select via silhouette
            max_clusters: max clusters to try in auto mode
            min_clusters: min clusters to try in auto mode
            sample_size: max images to use for silhouette scoring (speed)

        Returns:
            {
                "n_clusters": int,
                "labels": {image_path: cluster_id, ...},
                "clusters": [{id, size, images, representative, label}, ...],
                "method": "kmeans" or "minibatch"
            }
        """
        all_images = self.indexer.get_all()
        if len(all_images) < min_clusters:
            return {"n_clusters": 0, "labels": {}, "clusters": [], "method": "none",
                    "error": f"Need at least {min_clusters} indexed images, found {len(all_images)}"}

        # Extract embeddings
        paths = []
        embeddings = []
        for img in all_images:
            paths.append(img["path"])
            # Re-embed or get from ChromaDB — we need the vectors
            # ChromaDB stores them, so we query for all
            embeddings.append(None)  # placeholder

        # Get actual embeddings from ChromaDB
        self.indexer.connect()
        raw = self.indexer.collection.get(include=["embeddings"])
        paths = raw["ids"]
        embeddings = np.array(raw["embeddings"])

        if len(paths) < min_clusters:
            return {"n_clusters": 0, "labels": {}, "clusters": [], "method": "none",
                    "error": f"Need at least {min_clusters} images"}

        # Determine number of clusters
        if n_clusters is None:
            if auto and len(paths) >= min_clusters * 2:
                n_clusters = self._auto_k(embeddings, min_clusters, max_clusters, sample_size)
            else:
                n_clusters = max(min_clusters, min(max_clusters, len(paths) // 5))

        n_clusters = min(n_clusters, len(paths))

        # Run clustering
        if len(paths) > 5000:
            km = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=256, n_init=3)
            method = "minibatch"
        else:
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            method = "kmeans"

        labels = km.fit_predict(embeddings)

        # Build cluster info
        cluster_map: dict[int, list[str]] = {}
        for path, label in zip(paths, labels):
            cluster_map.setdefault(int(label), []).append(path)

        # Find representative image (closest to centroid) for each cluster
        clusters = []
        for cid in sorted(cluster_map.keys()):
            member_paths = cluster_map[cid]
            member_indices = [paths.index(p) for p in member_paths]
            member_embeddings = embeddings[member_indices]

            # Find closest to centroid
            centroid = km.cluster_centers_[cid]
            dists = np.linalg.norm(member_embeddings - centroid, axis=1)
            rep_idx = member_indices[int(np.argmin(dists))]

            # Auto-label from common tags
            label = self._auto_label(member_paths)

            clusters.append({
                "id": cid,
                "size": len(member_paths),
                "representative": paths[rep_idx],
                "label": label,
                "images": member_paths[:50],  # cap for API response
            })

        # Sort by size descending
        clusters.sort(key=lambda c: c["size"], reverse=True)

        # Assign readable labels if we didn't from tags
        for i, c in enumerate(clusters):
            if not c["label"]:
                c["label"] = f"Group {i + 1}"

        labels_dict = {path: int(label) for path, label in zip(paths, labels)}

        # Save clusters to SQLite
        self._save_clusters(labels_dict)

        return {
            "n_clusters": len(clusters),
            "labels": labels_dict,
            "clusters": clusters,
            "method": method,
            "total_images": len(paths),
        }

    def get_clusters(self) -> dict:
        """Get saved clusters from SQLite."""
        conn = self.meta._get_conn()

        # Check if cluster table exists
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='image_clusters'"
        ).fetchone()
        if not exists:
            return {"n_clusters": 0, "clusters": [], "message": "No clusters yet. Run clustering first."}

        rows = conn.execute("""
            SELECT cluster_id, cluster_label, COUNT(*) as size
            FROM image_clusters
            GROUP BY cluster_id
            ORDER BY size DESC
        """).fetchall()

        clusters = []
        for row in rows:
            # Get representative
            img_rows = conn.execute("""
                SELECT i.path, i.filename, i.caption, i.tags_json
                FROM image_clusters ic
                JOIN images i ON ic.path = i.path
                WHERE ic.cluster_id = ?
                ORDER BY ic.path
                LIMIT 50
            """, (row["cluster_id"],)).fetchall()

            images = []
            for ir in img_rows:
                tags = []
                try:
                    tags = json.loads(ir["tags_json"]) if ir["tags_json"] else []
                except (json.JSONDecodeError, TypeError):
                    pass
                images.append({
                    "path": ir["path"],
                    "filename": ir["filename"],
                    "caption": ir["caption"] or "",
                    "tags": tags,
                })

            clusters.append({
                "id": row["cluster_id"],
                "label": row["cluster_label"] or f"Group {row['cluster_id']}",
                "size": row["size"],
                "images": images,
            })

        return {"n_clusters": len(clusters), "clusters": clusters}

    def get_image_cluster(self, image_path: str) -> Optional[dict]:
        """Get the cluster assignment for a specific image."""
        conn = self.meta._get_conn()
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='image_clusters'"
        ).fetchone()
        if not exists:
            return None

        row = conn.execute(
            "SELECT cluster_id, cluster_label FROM image_clusters WHERE path = ?",
            (image_path,)
        ).fetchone()
        if row:
            return {"cluster_id": row["cluster_id"], "label": row["cluster_label"]}
        return None

    def _auto_k(self, embeddings: np.ndarray, min_k: int, max_k: int, sample_size: int) -> int:
        """Auto-select number of clusters using silhouette score."""
        n = len(embeddings)
        max_k = min(max_k, n - 1)

        if max_k <= min_k:
            return min_k

        # Subsample for speed
        if n > sample_size:
            indices = np.random.RandomState(42).choice(n, sample_size, replace=False)
            sample = embeddings[indices]
        else:
            sample = embeddings

        best_k = min_k
        best_score = -1

        for k in range(min_k, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = km.fit_predict(sample)
            score = silhouette_score(sample, labels, sample_size=min(1000, len(sample)))

            if score > best_score:
                best_score = score
                best_k = k

        return best_k

    def _auto_label(self, paths: list[str]) -> str:
        """Generate a label from the most common tags in a cluster."""
        tag_counts: dict[str, int] = {}

        for path in paths:
            row = self.meta.get_image(path)
            if not row:
                continue
            tags = row.get("tags", [])
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except (json.JSONDecodeError, TypeError):
                    tags = []
            for tag in tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        if not tag_counts:
            return ""

        # Top 2 tags as label
        top = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:2]
        return ", ".join(t[0] for t in top)

    def _save_clusters(self, labels: dict[str, int]):
        """Save cluster assignments to SQLite."""
        conn = self.meta._get_conn()

        conn.executescript("""
            CREATE TABLE IF NOT EXISTS image_clusters (
                path TEXT PRIMARY KEY,
                cluster_id INTEGER NOT NULL,
                cluster_label TEXT DEFAULT '',
                FOREIGN KEY (path) REFERENCES images(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_clusters_id ON image_clusters(cluster_id);
        """)

        # Get cluster labels (auto-generated from tags)
        cluster_labels: dict[int, str] = {}
        cluster_members: dict[int, list[str]] = {}
        for path, cid in labels.items():
            cluster_members.setdefault(cid, []).append(path)

        for cid, members in cluster_members.items():
            cluster_labels[cid] = self._auto_label(members)

        # Clear and re-insert
        conn.execute("DELETE FROM image_clusters")
        for path, cid in labels.items():
            conn.execute(
                "INSERT INTO image_clusters (path, cluster_id, cluster_label) VALUES (?, ?, ?)",
                (path, cid, cluster_labels.get(cid, ""))
            )
        conn.commit()
