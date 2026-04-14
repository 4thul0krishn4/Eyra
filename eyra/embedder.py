"""CLIP embedding generator — converts images to vector representations."""

import io
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from .config import CLIP_MODEL, CLIP_PRETRAINED, MODELS_DIR


class Embedder:
    """Generates CLIP embeddings for images and text queries."""

    def __init__(self, model_name: str = CLIP_MODEL, pretrained: str = CLIP_PRETRAINED):
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = "cpu"
        self.model = None
        self.preprocess = None
        self.tokenizer = None

    def load(self):
        """Load the CLIP model. Call once before using embed methods."""
        if self.model is not None:
            return

        print(f"  Loading CLIP model: {self.model_name}...")

        # Try Apple Silicon acceleration
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("  Using Apple Silicon GPU (MPS)")
        else:
            print("  Using CPU")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            cache_dir=str(MODELS_DIR),
        )
        self.model = self.model.to(self.device)
        self.model.eval()

        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        print("  Model loaded ✅")

    def embed_image(self, image_path: str | Path) -> np.ndarray:
        """Generate embedding for a single image.

        Returns a numpy array of shape (embedding_dim,).
        """
        self.load()

        img = Image.open(image_path).convert("RGB")
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_image(img_tensor)

        # Normalize to unit vector for cosine similarity
        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for a text query.

        Returns a numpy array of shape (embedding_dim,).
        """
        self.load()

        tokens = self.tokenizer([text]).to(self.device)

        with torch.no_grad():
            embedding = self.model.encode_text(tokens)

        embedding = embedding / embedding.norm(dim=-1, keepdim=True)
        return embedding.cpu().numpy().flatten()

    def embed_batch(self, image_paths: list[str | Path], batch_size: int = 32) -> list[np.ndarray]:
        """Generate embeddings for multiple images efficiently.

        Returns a list of numpy arrays.
        """
        self.load()
        results = []

        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            tensors = []

            for path in batch_paths:
                try:
                    img = Image.open(path).convert("RGB")
                    tensors.append(self.preprocess(img))
                except Exception:
                    tensors.append(torch.zeros(3, 224, 224))  # placeholder for failed images

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embeddings = self.model.encode_image(batch_tensor)

            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            results.extend(e.cpu().numpy() for e in embeddings)

        return results
