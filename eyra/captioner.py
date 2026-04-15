"""Vision model captioner — generates tags and captions using Florence-2 or BLIP-2."""

import json
import re
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from .config import (
    CAPTION_MODEL,
    CAPTION_MODEL_DIR,
    CAPTION_MAX_TAGS,
    CAPTION_DEVICE,
)


class Captioner:
    """Generates natural language captions and discrete tags for images.

    Supports two backends:
      - florence2 (default): Microsoft Florence-2 — fast, lightweight, multi-task
      - blip2: Salesforce BLIP-2 — heavier but high-quality captions
    """

    def __init__(self, model_name: str = CAPTION_MODEL, backend: str = "florence2"):
        self.model_name = model_name
        self.backend = backend
        self.device = self._detect_device()
        self.model = None
        self.processor = None

    def _detect_device(self) -> str:
        """Detect the best available device."""
        if CAPTION_DEVICE:
            return CAPTION_DEVICE
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load(self):
        """Load the vision model. Call once before using caption/tag methods."""
        if self.model is not None:
            return

        print(f"  Loading {self.backend} model: {self.model_name}...")
        print(f"  Device: {self.device}")

        if self.backend == "florence2":
            self._load_florence2()
        elif self.backend == "blip2":
            self._load_blip2()
        else:
            raise ValueError(f"Unknown backend: {self.backend}. Use 'florence2' or 'blip2'")

        print("  Captioner loaded ✅")

    def _load_florence2(self):
        """Load Florence-2 model."""
        from transformers import AutoProcessor, AutoModelForCausalLM

        model_path = self.model_name

        # Florence-2 requires CUDA or CPU — MPS has issues with it
        device_map = "auto" if self.device == "cuda" else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=str(CAPTION_MODEL_DIR),
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            cache_dir=str(CAPTION_MODEL_DIR),
            trust_remote_code=True,
        )

    def _load_blip2(self):
        """Load BLIP-2 model."""
        from transformers import Blip2Processor, Blip2ForConditionalGeneration

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32

        self.processor = Blip2Processor.from_pretrained(
            self.model_name,
            cache_dir=str(CAPTION_MODEL_DIR),
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name,
            cache_dir=str(CAPTION_MODEL_DIR),
            torch_dtype=dtype,
        )
        self.model = self.model.to(self.device)

    def caption(self, image_path: str | Path) -> str:
        """Generate a natural language caption for an image.

        Args:
            image_path: path to the image file

        Returns:
            Caption string like "a cat sitting on a windowsill"
        """
        self.load()
        img = Image.open(image_path).convert("RGB")

        if self.backend == "florence2":
            return self._caption_florence2(img)
        else:
            return self._caption_blip2(img)

    def _caption_florence2(self, img: Image.Image) -> str:
        """Caption using Florence-2."""
        task_prompt = "<MORE_DETAILED_CAPTION>"
        inputs = self.processor(text=task_prompt, images=img, return_tensors="pt")

        # Move inputs to the same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                num_beams=3,
                early_stopping=True,
            )

        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Parse Florence-2 output — it returns structured text with task tokens
        caption = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(img.width, img.height),
        )

        # Extract the caption text
        if isinstance(caption, dict) and task_prompt in caption:
            return caption[task_prompt].strip()
        elif isinstance(caption, str):
            return caption.strip()

        # Fallback: clean up the raw text
        text = generated_text.replace("</s>", "").replace("<s>", "").strip()
        # Remove task prompt tokens
        for token in ["<MORE_DETAILED_CAPTION>", "<CAPTION>", "<DETAILED_CAPTION>"]:
            text = text.replace(token, "").strip()
        return text

    def _caption_blip2(self, img: Image.Image) -> str:
        """Caption using BLIP-2."""
        inputs = self.processor(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=50)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()

    def tags(self, image_path: str | Path) -> list[str]:
        """Generate discrete tags/keywords for an image.

        Args:
            image_path: path to the image file

        Returns:
            List of tag strings like ["cat", "windowsill", "indoor", "daylight"]
        """
        self.load()
        img = Image.open(image_path).convert("RGB")

        if self.backend == "florence2":
            return self._tags_florence2(img)
        else:
            return self._tags_blip2(img)

    def _tags_florence2(self, img: Image.Image) -> list[str]:
        """Extract tags using Florence-2 region/phrase grounding."""
        # Use the caption and extract keywords from it
        caption = self._caption_florence2(img)

        # Florence-2 also supports "<REGION_TO_CATEGORY>" but it needs a region
        # So we generate a detailed caption and extract meaningful tags
        tags = self._extract_tags_from_caption(caption)

        # Also try region-level descriptions for richer tags
        try:
            task_prompt = "<REGION_TO_DESCRIPTION>"
            inputs = self.processor(text=task_prompt, images=img, return_tensors="pt")
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=128,
                    num_beams=3,
                )

            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            region_result = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(img.width, img.height),
            )

            if isinstance(region_result, dict) and task_prompt in region_result:
                region_desc = region_result[task_prompt]
                if isinstance(region_desc, str):
                    tags.extend(self._extract_tags_from_caption(region_desc))
        except Exception:
            pass  # Region tagging is optional enhancement

        # Deduplicate and limit
        seen = set()
        unique_tags = []
        for tag in tags:
            tag_lower = tag.lower().strip()
            if tag_lower and tag_lower not in seen and len(tag_lower) > 1:
                seen.add(tag_lower)
                unique_tags.append(tag_lower)

        return unique_tags[:CAPTION_MAX_TAGS]

    def _tags_blip2(self, img: Image.Image) -> list[str]:
        """Extract tags using BLIP-2."""
        # BLIP-2 can generate interrogative responses
        interrogative_prompts = [
            "a photo of",
            "this image shows",
            "the main subjects are",
        ]

        all_tags = []

        # Get the caption
        caption = self._caption_blip2(img)
        all_tags.extend(self._extract_tags_from_caption(caption))

        # Try asking about specific aspects
        for prompt in interrogative_prompts[:1]:  # Just use one to save time
            try:
                inputs = self.processor(
                    text=prompt,
                    images=img,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    out = self.model.generate(**inputs, max_new_tokens=30)

                response = self.processor.decode(out[0], skip_special_tokens=True)
                all_tags.extend(self._extract_tags_from_caption(response))
            except Exception:
                pass

        # Deduplicate
        seen = set()
        unique_tags = []
        for tag in all_tags:
            tag_lower = tag.lower().strip()
            if tag_lower and tag_lower not in seen and len(tag_lower) > 1:
                seen.add(tag_lower)
                unique_tags.append(tag_lower)

        return unique_tags[:CAPTION_MAX_TAGS]

    def _extract_tags_from_caption(self, caption: str) -> list[str]:
        """Extract meaningful keywords/tags from a caption string."""
        # Remove common stop words and filler
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "or", "but", "not", "no", "nor", "so", "yet", "it",
            "its", "this", "that", "these", "those", "there", "here",
            "which", "who", "whom", "what", "where", "when", "how",
            "image", "photo", "picture", "shows", "showing", "depicts",
            "contains", "seen", "visible", "background", "foreground",
        }

        # Clean the caption
        text = caption.lower()
        text = re.sub(r'[^a-z0-9\s\-]', ' ', text)

        # Split into words and meaningful phrases
        words = text.split()

        # Extract single words
        tags = []
        for word in words:
            word = word.strip('-').strip()
            if word and word not in stop_words and len(word) > 1:
                tags.append(word)

        # Extract 2-word phrases that are likely meaningful
        for i in range(len(words) - 1):
            if (words[i] not in stop_words and
                words[i+1] not in stop_words and
                len(words[i]) > 2 and len(words[i+1]) > 2):
                phrase = f"{words[i]} {words[i+1]}"
                tags.append(phrase)

        return tags

    def describe(self, image_path: str | Path) -> dict:
        """Generate both caption and tags for an image.

        Args:
            image_path: path to the image file

        Returns:
            Dict with 'caption' and 'tags' keys
        """
        caption = self.caption(image_path)
        tags = self.tags(image_path)

        return {
            "caption": caption,
            "tags": tags,
            "tags_json": json.dumps(tags),
            "backend": self.backend,
            "model": self.model_name,
        }

    def describe_batch(
        self,
        image_paths: list[str | Path],
        batch_size: int = 8,
    ) -> list[dict]:
        """Generate captions and tags for multiple images.

        Args:
            image_paths: list of image file paths
            batch_size: number of images to process at once (lower = less VRAM)

        Returns:
            List of dicts with 'caption' and 'tags' keys
        """
        results = []
        total = len(image_paths)

        for i, path in enumerate(image_paths):
            try:
                result = self.describe(path)
                result["path"] = str(path)
                results.append(result)
                print(f"  [{i+1}/{total}] Captioned: {Path(path).name}")
            except Exception as e:
                print(f"  [{i+1}/{total}] Failed: {Path(path).name}: {e}")
                results.append({
                    "path": str(path),
                    "caption": "",
                    "tags": [],
                    "tags_json": "[]",
                    "error": str(e),
                })

        return results
