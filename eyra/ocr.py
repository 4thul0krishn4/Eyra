"""OCR integration — extract text from images using Apple Vision or Tesseract.

On macOS: uses Apple's Vision framework via pyobjc (best quality, no install needed).
Fallback: uses pytesseract (requires `brew install tesseract`).
"""

import platform
from pathlib import Path
from typing import Optional


class OCREngine:
    """Extracts text from images via OCR."""

    def __init__(self, backend: str = "auto"):
        """
        Args:
            backend: "vision" (Apple), "tesseract", or "auto" (detect best)
        """
        self.backend = backend
        self._vision_request = None
        self._tesseract = None
        self._loaded = False

    def load(self):
        """Initialize the OCR backend."""
        if self._loaded:
            return

        if self.backend == "auto":
            if platform.system() == "Darwin":
                self.backend = "vision"
            else:
                self.backend = "tesseract"

        if self.backend == "vision":
            self._load_vision()
        else:
            self._load_tesseract()

        self._loaded = True

    def _load_vision(self):
        """Load Apple Vision framework via pyobjc."""
        try:
            import Vision
            import objc
            self._vision_request = Vision.VNRecognizeTextRequest.alloc().init()
            self._vision_request.setRecognitionLevel_(1)  # accurate
            self._vision_request.setRecognitionLanguages_(["en-US"])
            self._backend_name = "apple-vision"
            print("  OCR backend: Apple Vision ✅")
        except ImportError:
            print("  Apple Vision not available, falling back to Tesseract")
            self.backend = "tesseract"
            self._load_tesseract()

    def _load_tesseract(self):
        """Load pytesseract."""
        try:
            import pytesseract
            self._tesseract = pytesseract
            self._backend_name = "tesseract"
            print("  OCR backend: Tesseract ✅")
        except ImportError:
            raise ImportError(
                "No OCR backend available. Install one:\n"
                "  pip install pytesseract  # + brew install tesseract\n"
                "  pip install pyobjc-framework-Vision  # macOS only"
            )

    def extract_text(self, image_path: str | Path) -> str:
        """Extract all text from an image.

        Returns the extracted text as a string (empty if no text found).
        """
        self.load()
        image_path = str(Path(image_path).resolve())

        if self.backend == "vision":
            return self._extract_vision(image_path)
        else:
            return self._extract_tesseract(image_path)

    def _extract_vision(self, image_path: str) -> str:
        """Extract text using Apple Vision."""
        import Vision
        import objc
        from Foundation import NSURL

        url = NSURL.fileURLWithPath_(image_path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)

        success = handler.performRequests_error_([self._vision_request], None)
        if not success:
            return ""

        results = self._vision_request.results()
        if not results:
            return ""

        texts = []
        for observation in results:
            text = observation.topCandidates_(1)
            if text:
                texts.append(str(text[0].string()))

        return "\n".join(texts)

    def _extract_tesseract(self, image_path: str) -> str:
        """Extract text using Tesseract."""
        from PIL import Image

        img = Image.open(image_path)
        text = self._tesseract.image_to_string(img)
        return text.strip()

    def extract_structured(self, image_path: str | Path) -> dict:
        """Extract text with metadata (confidence, regions).

        Returns: {"text": str, "word_count": int, "backend": str}
        """
        text = self.extract_text(image_path)
        words = text.split()
        return {
            "text": text,
            "word_count": len(words),
            "has_text": len(words) > 0,
            "backend": self._backend_name if self._loaded else "unknown",
        }
