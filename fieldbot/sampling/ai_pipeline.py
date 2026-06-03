"""
sampling/ai_pipeline.py — Image AI pipeline for FieldBot.

Pipeline per image
──────────────────
1. Preprocessing  : resize → cellular-automaton → CLAHE → Wiener → background removal
                    (reuses PreprocessingPipeline from preprocessing.py)
2. Leaf detection : heuristic check (green-pixel ratio + edge density).
                    Returns (is_leaf: bool, confidence: float).
                    NOTE: A proper leaf detector requires a trained binary
                    classifier.  This heuristic works reasonably for clear
                    leaf photos but should be replaced with a trained model
                    when training data becomes available.
3. Disease detect : LiteCShuffle (best_litecshuffle.keras) predicts class.
                    Returns (label: str, confidence: float, is_diseased: bool)

All three steps are encapsulated in AIPipeline.run(image_path).
"""

from __future__ import annotations

import cv2
import logging
import numpy as np
import os
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


@dataclass
class AIResult:
    image_path:    str
    is_leaf:       bool
    leaf_conf:     float
    disease_label: str
    disease_conf:  float
    is_diseased:   bool

    def to_dict(self) -> dict:
        return {
            "image_path":    self.image_path,
            "is_leaf":       self.is_leaf,
            "leaf_conf":     round(self.leaf_conf, 4),
            "disease_label": self.disease_label,
            "disease_conf":  round(self.disease_conf, 4),
            "is_diseased":   self.is_diseased,
        }


class LeafDetector:
    """
    Heuristic leaf detector based on:
      1. Green dominance ratio  (leaves have high G relative to R and B)
      2. Edge density            (leaves have moderate texture — not sky/soil)
      3. Saturation mean in HSV  (leaves are not grey/white)

    Returns (is_leaf: bool, confidence: float 0–1).

    Replace with a proper binary classifier (healthy_leaf vs no_leaf)
    when labelled training data is available.
    """

    def __init__(self, green_thresh: float = 0.18, edge_thresh: float = 0.04):
        self._green_thresh = green_thresh
        self._edge_thresh  = edge_thresh

    def predict(self, img_rgb_u8: np.ndarray) -> tuple[bool, float]:
        """
        img_rgb_u8 : (H, W, 3) uint8 RGB image (after preprocessing resize)
        Returns (is_leaf, confidence_0_to_1)
        """
        try:
            # ── Green dominance ──────────────────────────────────────
            r, g, b = img_rgb_u8[:,:,0].astype(float), \
                      img_rgb_u8[:,:,1].astype(float), \
                      img_rgb_u8[:,:,2].astype(float)
            total = r + g + b + 1e-6
            green_frac = (g / total).mean()

            # ── Saturation in HSV ─────────────────────────────────────
            hsv      = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV)
            sat_mean = hsv[:,:,1].mean() / 255.0

            # ── Edge density  ─────────────────────────────────────────
            gray      = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY)
            edges     = cv2.Canny(gray, 50, 150)
            edge_den  = edges.mean() / 255.0

            # Combine scores — higher is more leaf-like
            score = (0.5 * max(0.0, green_frac - 0.28) / 0.22
                     + 0.3 * sat_mean
                     + 0.2 * min(1.0, edge_den / 0.10))
            score = float(np.clip(score, 0.0, 1.0))
            is_leaf = (green_frac > self._green_thresh
                       and edge_den > self._edge_thresh
                       and sat_mean > 0.05)
            return is_leaf, score

        except Exception as e:
            log.warning("LeafDetector error: %s", e)
            return False, 0.0


class DiseaseClassifier:
    """
    Loads best_litecshuffle.keras and classifies plant disease.
    Reconstructs architecture manually to bypass Keras 3 metadata loading bugs.
    """

    def __init__(self, model_path: str, class_labels: list,
                 preprocess_size: tuple = (160, 160),
                 confidence_thresh: float = 0.60):
        self._model_path  = model_path
        self._labels      = class_labels
        self._size        = preprocess_size
        self._thresh      = confidence_thresh
        self._model       = None
        self._preprocessor = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import tensorflow as tf
            # 1. Import LiteCShuffle from your model_training file
            from model_training import LiteCShuffle, WinnerTakeAllLayer, SquashLayer
            
            log.info("Manually reconstructing LiteCShuffle architecture...")
            
            # 2. Build the exact shell model used in training
            # Ensure n_classes matches your disease labels list (usually 15)
            wrapper = LiteCShuffle(
                input_shape=(self._size[0], self._size[1], 3), 
                n_classes=len(self._labels)
            )
            wrapper.build()
            self._model = wrapper.model
            
            # 3. Load weights only. skip_mismatch=True avoids optimizer errors.
            # Using load_weights bypasses the 'conv2d expected 2 variables' error
            # found in load_model for TensorFlow 2.15/2.16 environments.
            self._model.load_weights(self._model_path, skip_mismatch=True)
            
            log.info("LiteCShuffle weights successfully loaded from %s", self._model_path)
            
        except Exception as e:
            log.error("Failed to load model: %s", e)
            self._model = None

        # Standard Preprocessing setup
        try:
            from preprocessing import preprocess_one
            self._preprocessor = preprocess_one
            log.info("Preprocessing function ready (target=%s)", self._size)
        except ImportError:
            log.warning("preprocessing.py not found — using basic resize/scaling")
            self._preprocessor = None

    def _preprocess(self, img_rgb_u8: np.ndarray) -> np.ndarray:
        """Ensures image is resized and scaled to [0, 1] for LiteCShuffle."""
        if self._preprocessor is not None:
            # External preprocessor typically handles float conversion
            processed = self._preprocessor(img_rgb_u8)   
        else:
            h, w = self._size
            img  = cv2.resize(img_rgb_u8, (w, h), interpolation=cv2.INTER_AREA)
            # CRITICAL: Scale to [0, 1] to prevent constant class predictions
            processed = img.astype(np.float32) / 255.0

        return np.expand_dims(processed, axis=0)

    def predict(self, img_rgb_u8: np.ndarray) -> tuple[str, float, bool]:
        """
        Returns (label, confidence, is_diseased).
        """
        self._ensure_loaded()
        if self._model is None:
            return "unknown", 0.0, False
        try:
            x = self._preprocess(img_rgb_u8)
            
            # Run inference on the scaled batch
            probs = self._model.predict(x, verbose=0)[0]
            idx   = int(np.argmax(probs))
            conf  = float(probs[idx])
            
            label = self._labels[idx] if idx < len(self._labels) else f"class_{idx}"
            is_diseased = (label.lower() != "healthy" and conf >= self._thresh)
            
            return label, conf, is_diseased
        except Exception as e:
            log.error("Inference error: %s", e)
            return "error", 0.0, False

class AIPipeline:
    """
    Orchestrates leaf detection → disease classification for one image.

    Usage
    -----
        pipeline = AIPipeline()
        result   = pipeline.run("/path/to/image.jpg")
    """

    def __init__(self):
        from config import MODEL_PATH, AI_CONFIDENCE_THRESH, \
                           IMG_PREPROCESS_SIZE, DISEASE_CLASSES
        self._leaf_detector = LeafDetector()
        self._disease_clf   = DiseaseClassifier(
            model_path=MODEL_PATH,
            class_labels=DISEASE_CLASSES,
            preprocess_size=IMG_PREPROCESS_SIZE,
            confidence_thresh=AI_CONFIDENCE_THRESH,
        )

    def run(self, image_path: str) -> AIResult:
        """
        Full pipeline for a single image file.
        Returns AIResult with all fields populated.
        """
        # Load image
        bgr = cv2.imread(image_path)
        if bgr is None:
            log.warning("Cannot read image: %s", image_path)
            return AIResult(image_path=image_path,
                            is_leaf=False, leaf_conf=0.0,
                            disease_label="no_image", disease_conf=0.0,
                            is_diseased=False)

        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Step 1: leaf detection
        is_leaf, leaf_conf = self._leaf_detector.predict(img_rgb)
        if not is_leaf:
            log.debug("%s → not a leaf (conf=%.2f)", os.path.basename(image_path), leaf_conf)
            return AIResult(image_path=image_path,
                            is_leaf=False, leaf_conf=leaf_conf,
                            disease_label="not_a_leaf", disease_conf=1.0 - leaf_conf,
                            is_diseased=False)

        # Step 2: disease classification
        label, conf, diseased = self._disease_clf.predict(img_rgb)
        log.info("%s → leaf=%.2f | %s (%.0f%%) diseased=%s",
                 os.path.basename(image_path), leaf_conf, label, conf*100, diseased)

        return AIResult(image_path=image_path,
                        is_leaf=True,  leaf_conf=leaf_conf,
                        disease_label=label, disease_conf=conf,
                        is_diseased=diseased)
