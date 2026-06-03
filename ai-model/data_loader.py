"""
data_loader.py
Handles dataset discovery, filtering, and splitting for the plant disease pipeline.

Expected dataset structure:
    dataset/
    ├── Tomato/
    │   ├── Healthy/
    │   └── Bacterial_growth/
    ├── Pepper/
    │   ├── Healthy/
    │   └── Sperogia/
    └── ...
"""

import os
import glob
from pathlib import Path
from typing import Optional
import numpy as np
from sklearn.model_selection import train_test_split
import cv2


# ---------------------------------------------------------------------------
# Plant names that are used during TESTING / early experiments.
# Adjust or extend as needed.
# ---------------------------------------------------------------------------
SUBSET_PLANTS = ["Potato", "Pepper_Bell"]   # case-insensitive prefix match


class DatasetLoader:
    """
    Discovers images from a hierarchical plant-disease dataset folder,
    optionally filtering to a subset of plants.

    Parameters
    ----------
    dataset_root : str
        Path to the root directory (e.g. "dataset/").
    use_subset : bool
        If True, only load plants listed in SUBSET_PLANTS.
        Set to False to load the full dataset.
    image_size : tuple
        (height, width) to resize every loaded image.
    test_size : float
        Fraction of data to reserve for testing.
    val_size : float
        Fraction of *training* data to reserve for validation.
    random_state : int
        Seed for reproducible splits.
    """

    def __init__(
        self,
        dataset_root: str,
        use_subset: bool = True,
        image_size: tuple = (160, 160),
        test_size: float = 0.20,
        val_size: float = 0.15,
        random_state: int = 42,
    ):
        self.dataset_root = Path(dataset_root)
        self.use_subset = use_subset
        self.image_size = image_size
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state

        self.class_names: list[str] = []  # e.g. ["Tomato_Healthy", "Tomato_Bacterial_growth"]
        self.label_map: dict[str, int] = {}
        self._image_paths: list[str] = []
        self._labels: list[int] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def discover(self) -> "DatasetLoader":
        """Walk the dataset tree and collect (path, label) pairs."""
        self._image_paths.clear()
        self._labels.clear()
        self.class_names.clear()

        plant_dirs = sorted(
            [d for d in self.dataset_root.iterdir() if d.is_dir()]
        )

        if self.use_subset:
            plant_dirs = [
                d for d in plant_dirs
                if any(
                    d.name.lower().startswith(p.lower())
                    for p in SUBSET_PLANTS
                )
            ]
            if not plant_dirs:
                raise ValueError(
                    f"No matching plants found for subset {SUBSET_PLANTS}. "
                    "Check SUBSET_PLANTS or set use_subset=False."
                )

        for plant_dir in plant_dirs:
            for disease_dir in sorted(plant_dir.iterdir()):
                if not disease_dir.is_dir():
                    continue
                class_label = f"{plant_dir.name}__{disease_dir.name}"
                if class_label not in self.class_names:
                    self.class_names.append(class_label)
                label_idx = self.class_names.index(class_label)

                for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.JPG", "*.JPEG", "*.PNG", "*.BMP", "*.TIFF"):
                    for img_path in disease_dir.glob(ext):
                        self._image_paths.append(str(img_path))
                        self._labels.append(label_idx)

        self.label_map = {name: i for i, name in enumerate(self.class_names)}

        print(
            f"[DatasetLoader] Discovered {len(self._image_paths)} images "
            f"across {len(self.class_names)} classes."
        )
        print(f"  Classes: {self.class_names}")
        return self

    def get_splits(self) -> tuple:
        """
        Return train / val / test splits as (paths, labels) tuples.

        Returns
        -------
        (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        if not self._image_paths:
            raise RuntimeError("Call .discover() before .get_splits().")

        paths = np.array(self._image_paths)
        labels = np.array(self._labels)

        X_train_val, X_test, y_train_val, y_test = train_test_split(
            paths, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=self.random_state,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=self.val_size,
            stratify=y_train_val,
            random_state=self.random_state,
        )

        print(
            f"[DatasetLoader] Split sizes — "
            f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def load_image(self, path: str) -> Optional[np.ndarray]:
        """Load and resize a single image. Returns None on failure."""
        img = cv2.imread(path)
        if img is None:
            print(f"[DatasetLoader] Warning: could not read {path}")
            return None
        img = cv2.resize(img, (self.image_size[1], self.image_size[0]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def load_images(self, paths: np.ndarray) -> np.ndarray:
        """Batch load all images. Skips unreadable files."""
        images = []
        for p in paths:
            img = self.load_image(p)
            if img is not None:
                images.append(img)
        return np.array(images, dtype=np.uint8)
