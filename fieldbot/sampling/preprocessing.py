"""
litecshufflenet_preprocess.py
==============================
Standalone preprocessing module for the LiteCShuffle plant disease model.

Applies the exact same 5-step pipeline used during training, resizing
directly to 160×160 (the model's input resolution) instead of 512×512.

Feature selection (WOA-APSO) is NOT applied here — LiteCShuffle is an
image-input model that was trained on pixels, not on extracted features.

Steps
-----
1. Resize to 160 × 160  (INTER_AREA — best quality for downscaling)
2. Cellular automaton filter  (removes isolated noise pixels)
3. Histogram equalisation  (CLAHE in LAB colour space)
4. Wiener filter  (adaptive noise reduction)
5. Background removal  (unsharp-mask-style foreground sharpening)
6. Normalise to float32 in [0, 1]

Usage — Python API
------------------
    from litecshufflenet_preprocess import preprocess_one, preprocess_batch

    # Single image from a file path
    arr = preprocess_one("leaf.jpg")          # shape (160, 160, 3) float32

    # Single image from an existing numpy BGR array (e.g. from cv2.imread)
    arr = preprocess_one(bgr_array)

    # Batch from a list of file paths
    batch = preprocess_batch(["a.jpg", "b.jpg", "c.jpg"])   # (N, 160, 160, 3)

    # Batch with a progress callback
    batch = preprocess_batch(paths, on_progress=lambda i, n: print(f"{i}/{n}"))

Usage — Command line
--------------------
    # Preprocess a single image and save as .npy
    python litecshufflenet_preprocess.py --image leaf.jpg --output leaf.npy

    # Preprocess a folder and save a batched .npy file
    python litecshufflenet_preprocess.py --folder ./leaves/ --output batch.npy

    # Show a preview (requires a display)
    python litecshufflenet_preprocess.py --image leaf.jpg --show
"""

import os
import sys
import argparse
import time
from typing import Callable, List, Optional, Union

import cv2
import numpy as np
from scipy.signal import wiener as scipy_wiener


# ---------------------------------------------------------------------------
# Constants — must match what was used during training
# ---------------------------------------------------------------------------

MODEL_INPUT_SIZE   = (160, 160)    # (height, width)
CA_ITERATIONS      = 1             # cellular automaton passes
WIENER_NOISE       = None          # None = auto-estimate per channel
GAUSS_KSIZE        = 21            # must be odd; Gaussian kernel for background removal
CA_THRESHOLD       = 30            # pixel deviation threshold for CA filter
CLAHE_CLIP_LIMIT   = 2.0
CLAHE_TILE_SIZE    = (8, 8)


# ---------------------------------------------------------------------------
# Individual pipeline steps  (stateless functions — easy to test/inspect)
# ---------------------------------------------------------------------------

def step_resize(image: np.ndarray,
                size: tuple = MODEL_INPUT_SIZE) -> np.ndarray:
    """
    Resize to (height, width) using INTER_AREA interpolation.
    INTER_AREA gives the best quality when downscaling (avoids moiré).

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) uint8
    size  : (height, width) — defaults to MODEL_INPUT_SIZE = (160, 160)

    Returns
    -------
    np.ndarray  (size[0], size[1], 3) uint8
    """
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def step_cellular_automaton(image: np.ndarray,
                             iterations: int = CA_ITERATIONS,
                             threshold: int = CA_THRESHOLD) -> np.ndarray:
    """
    Majority-vote cellular automaton noise removal.

    Each pixel is replaced by its 3×3 neighbourhood median if it deviates
    from that median by more than `threshold` intensity counts.
    Removes isolated noise/error pixels without blurring edges.

    Parameters
    ----------
    image      : np.ndarray  (H, W, 3) uint8
    iterations : int         number of passes (1 is usually enough)
    threshold  : int         deviation threshold (0-255); higher = less aggressive

    Returns
    -------
    np.ndarray  (H, W, 3) uint8
    """
    result = image.copy()
    for _ in range(iterations):
        median = cv2.medianBlur(result, 3)
        diff   = np.abs(result.astype(np.int16) - median.astype(np.int16))
        mask   = diff > threshold
        result[mask] = median[mask]
    return result


def step_histogram_equalisation(image: np.ndarray) -> np.ndarray:
    """
    Contrast-limited adaptive histogram equalisation (CLAHE) in LAB space.

    Applied only to the Lightness channel (L) to improve contrast without
    shifting colours. clip_limit=2.0 prevents over-amplifying noise.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) uint8, RGB

    Returns
    -------
    np.ndarray  (H, W, 3) uint8, RGB
    """
    lab   = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT,
                             tileGridSize=CLAHE_TILE_SIZE)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def step_wiener_filter(image: np.ndarray,
                        noise: Optional[float] = WIENER_NOISE) -> np.ndarray:
    """
    Adaptive Wiener filter for noise reduction (applied per colour channel).

    The Wiener filter estimates the local signal statistics and suppresses
    noise while preserving edges. `noise=None` auto-estimates noise power.

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) uint8, RGB
    noise : float or None — estimated noise variance; None = auto

    Returns
    -------
    np.ndarray  (H, W, 3) uint8, RGB
    """
    out = np.zeros_like(image, dtype=np.float64)
    for c in range(image.shape[2]):
        out[:, :, c] = scipy_wiener(
            image[:, :, c].astype(np.float64), noise=noise)
    return np.clip(out, 0, 255).astype(np.uint8)


def step_background_removal(image: np.ndarray,
                              ksize: int = GAUSS_KSIZE) -> np.ndarray:
    """
    Foreground sharpening via unsharp-mask-style background suppression.

    Blurs the image to estimate the background, then subtracts it to
    make the foreground (leaf features) more prominent.
    Formula: foreground = 1.5 × image − 0.5 × blurred

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) uint8, RGB
    ksize : int         Gaussian kernel size (must be odd)

    Returns
    -------
    np.ndarray  (H, W, 3) uint8, RGB
    """
    k = ksize if ksize % 2 == 1 else ksize + 1
    blurred    = cv2.GaussianBlur(image, (k, k), sigmaX=0)
    foreground = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return np.clip(foreground, 0, 255).astype(np.uint8)


def step_normalise(image: np.ndarray) -> np.ndarray:
    """
    Convert uint8 [0, 255] to float32 [0, 1].

    Parameters
    ----------
    image : np.ndarray  (H, W, 3) uint8

    Returns
    -------
    np.ndarray  (H, W, 3) float32, values in [0, 1]
    """
    return image.astype(np.float32) / 255.0


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def _run_pipeline(image_rgb: np.ndarray) -> np.ndarray:
    """
    Apply all 5 preprocessing steps to a single RGB uint8 image.
    Returns float32 array of shape (MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3).
    """
    img = step_resize(image_rgb)
    img = step_cellular_automaton(img)
    img = step_histogram_equalisation(img)
    img = step_wiener_filter(img)
    img = step_background_removal(img)
    return step_normalise(img)


def _load_image(path: str) -> np.ndarray:
    """Load an image from disk and convert to RGB uint8."""
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Public API — single image
# ---------------------------------------------------------------------------

def preprocess_one(
    image: Union[str, np.ndarray],
    return_intermediate: bool = False,
) -> Union[np.ndarray, dict]:
    """
    Preprocess a single image for LiteCShuffle inference.

    Parameters
    ----------
    image : str or np.ndarray
        Either a file path (str) or a numpy array.
        Arrays can be RGB or BGR uint8 — if you pass a cv2.imread result
        (which is BGR) set `is_bgr=True` ... actually the function
        checks the path type and assumes BGR for arrays from cv2.
        To be safe, pass RGB arrays or a file path.
    return_intermediate : bool
        If True, also returns a dict with each pipeline step's output
        for inspection/debugging. Useful for checking what each step does.

    Returns
    -------
    np.ndarray  shape (160, 160, 3), dtype float32, values in [0, 1]
        Ready to pass to model.predict(arr[np.newaxis])

    — or, if return_intermediate=True —

    dict with keys:
        'original'      : original image resized to 160×160, uint8 RGB
        'after_ca'      : after cellular automaton filter
        'after_clahe'   : after histogram equalisation
        'after_wiener'  : after Wiener filter
        'after_bgremove': after background removal
        'final'         : float32 normalised output
    """
    # Load from disk if a path was given
    if isinstance(image, str):
        img_rgb = _load_image(image)
    elif isinstance(image, np.ndarray):
        img_rgb = image.copy()
    else:
        raise TypeError(f"image must be str or np.ndarray, got {type(image)}")

    # Ensure uint8 RGB
    if img_rgb.dtype != np.uint8:
        img_rgb = np.clip(img_rgb * 255, 0, 255).astype(np.uint8)

    if not return_intermediate:
        return _run_pipeline(img_rgb)

    # Step-by-step with intermediate outputs
    s1 = step_resize(img_rgb)
    s2 = step_cellular_automaton(s1)
    s3 = step_histogram_equalisation(s2)
    s4 = step_wiener_filter(s3)
    s5 = step_background_removal(s4)
    s6 = step_normalise(s5)

    return {
        "original":       s1,
        "after_ca":       s2,
        "after_clahe":    s3,
        "after_wiener":   s4,
        "after_bgremove": s5,
        "final":          s6,
    }


# ---------------------------------------------------------------------------
# Public API — batch images
# ---------------------------------------------------------------------------

def preprocess_batch(
    images: List[Union[str, np.ndarray]],
    on_progress: Optional[Callable[[int, int], None]] = None,
    skip_errors: bool = True,
) -> np.ndarray:
    """
    Preprocess a batch of images for LiteCShuffle inference.

    Parameters
    ----------
    images : list of str or np.ndarray
        File paths and/or numpy arrays, in any combination.
    on_progress : callable(i, n) or None
        Called after each image with the 1-based index and total count.
        Use for progress bars or logging. Default prints to stdout.
    skip_errors : bool
        If True (default), failed images are skipped and a warning is printed.
        If False, any error raises immediately.

    Returns
    -------
    np.ndarray  shape (N, 160, 160, 3), dtype float32
        Where N = number of successfully processed images.
        If some images failed and skip_errors=True, N may be < len(images).

    Examples
    --------
    # Basic usage
    batch = preprocess_batch(["leaf1.jpg", "leaf2.jpg"])
    predictions = model.predict(batch)

    # With progress printing
    batch = preprocess_batch(paths, on_progress=lambda i, n: print(f"{i}/{n}"))

    # Custom progress bar using tqdm
    from tqdm import tqdm
    bar = tqdm(total=len(paths))
    batch = preprocess_batch(paths, on_progress=lambda i, n: bar.update(1))
    bar.close()
    """
    n       = len(images)
    results = []
    failed  = 0

    # Default progress: print every 10 images
    if on_progress is None:
        def on_progress(i, total):
            if i == 1 or i % 10 == 0 or i == total:
                print(f"  Preprocessing: {i}/{total} ...", flush=True)

    for i, img in enumerate(images, start=1):
        try:
            arr = preprocess_one(img)
            results.append(arr)
        except Exception as e:
            failed += 1
            name = img if isinstance(img, str) else f"array[{i-1}]"
            if skip_errors:
                print(f"  [WARN] Skipped {name}: {e}")
            else:
                raise
        on_progress(i, n)

    if failed > 0:
        print(f"  {failed}/{n} images failed and were skipped.")

    if not results:
        raise RuntimeError("No images were successfully preprocessed.")

    return np.stack(results, axis=0)   # (N, 160, 160, 3) float32


# ---------------------------------------------------------------------------
# Convenience: preprocess a folder
# ---------------------------------------------------------------------------

def preprocess_folder(
    folder: str,
    extensions: tuple = (".jpg", ".jpeg", ".png", ".bmp", ".tiff",
                          ".JPG", ".JPEG", ".PNG", ".BMP", ".TIFF"),
    on_progress: Optional[Callable[[int, int], None]] = None,
    skip_errors: bool = True,
) -> tuple:
    """
    Preprocess all images in a folder.

    Parameters
    ----------
    folder     : path to folder containing images
    extensions : file extensions to include
    on_progress: optional progress callback (see preprocess_batch)
    skip_errors: skip unreadable files if True, raise otherwise

    Returns
    -------
    batch      : np.ndarray  (N, 160, 160, 3) float32
    paths      : list of str — file paths in the same order as batch rows
    """
    paths = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(extensions)
    ])
    if not paths:
        raise FileNotFoundError(
            f"No images found in {folder} with extensions {extensions}")
    print(f"[preprocess_folder] Found {len(paths)} images in {folder}")
    batch = preprocess_batch(paths, on_progress=on_progress,
                              skip_errors=skip_errors)
    return batch, paths


# ---------------------------------------------------------------------------
# Visualisation helper (optional — requires matplotlib)
# ---------------------------------------------------------------------------

def show_pipeline_steps(image: Union[str, np.ndarray]):
    """
    Display a 6-panel grid showing each preprocessing step.
    Requires matplotlib.  Good for checking pipeline correctness.

    Parameters
    ----------
    image : str or np.ndarray — image to inspect
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for show_pipeline_steps. "
                          "Install with: pip install matplotlib")

    steps = preprocess_one(image, return_intermediate=True)
    titles = [
        "1. Original (resized)",
        "2. Cellular automaton",
        "3. CLAHE equalisation",
        "4. Wiener filter",
        "5. Background removal",
        "6. Final (normalised)",
    ]
    arrays = [
        steps["original"],
        steps["after_ca"],
        steps["after_clahe"],
        steps["after_wiener"],
        steps["after_bgremove"],
        (steps["final"] * 255).astype(np.uint8),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle("LiteCShuffle Preprocessing Pipeline", fontsize=14, fontweight="bold")
    for ax, title, arr in zip(axes.flat, titles, arrays):
        ax.imshow(arr)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess images for LiteCShuffle plant disease model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--image",  type=str,
                     help="Single image file path")
    src.add_argument("--folder", type=str,
                     help="Folder of images — preprocesses all jpg/png files")
    p.add_argument("--output",   type=str, default=None,
                   help="Save preprocessed array(s) as a .npy file. "
                        "Single image: shape (160,160,3). "
                        "Folder/batch: shape (N,160,160,3).")
    p.add_argument("--show",     action="store_true",
                   help="Display the 6-step pipeline grid (single image only, "
                        "requires matplotlib)")
    p.add_argument("--show_steps", action="store_true",
                   help="Print shape and value stats for each pipeline step")
    return p.parse_args()


def main():
    args   = _parse_args()
    t_start = time.time()

    if args.image:
        print(f"[Preprocess] Single image: {args.image}")
        if args.show:
            show_pipeline_steps(args.image)
            return

        if args.show_steps:
            steps = preprocess_one(args.image, return_intermediate=True)
            for name, arr in steps.items():
                print(f"  {name:<20} shape={arr.shape}  "
                      f"dtype={arr.dtype}  "
                      f"min={arr.min():.3f}  max={arr.max():.3f}")
            arr = steps["final"]
        else:
            arr = preprocess_one(args.image)

        print(f"  Output shape: {arr.shape}  dtype: {arr.dtype}")
        print(f"  Value range:  [{arr.min():.4f}, {arr.max():.4f}]")
        elapsed = time.time() - t_start
        print(f"  Time: {elapsed:.2f}s")

        if args.output:
            np.save(args.output, arr)
            size_kb = os.path.getsize(args.output) / 1024
            print(f"  Saved -> {args.output}  ({size_kb:.0f} KB)")

    else:  # --folder
        print(f"[Preprocess] Batch folder: {args.folder}")
        batch, paths = preprocess_folder(args.folder)
        elapsed = time.time() - t_start
        print(f"\n  Batch shape: {batch.shape}  dtype: {batch.dtype}")
        print(f"  Value range: [{batch.min():.4f}, {batch.max():.4f}]")
        print(f"  Time: {elapsed:.1f}s  "
              f"({elapsed/len(paths)*1000:.0f}ms per image)")

        if args.output:
            np.save(args.output, batch)
            size_mb = os.path.getsize(args.output) / 1024 / 1024
            print(f"  Saved -> {args.output}  ({size_mb:.1f} MB)")
        else:
            print("\n  Tip: add --output batch.npy to save the preprocessed arrays.")


if __name__ == "__main__":
    main()
