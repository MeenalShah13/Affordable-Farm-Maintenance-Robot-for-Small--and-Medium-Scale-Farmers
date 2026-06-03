"""
preprocessing.py
Implements the preprocessing layer described in the document:
  1. Resize image – 512x512
  2. Cellular automaton filter (noise/error pixel removal)
  3. Histogram equalisation (brightness normalisation)
  4. Wiener filtering
  5. Background removal via Gaussian filter

GPU acceleration
----------------
If CuPy is installed (`pip install cupy-cuda12x`), the Wiener filter and
background removal steps run on GPU automatically, giving a 3-10x speedup
on those steps. All other steps use OpenCV which already uses optimised
CPU SIMD instructions. Falls back to CPU silently if CuPy is unavailable.
"""

import cv2
import numpy as np
from scipy.signal import wiener

# Try to import CuPy for GPU-accelerated array ops — lazy to avoid CUDA
# context errors when this module is imported in worker subprocesses.
_CUPY_CHECKED = False
_CUPY_OK      = False

def _check_cupy():
    global _CUPY_CHECKED, _CUPY_OK
    if not _CUPY_CHECKED:
        try:
            import cupy              # noqa: F401
            import cupyx.scipy.ndimage  # noqa: F401
            _CUPY_OK = True
        except Exception:
            _CUPY_OK = False
        _CUPY_CHECKED = True
    return _CUPY_OK

def _cp():
    import cupy
    return cupy

def _cpnd():
    import cupyx.scipy.ndimage
    return cupyx.scipy.ndimage


class PreprocessingPipeline:
    """
    Sequential preprocessing pipeline for plant-disease images.

    The output image shape is (target_size[0], target_size[1], 3) as uint8.
    Calling an instance with a BGR or RGB uint8 array applies every step in
    the documented order and returns an RGB float32 array normalised to [0,1].

    Parameters
    ----------
    target_size : tuple
        (height, width) for the resize step.  Defaults to (512, 512) as per
        the specification, but can be overridden (e.g. 160×160 for the CNN).
    ca_iterations : int
        Number of iterations for the cellular-automaton denoising step.
    wiener_noise : float or None
        Estimated noise variance for Wiener filter.  None = auto-estimate.
    gauss_ksize : int
        Kernel size (odd) for the Gaussian background-removal step.
    use_gpu : bool
        If True and CuPy is available, run Wiener + background removal on GPU.
        Defaults to True (auto-detected).
    """

    def __init__(
        self,
        target_size: tuple = (512, 512),
        ca_iterations: int = 1,
        wiener_noise: float = None,
        gauss_ksize: int = 21,
        use_gpu: bool = True,
    ):
        self.target_size   = target_size
        self.ca_iterations = ca_iterations
        self.wiener_noise  = wiener_noise
        self.gauss_ksize   = gauss_ksize if gauss_ksize % 2 == 1 else gauss_ksize + 1
        self._use_gpu_requested = use_gpu
        self._use_gpu_resolved  = None  # resolved on first call

    def _gpu_ok(self) -> bool:
        if self._use_gpu_resolved is None:
            if not self._use_gpu_requested:
                self._use_gpu_resolved = False
            else:
                ok = _check_cupy()
                self._use_gpu_resolved = ok
                if ok:
                    print("[PreprocessingPipeline] GPU acceleration enabled (CuPy).")
                else:
                    print("[PreprocessingPipeline] CuPy unavailable — using CPU. "
                          "Install with: pip install cupy-cuda12x")
        return self._use_gpu_resolved

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """
        Apply the full preprocessing pipeline.

        Parameters
        ----------
        image : np.ndarray  (H, W, 3) uint8, RGB or BGR

        Returns
        -------
        np.ndarray  (target_H, target_W, 3) float32, values in [0, 1]
        """
        img = image.copy()
        img = self.resize(img)
        img = self.cellular_automaton_filter(img)
        img = self.histogram_equalisation(img)
        img = self.wiener_filter(img)
        img = self.remove_background(img)
        return img.astype(np.float32) / 255.0

    # ------------------------------------------------------------------
    # Step 1 – Resize
    # ------------------------------------------------------------------

    def resize(self, image: np.ndarray) -> np.ndarray:
        """Resize to target_size using area interpolation (good for downscale)."""
        h, w = self.target_size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    # ------------------------------------------------------------------
    # Step 2 – Cellular automaton filter (error-pixel removal)
    # ------------------------------------------------------------------

    def cellular_automaton_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Simple majority-vote cellular automaton to remove isolated noise pixels.
        Each pixel is replaced by the median of its 3x3 neighbourhood if it
        deviates significantly from the neighbourhood median.
        """
        result = image.copy()
        for _ in range(self.ca_iterations):
            median = cv2.medianBlur(result, 3)
            diff   = np.abs(result.astype(np.int16) - median.astype(np.int16))
            mask   = diff > 30
            result[mask] = median[mask]
        return result

    # ------------------------------------------------------------------
    # Step 3 – Histogram equalisation
    # ------------------------------------------------------------------

    def histogram_equalisation(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to the luminance channel in LAB colour space.
        """
        lab   = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # ------------------------------------------------------------------
    # Step 4 – Wiener filter  (GPU-accelerated if CuPy available)
    # ------------------------------------------------------------------

    def wiener_filter(self, image: np.ndarray) -> np.ndarray:
        if self._gpu_ok():
            return self._wiener_gpu(image)
        return self._wiener_cpu(image)

    def _wiener_cpu(self, image: np.ndarray) -> np.ndarray:
        out = np.zeros_like(image, dtype=np.float64)
        for c in range(image.shape[2]):
            out[:, :, c] = wiener(image[:, :, c].astype(np.float64),
                                  noise=self.wiener_noise)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _wiener_gpu(self, image: np.ndarray) -> np.ndarray:
        cp      = _cp()
        cpnd_m  = _cpnd()
        img_gpu = cp.asarray(image.astype(np.float64))
        out_gpu = cp.zeros_like(img_gpu)
        size    = 3
        for c in range(img_gpu.shape[2]):
            ch         = img_gpu[:, :, c]
            local_mean = cpnd_m.uniform_filter(ch, size=size)
            local_var  = cpnd_m.uniform_filter(ch ** 2, size=size) - local_mean ** 2
            local_var  = cp.maximum(local_var, 0)
            noise      = float(cp.mean(local_var).get()) if self.wiener_noise is None \
                         else self.wiener_noise
            out_gpu[:, :, c] = local_mean + \
                cp.where(local_var > noise,
                         (local_var - noise) / cp.maximum(local_var, noise) * (ch - local_mean),
                         0)
        return cp.asnumpy(cp.clip(out_gpu, 0, 255).astype(cp.uint8))

    # ------------------------------------------------------------------
    # Step 5 – Background removal (GPU-accelerated if CuPy available)
    # ------------------------------------------------------------------

    def remove_background(self, image: np.ndarray) -> np.ndarray:
        if self._gpu_ok():
            return self._background_gpu(image)
        return self._background_cpu(image)

    def _background_cpu(self, image: np.ndarray) -> np.ndarray:
        blurred    = cv2.GaussianBlur(image, (self.gauss_ksize, self.gauss_ksize), sigmaX=0)
        foreground = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
        return np.clip(foreground, 0, 255).astype(np.uint8)

    def _background_gpu(self, image: np.ndarray) -> np.ndarray:
        cp      = _cp()
        cpnd_m  = _cpnd()
        img_gpu = cp.asarray(image.astype(np.float32))
        sigma   = self.gauss_ksize / 6.0
        blurred = cp.stack([
            cpnd_m.gaussian_filter(img_gpu[:, :, c], sigma=sigma)
            for c in range(img_gpu.shape[2])
        ], axis=2)
        foreground = 1.5 * img_gpu - 0.5 * blurred
        return cp.asnumpy(cp.clip(foreground, 0, 255).astype(cp.uint8))

