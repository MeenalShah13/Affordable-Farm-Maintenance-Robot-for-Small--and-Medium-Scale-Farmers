"""
cache.py
Chunked disk caching for preprocessed features and images using HDF5.

RAM-safe design
---------------
Features and images are NEVER fully loaded into RAM.  Instead, this
module provides LazyH5Array — a wrapper around an open HDF5 dataset that
reads slices on demand.  It implements enough of the numpy array interface
(shape, dtype, __len__, __getitem__) for Keras .fit() / .evaluate() and
sklearn estimators to work without modification.

All existing code that does  X_train[start:end]  or  X_train[idx]
continues to work — it just reads from disk instead of RAM.

GPU safety
----------
Worker processes are re-instantiated with use_gpu=False to avoid
cudaErrorInvalidDevice (error code 9).
"""

import os
import numpy as np
import h5py


# ---------------------------------------------------------------------------
# LazyH5Array — read-on-demand wrapper around an HDF5 dataset
# ---------------------------------------------------------------------------

class LazyH5Array:
    """
    Wraps an HDF5 dataset so it looks like a numpy array.

    Reads are lazy: only the requested slice is loaded from disk.
    The underlying HDF5 file stays open for the lifetime of this object.

    Parameters
    ----------
    h5_path  : path to the .h5 file
    key      : dataset name inside the file (e.g. "X_train")

    Supports
    --------
    * len(a)
    * a.shape, a.dtype, a.ndim
    * a[i]          — single integer index
    * a[i:j]        — slice
    * a[i:j:k]      — slice with step
    * a[[1,3,7]]    — fancy (list/array) index
    * a[np.array([…])]
    * iteration: for row in a
    * np.array(a)   — explicit full load (use only when you know it fits)
    """

    def __init__(self, h5_path: str, key: str):
        self._path = h5_path
        self._key  = key
        self._file = h5py.File(h5_path, "r")
        self._ds   = self._file[key]

    # ---- numpy-like properties ----------------------------------------------

    @property
    def shape(self):
        return self._ds.shape

    @property
    def dtype(self):
        return self._ds.dtype

    @property
    def ndim(self):
        return len(self._ds.shape)

    def __len__(self):
        return self._ds.shape[0]

    # ---- indexing -----------------------------------------------------------

    def __getitem__(self, idx):
        """
        Read only the requested slice from disk.

        HDF5 requires fancy (list/array) indices to be in strictly increasing
        order. Keras shuffles its batch indices, so we sort before reading and
        then invert the sort to return rows in the original requested order.
        """
        import numpy as _np
        # Integer or slice — pass straight through
        if isinstance(idx, (int, slice)):
            return self._ds[idx]
        # Numpy array or list of indices — must sort for HDF5
        idx_arr = _np.asarray(idx)
        if idx_arr.ndim == 0:
            return self._ds[int(idx_arr)]
        sort_order = _np.argsort(idx_arr)
        sorted_idx = idx_arr[sort_order]
        data = self._ds[list(sorted_idx)]   # HDF5 happy: strictly increasing
        # Invert sort to restore the original requested row order
        unsort = _np.empty_like(sort_order)
        unsort[sort_order] = _np.arange(len(sort_order))
        return data[unsort]

    # ---- iteration ----------------------------------------------------------

    def __iter__(self):
        for i in range(len(self)):
            yield self._ds[i]

    # ---- explicit conversion ------------------------------------------------

    def __array__(self, dtype=None, copy=None):
        """np.array(lazy) — loads everything. Use only when it fits in RAM."""
        data = self._ds[:]
        return data.astype(dtype) if dtype is not None else data

    # ---- cleanup ------------------------------------------------------------

    def close(self):
        self._file.close()

    def __del__(self):
        try:
            self._file.close()
        except Exception:
            pass

    def sample(self, n: int, seed: int = 42) -> np.ndarray:
        """
        Draw n rows at random without loading the full dataset.
        Used to give feature selectors a representative sample.
        """
        rng  = np.random.default_rng(seed)
        idx  = np.sort(rng.choice(len(self), size=min(n, len(self)), replace=False))
        # HDF5 fancy indexing requires a sorted list
        return self._ds[list(idx)]

    def to_tf_dataset(self, labels: np.ndarray, batch_size: int = 32,
                      shuffle: bool = False, shuffle_buffer: int = 1000):
        """
        Return a tf.data.Dataset that reads this array from HDF5 one batch
        at a time.  Peak RAM = one batch, regardless of total dataset size.
        Works for both feature vectors and image arrays.
        """
        import tensorflow as tf
        path       = self._path
        key        = self._key
        labels_arr = np.array(labels, dtype=np.int32)
        shape      = self.shape[1:]   # per-sample shape, e.g. (143019,) or (160,160,3)

        def _gen():
            with h5py.File(path, "r") as f:
                ds = f[key]
                for i in range(len(labels_arr)):
                    yield ds[i], labels_arr[i]

        ds = tf.data.Dataset.from_generator(
            _gen,
            output_signature=(
                tf.TensorSpec(shape=shape, dtype=tf.float32),
                tf.TensorSpec(shape=(),    dtype=tf.int32),
            ))
        if shuffle:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(2)  # cap to 2 batches to bound RAM

    def astype(self, dtype):
        """
        Materialise and cast — matches numpy ndarray.astype() interface.
        Called by model code that does X.astype(np.float32) directly.
        Loads the full array, so only use when it fits in RAM (e.g. small
        val/test splits, or post-selection feature arrays).
        """
        return self._ds[:].astype(dtype)

    def __repr__(self):
        return (f"LazyH5Array(path={self._path!r}, key={self._key!r}, "
                f"shape={self.shape}, dtype={self.dtype})")


# ---------------------------------------------------------------------------
# (Multiprocessing pool removed — see build_features docstring for reason)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DataCache
# ---------------------------------------------------------------------------

class DataCache:
    """
    Manages HDF5 cache files for features and preprocessed images.

    cache_dir/
        features.h5   — X_train, X_val, X_test, y_train, y_val, y_test
        images.h5     — X_train, X_val, X_test  (float32, normalised 0-1)

    After building, data is accessed via LazyH5Array objects that read
    slices on demand — the full arrays are never held in RAM.
    """

    def __init__(self, cache_dir: str = "outputs/cache"):
        self.cache_dir = cache_dir
        self.feat_path = os.path.join(cache_dir, "features.h5")
        self.img_path  = os.path.join(cache_dir, "images.h5")
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[DataCache] Sequential feature extraction (no worker pool).")

    # ------------------------------------------------------------------ #
    #  Features                                                            #
    # ------------------------------------------------------------------ #

    def features_cached(self) -> bool:
        if not os.path.exists(self.feat_path):
            return False
        with h5py.File(self.feat_path, "r") as f:
            return all(k in f for k in
                       ("X_train", "X_val", "X_test", "y_train", "y_val", "y_test"))

    def _get_feat_dim(self, paths, preprocessor, extractor, loader) -> int:
        for p in paths:
            img = loader.load_image(p)
            if img is not None:
                return len(extractor.extract(preprocessor(img)))
        raise RuntimeError("No valid images to determine feature dimension.")

    def build_features(self, tr_paths, val_paths, te_paths,
                       y_train, y_val, y_test,
                       preprocessor, extractor, loader,
                       chunk_size: int = 32):
        """
        Build the feature HDF5 cache sequentially in the main process.

        Multiprocessing (spawn Pool) was removed because on shared lab servers
        it causes semaphore exhaustion — each spawned process allocates OS
        semaphores that are not reliably released when the pool is destroyed
        inside an open HDF5 context. With 3 splits × N workers, the semaphore
        count hits the system limit and the process dies silently.

        Sequential extraction is safer and still fast enough: feature
        extraction is CPU-bound and the GIL prevents true parallelism anyway.
        Progress is printed every 50 images so you can see the job is alive.
        """
        print(f"\n[Cache] Building feature cache -> {self.feat_path}")
        feat_dim = self._get_feat_dim(tr_paths, preprocessor, extractor, loader)
        print(f"        Feature dim: {feat_dim}  (sequential, main process)")
        splits = [("train", tr_paths, y_train),
                  ("val",   val_paths, y_val),
                  ("test",  te_paths,  y_test)]

        with h5py.File(self.feat_path, "w") as f:
            # Pre-create all datasets so the file is valid even if we crash
            for name, paths, labels in splits:
                n = len(paths)
                f.create_dataset(f"X_{name}", shape=(n, feat_dim), dtype="float32",
                                 chunks=(min(chunk_size, n), feat_dim))
                f.create_dataset(f"y_{name}", data=np.array(labels, dtype=np.int32))

            for name, paths, _ in splits:
                ds  = f[f"X_{name}"]
                n   = len(paths)
                print(f"\n  [{name}] {n} images ...")
                buf, write_idx = [], 0

                for i, path in enumerate(paths):
                    try:
                        img = loader.load_image(path)
                        if img is None:
                            continue
                        feat = extractor.extract(preprocessor(img)).astype(np.float32)
                        buf.append(feat)
                    except Exception as e:
                        print(f"    [WARN] skipping {path}: {e}")
                        continue

                    if len(buf) == chunk_size:
                        arr = np.array(buf, dtype=np.float32)
                        ds[write_idx:write_idx + len(arr)] = arr
                        write_idx += len(arr)
                        buf = []

                    if (i + 1) % 50 == 0:
                        pct = 100 * (i + 1) / n
                        print(f"    {i+1}/{n}  ({pct:.0f}%)")
                        f.flush()   # flush to disk periodically — safe recovery point

                if buf:
                    arr = np.array(buf, dtype=np.float32)
                    ds[write_idx:write_idx + len(arr)] = arr

                f.flush()
                print(f"  [{name}] done  ({write_idx + len(buf)} features written)")

        print(f"\n[Cache] Features saved -> {self.feat_path}")

    def open_features(self):
        """
        Return LazyH5Array handles — reads slices on demand, never loads all.
        Returns (X_train, X_val, X_test, y_train, y_val, y_test).
        y_* are small integer arrays loaded fully (labels are tiny).
        """
        with h5py.File(self.feat_path, "r") as f:
            y_tr  = f["y_train"][:]
            y_val = f["y_val"][:]
            y_te  = f["y_test"][:]
        return (LazyH5Array(self.feat_path, "X_train"),
                LazyH5Array(self.feat_path, "X_val"),
                LazyH5Array(self.feat_path, "X_test"),
                y_tr, y_val, y_te)

    def get_features(self, tr_paths, val_paths, te_paths,
                     y_train, y_val, y_test,
                     preprocessor, extractor, loader,
                     chunk_size: int = 32, force_rebuild: bool = False):
        """Build cache if missing; return LazyH5Array handles."""
        if force_rebuild or not self.features_cached():
            self.build_features(tr_paths, val_paths, te_paths,
                                y_train, y_val, y_test,
                                preprocessor, extractor, loader,
                                chunk_size=chunk_size)
        else:
            print(f"[Cache] Opening features from {self.feat_path}")
        return self.open_features()

    # ------------------------------------------------------------------ #
    #  Images                                                              #
    # ------------------------------------------------------------------ #

    def images_cached(self, image_size: int) -> bool:
        if not os.path.exists(self.img_path):
            return False
        with h5py.File(self.img_path, "r") as f:
            return "X_train" in f and f.attrs.get("image_size", -1) == image_size

    def build_images(self, tr_paths, val_paths, te_paths,
                     preprocessor, loader, image_size: int = 160,
                     chunk_size: int = 16):
        """Single-process so GPU preprocessing runs in the main process."""
        print(f"\n[Cache] Building image cache -> {self.img_path}")
        C      = 3
        splits = [("train", tr_paths), ("val", val_paths), ("test", te_paths)]
        with h5py.File(self.img_path, "w") as f:
            f.attrs["image_size"] = image_size
            for name, paths in splits:
                n = len(paths)
                f.create_dataset(f"X_{name}",
                                 shape=(n, image_size, image_size, C),
                                 dtype="float32",
                                 chunks=(min(chunk_size, n),
                                         image_size, image_size, C))
            for name, paths in splits:
                ds = f[f"X_{name}"]
                buf, write_idx = [], 0
                for i, path in enumerate(paths):
                    img = loader.load_image(path)
                    if img is None:
                        continue
                    buf.append(preprocessor(img).astype(np.float32))
                    if len(buf) == chunk_size:
                        ds[write_idx:write_idx + len(buf)] = np.array(buf)
                        write_idx += len(buf)
                        buf = []
                    if (i + 1) % 50 == 0:
                        print(f"  [{name}] {i+1}/{len(paths)} ...")
                if buf:
                    ds[write_idx:write_idx + len(buf)] = np.array(buf)
                print(f"  [{name}] done.")
        print(f"[Cache] Images saved -> {self.img_path}")

    def open_images(self):
        """
        Return LazyH5Array handles for image splits.
        Reads slices on demand — never loads all images into RAM.
        """
        return (LazyH5Array(self.img_path, "X_train"),
                LazyH5Array(self.img_path, "X_val"),
                LazyH5Array(self.img_path, "X_test"))

    def get_images(self, tr_paths, val_paths, te_paths,
                   preprocessor, loader, image_size: int = 160,
                   chunk_size: int = 16, force_rebuild: bool = False):
        """Build cache if missing; return LazyH5Array handles."""
        if force_rebuild or not self.images_cached(image_size):
            self.build_images(tr_paths, val_paths, te_paths, preprocessor, loader,
                              image_size=image_size, chunk_size=chunk_size)
        else:
            print(f"[Cache] Opening images from {self.img_path}")
        return self.open_images()

    # ------------------------------------------------------------------ #
    #  tf.data streaming (alternative to LazyH5Array for Keras models)   #
    # ------------------------------------------------------------------ #

    def image_dataset(self, split: str, labels: np.ndarray,
                      batch_size: int = 32, shuffle: bool = False,
                      shuffle_buffer: int = 500):
        """Stream images from HDF5 as tf.data.Dataset."""
        import tensorflow as tf
        key_map    = {"train": "X_train", "val": "X_val", "test": "X_test"}
        img_path   = self.img_path
        hdf5_key   = key_map[split]
        labels_arr = np.array(labels, dtype=np.int32)
        with h5py.File(self.img_path, "r") as f:
            img_shape = tuple(f[hdf5_key].shape[1:])
        def _gen():
            with h5py.File(img_path, "r") as f:
                ds = f[hdf5_key]
                for i in range(len(labels_arr)):
                    yield ds[i], labels_arr[i]
        ds = tf.data.Dataset.from_generator(
            _gen,
            output_signature=(tf.TensorSpec(shape=img_shape, dtype=tf.float32),
                               tf.TensorSpec(shape=(),         dtype=tf.int32)))
        if shuffle:
            ds = ds.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
        return ds.batch(batch_size).prefetch(2)  # cap to 2 batches to bound RAM

    def feature_dataset(self, split: str, labels: np.ndarray,
                        batch_size: int = 256):
        """Stream feature vectors from HDF5 as tf.data.Dataset."""
        import tensorflow as tf
        key_map    = {"train": "X_train", "val": "X_val", "test": "X_test"}
        feat_path  = self.feat_path
        hdf5_key   = key_map[split]
        labels_arr = np.array(labels, dtype=np.int32)
        with h5py.File(self.feat_path, "r") as f:
            feat_dim = f[hdf5_key].shape[1]
        def _gen():
            with h5py.File(feat_path, "r") as f:
                ds = f[hdf5_key]
                for i in range(len(labels_arr)):
                    yield ds[i], labels_arr[i]
        ds = tf.data.Dataset.from_generator(
            _gen,
            output_signature=(tf.TensorSpec(shape=(feat_dim,), dtype=tf.float32),
                               tf.TensorSpec(shape=(),           dtype=tf.int32)))
        return ds.batch(batch_size).prefetch(2)  # cap to 2 batches to bound RAM

    def cache_info(self):
        print("\n[Cache] Cache directory:", self.cache_dir)
        for path, name in [(self.feat_path, "Features"), (self.img_path, "Images")]:
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / 1024 / 1024
                with h5py.File(path, "r") as f:
                    shapes = {k: f[k].shape for k in f.keys()}
                print(f"  {name}: {path} ({size_mb:.1f} MB)")
                for k, s in shapes.items():
                    print(f"    {k}: {s}")
            else:
                print(f"  {name}: not cached")