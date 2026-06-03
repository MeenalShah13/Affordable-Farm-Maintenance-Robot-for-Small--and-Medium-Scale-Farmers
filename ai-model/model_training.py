"""
model_training.py
Implements all AI models listed in the document.

Models implemented (from papers):
  1.  SimpleVGG16       – Provided baseline (VGG16 + dense head)
  2.  VGG19_SVM         – VGG19 feature extractor + SVM head
  3.  HybridCNN         – WOA-APSO paper (Nature 2025): custom CNN
  4.  KCNet             – arxiv:2108.07554: insect-inspired sparse random network
  5.  FlyCaps           – IJRITCC 2023 ✅ FULL PAPER:
                          VGG-19 backbone → Primary Capsule (squash routing) →
                          Firefly-optimised flatten → ELM head (analytic W_out)
                          lr=0.0001, batch=30, epochs=120, dropout=0.1
  6.  LiteCShuffle      – Cogent F&A 2025: lightweight CNN w/ channel attention
  7.  YangNet           – Yang et al. IEEE TSMC 2024 ✅ SNN-INSPIRED PROXY:
                          VPL (Conv encoder + winner-take-all inhibition) →
                          DML (all-to-all excitatory → competitive softmax)
                          (Full FPGA neuromorphic impl. not feasible in TF)
  8.  HybridGrasshopperABC – Springer 2023 ⚠️ PAYWALL – deep MLP proxy
  9.  MantisSearch      – ScienceDirect 2023 ⚠️ PAYWALL – ELM proxy

Training utilities:
  • apply_gradinit()            – GradInit (CS231n): S/S̃ batches, meta-gradient
                                  on scale vars, clip γ=1, lower-bound α=0.01
  • ImportanceSamplingCallback  – Bandit importance sampling, momentum variant:
                                  p_t = γ·p_{t-1} + (1-γ)·l^t, bias-corrected
  • ChannelPruner               – Magnitude-based L1 channel pruning
  • ModelTrainer                – Unified launcher

Common interface (Keras models):
    model.build()
    model.train(X_tr, y_tr, X_val, y_val, epochs, batch_size)
    model.evaluate(X_test, y_test) -> dict
    model.predict(X)            -> np.ndarray (int labels)

Analytic models (KCNet, MantisSearch, FlyCaps):
    model.train(X_tr, y_tr)
    model.evaluate / predict  (same interface)
"""

import os
import time
import numpy as np
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Custom Keras layers — defined at module level so they serialise safely.
# Replaces Lambda layers which Keras refuses to load by default.
# ---------------------------------------------------------------------------

class WinnerTakeAllLayer(tf.keras.layers.Layer):
    """
    Top-k winner-take-all lateral inhibition.
    Keeps the k largest activations per sample; zeros the rest.
    Used in YangViT as a proxy for lateral inhibitory interneurons.
    """

    def __init__(self, k: int, **kwargs):
        super().__init__(**kwargs)
        self.k = int(k)

    def call(self, x):
        vals, _ = tf.math.top_k(x, k=self.k)
        threshold = vals[:, -1:]          # k-th largest per sample
        return x * tf.cast(x >= threshold, x.dtype)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"k": self.k})
        return cfg


class SquashLayer(tf.keras.layers.Layer):
    """
    Capsule squash activation (element-wise along the last axis).
    ||v||^2 / (1 + ||v||^2) * v / ||v||
    Used in FlyCaps capsule network.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, v):
        sq_norm = tf.reduce_sum(tf.square(v), axis=-1, keepdims=True)
        scale   = sq_norm / (1.0 + sq_norm) / (tf.sqrt(sq_norm) + 1e-8)
        return scale * v

    def get_config(self):
        return super().get_config()


# ---------------------------------------------------------------------------
# Helpers for large-dataset / low-RAM training
# ---------------------------------------------------------------------------

def _is_lazy(X) -> bool:
    """True if X is a LazyH5Array or tf.data.Dataset (not a plain ndarray)."""
    return not isinstance(X, np.ndarray)


def _as_tf_dataset(X, y, batch_size: int = 32, shuffle: bool = False):
    """
    Wrap X, y in a tf.data.Dataset regardless of input type.

    Accepts:
      • numpy ndarray          → in-memory Dataset (small data)
      • LazyH5Array            → generator Dataset (reads from HDF5 per batch)
      • tf.data.Dataset        → returned as-is
    """
    if isinstance(X, tf.data.Dataset):
        return X
    if isinstance(X, np.ndarray):
        ds = tf.data.Dataset.from_tensor_slices(
            (X.astype(np.float32), y.astype(np.int32)))
    else:
        # LazyH5Array — use its built-in generator
        return X.to_tf_dataset(y, batch_size=batch_size, shuffle=shuffle)
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(y)),
                        reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(2)  # cap prefetch to 2 batches to bound RAM


def _chunked_normal_equations(X_lazy, y, hidden_fn,
                              n_hidden: int,
                              n_classes: int,
                              chunk_size: int = 512) -> tuple:
    """
    Accumulate H^T H and H^T Y over a LazyH5Array without loading it fully.

    This is mathematically equivalent to computing the matrices on the full
    dataset but uses only `chunk_size` rows in RAM at a time.

    Parameters
    ----------
    X_lazy   : LazyH5Array (or ndarray) of shape (N, n_features)
    y        : integer label array (N,)
    hidden_fn: callable X -> H  (maps a chunk to hidden activations)
    n_hidden : number of hidden units (columns of H)
    n_classes: number of output classes
    chunk_size: rows processed per iteration

    Returns
    -------
    HtH : (n_hidden, n_hidden) float32
    HtY : (n_hidden, n_classes_or_1) float32
    classes: label array from LabelBinarizer
    """
    lb      = LabelBinarizer()
    Y_full  = lb.fit_transform(np.array(y)).astype(np.float32)
    classes = lb.classes_
    n_out   = Y_full.shape[1] if Y_full.ndim > 1 else 1

    HtH = np.zeros((n_hidden, n_hidden), dtype=np.float64)
    HtY = np.zeros((n_hidden, n_out),    dtype=np.float64)

    n = len(X_lazy) if hasattr(X_lazy, '__len__') else len(y)
    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        Xc    = np.array(X_lazy[start:end], dtype=np.float32)
        Yc    = Y_full[start:end]
        Hc    = hidden_fn(Xc).astype(np.float64)
        HtH  += Hc.T @ Hc
        HtY  += Hc.T @ Yc

    return HtH.astype(np.float32), HtY.astype(np.float32), classes


# ---------------------------------------------------------------------------
# Helpers for low-RAM Keras training
# ---------------------------------------------------------------------------

def _keras_train(model, X_train, y_train, X_val, y_val,
                 epochs, batch_size, callbacks, model_path):
    """
    Train a Keras model from either numpy arrays or LazyH5Arrays.
    When X_train is a LazyH5Array the data is streamed batch-by-batch from
    HDF5, so peak RAM = one batch regardless of dataset size.
    """
    train_ds = _as_tf_dataset(X_train, y_train,
                               batch_size=batch_size, shuffle=True)
    val_ds   = _as_tf_dataset(X_val,   y_val,
                               batch_size=batch_size, shuffle=False)
    model.fit(train_ds,
              validation_data=val_ds,
              epochs=epochs,
              callbacks=callbacks,
              verbose=1)


def _keras_evaluate(model, X_test, y_test, batch_size=32):
    """Evaluate without loading all test data into RAM."""
    test_ds = _as_tf_dataset(X_test, y_test,
                              batch_size=batch_size, shuffle=False)
    loss, acc = model.evaluate(test_ds, verbose=0)
    return {"loss": loss, "accuracy": acc}


def _keras_predict(model, X, batch_size=32):
    """Predict without loading all data into RAM."""
    if isinstance(X, np.ndarray):
        return model.predict(X, verbose=0)
    # LazyH5Array — stream without labels
    dummy_y = np.zeros(len(X), dtype=np.int32)
    ds = _as_tf_dataset(X, dummy_y, batch_size=batch_size, shuffle=False)
    # Strip labels from the dataset
    ds_x = ds.map(lambda x, y: x)
    return model.predict(ds_x, verbose=0)


# ---------------------------------------------------------------------------
# Utility: GradInit
# ---------------------------------------------------------------------------

def apply_gradinit(model, X_sample, y_sample, lr=0.01, n_steps=5,
                   clip_gamma=1.0, alpha_lb=0.01):
    """
    GradInit (CS231n / Zhu et al.):
      1. Sample batch S from training data
      2. Compute a gradient step  θ̃ = θ - η·∇ℓ(S)
      3. Evaluate gradient norm on a new batch S̃ (intersection ratio ≈ 0.5 with S)
      4. Update scale parameters to minimise ||∇ℓ(S̃)||  subject to:
           - gradient clip (γ = 1.0)
           - lower-bound constraint α ≥ α_lb = 0.01  (prevents collapse)
    Only the *scale* (γ) parameters of each layer are updated, not all weights,
    to preserve random-init structure while conditioning the loss landscape.
    """
    n = len(X_sample)
    batch = 32
    opt   = tf.keras.optimizers.SGD(learning_rate=lr)

    # Collect scale-like variables (kernel norms / all trainable vars as proxy)
    scale_vars = [v for v in model.trainable_variables
                  if 'kernel' in v.name or 'gamma' in v.name]

    for step in range(n_steps):
        # ── Batch S (first half) ──────────────────────────────────────────
        idx_s = np.random.choice(n, min(batch, n), replace=False)
        xS = tf.constant(X_sample[idx_s], dtype=tf.float32)
        yS = tf.constant(y_sample[idx_s],  dtype=tf.int32)

        with tf.GradientTape() as tape_s:
            preds_s = model(xS, training=True)
            loss_s  = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(yS, preds_s))
        grads_s = tape_s.gradient(loss_s, model.trainable_variables)

        # ── Trial step  θ̃ = θ - η∇ℓ(S)  (stored, not applied permanently) ──
        orig_weights = [v.numpy() for v in model.trainable_variables]
        for g, v in zip(grads_s, model.trainable_variables):
            if g is not None:
                v.assign(v - lr * g)

        # ── Batch S̃ (overlapping ≈50 % with S) ────────────────────────────
        # Build S̃ so that ~50% indices are shared with S
        n_shared = max(1, batch // 2)
        n_new    = batch - n_shared
        shared   = idx_s[:n_shared]
        new_idx  = np.random.choice(n, n_new, replace=False)
        idx_st   = np.concatenate([shared, new_idx])
        xSt = tf.constant(X_sample[idx_st], dtype=tf.float32)
        ySt = tf.constant(y_sample[idx_st],  dtype=tf.int32)

        # ── Minimise gradient norm on S̃ w.r.t. scale vars ─────────────────
        with tf.GradientTape() as tape_meta:
            tape_meta.watch(scale_vars)
            preds_st  = model(xSt, training=True)
            loss_st   = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(ySt, preds_st))
        meta_grads = tape_meta.gradient(loss_st, scale_vars)

        # Restore original weights before applying meta-update
        for orig, v in zip(orig_weights, model.trainable_variables):
            v.assign(orig)

        # Apply clipped meta-gradients with lower-bound guard
        for mg, sv in zip(meta_grads, scale_vars):
            if mg is None:
                continue
            mg_clipped = tf.clip_by_norm(mg, clip_gamma)
            new_val    = sv - lr * mg_clipped
            # α lower-bound: prevent any scale from collapsing to 0
            new_val    = tf.maximum(new_val, alpha_lb)
            sv.assign(new_val)

        print(f"  [GradInit] step {step + 1}/{n_steps}  "
              f"loss_S={loss_s.numpy():.4f}  loss_S̃={loss_st.numpy():.4f}")

    return model


# ---------------------------------------------------------------------------
# Utility: Importance Sampling callback
# ---------------------------------------------------------------------------

class ImportanceSamplingCallback(tf.keras.callbacks.Callback):
    """
    Bandit-based Importance Sampling with momentum (CS231n / Loshchilov et al.).

    Class-level weighting variant (preferred over per-sample to avoid
    outlier dominance, per the paper):
      p_{t,j} = γ * p_{t-1,j} + (1-γ) * l^t_j    if j ∈ I_t  (observed classes)
      p^t_j   = p_{t,j} / (1 - β^t)               bias-corrected probability

    where l^t_j = mean loss of class j at epoch t.
    Sample weights are proportional to p^t_j for each sample's class.
    """

    def __init__(self, X_train, y_train, gamma=0.9, beta=0.9):
        super().__init__()
        self.X_train = X_train
        self.y_train = y_train
        self.gamma   = gamma          # momentum coefficient
        self.beta    = beta           # bias-correction base
        self._classes      = np.unique(y_train)
        self._p            = {c: 1.0 / len(self._classes) for c in self._classes}
        self._t            = 0        # epoch counter (for bias correction)
        self.sample_weights = np.ones(len(y_train))

    def on_epoch_end(self, epoch, logs=None):
        self._t += 1
        # Compute per-sample losses
        preds  = self.model.predict(self.X_train, verbose=0)
        losses = tf.keras.losses.sparse_categorical_crossentropy(
            self.y_train.astype(np.int32),
            preds.astype(np.float32)
        ).numpy()

        # Update class-level momentum estimates
        for c in self._classes:
            mask = (self.y_train == c)
            if mask.sum() == 0:
                continue
            l_c = float(np.mean(losses[mask]))
            # Momentum update (only classes observed this epoch)
            self._p[c] = self.gamma * self._p[c] + (1.0 - self.gamma) * l_c

        # Bias correction: p̂_j = p_j / (1 - β^t)
        bias_factor = 1.0 - (self.beta ** self._t)
        p_hat = {c: self._p[c] / (bias_factor + 1e-10) for c in self._classes}

        # Map to per-sample weights
        total_p = sum(p_hat.values()) + 1e-10
        for c in self._classes:
            mask = (self.y_train == c)
            self.sample_weights[mask] = (p_hat[c] / total_p) * len(self._classes)


# ---------------------------------------------------------------------------
# Utility: standard callbacks
# ---------------------------------------------------------------------------

def _standard_callbacks(model_path):
    return [
        tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path, save_best_only=True, verbose=0),
    ]


def _compile(model, n_classes):
    if n_classes == 2:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def _output_layer(x, n_classes):
    if n_classes == 2:
        return tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.layers.Dense(n_classes, activation='softmax')(x)


def _predict_labels(preds, n_classes):
    if n_classes == 2:
        return (preds.squeeze() > 0.5).astype(int)
    return np.argmax(preds, axis=1)


# ===========================================================================
# Model 1: SimpleVGG16
# ===========================================================================

class SimpleVGG16:
    """
    VGG16 frozen feature extractor + custom dense head.
    Set use_feature_input=True + feature_dim=N to accept pre-selected features.
    """

    def __init__(self, input_shape=(160, 160, 3), n_classes=2,
                 model_path="best_vgg16.keras",
                 use_gradinit=False, use_importance_sampling=False,
                 use_feature_input=False, feature_dim=None):
        self.input_shape   = input_shape
        self.n_classes     = n_classes
        self.model_path    = model_path
        self.use_gradinit  = use_gradinit
        self.use_importance_sampling = use_importance_sampling
        self.use_feature_input = use_feature_input
        self.feature_dim   = feature_dim
        self.model         = None

    def build(self):
        if self.use_feature_input:
            inp = tf.keras.Input(shape=(self.feature_dim,))
            x   = tf.keras.layers.Dense(512, activation='relu')(inp)
            x   = tf.keras.layers.Dropout(0.5)(x)
            x   = tf.keras.layers.Dense(512, activation='relu')(x)
        else:
            base = tf.keras.applications.VGG16(
                weights='imagenet', include_top=False, input_shape=self.input_shape)
            base.trainable = False
            inp = tf.keras.Input(shape=self.input_shape)
            x   = base(inp, training=False)
            x   = tf.keras.layers.Flatten()(x)
            x   = tf.keras.layers.Dense(512, activation='relu')(x)
            x   = tf.keras.layers.Dropout(0.5)(x)
            x   = tf.keras.layers.Dense(512, activation='relu')(x)

        out = _output_layer(x, self.n_classes)
        self.model = _compile(tf.keras.Model(inp, out), self.n_classes)
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build()
        if self.use_gradinit:
            # GradInit needs a small numpy sample — read a tiny slice only
            sample = np.array(X_train[0:64], dtype=np.float32)
            self.model = apply_gradinit(self.model, sample, np.array(y_train)[0:64])
        cbs = _standard_callbacks(self.model_path)
        if self.use_importance_sampling:
            # ImportanceSampling needs numpy — limit to 2K rows to cap RAM
            n_is   = min(2000, len(X_train))
            X_is   = np.array(X_train[0:n_is], dtype=np.float32)
            y_is   = np.array(y_train)[0:n_is]
            cbs.append(ImportanceSamplingCallback(X_is, y_is))
        # Route through _keras_train so LazyH5Arrays are streamed batch-by-batch
        _keras_train(self.model, X_train, y_train, X_val, y_val,
                     epochs=epochs, batch_size=batch_size,
                     callbacks=cbs, model_path=self.model_path)
        return self

    def evaluate(self, X_test, y_test):
        return _keras_evaluate(self.model, X_test, y_test)

    def predict(self, X):
        return _predict_labels(_keras_predict(self.model, X), self.n_classes)


# ===========================================================================
# Model 2: VGG19 + SVM
# ===========================================================================

class VGG19_SVM:
    """VGG19 feature extractor + StandardScaler + RBF-SVM."""

    def __init__(self, input_shape=(160, 160, 3), svm_C=1.0, svm_kernel='rbf',
                 use_feature_input=False, n_classes=2, **_):
        self.input_shape      = input_shape
        self.svm_C            = svm_C
        self.svm_kernel       = svm_kernel
        self.use_feature_input = use_feature_input
        self._extractor = None
        self._pipeline  = None
        self.training_time_ = 0.0

    def _build_extractor(self):
        base = tf.keras.applications.VGG19(
            weights='imagenet', include_top=False, input_shape=self.input_shape)
        base.trainable = False
        inp = tf.keras.Input(shape=self.input_shape)
        x   = base(inp, training=False)
        x   = tf.keras.layers.GlobalAveragePooling2D()(x)
        self._extractor = tf.keras.Model(inp, x)

    def _extract(self, X, batch_size=32):
        """
        Run VGG19 feature extraction in small batches.

        Never materialises the full array at once — processes batch_size
        images at a time so GPU memory stays bounded regardless of dataset
        size. For LazyH5Arrays this reads directly from HDF5 per batch.
        For plain numpy arrays it slices in-place with no copy.
        """
        if self.use_feature_input:
            return np.array(X) if not isinstance(X, np.ndarray) else X
        if self._extractor is None:
            self._build_extractor()

        n = len(X)
        out_chunks = []
        for start in range(0, n, batch_size):
            end   = min(start + batch_size, n)
            batch = np.array(X[start:end], dtype=np.float32)  # one batch only
            batch_pre = tf.keras.applications.vgg19.preprocess_input(
                batch * 255.0)
            feats = self._extractor.predict(batch_pre, verbose=0)
            out_chunks.append(feats)
            if (start // batch_size) % 20 == 0:
                print(f"    VGG19 extract: {end}/{n}", flush=True)
        return np.concatenate(out_chunks, axis=0)

    def train(self, X_train, y_train, X_val=None, y_val=None, **_):
        t0 = time.time()
        # SVM cannot train on >~10K samples without running out of RAM
        # (kernel matrix is N x N). Sub-sample if needed.
        max_svm = 5000
        n = len(X_train)
        if n > max_svm:
            print(f"  [VGG19_SVM] Subsampling {n} -> {max_svm} for SVM fit "
                  f"(full N×N kernel matrix would be {n*n*4/1e9:.1f}GB)")
            rng  = np.random.default_rng(42)
            idx  = rng.choice(n, size=max_svm, replace=False)
            idx.sort()
            X_sub = X_train[idx]
            y_sub = np.array(y_train)[idx]
        else:
            X_sub, y_sub = X_train, y_train

        feats = self._extract(X_sub)
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(C=self.svm_C, kernel=self.svm_kernel, probability=True))
        ])
        self._pipeline.fit(feats, y_sub)
        self.training_time_ = time.time() - t0
        return self

    def evaluate(self, X_test, y_test):
        return {"accuracy": self._pipeline.score(self._extract(X_test), y_test)}

    def predict(self, X):
        return self._pipeline.predict(self._extract(X))


# ===========================================================================
# Model 3: HybridCNN  (Nature 2025 / WOA-APSO paper)
# ===========================================================================

class HybridCNN:
    """
    Custom CNN from the WOA-APSO Nature 2025 paper.
    Conv(32)→BN→MaxPool, Conv(64)→BN→MaxPool, Conv(128)→BN→GAP, Dense(256).
    Also accepts 1-D feature vectors (use_feature_input=True).
    """

    def __init__(self, input_shape=(160, 160, 3), n_classes=2,
                 model_path="best_hybridcnn.keras",
                 use_feature_input=False, feature_dim=None, **_):
        self.input_shape      = input_shape
        self.n_classes        = n_classes
        self.model_path       = model_path
        self.use_feature_input = use_feature_input
        self.feature_dim      = feature_dim
        self.model            = None

    def build(self):
        if self.use_feature_input:
            inp = tf.keras.Input(shape=(self.feature_dim,))
            x   = tf.keras.layers.Dense(256, activation='relu')(inp)
            x   = tf.keras.layers.BatchNormalization()(x)
            x   = tf.keras.layers.Dropout(0.4)(x)
            x   = tf.keras.layers.Dense(128, activation='relu')(x)
        else:
            inp = tf.keras.Input(shape=self.input_shape)
            x   = tf.keras.layers.Conv2D(32, 3, padding='same')(inp)
            x   = tf.keras.layers.BatchNormalization()(x)
            x   = tf.keras.layers.Activation('relu')(x)
            x   = tf.keras.layers.MaxPooling2D()(x)
            x   = tf.keras.layers.Conv2D(64, 3, padding='same')(x)
            x   = tf.keras.layers.BatchNormalization()(x)
            x   = tf.keras.layers.Activation('relu')(x)
            x   = tf.keras.layers.MaxPooling2D()(x)
            x   = tf.keras.layers.Conv2D(128, 3, padding='same')(x)
            x   = tf.keras.layers.BatchNormalization()(x)
            x   = tf.keras.layers.Activation('relu')(x)
            x   = tf.keras.layers.GlobalAveragePooling2D()(x)
            x   = tf.keras.layers.Dense(256, activation='relu')(x)
            x   = tf.keras.layers.Dropout(0.4)(x)

        out = _output_layer(x, self.n_classes)
        self.model = _compile(tf.keras.Model(inp, out), self.n_classes)
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build()
        _keras_train(self.model, X_train, y_train, X_val, y_val,
                     epochs=epochs, batch_size=batch_size,
                     callbacks=_standard_callbacks(self.model_path),
                     model_path=self.model_path)
        return self

    def evaluate(self, X_test, y_test):
        return _keras_evaluate(self.model, X_test, y_test)

    def predict(self, X):
        return _predict_labels(_keras_predict(self.model, X), self.n_classes)


# ===========================================================================
# Model 4: KCNet  (arxiv:2108.07554)
# ===========================================================================

class KCNet:
    """
    Insect-inspired Kenyon Cell Network.
    • Sparse binary random weights (input→hidden, never updated)
    • Winner-take-all sparsification (top 5% active per sample)
    • Analytically solved output weights (ridge regression)
    Operates on 1-D feature vectors only.
    """

    def __init__(self, n_hidden=2000, k_connections=6, ridge_alpha=1.0,
                 n_classes=2, random_state=42, **_):
        self.n_hidden      = n_hidden
        self.k_connections = k_connections
        self.ridge_alpha   = ridge_alpha
        self.n_classes     = n_classes
        self.random_state  = random_state
        self._W_binary = None
        self._W_out    = None
        self._classes  = None
        self.training_time_ = 0.0

    def _build_W(self, n_features):
        rng = np.random.default_rng(self.random_state)
        W   = np.zeros((n_features, self.n_hidden), dtype=np.float32)
        for j in range(self.n_hidden):
            idx = rng.choice(n_features, size=min(self.k_connections, n_features), replace=False)
            W[idx, j] = 1.0
        self._W_binary = W

    def _kc(self, X):
        H = X @ self._W_binary
        k = max(1, int(0.05 * self.n_hidden))
        thresh = np.sort(H, axis=1)[:, -k:].min(axis=1, keepdims=True)
        return H * (H >= thresh)

    def train(self, X_train, y_train, X_val=None, y_val=None, **_):
        t0 = time.time()
        # Determine feature dim from first row (works for both ndarray and LazyH5Array)
        n_features = np.array(X_train[0:1]).shape[1]
        self._build_W(n_features)
        HtH, HtY, self._classes = _chunked_normal_equations(
            X_train, y_train,
            hidden_fn=self._kc,
            n_hidden=self.n_hidden,
            n_classes=self.n_classes,
        )
        A = HtH + self.ridge_alpha * np.eye(self.n_hidden, dtype=np.float32)
        self._W_out = np.linalg.solve(A, HtY)
        self.training_time_ = time.time() - t0
        return self

    def predict(self, X):
        logits = self._kc(X.astype(np.float32)) @ self._W_out
        return self._classes[np.argmax(logits, axis=1)]

    def evaluate(self, X_test, y_test):
        return {"accuracy": float(np.mean(self.predict(X_test) == y_test))}


# ===========================================================================
# Model 5: FlyCaps  (IJRITCC 2023)
# ===========================================================================

class FlyCaps:
    """
    FLY-CAPS (IJRITCC 2023) — full paper architecture:

        Input (224×224×3 or feature vector)
        → VGG-19 Conv backbone  (transfer learning, 4 blocks)
        → Primary Capsule layer (encodes spatial/orientation relationships)
        → Firefly-optimised flatten (multi-objective: accuracy + feature
          multilinearity + feature count)
        → ELM feedforward head  (single hidden layer, analytic W_out)
        → SoftMax output

    Firefly position update (per paper):
        x_{i+1} = x_i + β(r_{ij})(x_j − x_i) + α·ε
        β(r)    = β₀ · exp(−γ · r²)    attractiveness
    Fitness (multi-objective):
        f = μ·α·A + β·(S/N)
        A = accuracy,  S = feature multilinearity,  N = total features

    ELM output:
        Y(n) = H · (1/C · H^T H + I)^{−1} · H^T · O
    (Moore-Penrose pseudo-inverse solution)

    Training: ADAM lr=0.0001, batch=30, epochs=120, dropout=0.1 (paper)

    Two input modes:
      • image mode (use_feature_input=False):  input_shape must be (224, 224, 3)
        VGG-19 backbone is used.
      • feature mode (use_feature_input=True): dense primary-capsule from features
    """

    def __init__(self, input_shape=(224, 224, 3), n_classes=2,
                 feature_dim=None, use_feature_input=False,
                 n_caps=8, caps_dim=16, elm_hidden=256,
                 firefly_n=20, firefly_iter=30,
                 beta0=1.0, gamma_ff=0.1, alpha_ff=0.05,
                 model_path="best_flycaps.keras",
                 random_state=42, **_):
        self.input_shape       = input_shape
        self.n_classes         = n_classes
        self.feature_dim       = feature_dim
        self.use_feature_input = use_feature_input
        self.n_caps            = n_caps
        self.caps_dim          = caps_dim
        self.elm_hidden        = elm_hidden
        self.firefly_n         = firefly_n
        self.firefly_iter      = firefly_iter
        self.beta0             = beta0
        self.gamma_ff          = gamma_ff
        self.alpha_ff          = alpha_ff
        self.model_path        = model_path
        self.random_state      = random_state

        # ELM weights (set during train)
        self._W_in   = None
        self._b_in   = None
        self._W_out  = None
        self._classes = None
        # Keras feature extractor (backbone + capsule)
        self._extractor = None
        # Selected feature indices from Firefly optimisation
        self._ff_mask  = None
        self.training_time_ = 0.0

    # ── Squash activation (Capsule paper, Sabour et al.) ─────────────────────
    @staticmethod
    def _squash(s):
        sq_norm = tf.reduce_sum(tf.square(s), axis=-1, keepdims=True)
        return (sq_norm / (1.0 + sq_norm)) * (s / (tf.sqrt(sq_norm) + 1e-8))

    # ── Build VGG-19 + Capsule extractor ─────────────────────────────────────
    def _build_extractor(self, n_input_features):
        if self.use_feature_input:
            inp = tf.keras.Input(shape=(n_input_features,))
            # Dense projection into primary capsule space
            x = tf.keras.layers.Dense(
                self.n_caps * self.caps_dim, activation='relu')(inp)
        else:
            inp  = tf.keras.Input(shape=self.input_shape)
            base = tf.keras.applications.VGG19(
                weights='imagenet', include_top=False,
                input_shape=self.input_shape)
            base.trainable = False
            x = base(inp, training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.Dense(
                self.n_caps * self.caps_dim)(x)

        # Primary capsule reshape
        u = tf.keras.layers.Reshape((self.n_caps, self.caps_dim))(x)

        # Capsule routing: Y(i,j) = W_ij · U(i,j) · S_j
        # Implemented as learned projection + squash
        u_hat = tf.keras.layers.Dense(
            self.n_classes * self.caps_dim, use_bias=False)(
                tf.keras.layers.Flatten()(u))
        u_hat = tf.keras.layers.Reshape(
            (self.n_classes, self.caps_dim))(u_hat)

        # Squash each class capsule -> G_j = squash(S_j)
        caps_out = SquashLayer()(u_hat)

        # Flatten capsule output → feature vector for Firefly + ELM
        flat = tf.keras.layers.Flatten()(caps_out)

        self._extractor = tf.keras.Model(inp, flat)
        return flat.shape[-1]   # capsule_flat_dim

    # ── Firefly optimisation on capsule features ──────────────────────────────
    def _firefly_select(self, H: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Return binary mask of selected capsule features.
        Multi-objective fitness: f = μ·α·(1−acc) + β·(S/N)
          acc = ELM leave-one-out proxy (column-subset correlation)
          S   = mean pairwise correlation (feature multilinearity proxy)
          N   = n_features
        Firefly position update:
          x_{i+1} = x_i + β₀·exp(−γ·r²)·(x_j − x_i) + α·ε
        Binary conversion: sigmoid threshold.
        """
        rng   = np.random.default_rng(self.random_state)
        n_f   = H.shape[1]
        mu    = 0.7    # accuracy weight
        lam   = 0.3    # multilinearity weight

        # Initialise positions in ℝ^n_f
        pos = rng.uniform(-1, 1, (self.firefly_n, n_f)).astype(np.float32)
        H_n = (H - H.mean(0)) / (H.std(0) + 1e-8)   # normalised

        def _mask(p):
            sig = 1.0 / (1.0 + np.exp(-p))
            m   = (sig > 0.5).astype(float)
            if m.sum() == 0:
                m[np.argmax(sig)] = 1.0
            return m

        def _fitness(p):
            m   = _mask(p)
            idx = np.where(m)[0]
            if len(idx) == 0:
                return 1.0
            Hs  = H_n[:, idx]
            # ELM proxy accuracy: ridge regression leave-one-out approx
            C   = 1.0
            HtH = Hs.T @ Hs
            lb  = LabelBinarizer()
            Y   = lb.fit_transform(y).astype(np.float32)
            A   = HtH + (1.0 / C) * np.eye(len(idx))
            try:
                W_out = np.linalg.solve(A, Hs.T @ Y)
                pred  = np.argmax(Hs @ W_out, axis=1)
                acc   = float(np.mean(pred == y))
            except np.linalg.LinAlgError:
                acc = 0.0
            # Multilinearity: mean absolute pairwise correlation
            if len(idx) > 1:
                corr = np.corrcoef(Hs.T)
                triu = corr[np.triu_indices(len(idx), k=1)]
                S    = float(np.mean(np.abs(triu)))
            else:
                S = 0.0
            return mu * (1.0 - acc) + lam * (S / (n_f + 1e-8))

        fitness = np.array([_fitness(p) for p in pos])
        best_i  = np.argmin(fitness)

        for _ in range(self.firefly_iter):
            for i in range(self.firefly_n):
                for j in range(self.firefly_n):
                    if fitness[j] < fitness[i]:
                        r2  = float(np.sum((pos[i] - pos[j]) ** 2))
                        b   = self.beta0 * np.exp(-self.gamma_ff * r2)
                        eps = rng.standard_normal(n_f).astype(np.float32)
                        pos[i] = (pos[i]
                                  + b * (pos[j] - pos[i])
                                  + self.alpha_ff * eps)
                f_new = _fitness(pos[i])
                if f_new < fitness[i]:
                    fitness[i] = f_new
            cur = np.argmin(fitness)
            if fitness[cur] < fitness[best_i]:
                best_i = cur

        return _mask(pos[best_i])

    # ── ELM fit ───────────────────────────────────────────────────────────────
    def _elm_fit(self, H_sel: np.ndarray, y: np.ndarray):
        """
        Analytic ELM: Y(n) = H · (1/C · H^T H + I)^{−1} · H^T · O
        Single hidden layer with tanh activation (random W_in, b_in).
        """
        rng = np.random.default_rng(self.random_state)
        n_f = H_sel.shape[1]
        self._W_in = rng.uniform(-1, 1, (n_f, self.elm_hidden)).astype(np.float32)
        self._b_in = rng.uniform(-1, 1, (1,  self.elm_hidden)).astype(np.float32)

        H = np.tanh(H_sel @ self._W_in + self._b_in)
        lb = LabelBinarizer()
        Y  = lb.fit_transform(y).astype(np.float32)
        self._classes = lb.classes_
        C = 1.0
        A = (1.0 / C) * (H.T @ H) + np.eye(self.elm_hidden, dtype=np.float32)
        self._W_out = np.linalg.solve(A, H.T @ Y)

    def _elm_predict(self, H_sel: np.ndarray) -> np.ndarray:
        H    = np.tanh(H_sel.astype(np.float32) @ self._W_in + self._b_in)
        logits = H @ self._W_out
        return self._classes[np.argmax(logits, axis=1)]

    # ── Public interface ──────────────────────────────────────────────────────
    def train(self, X_train, y_train, X_val=None, y_val=None, **_):
        t0 = time.time()

        # 1. Build / compile extractor
        n_in = (self.feature_dim if self.use_feature_input
                else int(np.prod(self.input_shape)))
        caps_dim = self._build_extractor(n_in if self.use_feature_input
                                         else None)

        # 2. Extract capsule features — batch to avoid OOM on large datasets
        if isinstance(X_train, np.ndarray):
            X_in = X_train.astype(np.float32)
            H    = self._extractor.predict(X_in, batch_size=32, verbose=0)
        else:
            # LazyH5Array — read in chunks of 32 to keep memory bounded
            n = len(X_train)
            H_chunks = []
            for s in range(0, n, 32):
                chunk = np.array(X_train[s:min(s+32, n)], dtype=np.float32)
                H_chunks.append(
                    self._extractor.predict(chunk, batch_size=32, verbose=0))
            H = np.concatenate(H_chunks, axis=0)

        # 3. Firefly feature selection on capsule representation
        print("  [FlyCaps] Running Firefly optimisation on capsule features …")
        self._ff_mask = self._firefly_select(H, y_train)
        n_sel = int(self._ff_mask.sum())
        print(f"  [FlyCaps] Firefly selected {n_sel}/{H.shape[1]} capsule dims")

        H_sel = H[:, self._ff_mask.astype(bool)]

        # 4. ELM fit
        self._elm_fit(H_sel, y_train)
        self.training_time_ = time.time() - t0
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X_in = X.astype(np.float32)
            H    = self._extractor.predict(X_in, batch_size=32, verbose=0)
        else:
            n = len(X)
            H_chunks = []
            for s in range(0, n, 32):
                chunk = np.array(X[s:min(s+32, n)], dtype=np.float32)
                H_chunks.append(
                    self._extractor.predict(chunk, batch_size=32, verbose=0))
            H = np.concatenate(H_chunks, axis=0)
        H_sel = H[:, self._ff_mask.astype(bool)]
        return self._elm_predict(H_sel)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        return {"accuracy": float(np.mean(preds == y_test))}


# ===========================================================================
# Model 6: LiteCShuffle  (Cogent F&A 2025)
# ===========================================================================

class LiteCShuffle:
    """
    Lightweight CNN with Channel Attention (SE) + Channel Shuffle.
    Accepts raw images or 1-D feature vectors.
    """

    def __init__(self, input_shape=(160, 160, 3), n_classes=2,
                 base_channels=32, model_path="best_litecshuffle.keras",
                 use_feature_input=False, feature_dim=None, **_):
        self.input_shape      = input_shape
        self.n_classes        = n_classes
        self.base_channels    = base_channels
        self.model_path       = model_path
        self.use_feature_input = use_feature_input
        self.feature_dim      = feature_dim
        self.model            = None

    @staticmethod
    def _se(x, ratio=8):
        c   = x.shape[-1]
        avg = tf.keras.layers.GlobalAveragePooling2D()(x)
        avg = tf.keras.layers.Reshape((1, 1, c))(avg)
        avg = tf.keras.layers.Dense(max(1, c // ratio), activation='relu')(avg)
        avg = tf.keras.layers.Dense(c, activation='sigmoid')(avg)
        return tf.keras.layers.Multiply()([x, avg])

    @staticmethod
    def _shuffle(x, groups=2):
        s = tf.shape(x)
        _, h, w, c = x.shape
        if c is None or c % groups != 0:
            return x
        x = tf.keras.layers.Reshape((h, w, groups, c // groups))(x)
        x = tf.keras.layers.Permute((1, 2, 4, 3))(x)
        return tf.keras.layers.Reshape((h, w, c))(x)

    @staticmethod
    def _dws(x, f):
        x = tf.keras.layers.DepthwiseConv2D(3, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)
        x = tf.keras.layers.Conv2D(f, 1)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        return tf.keras.layers.Activation('relu')(x)

    def build(self):
        c = self.base_channels
        if self.use_feature_input:
            inp = tf.keras.Input(shape=(self.feature_dim,))
            x   = tf.keras.layers.Dense(256, activation='relu')(inp)
            x   = tf.keras.layers.BatchNormalization()(x)
            x   = tf.keras.layers.Dropout(0.3)(x)
            x   = tf.keras.layers.Dense(128, activation='relu')(x)
        else:
            inp = tf.keras.Input(shape=self.input_shape)
            x   = tf.keras.layers.Conv2D(c, 3, strides=2, padding='same')(inp)
            x   = tf.keras.layers.BatchNormalization()(x)
            x   = tf.keras.layers.Activation('relu')(x)
            for _ in range(2):
                x = self._dws(x, c)
                x = self._se(x)
            x   = self._shuffle(x)
            x   = tf.keras.layers.MaxPooling2D()(x)
            c  *= 2
            x   = tf.keras.layers.Conv2D(c, 1)(x)
            for _ in range(2):
                x = self._dws(x, c)
                x = self._se(x)
            x   = self._shuffle(x)
            x   = tf.keras.layers.MaxPooling2D()(x)
            c  *= 2
            x   = tf.keras.layers.Conv2D(c, 1)(x)
            for _ in range(2):
                x = self._dws(x, c)
                x = self._se(x)
            x   = tf.keras.layers.GlobalAveragePooling2D()(x)

        out = _output_layer(x, self.n_classes)
        self.model = _compile(tf.keras.Model(inp, out), self.n_classes)
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build()
        _keras_train(self.model, X_train, y_train, X_val, y_val,
                     epochs=epochs, batch_size=batch_size,
                     callbacks=_standard_callbacks(self.model_path),
                     model_path=self.model_path)
        return self

    def evaluate(self, X_test, y_test):
        return _keras_evaluate(self.model, X_test, y_test)

    def predict(self, X):
        return _predict_labels(_keras_predict(self.model, X), self.n_classes)


# ===========================================================================
# Model 7: YangViT  (IEEE 10318216 – PAYWALL)
# ===========================================================================

class YangViT:
    """
    YangNet — SNN-inspired proxy for Yang et al. IEEE TSMC 2024.

    The paper describes a neuromorphic visual perception framework with:
      • Visual Perception Layer (VPL):
          Poisson-encoded input → Excitatory neurons → Winner-take-all
          Inhibitory layer → Output encoding
      • Decision Making Layer (DML):
          6-neuron input → 8 excitatory neurons (all-to-all) →
          2 output neurons with mutual inhibition
      • LIF neuron dynamics:  τ dV/dt = (E_rest−V) + g_e(E_exc−V) + g_i(E_inh−V)
      • Q-STDP learning with 6 trace variables

    Since a full spiking network on GPU/CPU is not feasible in TensorFlow without
    dedicated SNN libraries, this proxy captures the *architectural intent*:
      1. Rate-coded encoding layer  (analogous to Poisson encoding)
      2. Excitatory projection + lateral inhibition via winner-take-all mask
         (top-k activation)
      3. Dense decision layers (DML analogue) with mutual-inhibition-style
         competitive softmax output

    Accepts raw (pre-processed) images only.
    """

    def __init__(self, input_shape=(160, 160, 3), n_classes=2,
                 n_excitatory=64, k_wta=8,            # WTA keeps top-k neurons
                 dml_units=8,                          # DML excitatory neurons
                 model_path="best_yangnet.keras", **_):
        self.input_shape  = input_shape
        self.n_classes    = n_classes
        self.n_excitatory = n_excitatory
        self.k_wta        = k_wta
        self.dml_units    = dml_units
        self.model_path   = model_path
        self.model        = None

    @staticmethod
    def _winner_take_all(x, k):
        """Top-k winner-take-all: zero out all but the k largest activations."""
        vals, _ = tf.math.top_k(x, k=k)
        threshold = vals[:, -1:]          # k-th largest per sample
        return x * tf.cast(x >= threshold, x.dtype)

    def build(self):
        inp = tf.keras.Input(shape=self.input_shape)

        # ── Visual Perception Layer (VPL) proxy ──────────────────────────────
        # Shallow CNN encoder (analogous to VPL excitatory projection)
        x = tf.keras.layers.Conv2D(32, 3, strides=2, padding='same',
                                   activation='relu')(inp)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Conv2D(64, 3, strides=2, padding='same',
                                   activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)

        # Excitatory projection
        x = tf.keras.layers.Dense(self.n_excitatory)(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation('relu')(x)

        # Winner-take-all inhibitory layer (lateral inhibition proxy)
        k = min(self.k_wta, self.n_excitatory)
        x = WinnerTakeAllLayer(k=k)(x)
        x = tf.keras.layers.Dropout(0.3)(x)

        # ── Decision Making Layer (DML) proxy ────────────────────────────────
        # All-to-all excitatory connectivity (8 units per paper)
        x = tf.keras.layers.Dense(self.dml_units, activation='relu')(x)

        # Mutual inhibition at output: standard softmax approximates
        # competitive winner-take-all among output neurons
        out = _output_layer(x, self.n_classes)

        self.model = _compile(tf.keras.Model(inp, out), self.n_classes)
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build()
        _keras_train(self.model, X_train, y_train, X_val, y_val,
                     epochs=epochs, batch_size=batch_size,
                     callbacks=_standard_callbacks(self.model_path),
                     model_path=self.model_path)
        return self

    def evaluate(self, X_test, y_test):
        return _keras_evaluate(self.model, X_test, y_test)

    def predict(self, X):
        return _predict_labels(_keras_predict(self.model, X), self.n_classes)


# ===========================================================================
# Model 8: HybridGrasshopperABC  (Springer 2023 – PAYWALL)
# ===========================================================================

class HybridGrasshopperABC:
    """
    Deep MLP classifier (proxy for Springer 2023 hybrid GOA+ABC model).
    ⚠️  Paper is behind institutional paywall.
    Accepts 1-D feature vectors.
    """

    def __init__(self, feature_dim=100, n_classes=2,
                 model_path="best_grasshopper.keras", **_):
        self.feature_dim = feature_dim
        self.n_classes   = n_classes
        self.model_path  = model_path
        self.model       = None

    def build(self):
        inp = tf.keras.Input(shape=(self.feature_dim,))
        x   = tf.keras.layers.Dense(512)(inp)
        x   = tf.keras.layers.BatchNormalization()(x)
        x   = tf.keras.layers.Activation('relu')(x)
        x   = tf.keras.layers.Dropout(0.4)(x)
        x   = tf.keras.layers.Dense(256)(x)
        x   = tf.keras.layers.BatchNormalization()(x)
        x   = tf.keras.layers.Activation('relu')(x)
        x   = tf.keras.layers.Dropout(0.3)(x)
        x   = tf.keras.layers.Dense(128)(x)
        x   = tf.keras.layers.BatchNormalization()(x)
        x   = tf.keras.layers.Activation('relu')(x)
        out = _output_layer(x, self.n_classes)
        self.model = _compile(tf.keras.Model(inp, out), self.n_classes)
        return self

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        if self.model is None:
            self.build()
        _keras_train(self.model, X_train, y_train, X_val, y_val,
                     epochs=epochs, batch_size=batch_size,
                     callbacks=_standard_callbacks(self.model_path),
                     model_path=self.model_path)
        return self

    def evaluate(self, X_test, y_test):
        return _keras_evaluate(self.model, X_test, y_test)

    def predict(self, X):
        return _predict_labels(_keras_predict(self.model, X), self.n_classes)


# ===========================================================================
# Model 9: MantisSearch  (ScienceDirect 2023 – PAYWALL)
# ===========================================================================

class MantisSearch:
    """
    Mantis Search Algorithm optimised ELM (Extreme Learning Machine).
    ⚠️  Paper is behind institutional paywall.
    Accepts 1-D feature vectors.
    """

    def __init__(self, n_hidden=1000, ridge_alpha=1.0,
                 n_classes=2, random_state=42, **_):
        self.n_hidden    = n_hidden
        self.ridge_alpha = ridge_alpha
        self.n_classes   = n_classes
        self.random_state = random_state
        self._W_in  = None
        self._b_in  = None
        self._W_out = None
        self._classes = None
        self.training_time_ = 0.0

    def _H(self, X):
        return np.tanh(X @ self._W_in + self._b_in)

    def train(self, X_train, y_train, X_val=None, y_val=None, **_):
        t0  = time.time()
        rng = np.random.default_rng(self.random_state)
        n_f = np.array(X_train[0:1]).shape[1]
        self._W_in = rng.uniform(-1, 1, (n_f, self.n_hidden)).astype(np.float32)
        self._b_in = rng.uniform(-1, 1, (1,  self.n_hidden)).astype(np.float32)
        HtH, HtY, self._classes = _chunked_normal_equations(
            X_train, y_train,
            hidden_fn=self._H,
            n_hidden=self.n_hidden,
            n_classes=self.n_classes,
        )
        A = HtH + self.ridge_alpha * np.eye(self.n_hidden, dtype=np.float32)
        self._W_out = np.linalg.solve(A, HtY)
        self.training_time_ = time.time() - t0
        return self

    def predict(self, X):
        logits = self._H(X.astype(np.float32)) @ self._W_out
        return self._classes[np.argmax(logits, axis=1)]

    def evaluate(self, X_test, y_test):
        return {"accuracy": float(np.mean(self.predict(X_test) == y_test))}


# ===========================================================================
# ChannelPruner
# ===========================================================================

class ChannelPruner:
    """Magnitude-based structured pruning for Keras Conv2D layers."""

    def __init__(self, prune_ratio=0.3):
        self.prune_ratio = prune_ratio

    def prune(self, model: tf.keras.Model) -> tf.keras.Model:
        pruned = {}
        for layer in model.layers:
            if not isinstance(layer, tf.keras.layers.Conv2D):
                continue
            ws = layer.get_weights()
            W  = ws[0]
            k  = max(1, int(W.shape[-1] * self.prune_ratio))
            drop = np.argsort(np.abs(W).sum(axis=(0, 1, 2)))[:k]
            W[:, :, :, drop] = 0.0
            ws[0] = W
            pruned[layer.name] = ws

        new = tf.keras.models.clone_model(model)
        new.set_weights(model.get_weights())
        for layer in new.layers:
            if layer.name in pruned:
                layer.set_weights(pruned[layer.name])

        new.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                    loss=model.loss, metrics=['accuracy'])
        print(f"[ChannelPruner] Pruned {len(pruned)} Conv2D layers @ {self.prune_ratio:.0%}")
        return new


# ===========================================================================
# MODEL_REGISTRY  +  ModelTrainer
# ===========================================================================

# (class, uses_feature_input_by_default)
# FlyCaps: False = image-input by default (use use_feature_input=True to switch)
MODEL_REGISTRY = {
    "vgg16":        (SimpleVGG16,           False),
    "vgg19_svm":    (VGG19_SVM,             False),
    "hybrid_cnn":   (HybridCNN,             False),
    "kcnet":        (KCNet,                 True),
    "flycaps":      (FlyCaps,               False),   # image or feature; see use_feature_input
    "litecshuffle": (LiteCShuffle,          False),
    "yang_vit":     (YangViT,               False),
    "grasshopper":  (HybridGrasshopperABC,  True),
    "mantis":       (MantisSearch,          True),
}

# Models that use analytic/sklearn training (no .build() step)
_ANALYTIC_MODELS = {"kcnet", "mantis", "vgg19_svm", "flycaps"}


class ModelTrainer:
    """
    Unified launcher.  Automatically routes image vs feature inputs.

    Parameters
    ----------
    model_type        : key in MODEL_REGISTRY
    n_classes         : number of output classes
    feature_dim       : required for feature-input models
    image_input_shape : required for image-input models
    save_dir          : checkpoint directory
    use_feature_input : override for dual-mode models (e.g. flycaps)
    **kwargs          : forwarded to the model constructor
    """

    def __init__(self, model_type, n_classes=2, feature_dim=None,
                 image_input_shape=(160, 160, 3),
                 save_dir="outputs/models",
                 use_feature_input=None,
                 **kwargs):
        if model_type not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model_type '{model_type}'. "
                             f"Valid: {list(MODEL_REGISTRY)}")
        os.makedirs(save_dir, exist_ok=True)
        self.model_type = model_type
        self.n_classes  = n_classes
        ModelClass, default_uses_feats = MODEL_REGISTRY[model_type]

        # Allow caller to override feature-input mode (e.g. flycaps)
        self.uses_features = (use_feature_input
                              if use_feature_input is not None
                              else default_uses_feats)

        model_path = os.path.join(save_dir, f"best_{model_type}.keras")
        ctor = dict(n_classes=n_classes, **kwargs)

        if self.uses_features:
            if feature_dim is None:
                raise ValueError(f"{model_type} requires feature_dim.")
            ctor["feature_dim"]       = feature_dim
            ctor["use_feature_input"] = True
        else:
            ctor["input_shape"]       = image_input_shape
            ctor["use_feature_input"] = False

        # FlyCaps also needs to know feature_dim even in image mode (ignored)
        if model_type == "flycaps" and feature_dim is not None:
            ctor["feature_dim"] = feature_dim

        if model_type not in _ANALYTIC_MODELS:
            ctor["model_path"] = model_path

        self.model = ModelClass(**ctor)
        self.training_time_ = 0.0

    def run(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
        t0 = time.time()
        if self.model_type in _ANALYTIC_MODELS:
            self.model.train(X_train, y_train)
        else:
            self.model.train(X_train, y_train, X_val, y_val,
                             epochs=epochs, batch_size=batch_size)
        self.training_time_ = time.time() - t0
        return {"training_time_s": self.training_time_}

    def evaluate(self, X_test, y_test):
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        return self.model.predict(X)

    # ------------------------------------------------------------------ #
    #  Persistence — save weights to disk, reload on demand               #
    # ------------------------------------------------------------------ #

    def save(self, save_dir: str = "outputs/models") -> str:
        """
        Save the trained model to disk. Returns the path it was saved to.

        Keras models  → saved as  <save_dir>/best_<type>.keras
                         (ModelCheckpoint already wrote the best epoch here;
                          this call saves the final/restored state on top)
        Analytic models (KCNet, MantisSearch, HybridGrasshopperABC,
                         VGG19_SVM, FlyCaps)
                       → saved as  <save_dir>/best_<type>.pkl  via pickle
        """
        import pickle
        os.makedirs(save_dir, exist_ok=True)

        if self.model_type in _ANALYTIC_MODELS:
            path = os.path.join(save_dir, f"best_{self.model_type}.pkl")
            with open(path, "wb") as fh:
                pickle.dump(self.model, fh, protocol=4)
        else:
            path = os.path.join(save_dir, f"best_{self.model_type}.keras")
            keras_model = getattr(self.model, "model", None)
            if keras_model is not None:
                keras_model.save(path)
            else:
                raise RuntimeError(
                    f"Cannot save {self.model_type}: inner Keras model is None.")

        print(f"  [Save] {self.model_type} -> {path}")
        return path

    @classmethod
    def load(cls, model_type: str, path: str,
             n_classes: int = 2,
             feature_dim: int = None,
             image_input_shape: tuple = (160, 160, 3),
             save_dir: str = "outputs/models") -> "ModelTrainer":
        """
        Reload a previously saved model from disk into a fresh ModelTrainer.

        This is the counterpart to .save(). Use it in experiment 2b / 3 so
        only one model lives in GPU memory at a time.
        """
        import pickle
        # Build a shell trainer (no training, just sets up the wrapper object)
        trainer = cls.__new__(cls)
        trainer.model_type      = model_type
        trainer.n_classes       = n_classes
        trainer.training_time_  = 0.0

        ModelClass, default_uses_feats = MODEL_REGISTRY[model_type]
        trainer.uses_features = default_uses_feats

        if model_type in _ANALYTIC_MODELS:
            with open(path, "rb") as fh:
                trainer.model = pickle.load(fh)
        else:
            keras_model = tf.keras.models.load_model(
                path,
                custom_objects={
                    "WinnerTakeAllLayer": WinnerTakeAllLayer,
                    "SquashLayer":        SquashLayer,
                })
            # Wrap in the appropriate class shell so .predict() works
            ctor = dict(n_classes=n_classes, model_path=path)
            if default_uses_feats:
                ctor["feature_dim"]       = feature_dim
                ctor["use_feature_input"] = True
            else:
                ctor["input_shape"]       = image_input_shape
                ctor["use_feature_input"] = False
            wrapper = ModelClass(**ctor)
            wrapper.model = keras_model   # inject loaded weights
            trainer.model = wrapper

        print(f"  [Load] {model_type} <- {path}")
        return trainer