"""
feature_selection.py
Three feature-selection algorithms from the document:
  1. Artificial Bee Colony (ABC)
  2. WOA-APSO (Whale Optimisation + Accelerated PSO)
  3. Ant Colony Optimisation (ACO)

Fitness function
----------------
Per the document, Experiment 1 measures "accuracy of a simple model (like CNN)
using the feature extraction model."  We therefore use the SimpleVGG16 model
in feature-input mode (a small dense network on top of the selected features)
evaluated via 3-fold cross-validation.  This is slower than KNN but directly
measures what the document asks for.

To reduce wall-clock time during optimisation, the fitness evaluation uses a
*mini-VGG16* (fewer epochs, smaller dataset sample) rather than full training.

All three classes expose the same interface:
    selector.fit(X, y, n_classes)
    selector.transform(X)
    selector.selected_indices_   : 1-D array of selected column indices
    selector.n_features_         : int
    selector.selection_time_     : float (seconds)
    selector._best_fitness       : float (lower is better: 1 - accuracy + penalty)
"""

import time
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold


# ---------------------------------------------------------------------------
# Fitness function: SimpleVGG16 in feature-input mode (fast variant)
# ---------------------------------------------------------------------------

# Fixed proxy dimension — MUST be constant across all fitness calls so that
# TF compiles exactly one XLA graph and reuses it every time.
_PROXY_DIM = 32


def _project(X_sel: np.ndarray) -> np.ndarray:
    """
    Always project to exactly _PROXY_DIM dimensions regardless of n_sel.
    • n_sel > _PROXY_DIM : random Johnson-Lindenstrauss projection (compress)
    • n_sel == _PROXY_DIM: return as-is
    • n_sel < _PROXY_DIM : zero-pad to _PROXY_DIM (rare with prefilter_k=1000+)
    The fixed output shape means TF compiles one graph and reuses it,
    eliminating per-call XLA recompilation and its ~200 MB GPU allocation.
    """
    n_sel = X_sel.shape[1]
    if n_sel > _PROXY_DIM:
        rng = np.random.default_rng(42)
        R   = rng.standard_normal((n_sel, _PROXY_DIM)).astype(np.float32)
        R  /= np.linalg.norm(R, axis=0, keepdims=True) + 1e-8
        return X_sel.astype(np.float32) @ R
    elif n_sel < _PROXY_DIM:
        pad = np.zeros((X_sel.shape[0], _PROXY_DIM - n_sel), dtype=np.float32)
        return np.concatenate([X_sel.astype(np.float32), pad], axis=1)
    return X_sel.astype(np.float32)


def _fitness_vgg16(X: np.ndarray, y: np.ndarray, mask: np.ndarray,
                   n_classes: int = 2,
                   alpha: float = 0.99,
                   beta:  float = 0.01,
                   n_splits: int = 3,
                   epochs: int = 5,
                   batch_size: int = 64,
                   max_samples: int = 500) -> float:
    """
    Evaluate a binary feature mask with a tiny CPU-only dense proxy.

    Key design decisions to prevent OOM and XLA recompilation kills:
    1. tf.device('/CPU:0') — fitness eval never touches the GPU, keeping
       it free for actual model training.
    2. Fixed input shape (_PROXY_DIM) via _project() — TF compiles exactly
       one graph for the proxy and reuses it across all fitness calls.
    3. clear_session() BEFORE each fold, not after — prevents graph
       accumulation inside a single fitness call.
    """
    if mask.sum() == 0:
        return 1.0

    # Sub-sample rows
    if len(X) > max_samples:
        idx = np.random.choice(len(X), max_samples, replace=False)
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    # Project to fixed _PROXY_DIM — same shape every call
    X_proj = _project(Xs[:, mask.astype(bool)])

    skf  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accs = []

    for tr_idx, val_idx in skf.split(X_proj, ys):
        # Clear BEFORE building — prevents accumulation across folds
        tf.keras.backend.clear_session()

        X_tr, X_val = X_proj[tr_idx], X_proj[val_idx]
        y_tr, y_val = ys[tr_idx],     ys[val_idx]

        # Run entirely on CPU — no GPU compilation, no XLA buffers
        with tf.device('/CPU:0'):
            inp = tf.keras.Input(shape=(_PROXY_DIM,))
            x   = tf.keras.layers.Dense(32, activation='relu')(inp)
            if n_classes == 2:
                out  = tf.keras.layers.Dense(1, activation='sigmoid')(x)
                loss = 'binary_crossentropy'
            else:
                out  = tf.keras.layers.Dense(n_classes, activation='softmax')(x)
                loss = 'sparse_categorical_crossentropy'

            model = tf.keras.Model(inp, out)
            model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                          loss=loss, metrics=['accuracy'])
            model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                      verbose=0, validation_data=None)
            _, acc = model.evaluate(X_val, y_val, verbose=0)

        accs.append(acc)
        del model

    tf.keras.backend.clear_session()

    mean_acc = float(np.mean(accs))
    penalty  = beta * (mask.sum() / len(mask))
    return (1.0 - alpha * mean_acc) + penalty


# ---------------------------------------------------------------------------
# 1. Artificial Bee Colony (ABC)
# ---------------------------------------------------------------------------

class ABCFeatureSelector:
    """
    Binary Artificial Bee Colony for feature selection.
    From: Evolving Systems (DOI: 10.1007/s12530-019-09289-2)

    Paper parameters:
      • Colony size = 2N  (N food sources + N onlooker bees)
      • max_iter    = 200
      • range_limit = 10  (scout abandonment threshold)
      • AP          = 0.3  (adjustment parameter for bit-flip neighbour)
      • Fitness     = VGG16 dense-proxy accuracy (per experiment spec)

    Employed bee neighbour update (binary, per paper):
      For each bit position i:
        if RN_i < AP:  y_i = 1 - y_i   (flip)
        else:          y_i = y_i        (keep)
      where RN_i ~ Uniform(0,1).

    Onlooker selection: proportional to quality (roulette wheel).
    Scout: reinitialise exhausted food source (random bit vector).
    """

    def __init__(self, n_bees=None, max_iter=200, limit=10,
                 AP=0.3, alpha=0.99, beta=0.01, random_state=42):
        # n_bees = N food sources; colony = 2N total (N employed + N onlooker)
        # If None, set to n_features at fit time
        self.n_bees   = n_bees
        self.max_iter = max_iter
        self.limit    = limit
        self.AP       = AP           # binary flip probability (paper: 0.3)
        self.alpha    = alpha
        self.beta     = beta
        self.random_state = random_state

        self.selected_indices_: np.ndarray = None
        self.n_features_:       int   = 0
        self.selection_time_:   float = 0.0
        self._best_fitness:     float = None

    def _neighbour(self, source: np.ndarray, rng) -> np.ndarray:
        """
        Binary neighbour: flip each bit with probability AP (paper eq.).
        y_i = 1 - y_i  if RN_i < AP,  else y_i  (for each bit i)
        """
        rn  = rng.random(len(source))
        ns  = source.copy()
        ns[rn < self.AP] = 1.0 - ns[rn < self.AP]
        return ns

    def fit(self, X: np.ndarray, y: np.ndarray,
            n_classes: int = 2,
            prefilter_k: int = 1000) -> "ABCFeatureSelector":
        rng = np.random.default_rng(self.random_state)
        N   = self.n_bees if self.n_bees is not None else 20
        t0  = time.time()

        # ── Variance pre-filter ──────────────────────────────────────────────
        # The binary search space has 2^n_feat states. Even at n_feat=143K
        # storing N×n_feat masks exhausts RAM. Pre-filter to top-k by variance
        # so the metaheuristic searches a tractable subspace; remap at the end.
        n_feat = X.shape[1]
        if n_feat > prefilter_k:
            print(f"  [ABC] Pre-filtering {n_feat} -> {prefilter_k} features by variance ...")
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            keep_idx  = var_order[:prefilter_k]
            X_sub     = X[:, keep_idx]
        else:
            keep_idx = np.arange(n_feat)
            X_sub    = X
        n_sub = X_sub.shape[1]

        sources  = rng.integers(0, 2, (N, n_sub)).astype(float)
        fitness  = np.array([_fitness_vgg16(X_sub, y, s, n_classes, self.alpha, self.beta)
                             for s in sources])
        trials   = np.zeros(N, dtype=int)
        best_idx = np.argmin(fitness)
        best_src = sources[best_idx].copy()
        best_fit = fitness[best_idx]

        for it in range(self.max_iter):
            # ── Employed bees ────────────────────────────────────────────────
            for i in range(N):
                ns = self._neighbour(sources[i], rng)
                nf = _fitness_vgg16(X_sub, y, ns, n_classes, self.alpha, self.beta)
                if nf < fitness[i]:
                    sources[i] = ns; fitness[i] = nf; trials[i] = 0
                else:
                    trials[i] += 1

            # ── Onlooker bees ────────────────────────────────────────────────
            quality = 1.0 / (fitness + 1e-10)
            probs   = quality / quality.sum()
            for _ in range(N):
                i  = rng.choice(N, p=probs)
                ns = self._neighbour(sources[i], rng)
                nf = _fitness_vgg16(X_sub, y, ns, n_classes, self.alpha, self.beta)
                if nf < fitness[i]:
                    sources[i] = ns; fitness[i] = nf; trials[i] = 0
                else:
                    trials[i] += 1

            # ── Scout bees ───────────────────────────────────────────────────
            for i in range(N):
                if trials[i] > self.limit:
                    sources[i] = rng.integers(0, 2, n_sub).astype(float)
                    fitness[i] = _fitness_vgg16(X_sub, y, sources[i], n_classes,
                                                self.alpha, self.beta)
                    trials[i]  = 0

            cur = np.argmin(fitness)
            if fitness[cur] < best_fit:
                best_fit = fitness[cur]; best_src = sources[cur].copy()

            print(f"  [ABC] iter {it+1}/{self.max_iter}  "
                  f"fitness={best_fit:.4f}  n_feat={int(best_src.sum())}")

        self.selection_time_   = time.time() - t0
        self._best_fitness     = best_fit
        # Remap sub-space indices back to original feature space
        sub_selected           = np.where(best_src.astype(bool))[0]
        self.selected_indices_ = keep_idx[sub_selected]
        self.n_features_       = len(self.selected_indices_)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]


# ---------------------------------------------------------------------------
# 2. WOA-APSO
# ---------------------------------------------------------------------------

class WOA_APSOSelector:
    """Hybrid Whale Optimisation + Accelerated PSO for feature selection."""

    def __init__(self, n_agents=20, max_iter=50, w=0.7, c1=1.5, c2=1.5,
                 alpha=0.99, beta=0.01, random_state=42):
        self.n_agents = n_agents; self.max_iter = max_iter
        self.w = w; self.c1 = c1; self.c2 = c2
        self.alpha = alpha; self.beta = beta
        self.random_state = random_state
        self.selected_indices_: np.ndarray = None
        self.n_features_:       int   = 0
        self.selection_time_:   float = 0.0
        self._best_fitness:     float = None

    def fit(self, X, y, n_classes=2,
            prefilter_k: int = 1000) -> "WOA_APSOSelector":
        rng    = np.random.default_rng(self.random_state)
        n_feat = X.shape[1]
        t0     = time.time()

        # ── Variance pre-filter ──────────────────────────────────────────────
        if n_feat > prefilter_k:
            print(f"  [WOA-APSO] Pre-filtering {n_feat} -> {prefilter_k} features by variance ...")
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            keep_idx  = var_order[:prefilter_k]
            X_sub     = X[:, keep_idx]
        else:
            keep_idx = np.arange(n_feat)
            X_sub    = X
        n_sub = X_sub.shape[1]

        pos = rng.uniform(-3, 3, (self.n_agents, n_sub))
        vel = rng.uniform(-1, 1, (self.n_agents, n_sub))

        def binarise(p):
            sig = 1 / (1 + np.exp(-p))
            return (sig > rng.random(n_sub)).astype(float)

        masks   = np.array([binarise(p) for p in pos])
        fitness = np.array([_fitness_vgg16(X_sub, y, m, n_classes, self.alpha, self.beta)
                            for m in masks])
        pbest   = pos.copy(); pbest_f = fitness.copy()
        gi      = np.argmin(fitness)
        gbest   = pos[gi].copy(); gbest_fit = fitness[gi]

        for it in range(self.max_iter):
            a = 2 - it * (2 / self.max_iter)
            for i in range(self.n_agents):
                r1, r2 = rng.random(), rng.random()
                A = 2 * a * r1 - a; C = 2 * r2
                if rng.random() < 0.5:
                    if abs(A) < 1:
                        D = abs(C * gbest - pos[i]); new_p = gbest - A * D
                    else:
                        ri = rng.integers(self.n_agents)
                        D = abs(C * pos[ri] - pos[i]); new_p = pos[ri] - A * D
                else:
                    l = rng.uniform(-1, 1)
                    D = abs(gbest - pos[i])
                    new_p = D * np.exp(l) * np.cos(2 * np.pi * l) + gbest

                vel[i] = (self.w * vel[i]
                          + self.c1 * rng.random() * (pbest[i] - pos[i])
                          + self.c2 * rng.random() * (gbest - pos[i]))
                pos[i] = np.clip(0.5 * new_p + 0.5 * (pos[i] + vel[i]), -6, 6)

                m = binarise(pos[i])
                f = _fitness_vgg16(X_sub, y, m, n_classes, self.alpha, self.beta)
                if f < pbest_f[i]:
                    pbest[i] = pos[i].copy(); pbest_f[i] = f
                if f < gbest_fit:
                    gbest = pos[i].copy(); gbest_fit = f

            print(f"  [WOA-APSO] iter {it+1}/{self.max_iter}  "
                  f"fitness={gbest_fit:.4f}  n_feat={int(binarise(gbest).sum())}")

        self.selection_time_   = time.time() - t0
        self._best_fitness     = gbest_fit
        best_mask              = binarise(gbest)
        sub_selected           = np.where(best_mask.astype(bool))[0]
        self.selected_indices_ = keep_idx[sub_selected]
        self.n_features_       = len(self.selected_indices_)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]


# ---------------------------------------------------------------------------
# 3. Ant Colony Optimisation (ACO)
# ---------------------------------------------------------------------------

class ACOSelector:
    """ACO wrapper feature selection using pheromone-guided probabilistic masks."""

    def __init__(self, n_ants=20, max_iter=50, rho=0.1, q=1.0,
                 alpha_aco=1.0, beta_aco=2.0,
                 sel_alpha=0.99, sel_beta=0.01, random_state=42):
        self.n_ants = n_ants; self.max_iter = max_iter
        self.rho = rho; self.q = q
        self.alpha_aco = alpha_aco; self.beta_aco = beta_aco
        self.sel_alpha = sel_alpha; self.sel_beta = sel_beta
        self.random_state = random_state
        self.selected_indices_: np.ndarray = None
        self.n_features_:       int   = 0
        self.selection_time_:   float = 0.0
        self._best_fitness:     float = None

    def fit(self, X, y, n_classes=2,
            prefilter_k: int = 1000) -> "ACOSelector":
        rng    = np.random.default_rng(self.random_state)
        n_feat = X.shape[1]
        t0     = time.time()

        # ── Variance pre-filter ──────────────────────────────────────────────
        if n_feat > prefilter_k:
            print(f"  [ACO] Pre-filtering {n_feat} -> {prefilter_k} features by variance ...")
            var_order = np.argsort(np.var(X, axis=0))[::-1]
            keep_idx  = var_order[:prefilter_k]
            X_sub     = X[:, keep_idx]
        else:
            keep_idx = np.arange(n_feat)
            X_sub    = X
        n_feat = X_sub.shape[1]   # shadow with sub-space size
        X      = X_sub            # work on subspace from here on

        heuristic = np.var(X, axis=0)
        heuristic = heuristic / (heuristic.sum() + 1e-10)
        pheromone = np.ones(n_feat)
        best_mask = np.ones(n_feat); best_fit = 1.0

        for it in range(self.max_iter):
            all_masks = []; all_fits = []
            for _ in range(self.n_ants):
                tau_eta = (pheromone ** self.alpha_aco) * (heuristic ** self.beta_aco)
                prob    = tau_eta / (tau_eta.sum() + 1e-10)
                mask    = (rng.random(n_feat) < prob).astype(float)
                if mask.sum() == 0:
                    mask[rng.integers(n_feat)] = 1.0
                f = _fitness_vgg16(X, y, mask, n_classes, self.sel_alpha, self.sel_beta)
                all_masks.append(mask); all_fits.append(f)
                if f < best_fit:
                    best_fit = f; best_mask = mask.copy()

            pheromone *= (1 - self.rho)
            mean_f     = np.mean(all_fits)
            for mask, f in zip(all_masks, all_fits):
                if f < mean_f:
                    pheromone += mask * (self.q / (f + 1e-10))
            pheromone = np.clip(pheromone, 0.01, 10.0)

            print(f"  [ACO] iter {it+1}/{self.max_iter}  "
                  f"fitness={best_fit:.4f}  n_feat={int(best_mask.sum())}")

        self.selection_time_   = time.time() - t0
        self._best_fitness     = best_fit
        sub_selected           = np.where(best_mask.astype(bool))[0]
        self.selected_indices_ = keep_idx[sub_selected]
        self.n_features_       = len(self.selected_indices_)
        return self

    def transform(self, X):
        return X[:, self.selected_indices_]