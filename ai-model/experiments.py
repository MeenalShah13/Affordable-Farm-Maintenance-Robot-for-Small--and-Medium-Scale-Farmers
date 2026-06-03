"""
experiments.py
Orchestrates Experiments 1, 2a, 2b, and 3.

Key design decisions (per user spec):
  • Experiment 1: use SimpleVGG16 (feature-input mode) to score each selector.
  • Experiments 2a/2b/3: feature-input models receive PREPROCESSED →
    FEATURE-EXTRACTED → FEATURE-SELECTED 1-D vectors.
    Image-input models (YangNet/YangViT, LiteCShuffle, HybridCNN image mode,
    VGG16, VGG19+SVM, FlyCaps image mode) receive images that have been passed
    through the full PreprocessingPipeline (resize → cellular automaton →
    histogram equalisation → Wiener filter → background removal).
    Raw/unprocessed images are NEVER fed to any model.
  • The best feature selector from Exp 1 is applied to all subsequent experiments.

Usage
-----
    python experiments.py --dataset_root ./dataset --use_subset
    python experiments.py --dataset_root ./dataset          # full dataset
    python experiments.py --dataset_root ./dataset --use_subset --skip_exp3
"""

import argparse
import os
import numpy as np

from data_loader        import DatasetLoader
from preprocessing      import PreprocessingPipeline
from feature_extraction import FeatureExtractor
from feature_selection  import ABCFeatureSelector, WOA_APSOSelector, ACOSelector
from model_training     import (
    SimpleVGG16, VGG19_SVM, HybridCNN, KCNet, FlyCaps,
    LiteCShuffle, YangViT, HybridGrasshopperABC, MantisSearch,
    ChannelPruner, ModelTrainer, MODEL_REGISTRY
)
from metrics import FeatureSelectionMetrics, ModelMetrics, ResultsLogger
from cache   import DataCache
import time

# ---------------------------------------------------------------------------
# Pipeline-wide timing utility
# ---------------------------------------------------------------------------

class Timer:
    """
    Context manager that measures wall-clock time for any named step and
    records it in a shared dict so a summary can be printed at the end.

    Usage:
        timings = {}
        with Timer("Feature extraction", timings):
            ...  # code to time

        Timer.print_summary(timings)
    """

    def __init__(self, name: str, timings: dict):
        self.name    = name
        self.timings = timings

    def __enter__(self):
        self._start = time.time()
        print(f"[Timer] {self.name} started ...", flush=True)
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._start
        self.timings[self.name] = elapsed
        if elapsed < 120:
            fmt = f"{elapsed:.1f}s"
        elif elapsed < 7200:
            fmt = f"{elapsed/60:.1f}m"
        else:
            fmt = f"{elapsed/3600:.2f}h"
        print(f"[Timer] {self.name} done -- {fmt}", flush=True)

    @staticmethod
    def fmt_secs(secs):
        if secs < 120:
            return f"{secs:.1f}s"
        elif secs < 7200:
            return f"{secs/60:.1f}m"
        return f"{secs/3600:.2f}h"

    @staticmethod
    def print_summary(timings: dict):
        if not timings:
            return
        total = sum(timings.values())
        print("")
        print("=" * 64)
        print("PIPELINE TIMING SUMMARY")
        print("=" * 64)
        max_name = max(len(k) for k in timings)
        for name, secs in timings.items():
            pct = 100 * secs / total if total > 0 else 0
            bar = chr(9608) * int(pct / 2.5)
            print(f"  {name:<{max_name}}  {Timer.fmt_secs(secs):>8}  {pct:5.1f}%  {bar}")
        print("-" * 64)
        print(f"  {'TOTAL':<{max_name}}  {Timer.fmt_secs(total):>8}")
        print("=" * 64)




# ---------------------------------------------------------------------------
# Helper – build feature matrix (preprocess + extract)
# ---------------------------------------------------------------------------

def build_feature_matrix(paths, preprocessor, extractor, loader):
    features = []
    for i, path in enumerate(paths):
        img = loader.load_image(path)
        if img is None:
            continue
        img_pre = preprocessor(img)
        feat    = extractor.extract(img_pre)
        features.append(feat)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(paths)} images processed …")
    return np.array(features)


def build_image_array(paths, preprocessor, loader):
    imgs = []
    for path in paths:
        img = loader.load_image(path)
        if img is not None:
            imgs.append(preprocessor(img))
    return np.array(imgs)


# ---------------------------------------------------------------------------
# Experiment 1 – compare feature selectors using SimpleVGG16 as evaluator
# ---------------------------------------------------------------------------

def experiment_1(X_train, y_train, n_classes, logger, timings=None):
    """
    Fit each selector and evaluate using SimpleVGG16 in feature-input mode.
    Returns (best_selector_name, fitted_selector_object).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1 - Feature Selection (SimpleVGG16 evaluator)")
    print("="*60)

    selectors = {
        "ABC":      ABCFeatureSelector(n_bees=10,   max_iter=20),
        "WOA-APSO": WOA_APSOSelector(n_agents=10,   max_iter=20),
        "ACO":      ACOSelector(n_ants=10,           max_iter=20),
    }

    fs_eval    = FeatureSelectionMetrics(n_classes=n_classes)
    best_name  = None
    best_score = -1.0
    best_sel   = None
    _t         = timings if timings is not None else {}

    for name, sel in selectors.items():
        print(f"\n--- {name} ---")
        with Timer(f"  Exp1 selector: {name}", _t):
            sel.fit(X_train, y_train, n_classes=n_classes)
            X_sel   = sel.transform(X_train)
            metrics = fs_eval.evaluate(
                X_sel, y_train, n_classes=n_classes,
                selection_time=sel.selection_time_,
                n_original_features=X_train.shape[1],
            )
        logger.log(f"Exp1 - {name}", metrics)
        print(f"  {name}: accuracy={metrics['vgg16_accuracy']:.4f}  "
              f"n_features={metrics['n_selected']}  "
              f"time={Timer.fmt_secs(metrics['selection_time_s'])}")

        if metrics["vgg16_accuracy"] > best_score:
            best_score = metrics["vgg16_accuracy"]
            best_name  = name
            best_sel   = sel

    print(f"\n[Exp 1] Best selector: {best_name}  "
          f"(VGG16 accuracy = {best_score:.4f})")
    return best_name, best_sel


# ---------------------------------------------------------------------------
# Experiment 2a – compare all AI models with best-selected features
# ---------------------------------------------------------------------------

def experiment_2a(X_train_feats, y_train, X_val_feats, y_val,
                  X_test_feats,  y_test,
                  X_train_imgs, X_val_imgs, X_test_imgs,
                  n_classes, feature_dim, image_shape,
                  logger, save_dir="outputs/models", timings=None):
    """
    Train all models. Feature-input models get selected feature vectors;
    image-input models get preprocessed images.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2a - AI Model Comparison")
    print("="*60)

    # Split models by input type
    # flycaps uses image input by default (use_feature_input=False in registry)
    IMAGE_MODELS   = {"vgg16", "vgg19_svm", "hybrid_cnn",
                      "litecshuffle", "yang_vit", "flycaps"}
    FEATURE_MODELS = {"kcnet", "grasshopper", "mantis"}

    mm      = ModelMetrics()
    results = {}
    _t      = timings if timings is not None else {}

    for model_type in MODEL_REGISTRY:
        import gc
        is_feat = model_type in FEATURE_MODELS
        print(f"\n--- Training: {model_type} ---", flush=True)
        _success = False

        # ── Resume check ─────────────────────────────────────────────────────
        # If a saved model file exists on disk AND we have metrics in the
        # results JSON, skip retraining and reload from disk instead.
        _keras_path = os.path.join(save_dir, f"best_{model_type}.keras")
        _pkl_path   = os.path.join(save_dir, f"best_{model_type}.pkl")
        _saved_path = _keras_path if os.path.exists(_keras_path)                       else (_pkl_path if os.path.exists(_pkl_path) else None)
        _prior_metrics = logger.get_existing(f"Exp2a - {model_type}")

        if _saved_path and _prior_metrics:
            print(f"  [Resume] Found saved model at {_saved_path} "
                  f"with prior accuracy={_prior_metrics.get('accuracy', '?'):.4f} "
                  f"-- skipping retraining.", flush=True)
            results[model_type] = {"model_path": _saved_path,
                                   "metrics":    _prior_metrics,
                                   "is_feat":    is_feat}
            continue   # skip the entire try/except/finally block

        try:
            trainer = ModelTrainer(
                model_type=model_type,
                n_classes=n_classes,
                feature_dim=feature_dim if is_feat else None,
                image_input_shape=image_shape,
                save_dir=save_dir,
            )

            with Timer(f"  Exp2a model: {model_type}", _t):
                if is_feat:
                    trainer.run(X_train_feats, y_train,
                                X_val_feats, y_val)
                    full = mm.evaluate(trainer.model, X_test_feats, y_test)
                else:
                    trainer.run(X_train_imgs, y_train,
                                X_val_imgs, y_val, epochs=30)
                    full = mm.evaluate(trainer.model, X_test_imgs, y_test)

            train_info = {"training_time_s": trainer.training_time_}
            full.update(train_info)
            logger.log(f"Exp2a - {model_type}", full)

            # Save weights to disk immediately after evaluate.
            # Storing paths (not live objects) means only one model lives
            # in GPU/RAM at a time — subsequent models get a clean slate.
            model_path = trainer.save(save_dir=save_dir)
            results[model_type] = {"model_path": model_path,
                                   "metrics":    full,
                                   "is_feat":    is_feat}
            _success = True

        except Exception as e:
            import traceback
            print(f"  [!] {model_type} failed: {e}")
            traceback.print_exc()
            results[model_type] = {"model_path": None,
                                   "metrics":    {"accuracy": 0.0},
                                   "is_feat":    is_feat}

        finally:
            # Always clear session and free memory after each model.
            try:
                import tensorflow as _tf
                _tf.keras.backend.clear_session()
            except Exception:
                pass
            gc.collect()
            print(f"  [Mem] session cleared after {model_type}", flush=True)

    best = max(results, key=lambda n: results[n]["metrics"].get("accuracy", 0))
    print(f"\n[Exp 2a] Best model: {best}  "
          f"(accuracy = {results[best]['metrics']['accuracy']:.4f})")
    return results, best


# ---------------------------------------------------------------------------
# Experiment 2b – channel pruning
# ---------------------------------------------------------------------------

def experiment_2b(exp2a_results, X_train_feats, y_train, X_val_feats, y_val,
                  X_test_feats, y_test, X_train_imgs, X_val_imgs, X_test_imgs,
                  n_classes, logger, prune_ratio=0.3,
                  image_input_shape=(160, 160, 3), feature_dim=None,
                  save_dir="outputs/models"):
    print("\n" + "="*60)
    print("EXPERIMENT 2b – Channel Pruning")
    print("="*60)

    from model_training import _as_tf_dataset, ModelTrainer
    import tensorflow as tf
    import gc

    pruner = ChannelPruner(prune_ratio=prune_ratio)
    mm     = ModelMetrics()

    for name, info in exp2a_results.items():
        model_path = info.get("model_path")
        if model_path is None:
            print(f"  Skipping {name} (no saved model).")
            continue
        # Only Keras image models (.keras files) have Conv2D layers to prune
        if not model_path.endswith(".keras"):
            print(f"  Skipping {name} (analytic model, no Conv2D layers).")
            continue

        # Reload from disk — only this one model in GPU memory at a time
        try:
            from model_training import WinnerTakeAllLayer, SquashLayer
            keras_model = tf.keras.models.load_model(
                model_path,
                custom_objects={
                    "WinnerTakeAllLayer": WinnerTakeAllLayer,
                    "SquashLayer":        SquashLayer,
                })
        except Exception as e:
            print(f"  Skipping {name} — could not reload: {e}")
            continue

        has_conv = any("conv" in layer.name.lower()
                       for layer in keras_model.layers
                       if hasattr(layer, "filters"))
        if not has_conv:
            print(f"  Skipping {name} (no Conv2D layers).")
            tf.keras.backend.clear_session(); gc.collect()
            continue

        print(f"\n--- Pruning {name} ---")
        pruned = pruner.prune(keras_model)
        del keras_model   # free unpruned weights immediately

        X_tr = X_train_feats if info["is_feat"] else X_train_imgs
        X_v  = X_val_feats   if info["is_feat"] else X_val_imgs
        X_te = X_test_feats  if info["is_feat"] else X_test_imgs

        train_ds = _as_tf_dataset(X_tr, y_train, batch_size=32, shuffle=True)
        val_ds   = _as_tf_dataset(X_v,  y_val,   batch_size=32, shuffle=False)
        pruned.fit(train_ds, validation_data=val_ds, epochs=5, verbose=1)

        class _W:
            def __init__(self, m, nc):
                self.model = m; self.nc = nc
            def predict(self, X):
                p = self.model.predict(X, verbose=0)
                return (p.squeeze() > 0.5).astype(int) if self.nc == 2 \
                       else np.argmax(p, axis=1)

        full  = mm.evaluate(_W(pruned, n_classes), X_te, y_test)
        delta = full["accuracy"] - info["metrics"]["accuracy"]
        full["accuracy_delta_vs_unpruned"] = delta
        logger.log(f"Exp2b – {name} pruned@{prune_ratio:.0%}", full)
        print(f"  Accuracy delta: {delta:+.4f}")

        # Free pruned model before next iteration
        del pruned
        tf.keras.backend.clear_session()
        gc.collect()


# ---------------------------------------------------------------------------
# Experiment 3 – alternate feature selection for near-best models
# ---------------------------------------------------------------------------

def experiment_3(exp2a_results, best_model_name,
                 X_train_feats, y_train, X_val_feats, y_val,
                 X_test_feats,  y_test,
                 n_classes, feature_dim, logger,
                 threshold=0.02):
    print("\n" + "="*60)
    print("EXPERIMENT 3 – Alternate Feature Selection for Near-Best Models")
    print("="*60)

    best_acc  = exp2a_results[best_model_name]["metrics"].get("accuracy", 0)
    near_best = [
        n for n, v in exp2a_results.items()
        if n != best_model_name and n != "vgg19_svm"   # skip non-Keras sklearn
        and (best_acc - v["metrics"].get("accuracy", 0)) <= threshold
    ]

    if not near_best:
        print("  No near-best models within threshold. Skipping.")
        return

    print(f"  Near-best models: {near_best}")

    alt_sel = WOA_APSOSelector(n_agents=10, max_iter=20)
    print("  Running alternate selector: WOA-APSO")
    alt_sel.fit(X_train_feats, y_train, n_classes=n_classes)

    X_alt_tr = alt_sel.transform(X_train_feats)
    X_alt_v  = alt_sel.transform(X_val_feats)
    X_alt_te = alt_sel.transform(X_test_feats)
    alt_dim  = X_alt_tr.shape[1]

    mm = ModelMetrics()
    for name in near_best:
        if not exp2a_results[name]["is_feat"]:
            print(f"  Skipping {name} (image model, not retrained on alt features).")
            continue
        print(f"\n  Re-training {name} with alternate features …")
        try:
            import gc
            trainer = ModelTrainer(
                model_type=name, n_classes=n_classes,
                feature_dim=alt_dim, save_dir="outputs/models_exp3"
            )
            trainer.run(X_alt_tr, y_train, X_alt_v, y_val)
            full = mm.evaluate(trainer.model, X_alt_te, y_test)
            delta = full["accuracy"] - exp2a_results[name]["metrics"].get("accuracy", 0)
            full["accuracy_delta_vs_original_selector"] = delta
            logger.log(f"Exp3 – {name} + WOA-APSO", full)
            print(f"  Accuracy delta: {delta:+.4f}")
            trainer.save(save_dir="outputs/models_exp3")
        except Exception as e:
            import traceback
            print(f"  [!] {name} Exp3 failed: {e}")
            traceback.print_exc()
        finally:
            import gc, tensorflow as _tf
            _tf.keras.backend.clear_session()
            gc.collect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plant Disease AI Pipeline")
    p.add_argument("--dataset_root",  default="./dataset")
    p.add_argument("--use_subset",    action="store_true",
                   help="Use only potato + pepper_bell")
    p.add_argument("--image_size",    type=int, default=160,
                   help="CNN input image size (square)")
    p.add_argument("--skip_exp3",     action="store_true")
    p.add_argument("--results_path",  default="outputs/experiment_results.json")
    p.add_argument("--prune_ratio",    type=float, default=0.3)
    p.add_argument("--rebuild_cache",    action="store_true",
                   help="Force rebuild of HDF5 cache even if it exists")
    p.add_argument("--load_chunk_size",  type=int, default=512,
                   help="Rows per chunk for lazy_transform (lower = less RAM)")
    p.add_argument("--selector_sample",  type=int, default=2000,
                   help="Rows sampled from training set for feature selector fit")
    p.add_argument("--resume",  action="store_true",
                   help="Resume from a previous partial run — skip already-saved "
                        "models and reload their metrics from --results_path")
    p.add_argument("--skip_exp2b", action="store_true",
                   help="Skip experiment 2b (channel pruning)")
    return p.parse_args()


def _lazy_transform(selector, lazy_arr, chunk_size=512):
    """
    Apply a fitted feature selector to a LazyH5Array chunk-by-chunk.
    Never loads the full array into RAM.
    Returns a numpy array of selected features (small: only selected columns).
    """
    n = len(lazy_arr)
    out_chunks = []
    for start in range(0, n, chunk_size):
        chunk = np.array(lazy_arr[start:min(start + chunk_size, n)])
        out_chunks.append(selector.transform(chunk))
    return np.concatenate(out_chunks, axis=0)


def main():
    args    = parse_args()
    timings = {}   # collects wall-clock seconds for every named step
    os.makedirs("outputs/models",      exist_ok=True)
    os.makedirs("outputs/models_exp3", exist_ok=True)
    os.makedirs("outputs/cache",       exist_ok=True)
    logger = ResultsLogger(args.results_path)

    # ── 1. Load dataset ─────────────────────────────────────────────────────
    with Timer("1. Dataset discovery", timings):
        loader = DatasetLoader(
            dataset_root=args.dataset_root,
            use_subset=args.use_subset,
            image_size=(args.image_size, args.image_size),
        )
        loader.discover()
        tr_paths, val_paths, te_paths, y_train, y_val, y_test = loader.get_splits()
        n_classes   = len(loader.class_names)
        image_shape = (args.image_size, args.image_size, 3)
        print(f"  n_classes={n_classes}  classes={loader.class_names}")

    # ── 2. Build / open cache ───────────────────────────────────────────────
    cache = DataCache(cache_dir="outputs/cache")
    cache.cache_info()

    feat_pre  = PreprocessingPipeline(target_size=(512, 512))
    extractor = FeatureExtractor()
    cnn_pre   = PreprocessingPipeline(target_size=(args.image_size, args.image_size))

    with Timer("2a. Feature cache (build or load)", timings):
        X_tr_raw, X_val_raw, X_te_raw, y_train, y_val, y_test = cache.get_features(
            tr_paths, val_paths, te_paths,
            y_train, y_val, y_test,
            feat_pre, extractor, loader,
            chunk_size=32,
            force_rebuild=args.rebuild_cache,
        )

    with Timer("2b. Image cache (build or load)", timings):
        X_tr_imgs, X_val_imgs, X_te_imgs = cache.get_images(
            tr_paths, val_paths, te_paths,
            cnn_pre, loader,
            image_size=args.image_size,
            chunk_size=16,
            force_rebuild=args.rebuild_cache,
        )

    # ── 3. Experiment 1 – feature selection ─────────────────────────────────
    print(f"\n[Memory] Sampling {args.selector_sample} rows for feature selection ...")
    X_tr_sample = X_tr_raw.sample(args.selector_sample)
    rng_idx = np.sort(np.random.default_rng(42).choice(
        len(y_train), size=min(args.selector_sample, len(y_train)), replace=False))
    y_tr_sample = np.array(y_train)[rng_idx]
    print(f"  Sample shape: {X_tr_sample.shape}")

    with Timer("3. Experiment 1 -- Feature selection", timings):
        best_sel_name, best_sel = experiment_1(
            X_tr_sample, y_tr_sample, n_classes, logger, timings=timings)
    del X_tr_sample, y_tr_sample

    # ── 4. Apply best selector ───────────────────────────────────────────────
    with Timer("4. Apply best selector (transform all splits)", timings):
        print(f"  Applying selector ({best_sel_name}) chunk-by-chunk ...")
        X_tr_sel  = _lazy_transform(best_sel, X_tr_raw,
                                    chunk_size=args.load_chunk_size)
        X_val_sel = _lazy_transform(best_sel, X_val_raw,
                                    chunk_size=args.load_chunk_size)
        X_te_sel  = _lazy_transform(best_sel, X_te_raw,
                                    chunk_size=args.load_chunk_size)
        feat_dim = X_tr_sel.shape[1]
        print(f"  Feature dim after selection: {feat_dim} (from {X_tr_raw.shape[1]})")

    # ── 4b. Pre-download VGG weights before experiment 2a ───────────────────
    _vgg_cache = os.path.join(
        os.path.expanduser("~"), ".keras", "models",
        "vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
    if not os.path.exists(_vgg_cache):
        print("\n[Setup] Pre-downloading VGG19 ImageNet weights (120s timeout)...")
        try:
            import signal as _sig
            import tensorflow as _tf2
            def _th(s, f): raise TimeoutError("VGG19 download timed out")
            _sig.signal(_sig.SIGALRM, _th)
            _sig.alarm(120)
            _tf2.keras.applications.VGG19(
                weights="imagenet", include_top=False, input_shape=(160,160,3))
            _sig.alarm(0)
            _tf2.keras.backend.clear_session()
            print("  VGG19 weights downloaded.")
        except TimeoutError:
            print("  [WARN] VGG19 download timed out. vgg19_svm will be skipped.")
            print(f"  To fix: manually copy vgg19 weights to {_vgg_cache}")
        except Exception as _de:
            print(f"  [WARN] VGG19 pre-download failed: {_de}")
    else:
        print(f"\n[Setup] VGG19 weights already cached.")

    # ── 5. Experiment 2a ────────────────────────────────────────────────────
    with Timer("5. Experiment 2a -- Model training & evaluation", timings):
        exp2a, best_model = experiment_2a(
            X_tr_sel,  y_train, X_val_sel, y_val, X_te_sel,  y_test,
            X_tr_imgs, X_val_imgs, X_te_imgs,
            n_classes, feat_dim, image_shape, logger,
            timings=timings,
        )

    # ── 6. Experiment 2b ────────────────────────────────────────────────────
    if args.skip_exp2b:
        print("\n[Skip] Experiment 2b skipped (--skip_exp2b).")
    else:
        with Timer("6. Experiment 2b -- Channel pruning", timings):
            experiment_2b(
                exp2a,
                X_tr_sel, y_train, X_val_sel, y_val, X_te_sel, y_test,
                X_tr_imgs, X_val_imgs, X_te_imgs,
                n_classes, logger,
                prune_ratio=args.prune_ratio,
                image_input_shape=image_shape,
                feature_dim=feat_dim,
            )

    # ── 7. Experiment 3 ─────────────────────────────────────────────────────
    if not args.skip_exp3:
        with Timer("7. Experiment 3 -- Alternate feature selection", timings):
            experiment_3(
                exp2a, best_model,
                X_tr_sel, y_train, X_val_sel, y_val, X_te_sel, y_test,
                n_classes, feat_dim, logger,
            )

    logger.save()
    logger.print_summary()
    Timer.print_summary(timings)


if __name__ == "__main__":
    main()