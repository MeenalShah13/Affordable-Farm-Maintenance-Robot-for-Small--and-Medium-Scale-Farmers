"""
retrain_two_models.py
=====================
Retrains HybridCNN and LiteCShuffle, running WOA-APSO feature selection
from scratch (since the selector is not saved from the original run).

What this script does
---------------------
1.  Load dataset and open existing HDF5 caches (image + feature)
2.  Run WOA-APSO on a sample of the training features
3.  Save the fitted selector to outputs/selector/
4.  Apply selector to all feature splits
5.  Train and evaluate:
      - LiteCShuffle          (image-input)
      - HybridCNN             (image-input)   <-- original mode
      - HybridCNN + WOA-APSO (feature-input) <-- new comparison mode
6.  Save all three models to --save_dir
7.  Print results table and compare with original run

Usage
-----
python retrain_two_models.py --dataset_root ../datasets --use_subset

python retrain_two_models.py \\
    --dataset_root ../datasets --use_subset \\
    --epochs 30 --woa_agents 10 --woa_iters 20

# Resume if interrupted
python retrain_two_models.py --dataset_root ../datasets --use_subset --resume

# Force redo everything
python retrain_two_models.py --dataset_root ../datasets --use_subset --force_retrain
"""

import os, sys, time, json, pickle, gc, argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from data_loader        import DatasetLoader
from preprocessing      import PreprocessingPipeline
from feature_extraction import FeatureExtractor
from cache              import DataCache
from feature_selection  import WOA_APSOSelector
from model_training     import ModelTrainer, WinnerTakeAllLayer, SquashLayer
from metrics            import ModelMetrics, ResultsLogger


# ---------------------------------------------------------------------------
# Timer
# ---------------------------------------------------------------------------

class Timer:
    def __init__(self, name, timings):
        self.name = name
        self.timings = timings

    def __enter__(self):
        self._t0 = time.time()
        print(f"\n[Timer] {self.name} started ...", flush=True)
        return self

    def __exit__(self, *_):
        elapsed = time.time() - self._t0
        self.timings[self.name] = elapsed
        print(f"[Timer] {self.name} done -- {Timer.fmt(elapsed)}", flush=True)

    @staticmethod
    def fmt(secs):
        if secs < 120:  return f"{secs:.1f}s"
        if secs < 7200: return f"{secs/60:.1f}m"
        return f"{secs/3600:.2f}h"

    @staticmethod
    def print_summary(timings):
        if not timings: return
        total    = sum(timings.values())
        max_name = max(len(k) for k in timings)
        print("\n" + "=" * 64)
        print("TIMING SUMMARY")
        print("=" * 64)
        for name, secs in timings.items():
            pct = 100 * secs / total if total else 0
            bar = "\u2588" * int(pct / 2.5)
            print(f"  {name:<{max_name}}  {Timer.fmt(secs):>8}  {pct:5.1f}%  {bar}")
        print("-" * 64)
        print(f"  {'TOTAL':<{max_name}}  {Timer.fmt(total):>8}")
        print("=" * 64)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lazy_transform(selector, arr, chunk_size=512):
    n = len(arr)
    chunks = []
    for start in range(0, n, chunk_size):
        chunk = np.array(arr[start:min(start + chunk_size, n)])
        chunks.append(selector.transform(chunk))
    return np.concatenate(chunks, axis=0)


def _print_results_table(results):
    print("\n" + "=" * 88)
    print("RETRAIN RESULTS SUMMARY")
    print("=" * 88)
    hdr = (f"  {'Run':<36} {'Accuracy':>10} {'F1 Macro':>10} "
           f"{'Pred Time':>11} {'FLOPs':>11} {'Train Time':>11}")
    print(hdr)
    print("  " + "-" * 84)
    for run_name, info in results.items():
        m     = info["metrics"]
        acc   = m.get("accuracy", 0)
        f1    = m.get("f1_score_macro", 0)
        pred  = m.get("prediction_time_s", 0)
        flops = m.get("flops", 0)
        train = m.get("training_time_s", 0)
        fstr  = f"{flops/1e6:.1f}M" if flops > 0 else "N/A"
        print(f"  {run_name:<36} {acc:>10.4f} {f1:>10.4f} "
              f"{Timer.fmt(pred):>11} {fstr:>11} {Timer.fmt(train):>11}")
    print("=" * 88)
    for run_name, info in results.items():
        print(f"\n  {run_name}")
        for k, v in info["metrics"].items():
            print(f"    {k:<34}: {v}")
        if info.get("model_path"):
            print(f"    {'saved_to':<34}: {info['model_path']}")


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def _train_model(run_name, model_type,
                 X_tr, y_train, X_val, y_val, X_te, y_test,
                 n_classes, image_shape, feat_dim, is_feat_input,
                 save_dir, epochs, logger, timings, force_retrain):
    import tensorflow as _tf

    safe_name  = run_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "")
    model_file = os.path.join(save_dir, f"{safe_name}.keras")
    prior      = logger.get_existing(run_name)

    if not force_retrain and os.path.exists(model_file) and prior:
        print(f"\n  [Resume] {run_name} already done "
              f"(accuracy={prior.get('accuracy', '?'):.4f}) -- skipping.",
              flush=True)
        return {"model_path": model_file, "metrics": prior}

    print(f"\n--- {run_name} ---", flush=True)
    mm      = ModelMetrics()
    result  = {}

    try:
        trainer = ModelTrainer(
            model_type=model_type,
            n_classes=n_classes,
            feature_dim=feat_dim if is_feat_input else None,
            image_input_shape=image_shape,
            save_dir=save_dir,
            use_feature_input=is_feat_input,   # passed directly to constructor
        )
        # ModelTrainer.__init__ already sets use_feature_input and feature_dim
        # correctly on the wrapped model — no post-construction override needed

        with Timer(f"  Train {run_name}", timings):
            trainer.run(X_tr, y_train, X_val, y_val, epochs=epochs)
            metrics = mm.evaluate(trainer.model, X_te, y_test)

        metrics["training_time_s"] = trainer.training_time_
        logger.log(run_name, metrics)

        keras_model = getattr(trainer.model, "model", None)
        if keras_model is not None:
            keras_model.save(model_file)
            print(f"  [Save] {run_name} -> {model_file}")
        else:
            model_file = None

        result = {"model_path": model_file, "metrics": metrics}

    except Exception as e:
        import traceback
        print(f"  [!] {run_name} failed: {e}")
        traceback.print_exc()
        result = {"model_path": None, "metrics": {"accuracy": 0.0}}

    finally:
        try:
            _tf.keras.backend.clear_session()
        except Exception:
            pass
        gc.collect()
        print(f"  [Mem] session cleared after {run_name}", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def retrain(args):
    timings = {}
    os.makedirs(args.save_dir,     exist_ok=True)
    os.makedirs(args.selector_dir, exist_ok=True)

    logger = ResultsLogger(output_path=args.results_path, resume=args.resume)

    # ── 1. Dataset ────────────────────────────────────────────────────────────
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

    # ── 2. Open caches ────────────────────────────────────────────────────────
    cache     = DataCache(cache_dir=args.cache_dir)
    cnn_pre   = PreprocessingPipeline(target_size=(args.image_size, args.image_size))
    feat_pre  = PreprocessingPipeline(target_size=(512, 512))
    extractor = FeatureExtractor()

    with Timer("2a. Image cache", timings):
        X_tr_imgs, X_val_imgs, X_te_imgs = cache.get_images(
            tr_paths, val_paths, te_paths, cnn_pre, loader,
            image_size=args.image_size, chunk_size=16, force_rebuild=False)

    with Timer("2b. Feature cache", timings):
        X_tr_raw, X_val_raw, X_te_raw, y_train, y_val, y_test = cache.get_features(
            tr_paths, val_paths, te_paths, y_train, y_val, y_test,
            feat_pre, extractor, loader, chunk_size=32, force_rebuild=False)

    # ── 3. WOA-APSO ───────────────────────────────────────────────────────────
    sel_pkl  = os.path.join(args.selector_dir, "retrain_woa_selector.pkl")
    sel_meta = os.path.join(args.selector_dir, "retrain_woa_selector_meta.json")

    if not args.force_retrain and os.path.exists(sel_pkl):
        print(f"\n[WOA-APSO] Loading saved selector from {sel_pkl} ...", flush=True)
        with open(sel_pkl, "rb") as fh:
            selector = pickle.load(fh)
        with open(sel_meta) as fh:
            meta = json.load(fh)
        print(f"  {meta['n_selected']} features selected "
              f"in {Timer.fmt(meta['selection_time_s'])}")
    else:
        print(f"\n[WOA-APSO] Sampling {args.selector_sample} rows ...", flush=True)
        rng_idx  = np.sort(np.random.default_rng(42).choice(
            len(y_train), size=min(args.selector_sample, len(y_train)), replace=False))
        X_sample = X_tr_raw.sample(args.selector_sample)
        y_sample = np.array(y_train)[rng_idx]
        print(f"  Sample shape: {X_sample.shape}")

        selector = WOA_APSOSelector(
            n_agents=args.woa_agents, max_iter=args.woa_iters,
            w=0.7, c1=1.5, c2=1.5, alpha=0.99, beta=0.01, random_state=42)

        with Timer("3. WOA-APSO feature selection", timings):
            selector.fit(X_sample, y_sample,
                         n_classes=n_classes,
                         prefilter_k=args.woa_prefilter_k)

        print(f"\n[WOA-APSO] Selected {selector.n_features_} of "
              f"{X_tr_raw.shape[1]} features "
              f"(fitness={selector._best_fitness:.4f})", flush=True)

        with open(sel_pkl, "wb") as fh:
            pickle.dump(selector, fh, protocol=4)
        with open(sel_meta, "w") as fh:
            json.dump({
                "selector_name":    "WOA-APSO",
                "n_agents":         args.woa_agents,
                "max_iter":         args.woa_iters,
                "prefilter_k":      args.woa_prefilter_k,
                "selector_sample":  args.selector_sample,
                "selected_indices": selector.selected_indices_.tolist(),
                "n_selected":       int(selector.n_features_),
                "n_original":       int(X_tr_raw.shape[1]),
                "best_fitness":     float(selector._best_fitness),
                "selection_time_s": float(selector.selection_time_),
            }, fh, indent=2)
        print(f"  Saved -> {sel_pkl}")

    # ── 4. Apply selector ─────────────────────────────────────────────────────
    with Timer("4. Apply WOA-APSO to all splits", timings):
        print("  Transforming splits ...", flush=True)
        X_tr_sel  = _lazy_transform(selector, X_tr_raw,  args.load_chunk_size)
        X_val_sel = _lazy_transform(selector, X_val_raw, args.load_chunk_size)
        X_te_sel  = _lazy_transform(selector, X_te_raw,  args.load_chunk_size)
        feat_dim  = X_tr_sel.shape[1]
        print(f"  Feature dim: {feat_dim} (from {X_tr_raw.shape[1]})")

    # ── 5. Train three runs ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING: LiteCShuffle (img), HybridCNN (img), HybridCNN (feat)")
    print("=" * 60)

    results = {}

    results["litecshuffle (image)"] = _train_model(
        run_name="litecshuffle (image)", model_type="litecshuffle",
        X_tr=X_tr_imgs, y_train=y_train, X_val=X_val_imgs, y_val=y_val,
        X_te=X_te_imgs, y_test=y_test,
        n_classes=n_classes, image_shape=image_shape,
        feat_dim=None, is_feat_input=False,
        save_dir=args.save_dir, epochs=args.epochs,
        logger=logger, timings=timings, force_retrain=args.force_retrain)

    results["hybrid_cnn (image)"] = _train_model(
        run_name="hybrid_cnn (image)", model_type="hybrid_cnn",
        X_tr=X_tr_imgs, y_train=y_train, X_val=X_val_imgs, y_val=y_val,
        X_te=X_te_imgs, y_test=y_test,
        n_classes=n_classes, image_shape=image_shape,
        feat_dim=None, is_feat_input=False,
        save_dir=args.save_dir, epochs=args.epochs,
        logger=logger, timings=timings, force_retrain=args.force_retrain)

    results["hybrid_cnn (WOA-APSO features)"] = _train_model(
        run_name="hybrid_cnn (WOA-APSO features)", model_type="hybrid_cnn",
        X_tr=X_tr_sel, y_train=y_train, X_val=X_val_sel, y_val=y_val,
        X_te=X_te_sel, y_test=y_test,
        n_classes=n_classes, image_shape=image_shape,
        feat_dim=feat_dim, is_feat_input=True,
        save_dir=args.save_dir, epochs=args.epochs,
        logger=logger, timings=timings, force_retrain=args.force_retrain)

    # ── 6. Print results ──────────────────────────────────────────────────────
    _print_results_table(results)
    Timer.print_summary(timings)

    # ── 7. Compare with original ──────────────────────────────────────────────
    if os.path.exists(args.orig_results_path):
        print("\n" + "=" * 64)
        print("COMPARISON WITH ORIGINAL TRAINING RUN")
        print("=" * 64)
        try:
            with open(args.orig_results_path) as fh:
                orig = {r["experiment"]: r["metrics"]
                        for r in json.load(fh)}

            pairs = [
                ("litecshuffle (image)",  "Exp2a - litecshuffle"),
                ("hybrid_cnn (image)",    "Exp2a - hybrid_cnn"),
            ]
            print(f"\n  {'Run':<36} {'Orig Acc':>10} {'New Acc':>10} {'Delta':>10}")
            print("  " + "-" * 68)
            for new_key, orig_key in pairs:
                new_acc  = results[new_key]["metrics"].get("accuracy", 0)
                orig_acc = orig.get(orig_key, {}).get("accuracy", None)
                if orig_acc is not None:
                    delta = new_acc - orig_acc
                    print(f"  {new_key:<36} {orig_acc:>10.4f} "
                          f"{new_acc:>10.4f} {delta:>+10.4f}")
                else:
                    print(f"  {new_key:<36} {'(not found)':>10} {new_acc:>10.4f}")

            img_acc  = results["hybrid_cnn (image)"]["metrics"].get("accuracy", 0)
            feat_acc = results["hybrid_cnn (WOA-APSO features)"]["metrics"].get("accuracy", 0)
            delta    = feat_acc - img_acc
            winner   = "feature-input" if feat_acc > img_acc else "image-input"
            print(f"\n  HybridCNN feature-input vs image-input: {delta:>+.4f}")
            print(f"  Winner for HybridCNN: {winner} "
                  f"({'%.4f' % max(feat_acc, img_acc)})")

        except Exception as e:
            print(f"  Could not load original results: {e}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Retrain HybridCNN and LiteCShuffle with fresh WOA-APSO",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    p.add_argument("--dataset_root",    type=str, required=True)
    p.add_argument("--use_subset",      action="store_true")
    p.add_argument("--image_size",      type=int, default=160)
    p.add_argument("--cache_dir",       type=str, default="outputs/cache")
    p.add_argument("--load_chunk_size", type=int, default=512)

    p.add_argument("--woa_agents",      type=int, default=10)
    p.add_argument("--woa_iters",       type=int, default=20)
    p.add_argument("--woa_prefilter_k", type=int, default=1000)
    p.add_argument("--selector_sample", type=int, default=2000)
    p.add_argument("--selector_dir",    type=str, default="outputs/selector")

    p.add_argument("--epochs",          type=int, default=30)
    p.add_argument("--save_dir",        type=str, default="outputs/models_retrain")
    p.add_argument("--force_retrain",   action="store_true")

    p.add_argument("--results_path",    type=str,
                   default="outputs/retrain_results.json")
    p.add_argument("--orig_results_path", type=str,
                   default="outputs/experiment_results.json")
    p.add_argument("--resume",          action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    retrain(parse_args())