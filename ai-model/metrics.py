"""
metrics.py
Gathers all evaluation metrics specified in the document:

  For Feature Selection:
    - Accuracy of simple model (KNN) using selected features
    - Time taken to select features
    - Number of features selected

  For AI Model:
    - Accuracy
    - F1 Score (macro)
    - Time taken to predict
    - Model size (MB)
    - FLOPS (approximate)
"""

import time
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
try:
    import tensorflow as tf
    _TF_OK2 = True
except ImportError:
    _TF_OK2 = False

try:
    import tensorflow as tf
    _TF_OK = True
except ImportError:
    _TF_OK = False


# ---------------------------------------------------------------------------
# Feature-selection metrics
# ---------------------------------------------------------------------------

class FeatureSelectionMetrics:
    """
    Evaluates a feature-selection result using SimpleVGG16 in feature-input
    mode (a small dense network), matching the document specification:
    "Accuracy of simple model (like CNN) using the feature extraction model."

    Parameters
    ----------
    n_classes : int  Number of output classes.
    cv_folds  : int  Number of cross-validation folds.
    epochs    : int  Training epochs per fold (kept small for speed).
    """

    def __init__(self, n_classes: int = 2, cv_folds: int = 3, epochs: int = 10):
        self.n_classes = n_classes
        self.cv_folds  = cv_folds
        self.epochs    = epochs

    def evaluate(self,
                 X_selected: np.ndarray,
                 y: np.ndarray,
                 n_classes: int = None,
                 selection_time: float = 0.0,
                 n_original_features: int = None) -> dict:
        """
        Parameters
        ----------
        X_selected           : Feature matrix after selection
        y                    : Integer labels
        n_classes            : Override n_classes if needed
        selection_time       : Wall-clock seconds taken by the selector
        n_original_features  : Total features before selection

        Returns
        -------
        dict with keys:
            vgg16_accuracy, selection_time_s, n_selected, n_original,
            reduction_ratio
        """
        nc    = n_classes or self.n_classes
        n_sel = X_selected.shape[1]
        if n_original_features is None:
            n_original_features = n_sel

        skf  = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        accs = []
        for tr_idx, val_idx in skf.split(X_selected, y):
            X_tr, X_v = X_selected[tr_idx], X_selected[val_idx]
            y_tr, y_v = y[tr_idx], y[val_idx]

            inp = tf.keras.Input(shape=(n_sel,))
            x   = tf.keras.layers.Dense(512, activation="relu")(inp)
            x   = tf.keras.layers.Dropout(0.5)(x)
            x   = tf.keras.layers.Dense(512, activation="relu")(x)
            if nc == 2:
                out  = tf.keras.layers.Dense(1, activation="sigmoid")(x)
                loss = "binary_crossentropy"
            else:
                out  = tf.keras.layers.Dense(nc, activation="softmax")(x)
                loss = "sparse_categorical_crossentropy"

            m = tf.keras.Model(inp, out)
            m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss=loss, metrics=["accuracy"])
            m.fit(X_tr, y_tr, epochs=self.epochs, batch_size=32, verbose=0)
            _, acc = m.evaluate(X_v, y_v, verbose=0)
            accs.append(acc)
            tf.keras.backend.clear_session()
            del m

        mean_acc = float(np.mean(accs))
        return {
            "vgg16_accuracy":   mean_acc,
            "selection_time_s": float(selection_time),
            "n_selected":       int(n_sel),
            "n_original":       int(n_original_features),
            "reduction_ratio":  float(n_sel / (n_original_features + 1e-10)),
        }


# ---------------------------------------------------------------------------
# AI model metrics
# ---------------------------------------------------------------------------

class ModelMetrics:
    """
    Gathers inference-time metrics for a trained model.

    Works with any sklearn-style .predict() model and with
    Keras models (for FLOPS / size estimation).
    """

    def evaluate(self,
                 model,
                 X_test,
                 y_test: np.ndarray,
                 model_path: str = None) -> dict:
        """
        Parameters
        ----------
        model       : Trained model with .predict(X) interface
        X_test      : Test features — numpy array or LazyH5Array
        y_test      : True labels
        model_path  : Optional path to saved model file (for size in MB)

        Returns
        -------
        dict with keys: accuracy, f1_score_macro, prediction_time_s,
                        model_size_mb, flops (if Keras)
        """
        # --- Prediction ---
        # Use the model's own .predict() which already handles LazyH5Array
        # correctly via _keras_predict / sklearn pipeline etc.
        t0    = time.time()
        preds = model.predict(X_test)
        prediction_time = time.time() - t0

        # Flatten in case of 2-D output (binary sigmoid squeeze)
        preds = np.array(preds).ravel() if np.array(preds).ndim > 1                 else np.array(preds)

        # --- Classification metrics ---
        acc = accuracy_score(y_test, preds)
        f1  = f1_score(y_test, preds, average="macro", zero_division=0)

        # --- Model size ---
        size_mb = 0.0
        if model_path and os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 ** 2)

        # --- FLOPS (Keras only, best-effort — skip on error) ---
        try:
            # Determine per-sample shape safely for both ndarray and LazyH5Array
            sample_shape = X_test.shape[1:]
            flops = self._estimate_flops(model, sample_shape)
        except Exception:
            flops = 0.0

        return {
            "accuracy":             float(acc),
            "f1_score_macro":       float(f1),
            "prediction_time_s":    float(prediction_time),
            "model_size_mb":        float(size_mb),
            "flops":                flops,
        }

    @staticmethod
    def _estimate_flops(model, input_shape) -> float:
        """
        Rough FLOPS estimate for Keras models using layer-wise calculation.
        Returns 0 for non-Keras models or on any error.
        """
        try:
            if not _TF_OK:
                return 0.0
            if not isinstance(model, tf.keras.Model) and \
               not hasattr(model, 'model'):
                return 0.0

            keras_model = model.model if hasattr(model, 'model') else model
            if keras_model is None:
                return 0.0

            total_flops = 0.0
            for layer in keras_model.layers:
                try:
                    if isinstance(layer, tf.keras.layers.Conv2D):
                        cfg     = layer.get_config()
                        filters = cfg["filters"]
                        kh, kw  = cfg["kernel_size"]
                        out_h = layer.output_shape[1] if len(layer.output_shape) > 2 else 1
                        out_w = layer.output_shape[2] if len(layer.output_shape) > 3 else 1
                        in_c  = layer.input_shape[-1]
                        total_flops += 2 * kh * kw * in_c * filters * out_h * out_w
                    elif isinstance(layer, tf.keras.layers.Dense):
                        in_units  = layer.input_shape[-1]
                        out_units = layer.units
                        total_flops += 2 * in_units * out_units
                except Exception:
                    continue   # skip layers whose shapes aren't inferrable
            return float(total_flops)
        except Exception:
            return 0.0


# ---------------------------------------------------------------------------
# Results logger
# ---------------------------------------------------------------------------

class ResultsLogger:
    """
    Collects results from multiple experiments and saves them as JSON.
    Saves after every log() call so partial results survive crashes.

    Usage
    -----
        logger = ResultsLogger("results.json", resume=True)
        logger.log("Experiment 1 - ABC selector", metrics_dict)
        logger.print_summary()
    """

    def __init__(self, output_path: str = "experiment_results.json",
                 resume: bool = False):
        self.output_path = output_path
        self._records: list = []
        # On resume, load existing records so we don't lose prior results
        if resume and os.path.exists(output_path):
            try:
                with open(output_path) as fh:
                    self._records = json.load(fh)
                print(f"[ResultsLogger] Loaded {len(self._records)} existing "
                      f"records from {output_path}")
            except Exception as e:
                print(f"[ResultsLogger] Could not load existing results: {e}")

    def get_existing(self, name: str) -> dict:
        """Return metrics for a previously logged experiment, or None."""
        for r in self._records:
            if r.get("experiment") == name:
                return r["metrics"]
        return None

    def log(self, name: str, metrics: dict):
        # Replace any existing record with the same name
        self._records = [r for r in self._records
                         if r.get("experiment") != name]
        record = {"experiment": name, "metrics": metrics}
        self._records.append(record)
        print(f"\n[ResultsLogger] {name}")
        for k, v in metrics.items():
            print(f"    {k:30s}: {v}")
        # Auto-save after every entry — crash-safe
        self._save_now()

    def _save_now(self):
        """Write current records to disk immediately."""
        try:
            os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
            with open(self.output_path, "w") as f:
                json.dump(self._records, f, indent=2, default=str)
        except Exception as e:
            print(f"[ResultsLogger] Warning: auto-save failed: {e}")

    def save(self):
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(self._records, f, indent=2)
        print(f"\n[ResultsLogger] Results saved to {self.output_path}")

    def print_summary(self):
        print("\n" + "=" * 60)
        print("EXPERIMENT SUMMARY")
        print("=" * 60)
        for rec in self._records:
            print(f"\n  {rec['experiment']}")
            for k, v in rec["metrics"].items():
                print(f"    {k:30s}: {v:.4f}" if isinstance(v, float) else
                      f"    {k:30s}: {v}")
        print("=" * 60)