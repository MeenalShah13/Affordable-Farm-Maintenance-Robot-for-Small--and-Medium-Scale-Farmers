# AI Model Training & Evaluation

Comprehensive framework for training, evaluating, and deploying 9 crop disease classification models for FieldBot.

## Quick Start

### Train a Single Model
```bash
python model_training.py
```

### Run Full Experiment (all 9 models + ablations)
```bash
python experiments.py
```

### Evaluate Existing Model
```python
from model_training import ModelTrainer

trainer = ModelTrainer.load(
    model_type="litecshuffle",
    path="outputs/models/best_litecshuffle.keras",
    n_classes=15
)
predictions = trainer.predict(X_test)
accuracy = trainer.evaluate(X_test, y_test)
```

---

## Models

All 9 models implement a common interface:

```python
model.build()                    # Initialize architecture
model.train(X_tr, y_tr, X_val, y_val, epochs, batch_size)
results = model.evaluate(X_test, y_test)  # → {"accuracy": ..., "loss": ...}
predictions = model.predict(X)   # → integer labels
```

### 1. SimpleVGG16
**Type**: Transfer learning  
**Input**: Images (160×160×3)  
**Architecture**: VGG16 (frozen backbone) + dense head (512→512→15)  
**Key features**: Baseline reference, fast training, moderate accuracy  
**Use case**: Prototyping, resource-constrained systems

```python
from model_training import SimpleVGG16
model = SimpleVGG16(n_classes=15)
model.build().train(X_tr, y_tr, X_val, y_val, epochs=50)
```

### 2. VGG19_SVM
**Type**: Feature extractor + classifier  
**Input**: Images (160×160×3)  
**Architecture**: VGG19 (frozen) → GlobalAveragePooling2D → StandardScaler → RBF-SVM  
**Key features**: Hybrid approach, interpretable features, subsamples for SVM (max 5K samples)  
**Use case**: Smaller datasets, feature importance analysis

```python
model = VGG19_SVM(n_classes=15, svm_C=1.0)
model.train(X_tr, y_tr)  # No validation needed (sklearn)
```

### 3. HybridCNN
**Type**: Custom CNN (Nature 2025 paper)  
**Input**: Images (160×160×3)  
**Architecture**: Conv(32)→BN→ReLU→MaxPool, Conv(64)→BN→ReLU→MaxPool, Conv(128)→BN→ReLU→GAP, Dense(256→15)  
**Key features**: Lightweight, good accuracy-speed tradeoff, batch norm stability  
**Use case**: Standard image classification

```python
model = HybridCNN(n_classes=15)
model.build().train(X_tr, y_tr, X_val, y_val, epochs=50)
```

### 4. KCNet
**Type**: Insect-inspired sparse random network (arxiv:2108.07554)  
**Input**: Features (1-D vector)  
**Architecture**: Sparse binary random weights (k=6 connections per hidden, 2000 hidden units) → winner-take-all (top 5%) → analytic ridge regression output  
**Key features**: No weight training, fast inference, biologically plausible  
**Use case**: Real-time edge inference (no GPU needed)

```python
model = KCNet(n_hidden=2000, k_connections=6, n_classes=15)
model.train(X_features_tr, y_tr)  # Analytic fit, no validation
```

### 5. FlyCaps
**Type**: Capsule network + Firefly optimization + ELM (IJRITCC 2023)  
**Input**: Images or Features  
**Architecture**:
- VGG19 backbone (frozen) or dense projection
- Primary capsule layer (squash routing)
- Firefly-optimised feature selection (multi-objective: accuracy + multilinearity + count)
- ELM head (analytic Moore-Penrose solution)

**Key features**:
- Sophisticated routing (squash activation)
- Metaheuristic feature selection
- Analytic output layer (fast training)
- Paper specifies: lr=0.0001, batch=30, epochs=120, dropout=0.1

**Use case**: State-of-the-art accuracy on plant disease, strong generalization

```python
model = FlyCaps(
    input_shape=(160, 160, 3),
    n_classes=15,
    n_caps=8,
    caps_dim=16,
    elm_hidden=256,
    firefly_n=20,
    firefly_iter=30
)
model.train(X_tr, y_tr)  # Image-based training
```

### 6. LiteCShuffle ⭐ DEPLOYED
**Type**: Lightweight attention CNN (Cogent F&A 2025)  
**Input**: Images (160×160×3)  
**Architecture**:
- Depthwise-separable convolutions (reduced filters per layer)
- Squeeze-and-Excitation (channel attention)
- Channel shuffle (spatial regularization)
- Adaptive pooling

**Key features**:
- **<50ms inference** on Raspberry Pi (CPU)
- **Smallest model** (< 5MB)
- Competitive accuracy (~90%+)
- Production-ready

**Deployed as**: `fieldbot/data/best_litecshuffle.keras`

```python
model = LiteCShuffle(input_shape=(160, 160, 3), base_channels=32, n_classes=15)
model.build().train(X_tr, y_tr, X_val, y_val, epochs=50)
```

### 7. YangViT
**Type**: SNN-inspired Visual Perception Network (IEEE TSMC 2024 proxy)  
**Input**: Images (160×160×3)  
**Architecture**:
- Rate-coded encoding (shallow CNN: Conv32→Conv64)
- Visual Perception Layer (VPL): excitatory projection → winner-take-all lateral inhibition
- Decision Making Layer (DML): 8-unit excitatory neurons → output softmax
- No spiking dynamics (TensorFlow limitation); captures architectural intent

**Key features**: Neuromorphic-inspired, competitive dynamics, biologically plausible  
**Use case**: Novel bio-inspired approaches, research

```python
model = YangViT(
    input_shape=(160, 160, 3),
    n_excitatory=64,
    k_wta=8,
    dml_units=8,
    n_classes=15
)
model.build().train(X_tr, y_tr, X_val, y_val, epochs=50)
```

### 8. HybridGrasshopperABC
**Type**: MLP proxy (Springer 2023 — paywall, not implemented in full)  
**Input**: Features (1-D vector)  
**Architecture**: Deep MLP with 4 hidden layers (512→256→128→output)  
**Key features**: Gradient-based optimization of GOA+ABC swarm algorithm (proxy only)  
**Use case**: Feature-based classification, comparison baseline

```python
model = HybridGrasshopperABC(feature_dim=100, n_classes=15)
model.build().train(X_features_tr, y_tr, X_val, y_val, epochs=50)
```

### 9. MantisSearch
**Type**: ELM with Mantis Search Algorithm (ScienceDirect 2023 — paywall, not implemented in full)  
**Input**: Features (1-D vector)  
**Architecture**: Random input weights + tanh hidden layer (1000 units) → analytic ridge regression output  
**Key features**: No training of hidden layer, fast inference, analytic solution  
**Use case**: Extreme Learning Machines, lightweight feature classifiers

```python
model = MantisSearch(n_hidden=1000, n_classes=15)
model.train(X_features_tr, y_tr)  # Analytic fit
```

---

## Training Data

### Expected Dataset Structure

```
data/
├── train/
│   ├── Bacterial_spot_Pepper_Bell/
│   │   ├── img1.jpg
│   │   └── ...
│   ├── healthy_Pepper_Bell/
│   ├── Early_blight_Potato/
│   └── ... (15 total classes)
├── val/
│   └── (same structure)
└── test/
    └── (same structure)
```

### Data Loader (data_loader.py)

**In-memory loading** (small datasets):
```python
from data_loader import ImageDataLoader
loader = ImageDataLoader("data/train", target_size=(160, 160))
X_train, y_train = loader.load_images()  # Returns numpy arrays
```

**Lazy HDF5 loading** (large datasets, low RAM):
```python
from data_loader import HDF5DataLoader
loader = HDF5DataLoader("data/train_images.h5", "data/train_labels.npy")
X_train = loader.get_lazy_array()  # Returns LazyH5Array (streams per batch)
y_train = loader.labels
```

### Disease Classes (15 total)

Organized by crop × disease:

**Pepper**:
- Bacterial_spot_Pepper_Bell
- healthy_Pepper_Bell

**Potato**:
- Early_blight_Potato
- Late_blight_Potato
- healthy_Potato

**Tomato**:
- Bacterial_spot_Tomato
- Early_blight_Tomato
- Late_blight_Tomato
- Leaf_Mold_Tomato
- Septoria_leaf_spot_Tomato
- Spider_mites_Two_spotted_spider_mite_Tomato
- Target_Spot_Tomato
- YellowLeaf__Curl_Virus_Tomato
- mosaic_virus_Tomato
- healthy_Tomato

---

## Preprocessing Pipeline (preprocessing.py)

Image preprocessing for all vision models:

```python
from preprocessing import PreprocessingPipeline
pipe = PreprocessingPipeline(target_size=(160, 160))
img_clean = pipe.process(image_array)
```

**Steps**:
1. **Cellular automaton denoise**: Reduces salt-and-pepper noise
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization): Enhances local contrast
3. **Wiener filter**: Noise reduction while preserving edges
4. **Background removal**: Segmentation mask → inpaint leaves only
5. **Resize**: to target_size (160×160)

---

## Feature Extraction & Selection

### Extract Features (feature_extraction.py)

```python
from feature_extraction import FeatureExtractor
extractor = FeatureExtractor(color_space="hsv")
features = extractor.extract_batch(X_preprocessed)  # (N, n_features)
```

**Feature types**:
- Color moments (R, G, B means/stds)
- Texture (LBP, SIFT descriptors)
- Morphology (area, perimeter, eccentricity)
- Spectral (Fourier descriptors)

### Feature Selection (feature_selection.py)

```python
from feature_selection import L1FeatureSelector, L2FeatureSelector
selector = L1FeatureSelector(n_features_keep=50)
X_selected = selector.fit_transform(X_features, y)
```

**Methods**:
- **L1** (Lasso): Sparsity-inducing
- **L2** (Ridge): Smooth penalty
- **Mutual Information**: Information-theoretic ranking

---

## Training Utilities

### GradInit (Initialization)
```python
from model_training import apply_gradinit
model = apply_gradinit(
    model,
    X_sample[:64],
    y_sample[:64],
    lr=0.01,
    n_steps=5
)
```

Improves initial gradient conditioning to speed training and improve convergence (CS231n technique).

### Importance Sampling Callback
```python
from model_training import ImportanceSamplingCallback
callback = ImportanceSamplingCallback(
    X_train[:2000],  # Subsample for RAM
    y_train[:2000],
    gamma=0.9,  # Momentum
    beta=0.9    # Bias correction
)
model.fit(callbacks=[callback, ...])
```

Weights hard-to-learn classes higher during training (CS231n bandit-based approach).

### Channel Pruning
```python
from model_training import ChannelPruner
pruner = ChannelPruner(prune_ratio=0.3)  # Remove 30% of channels
pruned = pruner.prune(model)
pruned.evaluate(X_test, y_test)
```

Magnitude-based structured pruning for efficiency.

---

## Experiments (experiments.py)

Run comprehensive benchmarks:

```bash
python experiments.py
```

**Included experiments**:
1. **All 9 models**: Training time, accuracy, inference speed
2. **Ablation studies**: Impact of preprocessing, feature selection
3. **Generalization**: Cross-dataset evaluation
4. **Efficiency**: Model size, RAM usage, latency

**Output**: JSON report with timing, accuracy curves, per-class metrics

---

## Model Persistence

### Save Model
```python
trainer = ModelTrainer("litecshuffle", n_classes=15)
trainer.run(X_tr, y_tr, X_val, y_val)
path = trainer.save("outputs/models")  # → "outputs/models/best_litecshuffle.keras"
```

**Keras models** (.keras format):
- Includes architecture, weights, optimizer state
- Loads seamlessly with custom layers (WinnerTakeAllLayer, SquashLayer)

**Analytic models** (.pkl format via pickle):
- KCNet, MantisSearch, VGG19_SVM, FlyCaps
- No Keras layer needed

### Load Model
```python
trainer = ModelTrainer.load(
    model_type="litecshuffle",
    path="outputs/models/best_litecshuffle.keras",
    n_classes=15
)
preds = trainer.predict(X_test)
```

---

## Inference Optimization

### Quantization (TensorFlow Lite)
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

Reduces model size by ~4× with minimal accuracy loss. Use on Raspberry Pi for **<20ms inference**.

### Batch Prediction
```python
predictions = trainer.predict(X_test)  # Automatically batches at 32 samples/batch
```

Streams inference through model in chunks to avoid OOM.

---

## Hyperparameter Tuning

Key parameters by model:

| Model | Key Hyperparams | Recommended Range |
|-------|-----------------|-------------------|
| SimpleVGG16 | epochs, batch_size, dropout | 50–100 epochs, 32 batch |
| VGG19_SVM | svm_C, kernel | C=0.1–10, kernel='rbf' |
| HybridCNN | batch_size, dropout | 32–64 batch, 0.3–0.5 dropout |
| KCNet | n_hidden, ridge_alpha | 1000–4000 hidden, 0.1–10 alpha |
| FlyCaps | firefly_n, firefly_iter | 20 fireflies, 30 iterations |
| LiteCShuffle | base_channels | 32–64 channels |
| YangViT | n_excitatory, k_wta | 64 excitatory, 8 top-k |
| Grasshopper | — | (fixed MLP) |
| Mantis | n_hidden, ridge_alpha | 500–2000 hidden, 0.1–10 alpha |

### Grid Search Example
```python
params_grid = {
    "n_classes": [15],
    "batch_size": [16, 32, 64],
    "epochs": [30, 50, 100]
}
for batch in params_grid["batch_size"]:
    for epochs in params_grid["epochs"]:
        trainer = ModelTrainer("hybrid_cnn", n_classes=15)
        trainer.run(X_tr, y_tr, X_val, y_val, epochs=epochs, batch_size=batch)
        print(f"batch={batch}, epochs={epochs}: {trainer.evaluate(X_test, y_test)}")
```

---

## Troubleshooting

### Model Training Crashes with OOM
**Cause**: Dataset too large to fit in GPU/CPU memory.  
**Fix**: Use LazyH5Array or reduce batch_size.
```python
loader = HDF5DataLoader("data/train_images.h5", "data/train_labels.npy")
X_lazy = loader.get_lazy_array()
trainer.run(X_lazy, y_train, X_val, y_val, batch_size=16)
```

### Low Accuracy on New Crops
**Cause**: Models trained only on Pepper/Tomato/Potato.  
**Fix**: Fine-tune with transfer learning or collect new labeled data.
```python
model.build()
model.model.layers[-1] = tf.keras.layers.Dense(n_new_classes, activation='softmax')
model.train(X_new, y_new, X_val, y_val, epochs=20)  # Lower epochs, higher LR
```

### Slow Inference on Raspberry Pi
**Cause**: Full-precision float32 model running on CPU.  
**Fix**: Quantize to int8 or use LiteCShuffle (native lightweight).
```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.int8]
tflite_model = converter.convert()
```

### Feature Model Training Fails
**Cause**: Feature dimension mismatch or missing preprocessing.  
**Fix**: Ensure features are normalized and match model input shape.
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_features_scaled = scaler.fit_transform(X_features)
model.train(X_features_scaled, y_train)
```

---

## Output & Results

Training results saved to `outputs/`:

```
outputs/
├── models/
│   ├── best_vgg16.keras
│   ├── best_litecshuffle.keras          ← Deployed model
│   ├── best_hybrid_cnn.keras
│   ├── best_kcnet.pkl
│   ├── best_flycaps.pkl
│   └── ...
├── cache/
│   ├── features.h5                      ← Extracted features
│   └── images.h5                        ← Preprocessed images
├── results_subset.json                  ← Experiment results
└── model_retrain/
    ├── hybrid_cnn_WOAAPSO_features.keras  ← Ablation variants
    ├── litecshuffle_image.keras
    └── ...
```

### Results JSON Format
```json
{
  "vgg16": {
    "accuracy": 0.92,
    "loss": 0.23,
    "training_time_s": 1234,
    "per_class_precision": {...},
    "per_class_recall": {...}
  },
  "litecshuffle": {
    "accuracy": 0.94,
    "loss": 0.18,
    ...
  }
}
```

---

## References

1. **LiteCShuffle** (Cogent F&A 2025): Lightweight CNN with channel attention
2. **FlyCaps** (IJRITCC 2023): Capsule network + Firefly optimization + ELM
3. **HybridCNN** (Nature 2025): WOA-APSO optimized custom architecture
4. **KCNet** (arxiv:2108.07554): Kenyon cell-inspired sparse networks
5. **YangViT** (IEEE TSMC 2024): SNN-inspired visual perception layer
6. **GradInit** (CS231n): Gradient-based initialization
7. **Importance Sampling** (Loshchilov et al., CS231n): Hard-example weighting
8. **Channel Pruning**: Magnitude-based structured pruning for efficiency

---

## License

GNU Affero General Public License v3 License. See root LICENSE file.
