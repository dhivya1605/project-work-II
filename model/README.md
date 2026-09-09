# SFOA–DLO–TabKANet Crop Recommendation Pipeline

Working, tested implementation of your architecture:

```
dataset.csv → SFOA (feature selection) → DLO (hyperparameter tuning) → TabKANet training → artifacts/
```

## Files

| File | What it does |
|---|---|
| `src/kan_layer.py` | `FastKANLayer` — the KAN building block (RBF-based learnable edge activations, the efficient/practical way to implement a KAN instead of raw B-splines). `NumericalKANEmbedding` embeds each numeric column independently through its own KAN layer. |
| `src/tabkanet.py` | `TabKANet` model: KAN embeddings → CLS token + positional embedding → Transformer encoder → MLP classifier head over your crop classes. |
| `src/sfoa.py` | Superb Fairy-wren Optimization Algorithm as a **binary wrapper feature selector**. Implements the paper's three behavioral phases (growth toward best solution, breeding/recombination, predator-avoidance escape jumps) over a population of candidate feature masks, fitness = weighted RF classification error + feature-count penalty. |
| `src/dlo.py` | Draco Lizard Optimizer as a **continuous hyperparameter optimizer** for TabKANet (embed_dim, n_heads, n_transformer_layers, ff_dim, dropout, learning_rate). Implements the paper's gliding (exploration) and perching/adaptive-refinement (exploitation) phases, with the gliding ratio decaying over iterations. |
| `src/train.py` | Orchestrates the full pipeline end-to-end and saves everything your Flask+SHAP dashboard needs. |

## Setup

```bash
pip install -r requirements.txt
```

## Running it on your data

1. Open `src/train.py`, edit the `CONFIG` block at the top:
   ```python
   "DATA_PATH": "dataset.csv",
   "FEATURE_COLUMNS": ["SOIL_PH", "N", "P", "K", "TEMP",
                       "WATERREQUIRED", "RELATIVE_HUMIDITY", "CROPDURATION"],
   "LABEL_COLUMN": "CROP",
   ```
   Point `DATA_PATH` at whichever synthetic dataset variant you're evaluating
   (Sobol / Beta / Copula / Hybrid, etc.) and adjust column names if yours
   differ.

2. Run:
   ```bash
   cd src
   python train.py
   ```

3. It prints SFOA's progress (features selected + fitness per iteration),
   then DLO's progress (best hyperparameters per iteration), then final
   training progress, then test accuracy.

## What gets saved to `artifacts/`

- `tabkanet_weights.pt` — trained model state dict
- `scaler.pkl` — the `StandardScaler` fit on the SFOA-selected columns (fit
  only on train+val, never on test — important for honest TSTR-style
  evaluation)
- `label_encoder.pkl` — maps crop names ↔ class indices
- `model_config.json` — selected columns, best hyperparameters, final
  test/val accuracy — everything you need to reconstruct
  `TabKANet(**hyperparameters)` for inference in your Flask app, and to know
  which columns to feed it (and in what order)

## Using it in your Flask/SHAP dashboard

```python
import json, joblib, torch
from tabkanet import TabKANet

cfg = json.load(open("artifacts/model_config.json"))
scaler = joblib.load("artifacts/scaler.pkl")
label_encoder = joblib.load("artifacts/label_encoder.pkl")

model = TabKANet(
    num_features=cfg["num_features"],
    num_classes=cfg["num_classes"],
    **cfg["hyperparameters"],
)
model.load_state_dict(torch.load("artifacts/tabkanet_weights.pt"))
model.eval()

# For a new input row, select cfg["selected_columns"] in the same order,
# scale with `scaler`, run through `model`, argmax -> label_encoder.inverse_transform(...)
# For SHAP: wrap model.forward in a function taking a numpy array and use
# shap.KernelExplainer or shap.DeepExplainer on the scaled numeric input.
```

## Notes on fidelity to the base papers

- **SFOA**: growth/breeding/predator-avoidance phases are reproduced
  behaviorally per Jia et al.'s description (Cluster Computing, 2025). The
  exact per-equation constants in their paper aren't reproduced verbatim here
  (I didn't have the full paper text) — if your review panel wants an
  equation-for-equation match, compare against the published pseudocode and
  I can adjust the update rules.
- **DLO**: gliding (exploration) vs. perching (exploitation) phases,
  controlled by a decaying gliding ratio, per Wang, X. (Evolutionary
  Intelligence, 2025). Same caveat as above applies.
- **TabKANet**: KAN layer here uses an RBF-based "FastKAN" approximation
  rather than raw B-splines — this is the standard practical substitution
  used in most working KAN implementations because it's far faster to train
  while preserving the "learnable per-edge activation function" idea that
  defines a KAN.

## Tuning knobs in `train.py`'s CONFIG

- `SFOA_POPULATION` / `SFOA_ITERATIONS` — increase for a more thorough
  feature search (slower)
- `DLO_POPULATION` / `DLO_ITERATIONS` / `DLO_PROBE_EPOCHS` — DLO trains a
  quick TabKANet for `DLO_PROBE_EPOCHS` per candidate hyperparameter set to
  score it; increase probe epochs for more reliable HPO signal at the cost
  of time
- `FINAL_EPOCHS` — full training length for the final model once
  hyperparameters are fixed
