# 📈 Improving AF/AT Recurrence Prediction on Small, Imbalanced ECG Data

## 1. Quick Diagnosis of Why ROC≈0.50

| Hypothesis | Evidence to Gather | How to Check |
|------------|-------------------|--------------|
| 1. **Severe class imbalance not handled** | ROCKET/ResNet pipeline *silently* skips `RandomOverSampler` because data are 3-D | Inspect log: `logger.warning("Oversampling not supported with rocket ..." )` → confirm minority ratio in `results["class_distribution"]` |
| 2. **Label / cohort mismatch** (e.g. pre-ECG window too far from recurrence event) | High label noise will keep AUC at chance | Plot label distribution vs. time-to-event; sanity-check a few records |
| 3. **Data leakage + small folds** cause overfit (train AUC≫val/holdout) | CV summary already logged—compare `train_metrics` to `val_metrics` | If train AUC≫0.9 but val≈0.5 → leakage/overfit |
| 4. **Model choice not suited** (ROCKET logistic is linear; signal may be subtle) | Try very small baseline (heartbeat averages) to see if any signal exists | If even deep networks fail, signal might truly be weak |

## 2. Guiding Principles for Improvements

1. **Keep pipeline simple & reproducible** – prefer scikit-learn primitives over bespoke deep nets.
2. **Exploit every label** but **never leak patients** across splits.
3. **Treat imbalance explicitly** via *sample weights* or *post-ROCKET oversampling* rather than hacking 3-D resampling.
4. **Measure properly** – use PR-AUC in addition to ROC-AUC; always compare to both *majority* and *random* baselines.

## 3. Minimal, High-Impact Changes

### 3.1 Swap `RocketClassifier` for `RocketTransformer → LogisticRegression`

```python
from sktime.transformations.panel.rocket import Rocket
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler

rocket = Rocket(num_kernels=10_000, random_state=42)
os = RandomOverSampler(random_state=42)
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
pipe = make_pipeline(rocket, os, logreg)
```

Why?
* `RocketClassifier` wraps a **fixed** logistic reg **without** class_weight. Breaking it apart lets us a) oversample **after** flattening to 2-D, b) tune `C`, and c) inject `class_weight="balanced"`.

### 3.2 Instance-level Sample Weighting (no oversampling needed)

If oversampling blows up memory, simply pass `sample_weight` when calling `fit()` on classifiers that support it.

### 3.3 Stratified, Grouped CV with Larger Folds

Use `StratifiedGroupKFold` (scikit-learn ≥1.2) to keep patient groups intact **and** balance labels across folds.

### 3.4 Hyper-Parameter Search (but lightweight)

Grid to explore and keep runtimes reasonable:

| Parameter | Values |
|-----------|--------|
| `num_kernels` | 2k, 10k (default 10k), 20k |
| `C` (logistic) | 0.1, 1, 10 |
| `class_weight` | `balanced`, custom {0:1,1:3} |

Use `RandomizedSearchCV(n_iter=8)` to finish in minutes.

### 3.5 Data-Level Tactics

1. **Patient Bootstrapping:** duplicate minority-class patients at the *patient* level so all their ECGs stay together ➜ avoid leakage.
2. **Noise Filtering:** drop recordings with missing leads / extreme artifacts before training.
3. **Time-Window Selection:** start with the closest window (`pre_ecg_1y`) where the signal is likely strongest; only add longer windows after success.

## 4. Concrete Step-by-Step Plan for the LLM

1. **Add utilities**
   1. Implement `scripts/analyse_class_balance.py` to print positive-ratio per window/outcome.
   2. Add `utils/cv_helpers.py` with `StratifiedGroupKFold` wrapper.
2. **Refactor `train.py`**
   1. Replace `get_model()` rocket branch with transformer + logistic approach above.
   2. Insert optional `--class-weight balanced` flag.
   3. Allow `--sample-weight` to toggle passing weights instead of oversampling.
3. **Implement Hyper-Parameter Search** (only when `--tune` flag passed):
   * Use `RandomizedSearchCV` on the pipeline; scoring = `average_precision`.
4. **Update `run_experiments.py`**
   * Add flags: `--tune`, `--class-weight`.
   * Reduce config grid size; focus on best window/outcome first.
5. **Validation Checks**
   * Write `notebooks/01_error_analysis.ipynb` to plot ROC & PR curves per fold and confusion matrices.
   * If ROC still ≈0.5, revisit label/censoring logic in `preprocess.py`.
6. **Documentation**
   * Create `docs/training_guidelines.md` summarizing best practices & the results of balance analysis.

## 5. Milestones & Expected Gains

| Milestone | Metric Target |
|-----------|---------------|
| After class weighting | ROC-AUC ≥0.60 |
| After hyper-parameter search | ROC-AUC ≥0.65, PR-AUC ↑ |
| After data cleaning & window tuning | ROC-AUC ≥0.70 |

These are realistic on small ECG datasets (literature: 0.68–0.75).

## 6. Future Ideas (if still plateauing)

* **Ensemble Averaging across Windows** – majority vote over multiple ECGs per patient.
* **Self-Supervised Pre-training** on all sinus rhythm ECGs ➜ finetune on recurrence labels (requires extra code, so postpone).
* **Feature Attribution Checks** with `SHAP` to verify the model sees physiologically plausible patterns.

---
*Prepared 2025-07-06 — v1.* 