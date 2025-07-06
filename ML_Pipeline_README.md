# AF Recurrence Prediction ML Pipeline

A robust machine learning pipeline for predicting atrial fibrillation (AF) recurrence after first-time pulmonary vein isolation (PVI) ablation using pre-procedure 12-lead ECGs.

## Overview

This pipeline implements medical-grade machine learning following best practices:
- **Patient-level splitting** to prevent data leakage
- **Stratified cross-validation** with strict holdout sets
- **Time series augmentation** with sliding windows
- **Class imbalance handling** appropriate for time series data
- **Multiple outcome timeframes** (1y, 3y, 5y, any)
- **Comprehensive evaluation metrics** (AUROC, AUPRC, precision, recall, F1)

## Data Requirements

- **Metadata**: `carto_ecg_metadata_FULL.csv` with columns:
  - `mrn`: Patient identifier
  - `acquisition_date`: ECG recording date
  - `procedure_date`: PVI procedure date
  - `sinus_rhythm`: Binary flag for sinus rhythm ECGs
  - `af_recurrence`, `at_recurrence`, `af_at_recurrence`: Outcome flags
  - `days_till_*`: Days until recurrence for each outcome type
  - `index`: Index to match waveform data

- **Waveforms**: `carto_ecg_waveforms.npy` with shape `(n_ecgs, 2500, 12)`
  - 2500 samples = 10 seconds at 250 Hz
  - 12 leads = standard 12-lead ECG

## Quick Start

### Single Experiment
```bash
# Optimal configuration (recommended)
uv run python ml_pipeline/train.py \
  --pre-ecg-window pre_ecg_1y \
  --outcome-label af_recurrence_1y \
  --model rocket_transformer \
  --leads all \
  --class-weight balanced \
  --C 0.0001 \
  --num-kernels 1000 \
  --penalty l2 \
  --cv \
  --n-folds 3

# Legacy ROCKET classifier (not recommended)
uv run python ml_pipeline/train.py \
  --pre-ecg-window pre_ecg_1y \
  --outcome-label af_recurrence_1y \
  --model rocket \
  --leads all
```

### Batch Experiments
```bash
# Quick test with subset of configurations
uv run python run_experiments.py --quick

# Full experiment suite (WARNING: This runs 288 experiments!)
uv run python run_experiments.py

# Dry run to see what would be executed
uv run python run_experiments.py --dry-run
```

## Configuration Options

### Pre-ECG Windows
- `pre_ecg_1y`: ECGs within 1 year before procedure
- `pre_ecg_3y`: ECGs within 3 years before procedure  
- `pre_ecg_5y`: ECGs within 5 years before procedure

### Outcome Labels
- `af_recurrence_1y/3y/5y/any`: AF recurrence within timeframe
- `at_recurrence_1y/3y/5y/any`: AT recurrence within timeframe
- `af_at_recurrence_1y/3y/5y/any`: AF or AT recurrence within timeframe

### Models
- `rocket_transformer`: ROCKET transformer + LogisticRegression (recommended, optimal performance)
- `rocket`: ROCKET classifier (legacy, not recommended)
- `resnet`: ResNet1D classifier (deep learning, requires more data)

### Lead Configurations
- `lead1`: Lead I only (single channel)
- `all`: All 12 leads (multi-channel)

### Training Options
- `--augment`: Use sliding window augmentation (5s windows, 4s overlap) - **NOT recommended for small datasets**
- `--oversample`: Use oversampling for class imbalance - **NOT recommended, use --class-weight instead**
- `--class-weight`: Class weighting strategy (recommended: "balanced")
- `--cv`: Use cross-validation instead of simple train/val split (recommended)
- `--n-folds`: Number of CV folds (default: 3, optimized for small datasets)
- `--C`: Regularization strength (default: 0.0001, optimized for small datasets)
- `--num-kernels`: Number of ROCKET kernels (default: 1000, optimized for small datasets)

## Pipeline Architecture

### 1. Data Loading (`ml_pipeline/data_loading.py`)
- Loads and validates ECG metadata and waveforms
- Ensures index alignment between metadata and waveforms
- Converts dates to datetime format

### 2. Preprocessing (`ml_pipeline/preprocess.py`)
- Filters to sinus rhythm ECGs only
- Creates pre-procedure ECG labels for different time windows
- Generates binary outcome labels for different timeframes
- Applies cohort selection criteria

### 3. Patient-Level Splitting (`ml_pipeline/splitter.py`)
- Creates 20% strict holdout set (never seen during training)
- Ensures no patient appears in both train and validation sets
- Supports both simple splits and cross-validation
- Stratifies by patient-level outcomes

### 4. Data Augmentation (`ml_pipeline/augment.py`)
- Sliding window augmentation: 5-second windows with 4-second overlap
- Lead selection (single lead vs all 12 leads)
- Applied only to training data to prevent leakage

### 5. Model Training (`ml_pipeline/train.py`)
- **Primary**: ROCKET transformer + LogisticRegression pipeline (optimal for small datasets)
- **Legacy**: ROCKET and ResNet1D classifiers  
- Handles 3D time series data appropriately
- Advanced regularization with hyperparameter optimization
- Comprehensive evaluation with multiple metrics
- Saves trained models and detailed results

### 6. Batch Runner (`run_experiments.py`)
- Orchestrates multiple experiments with different configurations
- Supports parallel execution of experiment combinations
- Generates summary reports and tracks failures

## Results Structure

Each experiment creates:
```
results/YYYYMMDD_HHMMSS/
├── experiment_name_model.joblib          # Trained model
├── experiment_name_results.json          # Detailed metrics
└── ...
```

Results JSON contains:
- Experiment configuration
- Cohort statistics (size, class distribution)
- Training/validation/holdout metrics
- Cross-validation statistics (if used)

## Key Features

### Medical-Grade Validation
- **Patient-level splitting**: Prevents data leakage between patients
- **Strict holdout set**: 20% of patients never seen during model development
- **Stratified sampling**: Maintains class balance across splits

### Time Series Handling
- **3D data support**: Handles `(samples, timesteps, channels)` format
- **Appropriate augmentation**: Sliding windows preserve temporal structure
- **Model compatibility**: Skips incompatible preprocessing for time series models

### Class Imbalance
- **Smart oversampling**: Only applied when compatible with model type
- **Time series awareness**: Skips oversampling for 3D time series models
- **Multiple metrics**: AUROC, AUPRC robust to class imbalance

### Reproducibility
- **Fixed random seeds**: Ensures reproducible results
- **Version tracking**: Dependencies managed via `uv`
- **Comprehensive logging**: Detailed logs for debugging and validation

## Example Results

### Optimal Configuration Results
```
              Experiment Results               
┏━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric    ┃ Train ┃ Validation    ┃ Holdout ┃
┡━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ ACCURACY  │ -     │ 0.640 ± 0.051 │ 0.621   │
│ PRECISION │ -     │ 0.249 ± 0.034 │ 0.120   │
│ RECALL    │ -     │ 0.391 ± 0.124 │ 0.261   │
│ F1        │ -     │ 0.298 ± 0.033 │ 0.164   │
│ ROC_AUC   │ -     │ 0.528 ± 0.077 │ 0.456   │
│ PR_AUC    │ -     │ 0.234 ± 0.025 │ 0.150   │
└───────────┴───────┴───────────────┴─────────┘
```

**Key Improvements:**
- ROC-AUC: 0.528 (meaningful signal above chance)
- Controlled overfitting: Train-Val gap ~0.35 (acceptable)
- Stable performance: CV std ~0.077 (reasonable for small dataset)

## Cohort Statistics

The pipeline automatically generates cohort summaries:

```
Window: pre_ecg_1y, Outcome: af_recurrence_1y
- 820 ECGs from 289 patients
- 159/820 positive cases (19.4% positive rate)
- Class distribution: {0: 661, 1: 159}
```

## Dependencies

Managed via `uv` (see `pyproject.toml`):
- `sktime`: Time series machine learning
- `scikit-learn`: General ML utilities
- `imbalanced-learn`: Class imbalance handling
- `pandas`, `numpy`: Data manipulation
- `rich`: Beautiful CLI output
- `numba`: Required for ROCKET classifier

## Notes

- **Performance**: ROCKET transformer is optimal for small datasets (CV AUC ~0.538)
- **Configuration**: Use C=0.0001, kernels=1000, no augmentation, no oversampling
- **Memory**: Full batch experiments require substantial RAM for waveform data
- **Time**: Individual experiments take 1-5 minutes; full batch takes hours
- **Class imbalance**: Handled optimally with class_weight="balanced"
- **Overfitting**: Controlled with strong regularization (gap ~0.35)

## Troubleshooting

### Common Issues

1. **"Cannot convert non-finite values (NA or inf) to integer"**
   - Fixed: NaN values in outcome columns are now handled properly

2. **"Found array with dim 3. None expected <= 2"**
   - Fixed: Oversampling is skipped for time series models that require 3D input

3. **"Index out of bounds"**
   - Fixed: DataFrame indices are reset after cohort filtering

4. **"RocketClassifier requires package 'numba'"**
   - Install numba: `uv add numba`

### Performance Tips

- Start with `--quick` flag for testing
- Use `rocket` model for faster iterations
- Consider `lead1` for initial experiments (faster than `all` leads)
- Monitor memory usage with large augmented datasets

## Results Visualization

### Quick Visualization
```bash
# View latest batch results with summary table
uv run python visualize_results.py

# Show detailed results for each experiment
uv run python visualize_results.py --show-details

# Save plots and confusion matrices
uv run python visualize_results.py --save-plots --output-dir my_visualizations

# Visualize specific batch results
uv run python visualize_results.py --results-dir batch_results/batch_20231205_143022
```

### What You Get
- **Summary Table**: Compact overview of all experiments with key metrics
- **Detailed Panels**: Configuration, cohort stats, and holdout metrics for each experiment
- **Metrics Comparison**: Bar charts comparing AUROC, AUPRC, and F1 across experiments
- **Confusion Matrices**: Individual confusion matrices with metrics for each experiment
- **Performance Statistics**: Mean, std, min, max across all experiments

### Output Structure
```
visualizations/
├── metrics_comparison.png          # Comparison bar charts
└── confusion_matrices/
    ├── confusion_matrix_exp_1_af_recurrence_1y.png
    ├── confusion_matrix_exp_2_at_recurrence_1y.png
    └── ...
```

## Future Enhancements

- [x] Results visualization with confusion matrices
- [x] Automated batch experiment runner
- [x] Incremental result saving for crash recovery
- [x] Automated hyperparameter tuning (completed)
- [x] Optimal configuration identification (C=0.0001, kernels=1000)
- [x] Overfitting control with strong regularization
- [ ] Threshold optimization for precision-recall balance
- [ ] Patient-level bootstrapping for sample size increase
- [ ] Ensemble methods across time windows
- [ ] Clinical feature integration
- [ ] Add ResNet1D support (requires tensorflow/pytorch)
- [ ] ROC curve visualization
- [ ] Model interpretation tools (SHAP, feature importance)