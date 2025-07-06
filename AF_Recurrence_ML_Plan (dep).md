# AF Recurrence Prediction – Machine-Learning Plan

## 1. Objective
Develop a reproducible, "medical-grade" machine-learning suite that predicts atrial-fibrillation (AF) recurrence after first-time pulmonary-vein isolation (PVI) ablation **using only pre-procedure 12-lead ECGs**.

## 2. Data Sources
1. **Waveforms**: `carto_ecg_waveforms.npy`  
   • Shape: `(n_ecgs, 10 s × 250 Hz, 12)` → `(n_ecgs, 2500, 12)`  
   • Every ECG is already resampled to 250 Hz and trimmed/padded to 10 s.
2. **Metadata**: `carto_ecg_metadata_FULL.csv`  
   • Columns include at least: `index` (matching waveform index), `mrn`, `acquisition_date`, `sinus_rhythm`, `procedure_date`, outcome flags (`at_recurrence`, `af_recurrence`, `af_at_recurrence`) and corresponding `days_till_*` fields.

## 3. Cohort Selection
* **Sinus-rhythm filter**: `sinus_rhythm == 1`.
* **Pre-procedure ECG definition** (performed **before** or **on** ablation day):
  1. Time window variants – within **1 y**, **3 y**, **5 y** before `procedure_date`.
  2. If multiple ECGs per MRN satisfy the window, take **all** for modelling
  3. Always include the **first ECG recorded on ablation day** (even if `< 0 days`) if it exists.

Implementation hint: reuse logic sketched in `pvi-paroxysmal_data.py` (see attached snippet) to compute boolean `pre_ecg` per window.

## 4. Outcome Definitions
For each ECG, create binary labels for:
1. `at_recurrence`  
2. `af_recurrence`  
3. `af_at_recurrence`

…each at three thresholds:
* **1 y**  (`days_till_* ≤ 365`)
* **3 y**  (`≤ 1095`)
* **5 y**  (`≤ 1825`)
* **any** (ignore days, i.e., > 5 y counts as positive too)

This yields **12 label variants**. The pipeline will loop over them.

## 5. Feature Variants
1. **Lead-1 only**: shape `(2500, 1)`.
2. **All 12 leads**: shape `(2500, 12)`.

## 6. Data Augmentation (optional)
Sliding windows – **5 s window (1250 samples)** with **4 s overlap (1000 samples)** → 6 windows per ECG.  
Apply only on training folds to avoid leakage. The non-augmented pipeline uses full 10 s traces.

## 7. Train / Validation Strategy
1. **Strict hold-out**: Randomly reserve **20 % of entire ECG set** (stratified by label) – never seen during model/threshold selection.
2. Remaining **80 % internal data**:
   • Simple split: 80 % train / 20 % validation.
   • AND 5-fold cross-validation experiments.  
Ensure that ECGs from the **same MRN** are kept within the *same* fold / split to prevent patient leakage.

## 8. Class Imbalance Handling
* Prefer **class-weighted loss** (for ResNet) / `class_weight="balanced"`.
* Compare with simple **RandomOverSampler** (imblearn) inside a `Pipeline`.

## 9. Model Architectures
| Library | Model | Notes |
|---------|-------|-------|
| sktime-dl | `ResNetClassifier` *(1-D)* | Works for univariate or multivariate; small depth for dataset size. |
| sktime | `RocketClassifier` | Fast; good baseline; supports multivariate. |

Both accept numpy arrays shaped `(n_instances, n_timesteps, n_channels)`.

## 10. Evaluation Metrics
* Primary: **AUROC** (class-imbalance robust).  
* Secondary: Accuracy, Precision, Recall, F1, AUPRC.
* Report per-fold, hold-out, and aggregate mean ± CI.

## 11. Implementation Outline
1. `data_loading.py`  
   – Load CSV + NPY → `pandas.DataFrame`, `np.ndarray`.  
   – Merge on `index`.
2. `preprocess.py`  
   – Apply cohort filters & outcome generation.
3. `splitter.py`  
   – Patient-level stratified splitting (20 % hold-out) + CV fold generator.
4. `augment.py`  
   – Windowing function (numpy slicing).
5. `train.py`  
   – Argument-driven script (`uv run train.py --label at_recurrence_1y --leads 12 --augment yes --cv 5`).  
   – Builds `Pipeline(resampler?, transformer?, classifier)`; handles class weights; logs metrics.
6. `evaluate.py`  
   – Generates aggregated reports & ROC curves.
7. `requirements` handled via **uv** (`uv add sktime sktime-dl imbalanced-learn pandas numpy scikit-learn joblib rich` …).

## 12. Reproducibility & Best Practices
* Fix random seeds, document versions in `pyproject.toml` (managed by uv).
* Use `mlflow` or simple `json` logs for runs.
* Store trained models (`joblib`) & metrics under `results/<timestamp>/`.
* Add CLI help & README for usage.

## 13. Next Steps
1. Scaffold directory & scripts per outline.
2. Implement data loader & cohort filter; validate counts for each window.
3. Build splitter ensuring no MRN leakage.
4. Prototype RocketClassifier (fast) to validate pipeline end-to-end.
5. Iterate with ResNetClassifier & augmentation.
6. Review class-imbalance impact; tune thresholds if necessary.
7. Finalize documentation & sanity-checks before clinical interpretation. 


## Appendix

Snippet from pvi-paroxysmal_data.py showing pre-ecg selection

```# Function to determine if ECG is pre-procedure
def is_pre_ecg(row):
    mrn = row["mrn"]
    ecg_date = row["acquisition_date"]

    # Get procedure date for this MRN
    procedure_date = mrn_to_procedure_date.get(mrn)

    if procedure_date is None:
        return 0  # No procedure date found

    # Calculate time difference
    time_diff = procedure_date - ecg_date

    # Check if ECG is before procedure date and within cutoff period
    if time_diff.days >= 0 and time_diff.days <= (PRE_ECG_CUTOFF_YEARS * 365):
        return 1
    else:
        return 0


# Apply the function to create pre_ecg column
print("Creating pre_ecg column...")
ecg_data["pre_ecg"] = ecg_data.apply(is_pre_ecg, axis=1)

# For each MRN, find the earliest ECG before the procedure and mark it as pre_ecg=1
# But only if it's within the cutoff period, OR if there are no ECGs within the cutoff period
print("Identifying earliest ECGs for each MRN...")
for mrn in ecg_data["mrn"].unique():
    if mrn in mrn_to_procedure_date:
        procedure_date = mrn_to_procedure_date[mrn]

        # Get all ECGs for this MRN that are before the procedure date
        mrn_ecgs = ecg_data[
            (ecg_data["mrn"] == mrn) & (ecg_data["acquisition_date"] < procedure_date)
        ]

        if not mrn_ecgs.empty:
            # Check if any ECGs are already marked as pre_ecg=1 (within cutoff period)
            ecgs_within_cutoff = mrn_ecgs[mrn_ecgs["pre_ecg"] == 1]

            if ecgs_within_cutoff.empty:
                # No ECGs within cutoff period, so mark the earliest ECG as pre_ecg=1
                earliest_ecg_idx = mrn_ecgs["acquisition_date"].idxmin()
                ecg_data.loc[earliest_ecg_idx, "pre_ecg"] = 1```