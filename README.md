# EquiScreen
### A Fairness Auditing Pipeline for Clinical Machine Learning

> *Auditing a hypertension prediction model for demographic bias — built on NHANES 2017–2018 data.*

---

## Overview

EquiScreen is a 12-week end-to-end project that builds, audits, and applies bias mitigation to a hypertension prediction model. The core question: does the model perform equally well for all demographic groups, and if not, can we fix it?

**Key result:** An XGBoost classifier (AUC 0.728, Recall 0.873) was found to significantly underserve Non-Hispanic Black patients (AUC 0.680, TPR 0.735 vs overall 0.873). Class weight adjustment improved NHB TPR by 17.7 percentage points at a cost of 0.9pp in overall AUC — illustrating the fundamental fairness-accuracy tradeoff in clinical ML.

---

## Project Structure

```
equiscreen/
│
├── data/
│   ├── raw/                  # NHANES .XPT source files (not tracked)
│   └── processed/            # Cleaned CSVs, train/test splits
│
├── notebooks/
│   ├── week02_eda.ipynb              # EDA, pandas fundamentals
│   ├── week03_feature_engineering.ipynb  # Preprocessing pipeline
│   ├── week04_logistic_regression.ipynb  # Baseline LR model
│   ├── week05_07_xgboost.ipynb       # XGBoost, model selection
│   ├── week08_09_fairness_audit.ipynb    # Subgroup analysis + fairness metrics
│   ├── week10_bias_mitigation.ipynb  # 3 mitigation strategies
│   └── week11_singapore_analysis.ipynb   # Transferability analysis
│
├── reports/
│   ├── EquiScreen_Final_Report.docx
│   └── EquiScreen_Portfolio_Summary.docx
│
├── requirements.txt
└── README.md
```

---

## Results at a Glance

| Metric | Value |
|--------|-------|
| Dataset | NHANES 2017–2018, 5,313 adults |
| Hypertension prevalence | 32.1% |
| Final model | XGBoost |
| Overall AUC | 0.7284 |
| Overall Recall | 0.8731 |
| Worst subgroup AUC | 0.680 (Non-Hispanic Black) |
| Post-mitigation NHB TPR | 0.765 (+17.7pp) |

### Fairness Audit Summary

| Group | AUC | TPR | Fairness Criteria Passed |
|-------|-----|-----|--------------------------|
| Non-Hispanic White (ref.) | 0.740 | 0.871 | 4/4 |
| **Non-Hispanic Black** | **0.680** | **0.735** | **1/4** |
| Mexican American | 0.712 | 0.851 | 3/4 |
| Other Hispanic | 0.751 | 0.892 | 4/4 |
| Other / Mixed Race | 0.723 | 0.862 | 3/4 |

---

## Methodology

**Fairness Frameworks Applied:**
- **Equalized Odds** — equal TPR and FPR across groups (±0.10 tolerance)
- **Calibration** — predicted probabilities reflect actual event rates per group
- **Disparate Impact Ratio** — four-fifths rule (DIR ≥ 0.80)

**Bias Mitigation Strategies Tested:**
1. SMOTE oversampling (pre-processing)
2. Class weight adjustment — `scale_pos_weight=8.0` ✅ *selected*
3. Threshold adjustment to 0.3 (post-processing)

**Key methodological decision:** MAP and pulse pressure were removed from the feature set after identifying them as data leakage (mathematically derived from the BP values used to define the hypertension label). This dropped AUC from ~0.9997 to 0.7284 — representing honest model performance.

---

## How to Run

### 1. Clone the repo
```bash
git clone https://github.com/[your-username]/equiscreen.git
cd equiscreen
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download NHANES data
Download the following files from the [CDC NHANES website](https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2017):
- `DEMO_J.XPT` — Demographics
- `BMX_J.XPT` — Body Measurements
- `BPX_J.XPT` — Blood Pressure

Place them in `data/raw/`.

### 4. Run notebooks in order
Start from `week02_eda.ipynb` and work through sequentially. Each notebook saves outputs used by the next.

---

## Requirements

```
pandas>=1.5
numpy>=1.23
scikit-learn>=1.1
xgboost>=1.7
imbalanced-learn>=0.10
fairlearn>=0.8
matplotlib>=3.6
seaborn>=0.12
jupyter>=1.0
```

---

## Key Learnings

- **Aggregate metrics lie.** A model with AUC 0.73 overall can perform at 0.68 for a specific demographic group. Always audit subgroup performance.
- **No single fairness metric is sufficient.** Disparate Impact Ratio passed for all groups; Equalized Odds revealed significant violations for the same groups.
- **No silver bullet for bias.** All three mitigation strategies improved NHB TPR. None eliminated the gap. Each introduced tradeoffs.
- **Data leakage is subtle.** Engineered features that are mathematically derived from the label are leakage, even if they look like independent predictors.

---

## Ethics

CITI 'Data or Specimens Only Research' training completed February 2026 (valid until 2029). NHANES data is publicly available and does not require individual IRB approval for secondary analysis.

---

## Author

**Syafiq Khan** — Biomedical Engineering, National University of Singapore  
March 2026
