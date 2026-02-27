# Bank Marketing — Term Deposit Subscription Prediction

A supervised machine learning project that predicts whether a bank client will subscribe to a term deposit, based on their demographic and campaign data.

---

## Overview

Banks run telephone marketing campaigns to get clients to subscribe to term deposits. Not every client says yes, and calling people who will never subscribe wastes time and money. This project builds a classifier that helps identify which clients are most likely to subscribe — so the bank can focus its effort where it matters.

**Dataset:** UCI Bank Marketing Dataset — 45,211 records, 16 features
**Target:** `y` — did the client subscribe? (`yes` / `no`)
**Class split:** ~88% No, ~12% Yes (imbalanced)

---

## Project Structure

```
ML_project/
├── ml_final_p_bank_marketing.ipynb   # main notebook
├── bank-full.csv                     # dataset
└── README.md
```

---

## What's Inside the Notebook

| Section | Description |
|---------|-------------|
| 1. Load & Preview | Load the CSV, check shape and dtypes |
| 2. EDA | Target distribution, correlation heatmap, histograms, categorical plots |
| 3. Feature Engineering | New binary flags, balance/age/campaign buckets |
| 4. Preprocessing | Drop data-leaking `duration`, encode target, train/test split (70/30, stratified) |
| 5. Baseline | Dummy classifier — sets the minimum bar to beat |
| 6. Pipeline Setup | `ColumnTransformer` + `Pipeline` for clean preprocessing |
| 7. Model Comparison | Cross-validate Logistic Regression, Decision Tree, Random Forest, SVC |
| 8. Hyperparameter Tuning | `GridSearchCV` on Random Forest, optimising for Recall |
| 9. Final Evaluation | Test-set metrics, confusion matrix, classification report |
| 10. Feature Importance | Top 15 features driving the model's predictions |

---

## Results

| Metric | Dummy Baseline | Best Random Forest |
|--------|:--------------:|:------------------:|
| Accuracy | 88.2% | 80.7% |
| Recall (Yes) | 0.0% | ~58% |
| Precision (Yes) | 0.0% | ~32% |
| F1-Score (Yes) | 0.0% | ~41% |

> The dummy classifier looks accurate but catches **zero subscribers**. The tuned Random Forest identifies roughly 6 in 10 actual subscribers — a meaningful improvement for a real campaign.

---

## Key Findings

- **Call-related features dominate**: number of contacts and previous campaign outcome are the strongest predictors
- **Financial features matter**: account balance and age are highly informative
- **Class imbalance is a real challenge**: using `class_weight="balanced"` and optimising for Recall over Accuracy gave much better results on the minority class

---

## Tech Stack

- **Python 3.12**
- **pandas** — data loading and manipulation
- **numpy** — numerical operations
- **matplotlib / seaborn** — visualisations
- **scikit-learn** — pipelines, models, cross-validation, GridSearchCV

---

## How to Run

1. Clone the repo and navigate to the project folder
2. Make sure you have the dependencies installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

3. Open the notebook:

```bash
jupyter notebook ml_final_p_bank_marketing.ipynb
```

4. Run all cells from top to bottom (`Kernel → Restart & Run All`)

> **Note:** The GridSearchCV step tests a large parameter grid and may take a few minutes to complete.

---

## Dataset

**UCI Bank Marketing Dataset**
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

The data is related to direct marketing campaigns of a Portuguese banking institution. Campaigns were phone-based; often multiple contacts were needed to assess whether a client would subscribe to a bank term deposit.

---

## Possible Next Steps

- Threshold tuning to further improve the precision-recall trade-off
- Try SMOTE or other oversampling techniques to address class imbalance
- Test gradient boosting models (XGBoost, LightGBM) — often outperform Random Forest on tabular data
- Build a simple prediction API or Streamlit app around the best model
