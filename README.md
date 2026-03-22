<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=30&duration=3000&pause=1000&color=2E86C1&center=true&vCenter=true&width=700&lines=Bank+Marketing+ML+Project;Predicting+Term+Deposit+Subscriptions;Supervised+Machine+Learning+%F0%9F%8F%A6" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-Data_Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ECC71?style=for-the-badge)

<br/>

> **Can we predict who will say YES to a term deposit — before making the call?**

</div>

---

## What This Project Does

Banks run telephone marketing campaigns to get clients to subscribe to term deposits. Calling everyone is expensive and inefficient. This project trains a **Random Forest classifier** on 45,211 real bank records to predict which clients are most likely to subscribe — so the bank can focus its effort where it matters.

---

## Live Demo Preview

<div align="center">

```
┌─────────────────────────────────────────────────────────────────┐
│              BANK MARKETING — PREDICTION PIPELINE               │
│─────────────────────────────────────────────────────────────────│
│  Raw CSV (45,211 rows)                                          │
│       │                                                         │
│       ▼                                                         │
│  [EDA + Visualisation]  →  distributions, heatmaps, patterns   │
│       │                                                         │
│       ▼                                                         │
│  [Feature Engineering]  →  balance buckets, age bins, flags    │
│       │                                                         │
│       ▼                                                         │
│  [Sklearn Pipeline]     →  scale + encode + model (clean)      │
│       │                                                         │
│       ▼                                                         │
│  [GridSearchCV Tuning]  →  optimise for Recall (catch more!)   │
│       │                                                         │
│       ▼                                                         │
│  [Random Forest Model]                                          │
│       │                                                         │
│       ├──── Accuracy  80.7%  ████████████████░░░░              │
│       ├──── Recall    ~58%   ████████████░░░░░░░░              │
│       └──── F1-Score  ~41%   ████████░░░░░░░░░░░░              │
└─────────────────────────────────────────────────────────────────┘
```

</div>

---

## Results at a Glance

<div align="center">

| Metric | Dummy Baseline | Best Random Forest |
|--------|:--------------:|:------------------:|
| Accuracy | 88.2% | 80.7% |
| Recall (Yes) | 0% | ~58% |
| Precision (Yes) | 0% | ~32% |
| F1-Score (Yes) | 0% | ~41% |

</div>

> The dummy classifier *looks* accurate but catches **zero subscribers**. The tuned Random Forest identifies roughly **6 in 10 actual subscribers** — a meaningful lift for a real campaign.

---

## Project Structure

```
ML_project/
├── ml_final_p_bank_marketing.ipynb   # full analysis notebook
├── bank-full.csv                     # UCI dataset (45,211 records)
└── README.md
```

---

## Notebook Walkthrough

| # | Section | What Happens |
|---|---------|--------------|
| 1 | Load & Preview | CSV loaded, shape/dtypes inspected |
| 2 | EDA | Target distribution, heatmap, histograms, categorical plots |
| 3 | Feature Engineering | Binary flags, balance/age/campaign buckets |
| 4 | Preprocessing | Drop leaking `duration`, encode target, 70/30 stratified split |
| 5 | Baseline | Dummy classifier sets the minimum bar to beat |
| 6 | Pipeline | `ColumnTransformer` + `Pipeline` for reproducible preprocessing |
| 7 | Model Comparison | Cross-validate LR, Decision Tree, Random Forest, SVC |
| 8 | Tuning | `GridSearchCV` on Random Forest, optimising for Recall |
| 9 | Final Evaluation | Test-set metrics, confusion matrix, classification report |
| 10 | Feature Importance | Top 15 features driving predictions |

---

## Key Findings

- **Call history dominates**: number of contacts and previous campaign outcome are the strongest predictors
- **Financial features matter**: account balance and age are highly informative
- **Class imbalance is the core challenge**: `class_weight="balanced"` + Recall optimisation gave much better real-world results than chasing Accuracy

---

## Tech Stack

<div align="center">

![Python](https://img.shields.io/badge/-Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![pandas](https://img.shields.io/badge/-pandas-150458?style=flat-square&logo=pandas)
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=flat-square&logo=numpy)
![Matplotlib](https://img.shields.io/badge/-Matplotlib-11557C?style=flat-square)
![seaborn](https://img.shields.io/badge/-seaborn-4C72B0?style=flat-square)
![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)

</div>

---

## How to Run

```bash
# 1. Clone the repo
git clone https://github.com/raselmian03-alt/bank-marketing-ml.git
cd bank-marketing-ml

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Launch the notebook
jupyter notebook ml_final_p_bank_marketing.ipynb
```

Then run all cells: `Kernel → Restart & Run All`

> **Note:** The GridSearchCV step tests a large parameter grid — give it a couple of minutes.

---

## Dataset

**UCI Bank Marketing Dataset**
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)

Phone-based marketing campaigns from a Portuguese bank. Multiple contacts were often needed per client to determine whether they would subscribe to a term deposit.

---

## Possible Next Steps

- Threshold tuning for better precision-recall trade-off
- SMOTE / oversampling to address class imbalance
- Gradient boosting models (XGBoost, LightGBM)
- Streamlit app for live predictions

---

<div align="center">

**Made by [raselmian03-alt](https://github.com/raselmian03-alt)**

</div>
