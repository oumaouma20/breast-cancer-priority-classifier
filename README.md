# 🩺 Predictive Analytics for Resource Allocation in Breast-Cancer Care  
*(Task 3 — Kaggle Breast Cancer Wisconsin Diagnostic Dataset)*

A machine-learning workflow that classifies breast-cancer cases into **high, medium, or low priority** to help hospitals allocate diagnostic and treatment resources more effectively.

---

## 📂 Table of Contents
1. [Project Motivation](#project-motivation)  
2. [Dataset](#dataset)  
3. [Specific Objectives](#specific-objectives)  
4. [Methodology](#methodology)  
5. [Results](#results)  
6. [Repository Structure](#repository-structure)  
7. [Environment & Installation](#environment--installation)  
8. [Running the Notebook](#running-the-notebook)  
9. [Future Work](#future-work)  
10. [License](#license)

---

## Project Motivation
Hospitals often face bottlenecks when deciding which breast-cancer cases require immediate attention. By predicting **priority levels (high / medium / low)** from diagnostic measurements, clinicians can triage patients faster and deploy limited resources (e.g., biopsy slots, imaging time, oncologist review) where they matter most.

---

## Dataset
- **Name:** *Breast Cancer Wisconsin (Diagnostic)*  
- **Source:** Kaggle (`uciml/breast-cancer-wisconsin-data`)  
- **Size:** 569 rows × 32 columns (30 numeric features + labels)  
- **Features:** Mean, worst, and standard-error measurements of cell nucleus radius, perimeter, area, concavity, texture, etc.  
- **Label Engineering:**  
  - `high`  → Malignant (`diagnosis == "M"`)  
  - `medium` → Benign with radius above benign median  
  - `low`  → Benign with radius below benign median  

---

## Specific Objectives
1. **Data Understanding** – load dataset, inspect distributions, flag quality issues.  
2. **Data Preparation** – clean, encode 3-class priority, scale features.  
3. **Dataset Partitioning** – stratified train/validation/test splits.  
4. **Model Development** – train a *Random Forest* (baseline Logistic Reg. optional).  
5. **Model Evaluation** – report Accuracy & Macro F1, visualize confusion matrices.  
6. **Feature Interpretation** – plot top feature importances.  
7. **Deployment-Readiness** – notebook formatted with clear narrative & code.  
8. **Deliverables** – annotated `.ipynb` + metrics summary for exec report.

---

## Methodology
| Step | Description |
|------|-------------|
| **Exploratory Analysis** | Checked distributions, class balance (benign ≈ 63%, malignant ≈ 37%), no missing values. |
| **Label Engineering** | Added `priority` column (`high / medium / low`) based on diagnosis and radius. |
| **Pre-processing** | Dropped *id* & unnamed column, applied `StandardScaler`. |
| **Model** | `RandomForestClassifier` (`class_weight="balanced"`, default hyper-params). |
| **Validation** | 80 / 20 stratified split; metrics computed on test set. |
| **Interpretability** | Used impurity-based feature importances (top 10 plotted). |

---

## Results
| Metric | Score |
|--------|-------|
| **Accuracy** | **0.965** |
| **Macro F1-Score** | **0.965** |

**Class-wise performance**

| Priority | Precision | Recall | F1 |
|----------|-----------|--------|----|
| High     | 0.97 | 0.93 | 0.95 |
| Medium   | 0.95 | 0.97 | 0.96 |
| Low      | 0.97 | 1.00 | 0.99 |

**Top 5 predictive features**

1. `radius_mean`  
2. `area_worst`  
3. `perimeter_mean`  
4. `area_mean`  
5. `perimeter_worst`

> **Impact:** The model offers reliable, interpretable triage guidance—no low-priority cases were misclassified as high, reducing the risk of over-allocating scarce resources.

---

## Repository Structure
├── README.md
├── breast_cancer_priority.ipynb # Main notebook (deliverable)
└── data/
└── data.csv # Raw Kaggle dataset (after unzip)


---

## Environment & Installation
```bash
# Clone repo
git clone https://github.com/<your-user>/breast-cancer-priority.git
cd breast-cancer-priority

# Create env (conda or venv)
conda create -n bc_priority python=3.11
conda activate bc_priority

# Install requirements
pip install -r requirements.txt

Running the Notebook
   Download kaggle.json and place in project root.

   Open breast_cancer_priority.ipynb in Jupyter or Colab.

    Run all cells (Runtime → Run all).

    Review results and export as HTML/PDF if needed.

Future Work
    Hyperparameter tuning (GridSearchCV, Bayesian search)

     Test additional models (XGBoost, LightGBM)

     Build REST API for real-time inference in hospital systems

    Validate on external datasets to assess generalizability

License
This project is released under the MIT License. Feel free to use and adapt for academic or clinical purposes.
Dataset license remains with the original Kaggle/UCI contributors.


