# Predicting Heart Disease Using Logistic Regression

**Author:** Andile Brian Sithole

---

## 📌 Overview

An end-to-end machine learning project that predicts the **presence of heart
disease** from 13 clinical features using **Logistic Regression**. The pipeline
covers exploratory data analysis, feature engineering, model training, and
clinical interpretation — achieving **86.9% test accuracy** with **91% recall**
on the positive (disease) class.

Built to demonstrate a **production-style ML workflow** in a high-stakes medical
screening context, where minimising false negatives is critical.

---

## 🎯 Key Results

| Metric | Value |
|---|---|
| **Test Accuracy** | **86.9 %** |
| Training Accuracy | 88.1 % |
| **Recall (Disease class)** | **0.91** ← minimises missed diagnoses |
| Precision (Disease class) | 0.85 |
| F1-Score (Disease class) | 0.88 |

Confusion matrix (30 test samples):

|                | Predicted No | Predicted Yes |
|----------------|--------------|---------------|
| **Actual No**  | 23           | 5             |
| **Actual Yes** | 3            | 30            |

> **Why this matters:** In medical screening, a missed disease case
> (false negative) is far costlier than a false alarm. A 91% recall means the
> model catches 9 out of 10 patients who actually have heart disease.

---

## 📊 Dataset

- **Source:** `data/heart.csv` — 303 patient records, 14 columns
- **Target:** `target` — 1 = heart disease present, 0 = no disease
- **Class balance:** ~54% disease, ~46% no disease (moderately balanced)

| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Age in years |
| `sex` | Binary | 1 = male, 0 = female |
| `cp` | Categorical | Chest pain type (0–3) |
| `trestbps` | Continuous | Resting blood pressure (mm Hg) |
| `chol` | Continuous | Serum cholesterol (mg/dL) |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dL |
| `restecg` | Categorical | Resting ECG result (0–2) |
| `thalach` | Continuous | Maximum heart rate achieved |
| `exang` | Binary | Exercise-induced angina |
| `oldpeak` | Continuous | ST depression induced by exercise |
| `slope` | Categorical | Slope of peak exercise ST segment |
| `ca` | Categorical | Number of major vessels coloured (0–3) |
| `thal` | Categorical | Thalassemia type |
| `target` | Binary | **Target variable** |

---

## 🔬 Methodology

### 1. Data Pre-processing
- **No missing values** — verified on load
- **Stratified 80/20 train–test split** — preserves class balance and prevents
  data leakage
- **One-hot encoding** for all categorical features (`sex, cp, fbs, restecg,
  exang, slope, ca, thal`) with `drop='first'` to avoid the dummy-variable trap
- **StandardScaler** for continuous features (`age, trestbps, chol, thalach,
  oldpeak`) to improve gradient-descent convergence
- All preprocessing wrapped in a **`ColumnTransformer`** fitted *only* on the
  training set — a critical safeguard against leakage

### 2. Exploratory Data Analysis
- Correlation heatmap of all 14 variables
- Distribution of `age` and `chol` split by disease status
- Heart-disease counts by gender
- Proportion of disease by resting ECG result

### 3. Model Training
- **Algorithm:** `LogisticRegression(max_iter=1000, random_state=42)`
- **Justification:** Binary classification, interpretable coefficients, well
  suited to small clinical datasets, standard baseline in medical ML

### 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion matrix
- Coefficient-based feature importance

---

## 📈 Feature Importance

Top predictors ranked by logistic-regression coefficient magnitude:

| Direction | Features |
|---|---|
| **Strongly positive** (favour disease) | `cp` (certain chest-pain types), `oldpeak`, `exang`, `ca`, `slope` |
| **Strongly negative** (favour no disease) | `thalach` (max heart rate), `sex` (male-biased sample), some `cp` |

These results align with clinical literature — reduced max heart rate,
exercise-induced angina, and ST depression are established markers of
cardiovascular risk.

---
