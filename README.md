# 🩺 Fetal Health Classifier

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://fetal-health-classifier-tanay.streamlit.app/)

A multi-model classification web application that predicts fetal health from Cardiotocogram (CTG) exam features. Built as an end-to-end ML deployment exercise for BITS Pilani M.Tech AIML — Machine Learning Assignment 2.

**🔗 Live App:** [fetal-health-classifier-tanay.streamlit.app](https://fetal-health-classifier-tanay.streamlit.app/)
**📦 GitHub Repository:** [T-Shandilya-2025ac05260/fetal-health-classifier](https://github.com/T-Shandilya-2025ac05260/fetal-health-classifier)

---

## a. Problem Statement

Cardiotocograms (CTGs) monitor fetal heart rate and uterine contractions during pregnancy, and are a primary tool for detecting fetal distress. Manual interpretation of CTG traces is time-consuming and prone to observer variability. This project builds and compares five machine-learning classifiers that predict fetal health status — **Normal**, **Suspect**, or **Pathological** — from 21 numerical features extracted from CTG exams. The goal is to help obstetric decision-making by flagging cases that need clinical attention.

The trained models are wrapped in an interactive Streamlit web app where users upload test data and see per-model metrics, confusion matrices, per-row prediction inspection, and feature importance — all in one dashboard.

---

## b. Dataset Description

- **Source:** [Kaggle — Fetal Health Classification](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification) by Andrew Mvd
- **Records:** 2,126 CTG exams
- **Features:** 21 numerical CTG-derived measurements (baseline fetal heart rate, accelerations, decelerations, uterine contractions, and histogram-based statistics of the FHR signal)
- **Target:** `fetal_health` — 3 classes labelled by expert obstetricians
  - `1` = Normal (1,655 records, ~78%)
  - `2` = Suspect (295 records, ~14%)
  - `3` = Pathological (176 records, ~8%)
- **Missing values:** None
- **Class imbalance:** Significant — the dataset is heavily skewed toward Normal cases, which is why we use macro-averaged Precision/Recall/F1 and MCC as our primary metrics rather than raw accuracy.

### Preprocessing pipeline

1. Stratified 80/20 train-test split (`random_state=42`) to preserve class ratios in both sets.
2. Feature standardization using `StandardScaler` (fit on train, transform on test — no leakage).
3. `class_weight='balanced'` on models that support it (Logistic Regression, Decision Tree, Random Forest) to counteract the imbalance.

---

## c. GitHub Repository Link

**Repository:** [https://github.com/T-Shandilya-2025ac05260/fetal-health-classifier](https://github.com/T-Shandilya-2025ac05260/fetal-health-classifier)

Contents:
- `app.py` — Streamlit application (dark-themed dashboard with 3 interactive tabs)
- `train_models.ipynb` — Training notebook: EDA, preprocessing, model training, metric computation, artifact saving
- `test_data.csv` — 426-row test set (stratified subset) for uploading to the live app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Custom dark theme configuration
- `model/` — Pickled model artifacts (5 trained models + fitted scaler)
- `confusion_matrices.png` — Composite visualization of confusion matrices for all 5 models

---

## d. Models Used

Five classification algorithms were trained on the same preprocessed training set and evaluated on a held-out test set (426 samples). All metrics use **macro-averaging** to give equal weight to each of the three classes — critical for interpreting performance on imbalanced data.

### Comparison Table — Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8568 | 0.9602 | 0.7406 | 0.8497 | 0.7798 | 0.6868 |
| Decision Tree | 0.8873 | 0.8786 | 0.8150 | 0.8496 | 0.8268 | 0.7188 |
| kNN | 0.8709 | 0.9563 | 0.8090 | 0.6900 | 0.7374 | 0.6168 |
| Naive Bayes | 0.8099 | 0.8759 | 0.6500 | 0.7011 | 0.6569 | 0.5737 |
| **Random Forest (Ensemble)** | **0.9319** | **0.9812** | **0.8920** | **0.8625** | **0.8764** | **0.8097** |

**Random Forest wins on all six metrics.** The bolded row indicates the top performer per column.

### Hyperparameter choices

| Model | Key hyperparameters |
|---|---|
| Logistic Regression | `max_iter=1000`, `class_weight='balanced'`, `random_state=42` |
| Decision Tree | `max_depth=8`, `class_weight='balanced'`, `random_state=42` |
| kNN | `n_neighbors=7` |
| Naive Bayes | Gaussian variant (default parameters) |
| Random Forest | `n_estimators=200`, `max_depth=12`, `class_weight='balanced'`, `random_state=42` |

---

## Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| **Logistic Regression** | Achieves the second-highest AUC (0.9602), showing it ranks samples by risk very well — but its middling accuracy (0.8568) suggests a suboptimal decision threshold for a 3-class problem. It handles class imbalance gracefully thanks to `class_weight='balanced'`, giving strong recall (0.8497) on rare classes. The linear decision boundary limits how well it can capture the non-linear CTG feature interactions that Random Forest exploits. |
| **Decision Tree** | Reaches 88.73% accuracy but has the **lowest AUC** (0.8786) among all models despite being competitive on accuracy. This is because trees produce step-function probability estimates (mostly near 0 or 1), which hurts AUC's ranking-quality measurement. `max_depth=8` prevents severe overfitting and gives good precision and recall balance (both ~0.83–0.85). |
| **kNN** | Middling overall (accuracy 0.8709) with the **lowest recall** (0.6900) of any model. The reason is structural: kNN doesn't support `class_weight='balanced'`, so its majority-vote mechanism naturally biases toward the Normal class (78% of the data). This hurts recall on the rare Suspect and Pathological cases — clinically the most important predictions to get right. |
| **Naive Bayes** | **Weakest performer** across all metrics (accuracy 0.8099, MCC 0.5737). The reason is fundamental: GaussianNB assumes features are conditionally independent given the class, but the CTG dataset has strongly correlated histogram features (e.g., `histogram_mean`, `histogram_median`, and `histogram_mode` all measure the same underlying FHR distribution). NB double-counts this correlated evidence, degrading its predictions. |
| **Random Forest (Ensemble)** | **Top performer on all six metrics** (accuracy 0.9319, MCC 0.8097, AUC 0.9812). The ensemble of 200 decision trees averages away individual-tree overfitting, handles non-linear feature interactions, and remains robust to correlated features — the three weaknesses that hurt the other four models. `class_weight='balanced'` ensures rare classes get proportional attention during training. The most confidence-calibrated model of the five (highest AUC), making its probability outputs trustworthy for clinical decision support. |
| **🏆 Overall Winner for this dataset** | **Random Forest.** MCC of 0.8097 is a strong result on a 3-class imbalanced problem, and its AUC of 0.9812 indicates excellent probability calibration. It wins every single metric in the comparison table by a clear margin (2–5 percentage points over the runner-up in most columns). For clinical CTG analysis, its ability to catch rare Pathological cases while maintaining high precision on Normal cases is the exact trade-off you want. |

---

## Reproducing the Results

### Local setup

```bash
git clone https://github.com/T-Shandilya-2025ac05260/fetal-health-classifier.git
cd fetal-health-classifier
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`. Upload `test_data.csv` to see per-model metrics and confusion matrices.

### Retraining the models

Open `train_models.ipynb` in Jupyter and run all cells. The notebook:
1. Loads `fetal_health.csv` (must be downloaded from Kaggle separately — not committed to this repo)
2. Performs EDA and preprocessing
3. Trains all 5 models on the stratified training set
4. Computes 6 metrics on the test set
5. Saves confusion matrices as `confusion_matrices.png`
6. Pickles the trained models and scaler into `model/`
7. Exports `test_data.csv` for use in the Streamlit app

---

## Tech Stack

- **Python 3.9+**
- **scikit-learn** — Model training and metrics
- **pandas / numpy** — Data manipulation
- **matplotlib / seaborn** — Visualizations (confusion matrices, feature importance charts)
- **Streamlit** — Interactive web application
- **Streamlit Community Cloud** — Deployment

---

## Author

**Tanay Shandilya**
BITS ID: 2025AC05260
BITS Pilani WILP — M.Tech (AIML)
Machine Learning Course — Assignment 2 (2025–2026)
