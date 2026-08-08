## Objective
Transition from regression (Tasks 1–3) to **binary classification**. Build a model that
predicts whether a breast tumor is malignant or benign, evaluate it using metrics that
actually matter for a medical use case (not just accuracy), test a technique for
handling class imbalance, and justify the final model choice.

## Dataset
- **Breast Cancer Wisconsin Dataset** (built into scikit-learn)
- 569 samples — 212 Malignant, 357 Benign (62.7% / 37.3% — moderate imbalance)
- Binary target: Malignant = 0, Benign = 1

## Models Compared
| Model | Notes |
|---|---|
| Logistic Regression (Baseline) | Default class weighting |
| Logistic Regression (Balanced) | `class_weight="balanced"` — tests imbalance handling |
| Decision Tree | Included as a non-linear comparison point |

## Why Not Just Accuracy?
In a medical diagnosis setting, a **False Positive** — telling a sick patient they're
healthy — is far more dangerous than a False Negative. So this task prioritizes:
- **Recall** — minimizes missed malignant cases
- **F1-Score** — balances precision and recall, resists imbalance
- **ROC-AUC** — measures discrimination ability across all thresholds, not just one

## Results

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| **Logistic Regression (Baseline)** | **0.9825** | **0.9861** | **0.9861** | **0.9861** | **0.9954** |
| Logistic Regression (Balanced) | 0.9561 | 0.9855 | 0.9444 | 0.9645 | 0.9954 |
| Decision Tree | 0.9123 | 0.9559 | 0.9028 | 0.9286 | 0.9157 |

## Key Finding: Class Weighting Didn't Help Here
`class_weight="balanced"` is a standard technique for imbalanced data — but on this
dataset it actually made Recall *worse* (0.9444 vs 0.9861). The imbalance here is
moderate (63/37), not severe, and the baseline model was already learning the minority
class well. Class weighting earns its keep on **severe** imbalance (e.g. fraud
detection with <5% positive cases), not moderate imbalance like this one — a useful,
counter-intuitive lesson in when a "standard fix" isn't actually needed.

## Final Model: Logistic Regression (Baseline)
Selected for the best score across every metric, no overfitting (unlike the Decision
Tree, which scored 100% on training data but only 91.2% on test data), and full
interpretability — clinicians can see exactly which features drive each prediction,
which matters as much as raw performance in a healthcare context.

## Tools
Python 3 · scikit-learn · pandas · NumPy · matplotlib · seaborn

## Files
- `Task4_Report.pdf` — full technical write-up (confusion matrix, metric comparison, ROC curves, imbalance analysis, final justification)
- Jupyter notebook with full implementation

---
*Part of a 12-week Data Analyst / AI-ML internship at Maincrafts Technology. See also: [Task 1 — Linear Regression](../task1) and [Task 3 — Cross-Validation & Hyperparameter Tuning](../task3).*
