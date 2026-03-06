# ml-classification-check 🔍

> **Catch silent data mistakes before training your classification model.**
> One function call. Plain English warnings. Paste-ready fix code.
> Works in VS Code, Google Colab, and Kaggle.

---

## The Problem

You build a classification model. You see **99% accuracy**. You feel great.
Then you lose marks. Or your Kaggle score tanks.

**Why? Silent data mistakes.** No error. No warning. Python ran fine.
But your data was secretly broken the whole time.

`ml-classification-check` catches these mistakes **before you train.**

---

## Install
```bash
pip install ml-classification-check
```

---

## Usage

### Binary Classification
```python
from ml_classification_check import check_binary_classification

check_binary_classification(X_train, X_test, y_train, y_test)
```

### Multi-class Classification
```python
from ml_classification_check import check_multiclass_classification

check_multiclass_classification(X_train, X_test, y_train, y_test)
```

### Multi-label Classification
```python
from ml_classification_check import check_multilabel_classification

check_multilabel_classification(X_train, X_test, y_train, y_test)
```

---

## What It Checks

### Data Integrity (All Types)
| Check | What It Catches |
|---|---|
| Train-Test Leakage | Same rows in both train and test — fake accuracy |
| Duplicate Rows | Repeated rows in training set — model memorization |
| Target Leakage | Feature too correlated with label — model cheating |
| Missing Values | Columns with too many NaN values — dirty data |

### Binary Classification
| Check | What It Catches |
|---|---|
| Class Validator | Confirms exactly 2 classes exist |
| Classes in Both Splits | Both classes present in train and test |
| Class Imbalance | One class dominates — fake accuracy |
| Label Type | y is numeric not string |

### Multi-class Classification
| Check | What It Catches |
|---|---|
| All Classes in Training | No class has zero training samples |
| Unseen Test Classes | Test has classes train never saw |
| Minimum Samples | Each class has enough samples |
| Class Imbalance | One class dominates all others |
| Distribution Consistency | Train and test class ratios match |

### Multi-label Classification
| Check | What It Catches |
|---|---|
| Label Matrix Format | y is proper 2D binary matrix |
| Per-label Imbalance | Individual label imbalance |
| Empty Labels | Labels never seen in training |
| Label Density | Too many or too few labels per sample |
| Rare Combinations | Label combos seen fewer than 5 times |

### Feature Quality (All Types)
| Check | What It Catches |
|---|---|
| Constant Columns | Zero variance — useless features |
| Feature Scaling | Incompatible feature scales |
| Distribution Shift | Train and test from different distributions |
| Correlated Features | Redundant feature pairs above 0.95 correlation |

---

## Output Format

Every run gives you 4 sections:
```
1. Dataset Summary    → What we found in your data
2. Check Results      → What is wrong and how serious
3. Final Score        → Critical / Warnings / Passed count
4. Fix Code           → Exact paste-ready code for every issue
```

---

## When To Run It
```
Load Data
    ↓
Train-Test Split
    ↓
🔍 Run ml-classification-check HERE
    ↓
Fix Issues
    ↓
Train Model
```

---

## Works Everywhere
```python
# Google Colab / Kaggle
!pip install ml-classification-check

# VS Code / Local
pip install ml-classification-check
```

---

## License

MIT — free to use, share, and modify.
```

---

