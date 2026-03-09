import pandas as pd
import numpy as np
from ml_classification_check import check_binary_classification

print("=" * 54)
print("TEST 1 — Pure Duplicate Test")
print("=" * 54)

# Clean simple duplicates — no NaN confusion
X_train = pd.DataFrame({
    'age':    [25, 25, 30, 35, 35, 40, 45, 50, 55, 60],
    'salary': [50000, 50000, 60000, 70000, 70000,
               80000, 90000, 55000, 65000, 75000]
})
X_test = pd.DataFrame({
    'age':    [28, 33],
    'salary': [55000, 65000]
})
y_train = pd.Series([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
y_test  = pd.Series([0, 1])

check_binary_classification(X_train, X_test, y_train, y_test)

print("=" * 54)
print("TEST 2 — All 4 Issues Together")
print("=" * 54)

# All 4 issues — keeping loan_status OUT of X_train
# to avoid confusion with target leakage check
X_train2 = pd.DataFrame({
    'age':          [25, 25, 30, 35, 35,
                     40, 45, 50, 55, 60],
    'salary':       [50000, 50000, 60000, 70000, 70000,
                     80000, 90000, 55000, 65000, 75000],
    'loan_status':  [0, 0, 1, 0, 0,
                     1, 0, 1, 0, 1],
    'credit_score': [None, None, None, None, None,
                     700, 750, None, 800, None]
})
X_test2 = pd.DataFrame({
    'age':          [25, 25, 50],
    'salary':       [50000, 50000, 55000],
    'loan_status':  [0, 0, 1],
    'credit_score': [None, None, 800]
})
y_train2 = pd.Series([0, 0, 1, 0, 0, 1, 0, 1, 0, 1])
y_test2  = pd.Series([0, 0, 0])

check_binary_classification(X_train2, X_test2, y_train2, y_test2)