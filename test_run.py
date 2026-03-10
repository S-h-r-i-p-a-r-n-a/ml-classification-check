import pandas as pd
import numpy as np
from ml_classification_check import check_multiclass_classification

print("=" * 54)
print("TEST 1 — Clean Multiclass Dataset")
print("=" * 54)

X_train = pd.DataFrame({
    'age':    [25,30,35,40,45,50,55,60,65,70,
               26,31,36,41,46],
    'salary': [50000,60000,70000,80000,90000,
               55000,65000,75000,85000,95000,
               52000,62000,72000,82000,92000]
})
X_test = pd.DataFrame({
    'age':    [28, 33, 38, 43, 48],
    'salary': [54000, 64000, 74000, 84000, 94000]
})
y_train = pd.Series([0,1,2,0,1,2,0,1,2,0,1,2,0,1,2])
y_test  = pd.Series([0,1,2,0,1])

check_multiclass_classification(X_train, X_test, y_train, y_test)

print("=" * 54)
print("TEST 2 — All Multiclass Issues")
print("=" * 54)

X_train2 = pd.DataFrame({
    'age':    [25,30,35,40,45,50,55,60,65,70],
    'salary': [50000,60000,70000,80000,90000,
               55000,65000,75000,85000,95000]
})
X_test2 = pd.DataFrame({
    'age':    [28, 33, 38],
    'salary': [54000, 64000, 99000]
})

# Imbalanced — class 0 dominates 80%
# Class 3 unseen in train but in test
y_train2 = pd.Series([0,0,0,0,0,0,0,0,1,2])
y_test2  = pd.Series([0,1,3])  # class 3 never in train!

check_multiclass_classification(X_train2, X_test2, y_train2, y_test2)