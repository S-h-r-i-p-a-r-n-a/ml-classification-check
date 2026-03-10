import pandas as pd
import numpy as np
from ml_classification_check import check_multilabel_classification

print("=" * 54)
print("TEST 1 — Clean Multilabel Dataset")
print("=" * 54)

X_train = pd.DataFrame({
    'age':    [25,30,35,40,45,50,55,60,65,70,
               26,31,36,41,46,51,56,61,66,71],
    'salary': [50000,60000,70000,80000,90000,
               55000,65000,75000,85000,95000,
               52000,62000,72000,82000,92000,
               53000,63000,73000,83000,93000]
})
X_test = pd.DataFrame({
    'age':    [28, 33, 38],
    'salary': [54000, 64000, 74000]
})
# Clean 2D binary label matrix
y_train = np.array([
    [1,0,1],[0,1,0],[1,1,0],[0,0,1],[1,0,0],
    [0,1,1],[1,0,1],[0,1,0],[1,1,1],[0,0,1],
    [1,0,0],[0,1,1],[1,0,1],[0,1,0],[1,1,0],
    [0,0,1],[1,0,0],[0,1,1],[1,1,0],[0,0,1]
])
y_test = np.array([[1,0,1],[0,1,0],[1,1,0]])

check_multilabel_classification(X_train, X_test, y_train, y_test)

print("=" * 54)
print("TEST 2 — All Multilabel Issues")
print("=" * 54)

X_train2 = pd.DataFrame({
    'age':    [25,30,35,40,45,50,55,60,65,70],
    'salary': [50000,60000,70000,80000,90000,
               55000,65000,75000,85000,95000]
})
X_test2 = pd.DataFrame({
    'age':    [28, 33],
    'salary': [54000, 64000]
})

# Problems:
# Label 2 → always 0 (empty label)
# Label 3 → appears in only 1 sample (rare)
# All combos appear only once (rare combinations)
y_train2 = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [1,0,0,1],
    [0,1,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [1,0,0,0],
    [0,1,0,0],
    [1,0,0,0],
    [0,0,0,0],  # no labels at all
])
y_test2  = np.array([[1,0,0,0],[0,1,0,0]])

check_multilabel_classification(X_train2, X_test2, y_train2, y_test2)