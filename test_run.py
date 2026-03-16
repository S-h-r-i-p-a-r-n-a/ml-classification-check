import pandas as pd
import numpy as np
from ml_classification_check import check_binary_classification

print("=" * 54)
print("TEST — Distribution Shift")
print("=" * 54)

# Train → young people age 20-30
# Test  → old people age 60-70 → clear shift
X_train = pd.DataFrame({
    'age':    [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    'salary': [30000, 31000, 32000, 33000, 34000,
               35000, 36000, 37000, 38000, 39000]
})
X_test = pd.DataFrame({
    'age':    [60, 65, 70],       # completely different range
    'salary': [31000, 32000, 33000]
})
y_train = pd.Series([0,1,0,1,0,1,0,1,0,1])
y_test  = pd.Series([0,1,0])

check_binary_classification(X_train, X_test, y_train, y_test)