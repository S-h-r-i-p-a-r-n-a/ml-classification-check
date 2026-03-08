import pandas as pd
import numpy as np
from ml_classification_check import check_binary_classification

# BAD data — columns with high missing values
X_train = pd.DataFrame({
    'age':    [25, np.nan, 35, np.nan, 45,
               np.nan, 55, np.nan, 65, np.nan],  # 50% missing
    'salary': [50000, 60000, np.nan, 80000, np.nan,
               55000, np.nan, 75000, 85000, 95000], # 30% missing
    'score':  [0.8, 0.6, 0.9, 0.7, 0.5,
               0.4, 0.8, 0.6, 0.9, 0.7]            # no missing
})
X_test = pd.DataFrame({
    'age':    [28, 33],
    'salary': [55000, 65000],
    'score':  [0.7, 0.8]
})
y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
y_test  = pd.Series([0, 1])

check_binary_classification(X_train, X_test, y_train, y_test)