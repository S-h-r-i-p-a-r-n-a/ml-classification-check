"""
Fix Suggester Module
────────────────────
Contains paste-ready fix code for every possible issue
the checkers can detect. Each function returns a clean
code string that gets printed in Section 4 of the report.
"""


# ─────────────────────────────────────────────────────
# DATA INTEGRITY FIXES
# ─────────────────────────────────────────────────────

def fix_train_test_leakage():
    return """
from sklearn.model_selection import train_test_split

# Re-split your data correctly
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    shuffle=True        # ensures no ordered leakage
)
"""


def fix_duplicate_rows():
    return """
# Remove duplicate rows before splitting
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Then re-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
"""


def fix_target_leakage(leaky_columns):
    cols = str(leaky_columns)
    return f"""
# Remove target-leaking columns
leaky_columns = {cols}

X_train = X_train.drop(columns=leaky_columns)
X_test  = X_test.drop(columns=leaky_columns)
"""


def fix_missing_values(high_missing_cols, low_missing_cols):
    high = str(high_missing_cols)
    low  = str(low_missing_cols)
    return f"""
# Option 1 — Drop columns with too many missing values (>50%)
X_train = X_train.drop(columns={high})
X_test  = X_test.drop(columns={high})

# Option 2 — Impute columns with moderate missing values
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')  # or 'median' or 'most_frequent'
X_train[{low}] = imputer.fit_transform(X_train[{low}])
X_test[{low}]  = imputer.transform(X_test[{low}])
"""


# ─────────────────────────────────────────────────────
# BINARY CLASSIFICATION FIXES
# ─────────────────────────────────────────────────────

def fix_binary_label_type():
    return """
# Convert string labels to numeric
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test  = le.transform(y_test)

# Check mapping
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
"""


def fix_binary_imbalance():
    return """
# Option 1 — Use class_weight (simplest, works with most sklearn models)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

model = RandomForestClassifier(class_weight='balanced', random_state=42)
# model = LogisticRegression(class_weight='balanced')

# Option 2 — Use SMOTE to oversample minority class
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Option 3 — Evaluate with better metrics than accuracy
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
"""


# ─────────────────────────────────────────────────────
# MULTI-CLASS CLASSIFICATION FIXES
# ─────────────────────────────────────────────────────

def fix_multiclass_imbalance():
    return """
# Option 1 — Use class_weight='balanced'
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(class_weight='balanced', random_state=42)

# Option 2 — Use SMOTE for multiclass
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Option 3 — Use better evaluation metrics
from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))
"""


def fix_unseen_test_classes(unseen_classes):
    classes = str(unseen_classes)
    return f"""
# Remove samples with classes not seen during training
unseen = {classes}

test_mask = ~y_test.isin(unseen)
X_test  = X_test[test_mask]
y_test  = y_test[test_mask]
"""


def fix_minimum_samples(low_sample_classes):
    classes = str(low_sample_classes)
    return f"""
# Classes with too few samples: {classes}

# Option 1 — Collect more data for these classes

# Option 2 — Merge rare classes into an 'Other' category
rare_classes = {classes}
y_train = y_train.replace({{c: 'Other' for c in rare_classes}})
y_test  = y_test.replace({{c: 'Other' for c in rare_classes}})
"""


# ─────────────────────────────────────────────────────
# MULTI-LABEL CLASSIFICATION FIXES
# ─────────────────────────────────────────────────────

def fix_multilabel_imbalance():
    return """
# Use label-specific class weights
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# Evaluate with label-aware metrics
from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

# Consider oversampling with MLSMOTE or collect more data
# for underrepresented labels
"""


def fix_empty_labels(empty_labels):
    labels = str(empty_labels)
    return f"""
# Remove labels that never appear in training
empty_labels = {labels}

# If y is a DataFrame
y_train = y_train.drop(columns=empty_labels)
y_test  = y_test.drop(columns=empty_labels)
"""


# ─────────────────────────────────────────────────────
# FEATURE QUALITY FIXES
# ─────────────────────────────────────────────────────

def fix_constant_columns(constant_cols):
    cols = str(constant_cols)
    return f"""
# Drop constant columns — they carry zero information
constant_columns = {cols}

X_train = X_train.drop(columns=constant_columns)
X_test  = X_test.drop(columns=constant_columns)
"""


def fix_feature_scaling():
    return """
# Standardize all features to same scale
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Use X_train_scaled and X_test_scaled for training
# IMPORTANT: Always fit scaler on train only, transform both
"""


def fix_distribution_shift(shifted_cols):
    cols = str(shifted_cols)
    return f"""
# Columns with distribution shift: {cols}

# Option 1 — Investigate why train and test differ
# Are they from different time periods or sources?

# Option 2 — Re-collect data so train and test
# come from the same distribution

# Option 3 — Use robust models less sensitive to shift
from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier(random_state=42)
"""


def fix_correlated_features(correlated_pairs):
    drop_cols = list(set([pair[1] for pair in correlated_pairs]))
    cols = str(drop_cols)
    return f"""
# Drop one column from each highly correlated pair
# Keeping the first, dropping the second in each pair
# Correlated pairs found: {correlated_pairs}

columns_to_drop = {cols}
X_train = X_train.drop(columns=columns_to_drop)
X_test  = X_test.drop(columns=columns_to_drop)
"""
