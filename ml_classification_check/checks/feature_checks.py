"""
Feature Quality Checks
──────────────────────
Checks the quality of features in X_train.
These run for all classification types.

Checks included:
6.1 → Constant columns checker
6.2 → Feature scaling checker
6.3 → Distribution shift checker
6.4 → Correlated features checker
"""

import pandas as pd
import numpy as np
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import (
    fix_constant_columns,
    fix_feature_scaling,
    fix_distribution_shift,
    fix_correlated_features
)


# ─────────────────────────────────────────────────────
# 6.1 — Constant Columns Checker
# ─────────────────────────────────────────────────────

class ConstantColumnsChecker(BaseChecker):
    """
    Finds columns where every value is identical.
    A constant column adds zero information to the model.
    It has no variance — nothing to learn from.

    Example:
    'country' column where every value is 'India'
    Model cannot use this to distinguish any class.
    It just wastes a feature slot.
    """

    def __init__(self, X_train):
        self.X_train = X_train

    def check(self):
        try:
            constant_cols = [
                col for col in self.X_train.columns
                if self.X_train[col].nunique(dropna=True) <= 1
            ]

            if constant_cols:
                details = "\n".join([
                    f"      → '{col}': "
                    f"only value = {self.X_train[col].dropna().unique().tolist()}"
                    for col in constant_cols
                ])
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Constant Columns",
                    group    = "feature_quality",
                    message  = (
                        f"   ⚠️  WARNING — Constant Columns Found!\n"
                        f"      {len(constant_cols)} column(s) have "
                        f"only one unique value:\n"
                        f"{details}\n"
                        f"      These columns add zero information.\n"
                        f"      Model wastes capacity on them."
                    ),
                    fix_code = fix_constant_columns(constant_cols)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Constant Columns",
                group    = "feature_quality",
                message  = (
                    "   ✅ No constant columns found. "
                    "All features have variance."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Constant Columns",
                group    = "feature_quality",
                message  = (
                    f"   ⚠️  Could not complete constant "
                    f"columns check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 6.2 — Feature Scaling Checker
# ─────────────────────────────────────────────────────

class FeatureScalingChecker(BaseChecker):
    """
    Checks if numeric features have very different scales.
    Example:
    'age'    → range 20–80    (scale of tens)
    'salary' → range 10000–200000 (scale of hundreds of thousands)

    Distance-based models (KNN, SVM, LogReg) are severely
    affected by scale differences.
    The large-scale feature dominates everything.
    """

    def __init__(self, X_train, scale_ratio_threshold=100):
        self.X_train              = X_train
        self.scale_ratio_threshold = scale_ratio_threshold

    def check(self):
        try:
            numeric_cols = self.X_train.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            if len(numeric_cols) < 2:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Feature Scaling",
                    group    = "feature_quality",
                    message  = (
                        "   ✅ Only one numeric feature — "
                        "scaling check not applicable."
                    ),
                    fix_code = None
                )

            # Calculate range for each numeric column
            ranges = {}
            for col in numeric_cols:
                col_data = self.X_train[col].dropna()
                if len(col_data) > 0:
                    col_range = col_data.max() - col_data.min()
                    if col_range > 0:
                        ranges[col] = col_range

            if len(ranges) < 2:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Feature Scaling",
                    group    = "feature_quality",
                    message  = (
                        "   ✅ Feature scaling looks consistent."
                    ),
                    fix_code = None
                )

            max_range = max(ranges.values())
            min_range = min(ranges.values())
            ratio     = max_range / min_range

            if ratio >= self.scale_ratio_threshold:
                largest_col  = max(ranges, key=ranges.get)
                smallest_col = min(ranges, key=ranges.get)

                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Feature Scaling",
                    group    = "feature_quality",
                    message  = (
                        f"   ⚠️  WARNING — Features Have Very "
                        f"Different Scales!\n"
                        f"      Largest range  : '{largest_col}' "
                        f"= {round(max_range, 2)}\n"
                        f"      Smallest range : '{smallest_col}' "
                        f"= {round(min_range, 2)}\n"
                        f"      Scale ratio    : {round(ratio, 1)}x\n"
                        f"      Distance-based models (KNN, SVM, "
                        f"LogReg) heavily affected."
                    ),
                    fix_code = fix_feature_scaling()
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Feature Scaling",
                group    = "feature_quality",
                message  = (
                    f"   ✅ Feature scales are comparable. "
                    f"Ratio = {round(ratio, 1)}x"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Feature Scaling",
                group    = "feature_quality",
                message  = (
                    f"   ⚠️  Could not complete scaling check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 6.3 — Distribution Shift Checker
# ─────────────────────────────────────────────────────

class DistributionShiftChecker(BaseChecker):
    """
    Checks if feature distributions differ significantly
    between train and test sets.

    Example:
    'age' in train → mean 25, std 3  (young people)
    'age' in test  → mean 60, std 5  (old people)

    Model trained on young people data
    being tested on old people data
    → predictions will be systematically wrong.

    Uses mean shift relative to standard deviation
    as a simple but effective measure.
    """

    def __init__(self, X_train, X_test, threshold=2.0):
        self.X_train   = X_train
        self.X_test    = X_test
        self.threshold = threshold

    def check(self):
        try:
            numeric_cols = self.X_train.select_dtypes(
                include=[np.number]
            ).columns.tolist()

            shifted_cols = []

            for col in numeric_cols:
                if col not in self.X_test.columns:
                    continue

                train_data = self.X_train[col].dropna()
                test_data  = self.X_test[col].dropna()

                if len(train_data) < 2 or len(test_data) < 2:
                    continue

                train_mean = train_data.mean()
                train_std  = train_data.std()

                if train_std == 0:
                    continue

                test_mean  = test_data.mean()
                shift      = abs(test_mean - train_mean) / train_std

                if shift >= self.threshold:
                    shifted_cols.append((
                        col,
                        round(train_mean, 2),
                        round(test_mean, 2),
                        round(shift, 2)
                    ))

            if shifted_cols:
                details = "\n".join([
                    f"      → '{col}': "
                    f"train mean={tr}  "
                    f"test mean={te}  "
                    f"(shift={sh} std devs)"
                    for col, tr, te, sh in shifted_cols
                ])
                col_names = [col for col, _, _, _ in shifted_cols]

                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Distribution Shift",
                    group    = "feature_quality",
                    message  = (
                        f"   ⚠️  WARNING — Distribution Shift Detected!\n"
                        f"      {len(shifted_cols)} feature(s) have very "
                        f"different distributions in train vs test:\n"
                        f"{details}\n"
                        f"      Model may perform poorly on test data."
                    ),
                    fix_code = fix_distribution_shift(col_names)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Distribution Shift",
                group    = "feature_quality",
                message  = (
                    "   ✅ No significant distribution shift "
                    "detected between train and test."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Distribution Shift",
                group    = "feature_quality",
                message  = (
                    f"   ⚠️  Could not complete distribution "
                    f"shift check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 6.4 — Correlated Features Checker
# ─────────────────────────────────────────────────────

class CorrelatedFeaturesChecker(BaseChecker):
    """
    Finds pairs of features that are highly correlated.
    If 'height_cm' and 'height_inches' both exist
    they carry identical information.

    Highly correlated features:
    - Cause multicollinearity in linear models
    - Waste feature capacity
    - Make feature importance misleading
    - Slow down training unnecessarily

    Flags pairs with correlation above 0.90.
    """

    def __init__(self, X_train, threshold=0.90):
        self.X_train   = X_train
        self.threshold = threshold

    def check(self):
        try:
            numeric_cols = self.X_train.select_dtypes(
                include=[np.number]
            )

            if numeric_cols.shape[1] < 2:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Correlated Features",
                    group    = "feature_quality",
                    message  = (
                        "   ✅ Less than 2 numeric features — "
                        "correlation check not applicable."
                    ),
                    fix_code = None
                )

            # Suppress numpy warnings for NaN correlation
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = numeric_cols.corr().abs()

            # Get upper triangle only to avoid duplicates
            upper      = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            corr_pairs = []

            for col in upper.columns:
                for idx in upper.index:
                    val = upper.loc[idx, col]
                    if pd.notna(val) and val >= self.threshold:
                        corr_pairs.append((
                            idx, col, round(val, 4)
                        ))

            if corr_pairs:
                details = "\n".join([
                    f"      → '{a}' & '{b}': "
                    f"correlation = {c}"
                    for a, b, c in corr_pairs
                ])
                pair_list = [(a, b) for a, b, _ in corr_pairs]

                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Correlated Features",
                    group    = "feature_quality",
                    message  = (
                        f"   ⚠️  WARNING — Highly Correlated "
                        f"Features Found!\n"
                        f"      {len(corr_pairs)} feature pair(s) "
                        f"with correlation ≥ {self.threshold}:\n"
                        f"{details}\n"
                        f"      These features carry duplicate information.\n"
                        f"      Consider dropping one from each pair."
                    ),
                    fix_code = fix_correlated_features(pair_list)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Correlated Features",
                group    = "feature_quality",
                message  = (
                    f"   ✅ No highly correlated feature pairs found. "
                    f"(threshold = {self.threshold})"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Correlated Features",
                group    = "feature_quality",
                message  = (
                    f"   ⚠️  Could not complete correlation "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )