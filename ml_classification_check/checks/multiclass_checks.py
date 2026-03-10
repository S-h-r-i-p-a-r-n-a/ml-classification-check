"""
Multiclass Classification Checks
──────────────────────────────────
Checks specific to multiclass classification problems.
These run only when check_multiclass_classification() is called.

Checks included:
4.1 → All classes present in training checker
4.2 → Unseen classes in test checker
4.3 → Minimum samples per class checker
4.4 → Multiclass imbalance checker
4.5 → Class distribution consistency checker
"""

import pandas as pd
import numpy as np
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import (
    fix_multiclass_imbalance,
    fix_unseen_test_classes,
    fix_minimum_samples
)


# ─────────────────────────────────────────────────────
# 4.1 — All Classes Present in Training Checker
# ─────────────────────────────────────────────────────

class AllClassesInTrainChecker(BaseChecker):
    """
    Confirms every class in y_train has at least
    one training sample.
    If a class has zero samples — model can never
    learn it — predictions for that class will always
    be wrong.
    """

    def __init__(self, y_train):
        self.y_train = y_train

    def check(self):
        try:
            value_counts  = pd.Series(self.y_train).value_counts()
            zero_sample   = value_counts[value_counts == 0]
            all_classes   = sorted(np.unique(self.y_train))
            n_classes     = len(all_classes)

            if len(zero_sample) > 0:
                missing = list(zero_sample.index)
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "All Classes in Training",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Classes With Zero Samples!\n"
                        f"      Classes {missing} have no training samples.\n"
                        f"      Model cannot learn these classes at all.\n"
                        f"      Every prediction for these classes will be wrong."
                    ),
                    fix_code = (
                        f"# Collect more data for these classes\n"
                        f"# Or remove them if not needed\n"
                        f"missing_classes = {missing}\n\n"
                        f"# Remove missing classes from dataset\n"
                        f"mask = ~y_train.isin(missing_classes)\n"
                        f"X_train = X_train[mask]\n"
                        f"y_train = y_train[mask]"
                    )
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "All Classes in Training",
                group    = "classification",
                message  = (
                    f"   ✅ All {n_classes} classes have training "
                    f"samples: {all_classes}"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "All Classes in Training",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete all classes check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 4.2 — Unseen Classes in Test Checker
# ─────────────────────────────────────────────────────

class UnseenTestClassesChecker(BaseChecker):
    """
    Checks if test set contains classes that
    never appeared in training.
    Model has never seen these classes —
    it has no idea how to classify them.
    """

    def __init__(self, y_train, y_test):
        self.y_train = y_train
        self.y_test  = y_test

    def check(self):
        try:
            if self.y_test is None:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Unseen Test Classes",
                    group    = "classification",
                    message  = (
                        "   ℹ️  Unseen classes check skipped "
                        "(y_test not provided)."
                    ),
                    fix_code = None
                )

            train_classes  = set(np.unique(self.y_train))
            test_classes   = set(np.unique(self.y_test))
            unseen_classes = test_classes - train_classes

            if unseen_classes:
                unseen_list = sorted(list(unseen_classes))
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Unseen Test Classes",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Unseen Classes in Test Set!\n"
                        f"      Classes {unseen_list} appear in test "
                        f"but NEVER in training.\n"
                        f"      Model has no knowledge of these classes.\n"
                        f"      Predictions for them will be random guesses."
                    ),
                    fix_code = fix_unseen_test_classes(unseen_list)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Unseen Test Classes",
                group    = "classification",
                message  = (
                    "   ✅ No unseen classes in test set. "
                    "Train and test share same classes."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Unseen Test Classes",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete unseen classes "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 4.3 — Minimum Samples Per Class Checker
# ─────────────────────────────────────────────────────

class MinSamplesPerClassChecker(BaseChecker):
    """
    Checks if every class has enough training samples.
    Classes with very few samples (less than 10)
    cause the model to learn them poorly —
    leading to terrible recall for those classes.
    """

    def __init__(self, y_train, min_samples=10):
        self.y_train    = y_train
        self.min_samples = min_samples

    def check(self):
        try:
            value_counts = pd.Series(self.y_train).value_counts()
            low_classes  = value_counts[
                value_counts < self.min_samples
            ]

            if len(low_classes) > 0:
                details = "\n".join([
                    f"      → Class '{cls}': only {cnt} sample(s)"
                    for cls, cnt in low_classes.items()
                ])
                low_list = list(low_classes.index)

                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Minimum Samples Per Class",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Low Sample Classes Found!\n"
                        f"      {len(low_classes)} class(es) have fewer "
                        f"than {self.min_samples} samples:\n"
                        f"{details}\n"
                        f"      Model will learn these classes poorly."
                    ),
                    fix_code = fix_minimum_samples(low_list)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Minimum Samples Per Class",
                group    = "classification",
                message  = (
                    f"   ✅ All classes have at least "
                    f"{self.min_samples} samples."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Minimum Samples Per Class",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete min samples "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 4.4 — Multiclass Imbalance Checker
# ─────────────────────────────────────────────────────

class MulticlassImbalanceChecker(BaseChecker):
    """
    Checks if one class dominates all others
    in a multiclass setting.
    Unlike binary imbalance, here we check if
    any single class exceeds 70% of all samples.
    """

    def __init__(self, y_train, threshold=0.70):
        self.y_train   = y_train
        self.threshold = threshold

    def check(self):
        try:
            values    = pd.Series(self.y_train).value_counts(
                normalize=True
            )
            max_class = values.idxmax()
            max_pct   = round(values.max() * 100, 1)

            dist = "  |  ".join([
                f"Class '{cls}' = {round(pct*100, 1)}%"
                for cls, pct in values.items()
            ])

            if max_pct >= self.threshold * 100:
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Multiclass Imbalance",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Multiclass Imbalance!\n"
                        f"      Distribution: {dist}\n"
                        f"      Class '{max_class}' dominates at {max_pct}%.\n"
                        f"      Model biased toward majority class.\n"
                        f"      Use macro F1 score not accuracy."
                    ),
                    fix_code = fix_multiclass_imbalance()
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Multiclass Imbalance",
                group    = "classification",
                message  = (
                    f"   ✅ Class distribution looks balanced.\n"
                    f"      {dist}"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Multiclass Imbalance",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete imbalance check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 4.5 — Class Distribution Consistency Checker
# ─────────────────────────────────────────────────────

class ClassDistributionConsistencyChecker(BaseChecker):
    """
    Compares class distribution between train and test.
    If train has 60% cats and test has 10% cats —
    the model was trained on a very different
    distribution than it will be tested on.
    This leads to misleading evaluation metrics.
    """

    def __init__(self, y_train, y_test, threshold=0.20):
        self.y_train   = y_train
        self.y_test    = y_test
        self.threshold = threshold

    def check(self):
        try:
            if self.y_test is None:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Class Distribution Consistency",
                    group    = "classification",
                    message  = (
                        "   ℹ️  Distribution consistency check skipped "
                        "(y_test not provided)."
                    ),
                    fix_code = None
                )

            train_dist = pd.Series(self.y_train).value_counts(
                normalize=True
            )
            test_dist  = pd.Series(self.y_test).value_counts(
                normalize=True
            )

            inconsistent = []
            for cls in train_dist.index:
                train_pct = train_dist.get(cls, 0)
                test_pct  = test_dist.get(cls, 0)
                diff      = abs(train_pct - test_pct)
                if diff >= self.threshold:
                    inconsistent.append((
                        cls,
                        round(train_pct * 100, 1),
                        round(test_pct  * 100, 1)
                    ))

            if inconsistent:
                details = "\n".join([
                    f"      → Class '{cls}': "
                    f"train={tr}%  test={te}%  "
                    f"(gap={round(abs(tr-te), 1)}%)"
                    for cls, tr, te in inconsistent
                ])
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Class Distribution Consistency",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Class Distribution Mismatch!\n"
                        f"      Train and test have different class ratios:\n"
                        f"{details}\n"
                        f"      Evaluation metrics may be misleading.\n"
                        f"      Use stratified split to fix this."
                    ),
                    fix_code = (
                        "# Fix distribution mismatch with stratified split\n"
                        "from sklearn.model_selection import train_test_split\n\n"
                        "X_train, X_test, y_train, y_test = train_test_split(\n"
                        "    X, y,\n"
                        "    test_size=0.2,\n"
                        "    random_state=42,\n"
                        "    stratify=y    "
                        "# preserves class ratios in both splits\n"
                        ")"
                    )
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Class Distribution Consistency",
                group    = "classification",
                message  = (
                    "   ✅ Class distribution is consistent "
                    "between train and test."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Class Distribution Consistency",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete distribution "
                    f"consistency check: {str(e)}"
                ),
                fix_code = None
            )