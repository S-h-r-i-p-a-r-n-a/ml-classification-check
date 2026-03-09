"""
Binary Classification Checks
─────────────────────────────
Checks specific to binary classification problems.
These run only when check_binary_classification() is called.

Checks included:
3.1 → Exactly 2 classes validator
3.2 → Both classes present in train AND test
3.3 → Binary class imbalance checker
3.4 → Binary label type validator
"""

import pandas as pd
import numpy as np
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import (
    fix_binary_imbalance,
    fix_binary_label_type
)


# ─────────────────────────────────────────────────────
# 3.1 — Exactly 2 Classes Validator
# ─────────────────────────────────────────────────────

class BinaryClassValidator(BaseChecker):
    """
    Confirms y_train has exactly 2 unique classes.
    If more or fewer classes found — flags it immediately.
    """

    def __init__(self, y_train):
        self.y_train = y_train

    def check(self):
        try:
            unique_classes = list(np.unique(self.y_train))
            n_classes      = len(unique_classes)

            if n_classes == 2:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Binary Class Validator",
                    group    = "classification",
                    message  = (
                        f"   ✅ Exactly 2 classes confirmed: "
                        f"{unique_classes}"
                    ),
                    fix_code = None
                )

            elif n_classes < 2:
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Binary Class Validator",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Less Than 2 Classes Found!\n"
                        f"      Only {n_classes} unique class found: "
                        f"{unique_classes}\n"
                        f"      Binary classification needs exactly 2 classes.\n"
                        f"      Check if your target column is correct."
                    ),
                    fix_code = None
                )

            else:
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Binary Class Validator",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — More Than 2 Classes Found!\n"
                        f"      Found {n_classes} classes: {unique_classes}\n"
                        f"      This is NOT a binary classification problem.\n"
                        f"      Use check_multiclass_classification() instead."
                    ),
                    fix_code = None
                )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Binary Class Validator",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete class validation: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 3.2 — Both Classes Present in Train AND Test
# ─────────────────────────────────────────────────────

class BothClassesChecker(BaseChecker):
    """
    Confirms both classes exist in BOTH train and test.
    If test is missing a class — model never gets evaluated
    on it — giving completely misleading metrics.
    """

    def __init__(self, y_train, y_test):
        self.y_train = y_train
        self.y_test  = y_test

    def check(self):
        try:
            # If no y_test provided skip this check
            if self.y_test is None:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Both Classes Present",
                    group    = "classification",
                    message  = (
                        "   ℹ️  Both classes check skipped "
                        "(y_test not provided)."
                    ),
                    fix_code = None
                )

            train_classes = set(np.unique(self.y_train))
            test_classes  = set(np.unique(self.y_test))

            missing_in_test  = train_classes - test_classes
            missing_in_train = test_classes  - train_classes

            issues = []

            if missing_in_test:
                issues.append(
                    f"      → Classes {missing_in_test} exist in "
                    f"train but NOT in test.\n"
                    f"        Model never gets evaluated on these classes."
                )
            if missing_in_train:
                issues.append(
                    f"      → Classes {missing_in_train} exist in "
                    f"test but NOT in train.\n"
                    f"        Model never learned these classes."
                )

            if issues:
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Both Classes Present",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Class Mismatch Between "
                        f"Train and Test!\n"
                        + "\n".join(issues) +
                        f"\n      Re-split your data to ensure both "
                        f"classes appear in both sets."
                    ),
                    fix_code = (
                        "# Fix class mismatch — use stratified split\n"
                        "from sklearn.model_selection import train_test_split\n\n"
                        "X_train, X_test, y_train, y_test = train_test_split(\n"
                        "    X, y,\n"
                        "    test_size=0.2,\n"
                        "    random_state=42,\n"
                        "    stratify=y    "
                        "# ensures both classes in train and test\n"
                        ")"
                    )
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Both Classes Present",
                group    = "classification",
                message  = (
                    f"   ✅ Both classes present in train and test: "
                    f"{sorted(train_classes)}"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Both Classes Present",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete class presence "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 3.3 — Binary Class Imbalance Checker
# ─────────────────────────────────────────────────────

class BinaryImbalanceChecker(BaseChecker):
    """
    Checks if one class dominates the dataset.
    Severe imbalance causes model to predict
    majority class always — looks accurate but useless.

    Example:
    95% not fraud, 5% fraud
    Model predicts "not fraud" every time
    Gets 95% accuracy — catches zero fraud
    """

    def __init__(self, y_train, threshold=0.85):
        self.y_train   = y_train
        self.threshold = threshold

    def check(self):
        try:
            values    = pd.Series(self.y_train).value_counts(normalize=True)
            max_class = values.idxmax()
            max_pct   = round(values.max() * 100, 1)
            min_class = values.idxmin()
            min_pct   = round(values.min() * 100, 1)

            dist = "  |  ".join([
                f"Class {cls} = {round(pct*100, 1)}%"
                for cls, pct in values.items()
            ])

            if max_pct >= self.threshold * 100:
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Binary Class Imbalance",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Severe Class Imbalance!\n"
                        f"      Distribution: {dist}\n"
                        f"      Class {max_class} dominates at {max_pct}%.\n"
                        f"      Model may predict class {max_class} always.\n"
                        f"      Accuracy metric will be misleading."
                    ),
                    fix_code = fix_binary_imbalance()
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Binary Class Imbalance",
                group    = "classification",
                message  = (
                    f"   ✅ Class balance looks healthy. {dist}"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Binary Class Imbalance",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete imbalance check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 3.4 — Binary Label Type Validator
# ─────────────────────────────────────────────────────

class BinaryLabelTypeChecker(BaseChecker):
    """
    Checks if y_train contains numeric labels (0/1)
    or string labels ("yes"/"no", "spam"/"ham" etc.)

    Most sklearn models require numeric labels.
    String labels cause cryptic errors or wrong results.
    """

    def __init__(self, y_train):
        self.y_train = y_train

    def check(self):
        try:
            unique_vals = np.unique(self.y_train)
            is_numeric  = all(
                isinstance(v, (int, float, np.integer, np.floating))
                for v in unique_vals
            )

            if is_numeric:
                # Check if values are actually 0 and 1
                vals = sorted([int(v) for v in unique_vals])
                if vals == [0, 1]:
                    return self._result(
                        passed   = True,
                        severity = "ok",
                        check    = "Binary Label Type",
                        group    = "classification",
                        message  = (
                            "   ✅ Binary labels are correct "
                            "numeric format (0 and 1)."
                        ),
                        fix_code = None
                    )
                else:
                    return self._result(
                        passed   = False,
                        severity = "warning",
                        check    = "Binary Label Type",
                        group    = "classification",
                        message  = (
                            f"   ⚠️  WARNING — Labels are numeric "
                            f"but not 0/1!\n"
                            f"      Found: {list(unique_vals)}\n"
                            f"      Most models expect exactly 0 and 1.\n"
                            f"      Consider remapping to 0 and 1."
                        ),
                        fix_code = fix_binary_label_type()
                    )

            else:
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Binary Label Type",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — String Labels Detected!\n"
                        f"      Found: {list(unique_vals)}\n"
                        f"      Most sklearn models need numeric labels.\n"
                        f"      Convert strings to 0 and 1 before training."
                    ),
                    fix_code = fix_binary_label_type()
                )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Binary Label Type",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete label type check: {str(e)}"
                ),
                fix_code = None
            )