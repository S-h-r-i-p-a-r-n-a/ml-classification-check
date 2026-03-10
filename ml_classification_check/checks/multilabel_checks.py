"""
Multilabel Classification Checks
──────────────────────────────────
Checks specific to multilabel classification problems.
These run only when check_multilabel_classification() is called.

In multilabel classification each sample can belong
to MULTIPLE classes simultaneously.
Example: A news article tagged as both 'politics' AND 'economy'

Checks included:
5.1 → Label matrix format validator
5.2 → Per-label imbalance checker
5.3 → Empty labels checker
5.4 → Label density checker
5.5 → Rare label combinations checker
"""

import pandas as pd
import numpy as np
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import (
    fix_multilabel_imbalance,
    fix_empty_labels
)


# ─────────────────────────────────────────────────────
# 5.1 — Label Matrix Format Validator
# ─────────────────────────────────────────────────────

class LabelMatrixFormatChecker(BaseChecker):
    """
    Confirms y_train is a valid multilabel matrix.
    Must be 2D with only binary values (0 and 1).

    Common mistakes:
    - Passing a 1D array (multiclass format)
    - Having non-binary values like 2, 3, -1
    - Having float values like 0.5
    """

    def __init__(self, y_train):
        self.y_train = y_train

    def check(self):
        try:
            y = np.array(self.y_train)

            # Must be 2D
            if y.ndim != 2:
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Label Matrix Format",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Wrong Label Format!\n"
                        f"      y_train is {y.ndim}D but multilabel "
                        f"needs a 2D matrix.\n"
                        f"      Expected shape: (n_samples, n_labels)\n"
                        f"      Got shape: {y.shape}\n"
                        f"      Use MultiLabelBinarizer to fix this."
                    ),
                    fix_code = (
                        "from sklearn.preprocessing import MultiLabelBinarizer\n\n"
                        "# If y is a list of label lists:\n"
                        "# e.g. [['cat','dog'], ['cat'], ['dog','bird']]\n"
                        "mlb = MultiLabelBinarizer()\n"
                        "y_train = mlb.fit_transform(y_train_raw)\n"
                        "y_test  = mlb.transform(y_test_raw)\n\n"
                        "print('Classes:', mlb.classes_)"
                    )
                )

            # Must contain only 0 and 1
            unique_vals = np.unique(y)
            valid_vals  = set(unique_vals).issubset({0, 1})

            if not valid_vals:
                bad_vals = [v for v in unique_vals if v not in [0, 1]]
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Label Matrix Format",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Non-Binary Values in Labels!\n"
                        f"      Found values: {bad_vals}\n"
                        f"      Multilabel matrix must contain only 0 and 1.\n"
                        f"      Each cell means: 1=label present, 0=absent."
                    ),
                    fix_code = (
                        "# Binarize label matrix\n"
                        "import numpy as np\n\n"
                        "# Threshold to binary if values are probabilities\n"
                        "y_train = (y_train >= 0.5).astype(int)\n"
                        "y_test  = (y_test  >= 0.5).astype(int)"
                    )
                )

            n_samples, n_labels = y.shape
            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Label Matrix Format",
                group    = "classification",
                message  = (
                    f"   ✅ Label matrix format is valid.\n"
                    f"      Shape: {n_samples} samples × "
                    f"{n_labels} labels."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Label Matrix Format",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not validate label matrix "
                    f"format: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 5.2 — Per-Label Imbalance Checker
# ─────────────────────────────────────────────────────

class PerLabelImbalanceChecker(BaseChecker):
    """
    Checks if individual labels are severely imbalanced.
    In multilabel problems each label is essentially
    a separate binary classification problem.
    If label 'rare_topic' appears in only 1% of samples
    the model will almost never predict it.
    """

    def __init__(self, y_train, threshold=0.05):
        self.y_train   = y_train
        self.threshold = threshold

    def check(self):
        try:
            y           = np.array(self.y_train)
            n_samples   = y.shape[0]
            label_freq  = y.mean(axis=0)

            rare_labels = []
            for i, freq in enumerate(label_freq):
                if freq < self.threshold:
                    rare_labels.append((i, round(freq * 100, 2)))

            if rare_labels:
                details = "\n".join([
                    f"      → Label {i}: appears in only {pct}% of samples"
                    for i, pct in rare_labels
                ])
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Per-Label Imbalance",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Rare Labels Detected!\n"
                        f"      {len(rare_labels)} label(s) appear in "
                        f"fewer than {self.threshold*100}% of samples:\n"
                        f"{details}\n"
                        f"      Model will rarely predict these labels."
                    ),
                    fix_code = fix_multilabel_imbalance()
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Per-Label Imbalance",
                group    = "classification",
                message  = (
                    f"   ✅ All labels appear in at least "
                    f"{self.threshold*100}% of samples."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Per-Label Imbalance",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete per-label "
                    f"imbalance check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 5.3 — Empty Labels Checker
# ─────────────────────────────────────────────────────

class EmptyLabelsChecker(BaseChecker):
    """
    Checks for labels that are always 0 — never active.
    An empty label column means no sample ever has
    that label — completely useless for training.
    Model wastes capacity trying to learn nothing.
    """

    def __init__(self, y_train):
        self.y_train = y_train

    def check(self):
        try:
            y          = np.array(self.y_train)
            label_sums = y.sum(axis=0)
            empty      = list(np.where(label_sums == 0)[0])

            if empty:
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Empty Labels",
                    group    = "classification",
                    message  = (
                        f"   ❌ CRITICAL — Empty Labels Found!\n"
                        f"      Labels {empty} have zero positive "
                        f"samples in training set.\n"
                        f"      These labels are completely useless.\n"
                        f"      Model wastes capacity learning nothing."
                    ),
                    fix_code = fix_empty_labels(empty)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Empty Labels",
                group    = "classification",
                message  = (
                    "   ✅ No empty labels found. "
                    "All labels have at least one positive sample."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Empty Labels",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete empty labels "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 5.4 — Label Density Checker
# ─────────────────────────────────────────────────────

class LabelDensityChecker(BaseChecker):
    """
    Checks average number of labels per sample.

    Too low density (< 0.02) → most samples have no labels
    → model has nothing meaningful to learn

    Too high density (> 0.8) → almost every label is always
    active → model predicts everything → precision tanks
    """

    def __init__(self, y_train,
                 min_density=0.02, max_density=0.80):
        self.y_train     = y_train
        self.min_density = min_density
        self.max_density = max_density

    def check(self):
        try:
            y       = np.array(self.y_train)
            density = y.mean()
            avg_labels_per_sample = round(
                y.sum(axis=1).mean(), 2
            )

            if density < self.min_density:
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Label Density",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Very Low Label Density!\n"
                        f"      Average labels per sample: "
                        f"{avg_labels_per_sample}\n"
                        f"      Label density: {round(density*100, 2)}%\n"
                        f"      Most samples have no active labels.\n"
                        f"      Model has very little signal to learn from."
                    ),
                    fix_code = (
                        "# Low label density — check your data pipeline\n"
                        "# Verify labels were correctly assigned\n"
                        "# Consider removing samples with zero labels\n\n"
                        "import numpy as np\n"
                        "y_arr    = np.array(y_train)\n"
                        "has_label = y_arr.sum(axis=1) > 0\n"
                        "X_train  = X_train[has_label]\n"
                        "y_train  = y_train[has_label]"
                    )
                )

            if density > self.max_density:
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Label Density",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Very High Label Density!\n"
                        f"      Average labels per sample: "
                        f"{avg_labels_per_sample}\n"
                        f"      Label density: {round(density*100, 2)}%\n"
                        f"      Almost every label is always active.\n"
                        f"      Model precision will be very low."
                    ),
                    fix_code = (
                        "# High label density — review your labeling\n"
                        "# Consider removing redundant labels\n"
                        "# Or splitting into more specific sub-labels"
                    )
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Label Density",
                group    = "classification",
                message  = (
                    f"   ✅ Label density is healthy.\n"
                    f"      Average labels per sample: "
                    f"{avg_labels_per_sample}  "
                    f"Density: {round(density*100, 2)}%"
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Label Density",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete label density "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )


# ─────────────────────────────────────────────────────
# 5.5 — Rare Label Combinations Checker
# ─────────────────────────────────────────────────────

class RareLabelCombinationsChecker(BaseChecker):
    """
    Checks if certain label combinations appear
    very rarely in the dataset.

    Example:
    Label combo [1,0,1,0] appears only once in 1000 samples.
    Model never sees enough examples of this combination
    to learn when it should predict both label 0 AND label 2.
    """

    def __init__(self, y_train, min_combo_count=2):
        self.y_train        = y_train
        self.min_combo_count = min_combo_count

    def check(self):
        try:
            y = np.array(self.y_train)

            # Convert each row to a tuple for counting
            combos        = [tuple(row) for row in y]
            combo_counts  = pd.Series(combos).value_counts()
            rare_combos   = combo_counts[
                combo_counts < self.min_combo_count
            ]
            total_combos  = len(combo_counts)
            rare_count    = len(rare_combos)

            if rare_count > 0:
                pct = round((rare_count / total_combos) * 100, 1)
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Rare Label Combinations",
                    group    = "classification",
                    message  = (
                        f"   ⚠️  WARNING — Rare Label Combinations!\n"
                        f"      {rare_count} of {total_combos} unique "
                        f"label combinations ({pct}%) appear only once.\n"
                        f"      Model may not generalise these "
                        f"combinations well.\n"
                        f"      Consider collecting more data."
                    ),
                    fix_code = (
                        "# Rare label combinations — collect more data\n"
                        "# Or use label powerset with caution\n"
                        "# Or switch to binary relevance approach\n\n"
                        "from sklearn.multioutput import "
                        "MultiOutputClassifier\n"
                        "from sklearn.ensemble import "
                        "RandomForestClassifier\n\n"
                        "# Binary relevance — treats each label independently\n"
                        "model = MultiOutputClassifier(\n"
                        "    RandomForestClassifier("
                        "class_weight='balanced', random_state=42)\n"
                        ")"
                    )
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Rare Label Combinations",
                group    = "classification",
                message  = (
                    f"   ✅ No rare label combinations found.\n"
                    f"      All {total_combos} unique combinations "
                    f"appear at least {self.min_combo_count} times."
                ),
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Rare Label Combinations",
                group    = "classification",
                message  = (
                    f"   ⚠️  Could not complete rare combinations "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )