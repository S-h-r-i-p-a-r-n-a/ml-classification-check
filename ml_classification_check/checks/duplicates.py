"""
Duplicate Rows Checker
──────────────────────
Checks if X_train contains repeated rows.
Duplicate rows make the model memorize specific
samples — inflating accuracy and hurting generalization.
"""

import pandas as pd
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import fix_duplicate_rows


class DuplicateChecker(BaseChecker):

    def __init__(self, X_train):
        self.X_train = X_train

    def check(self):
        try:
            # Fill NaN temporarily for comparison
            # so rows with NaN in same position
            # are correctly identified as duplicates
            temp            = self.X_train.fillna("__NaN__")
            duplicate_count = temp.duplicated().sum()
            total           = len(self.X_train)
            pct             = round((duplicate_count / total) * 100, 2)

            if duplicate_count > 0:
                return self._result(
                    passed   = False,
                    severity = "warning",
                    check    = "Duplicate Rows",
                    group    = "data_integrity",
                    message  = (
                        f"   ⚠️  WARNING — Duplicate Rows Found!\n"
                        f"      {duplicate_count} duplicate row(s) in X_train "
                        f"({pct}% of training data).\n"
                        f"      Model may memorize these samples.\n"
                        f"      This inflates accuracy and hurts real performance."
                    ),
                    fix_code = fix_duplicate_rows()
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Duplicate Rows",
                group    = "data_integrity",
                message  = "   ✅ No duplicate rows found in training set.",
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Duplicate Rows",
                group    = "data_integrity",
                message  = (
                    f"   ⚠️  Could not complete duplicate check: {str(e)}"
                ),
                fix_code = None
            )