"""
Train-Test Leakage Checker
──────────────────────────
Checks if any rows from X_train appear in X_test.
This is the most common cause of fake high accuracy
in beginner ML projects.
"""

import pandas as pd
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import fix_train_test_leakage


class LeakageChecker(BaseChecker):

    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test  = X_test

    def check(self):
        try:
            train_hashes = set(
                pd.util.hash_pandas_object(self.X_train, index=False)
            )
            test_hashes = set(
                pd.util.hash_pandas_object(self.X_test, index=False)
            )
            overlap = train_hashes & test_hashes
            count   = len(overlap)

            if count > 0:
                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Train-Test Leakage",
                    group    = "data_integrity",
                    message  = (
                        f"   ❌ CRITICAL — Train-Test Leakage Detected!\n"
                        f"      {count} row(s) appear in both X_train and X_test.\n"
                        f"      Your model already saw test answers during training.\n"
                        f"      Like memorizing exam answers — accuracy is FAKE."
                    ),
                    fix_code = fix_train_test_leakage()
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Train-Test Leakage",
                group    = "data_integrity",
                message  = "   ✅ No train-test leakage detected.",
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Train-Test Leakage",
                group    = "data_integrity",
                message  = f"   ⚠️  Could not complete leakage check: {str(e)}",
                fix_code = None
            )