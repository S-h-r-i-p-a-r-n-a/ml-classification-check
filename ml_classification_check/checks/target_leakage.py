"""
Target Leakage Checker
──────────────────────
Checks if any feature column is suspiciously
correlated with the target variable y_train.

A feature that is too correlated with the target
likely contains future or leaked information —
meaning the model is essentially cheating.

Example:
Predicting loan default.
Feature 'loan_written_off' correlates 0.99 with target.
Because loans that defaulted were written off — it's
the same information just named differently.
"""

import pandas as pd
import numpy as np
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import fix_target_leakage


class TargetLeakageChecker(BaseChecker):

    def __init__(self, X_train, y_train, threshold=0.95):
        self.X_train   = X_train
        self.y_train   = y_train
        self.threshold = threshold

    def check(self):
        try:
            suspicious = []

            for col in self.X_train.columns:
                try:
                    # Only check numeric columns
                    if pd.api.types.is_numeric_dtype(self.X_train[col]):
                        corr = abs(self.X_train[col].corr(
                            pd.Series(self.y_train.values,
                            index=self.X_train.index)
                        ))
                        if corr >= self.threshold:
                            suspicious.append((col, round(corr, 4)))
                except Exception:
                    continue

            if suspicious:
                details = "\n".join([
                    f"      → '{col}' has {corr} correlation with target"
                    for col, corr in suspicious
                ])
                leaky_cols = [col for col, _ in suspicious]

                return self._result(
                    passed   = False,
                    severity = "critical",
                    check    = "Target Leakage",
                    group    = "data_integrity",
                    message  = (
                        f"   ❌ CRITICAL — Target Leakage Detected!\n"
                        f"      {len(suspicious)} feature(s) suspiciously "
                        f"correlated with target:\n"
                        f"{details}\n"
                        f"      These features contain the answer inside "
                        f"the question.\n"
                        f"      Model is cheating — accuracy is FAKE."
                    ),
                    fix_code = fix_target_leakage(leaky_cols)
                )

            return self._result(
                passed   = True,
                severity = "ok",
                check    = "Target Leakage",
                group    = "data_integrity",
                message  = "   ✅ No target leakage detected.",
                fix_code = None
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Target Leakage",
                group    = "data_integrity",
                message  = (
                    f"   ⚠️  Could not complete target leakage check: {str(e)}"
                ),
                fix_code = None
            )