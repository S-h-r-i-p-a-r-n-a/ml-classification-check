"""
Missing Values Checker
──────────────────────
Checks for columns with too many missing values (NaN).

Two levels of severity:
- Above 50% missing → critical  (column is mostly empty)
- Above 30% missing → warning   (column has concerning NaN rate)

High NaN columns silently corrupt model training
because models handle missing values differently
and inconsistently without explicit instructions.
"""

import pandas as pd
import numpy as np
from ml_classification_check.checks import BaseChecker
from ml_classification_check.fix_suggester import fix_missing_values


class MissingValuesChecker(BaseChecker):

    def __init__(self, X_train, critical_threshold=0.50,
                 warning_threshold=0.30):
        self.X_train            = X_train
        self.critical_threshold = critical_threshold
        self.warning_threshold  = warning_threshold

    def check(self):
        try:
            missing_pct = self.X_train.isnull().mean()

            # Separate into critical and warning columns
            critical_cols = list(
                missing_pct[
                    missing_pct >= self.critical_threshold
                ].index
            )
            warning_cols = list(
                missing_pct[
                    (missing_pct >= self.warning_threshold) &
                    (missing_pct < self.critical_threshold)
                ].index
            )

            all_bad_cols = critical_cols + warning_cols

            # No missing value issues
            if not all_bad_cols:
                return self._result(
                    passed   = True,
                    severity = "ok",
                    check    = "Missing Values",
                    group    = "data_integrity",
                    message  = (
                        "   ✅ No high missing value columns found."
                    ),
                    fix_code = None
                )

            # Build details message
            details = ""
            if critical_cols:
                for col in critical_cols:
                    pct = round(missing_pct[col] * 100, 1)
                    details += (
                        f"\n      → '{col}': {pct}% missing "
                        f"[CRITICAL — drop this column]"
                    )
            if warning_cols:
                for col in warning_cols:
                    pct = round(missing_pct[col] * 100, 1)
                    details += (
                        f"\n      → '{col}': {pct}% missing "
                        f"[WARNING — consider imputing]"
                    )

            # Determine overall severity
            severity = "critical" if critical_cols else "warning"
            icon     = "❌ CRITICAL" if critical_cols else "⚠️  WARNING"

            return self._result(
                passed   = False,
                severity = severity,
                check    = "Missing Values",
                group    = "data_integrity",
                message  = (
                    f"   {icon} — Missing Values Detected!\n"
                    f"      {len(all_bad_cols)} column(s) have high "
                    f"missing value rates:{details}\n"
                    f"      Missing values silently corrupt model training."
                ),
                fix_code = fix_missing_values(
                    critical_cols, warning_cols
                )
            )

        except Exception as e:
            return self._result(
                passed   = False,
                severity = "warning",
                check    = "Missing Values",
                group    = "data_integrity",
                message  = (
                    f"   ⚠️  Could not complete missing values "
                    f"check: {str(e)}"
                ),
                fix_code = None
            )